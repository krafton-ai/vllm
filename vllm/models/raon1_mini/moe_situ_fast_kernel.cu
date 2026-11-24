// RAON_MOE_SITU_FAST: a faster launch of vLLM's SiTU-GLU kernel, same bits.
//
// Source of truth: vllm-project/vllm @ 2cf0a6915 (the 0.28.0 wheel's commit),
// csrc/libtorch_stable/activation_kernels.cu, `situ_and_mul_kernel`:
//
//   grid = num_tokens, block = min(d, 1024), one row per block, one scalar
//   bf16 load of gate and of up per thread per iteration, then
//     gate_out = beta * tanhf(g * inv_beta) / (1.0f + expf(-g));
//     up_out   = clamp_up ? linear_beta * tanhf(u * inv_linear_beta) : u;
//     out      = (scalar_t)(gate_out * up_out);
//
// For RAON (d = 1088) that is a 1024-thread block whose second loop trip runs
// 64 of 1024 threads, 2-byte loads, and one block per row (196,602 blocks for
// a 32K prefill at top-6).  This file keeps the per-element arithmetic
// character for character (same expressions, same operand order, same fp32
// temporaries, same c10 bf16 conversion, compiled WITHOUT --use_fast_math so
// tanhf/expf/the division stay the accurate IEEE realizations) and changes
// only the launch: rows are flattened, each thread owns 8 contiguous
// elements (one 16-byte load of gate, one of up, one 16-byte store), and the
// grid is sized to the element count.  The element function is a pure
// function of (g, u, beta, linear_beta), so the schedule cannot change bits;
// moe_situ_fast.py's self-test proves it over all 2^32 (g, u) bf16 pairs.
//
// Non-vectorizable calls (d % 8 != 0, or a pointer not 16-byte aligned) take
// a flattened scalar kernel with the same element function.  No host syncs,
// no allocation: CUDA-graph capturable.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

namespace raon_situ {

// Character-for-character the upstream loop body (see header).
template <typename scalar_t>
__device__ __forceinline__ scalar_t situ_elem(const scalar_t gs,
                                              const scalar_t us,
                                              const float beta,
                                              const float linear_beta,
                                              const bool clamp_up,
                                              const float inv_beta,
                                              const float inv_linear_beta) {
  const float g = (float)gs;
  const float u = (float)us;
  const float gate_out = beta * tanhf(g * inv_beta) / (1.0f + expf(-g));
  const float up_out =
      clamp_up ? linear_beta * tanhf(u * inv_linear_beta) : u;
  return (scalar_t)(gate_out * up_out);
}

constexpr int kVec = 8;  // 8 x 2-byte elements = one 16-byte access

template <typename scalar_t>
__global__ void situ_and_mul_vec_kernel(
    scalar_t* __restrict__ out,          // [rows, d]
    const scalar_t* __restrict__ input,  // [rows, 2, d]
    const int64_t total_vecs, const int vecs_per_row, const int d,
    const float beta, const float linear_beta) {
  static_assert(sizeof(scalar_t) * kVec == 16, "16-byte vectors");
  const int64_t v = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= total_vecs) return;
  const int64_t row = v / vecs_per_row;
  const int col = (int)(v - row * vecs_per_row) * kVec;
  const scalar_t* gate_ptr = input + row * 2 * d + col;
  const scalar_t* up_ptr = gate_ptr + d;
  scalar_t* out_ptr = out + row * d + col;

  // Same derived constants as upstream, computed the same way on device.
  const bool clamp_up = linear_beta > 0.0f;
  const float inv_beta = 1.0f / beta;
  const float inv_linear_beta = clamp_up ? 1.0f / linear_beta : 0.0f;

  union Pack {
    uint4 u;
    scalar_t e[kVec];
  };
  Pack gp, up, op;
  gp.u = __ldg(reinterpret_cast<const uint4*>(gate_ptr));
  up.u = __ldg(reinterpret_cast<const uint4*>(up_ptr));
#pragma unroll
  for (int j = 0; j < kVec; ++j) {
    op.e[j] = situ_elem<scalar_t>(gp.e[j], up.e[j], beta, linear_beta,
                                  clamp_up, inv_beta, inv_linear_beta);
  }
  *reinterpret_cast<uint4*>(out_ptr) = op.u;
}

template <typename scalar_t>
__global__ void situ_and_mul_scalar_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input,
    const int64_t total, const int d, const float beta,
    const float linear_beta) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t row = i / d;
  const int64_t idx = i - row * d;
  const bool clamp_up = linear_beta > 0.0f;
  const float inv_beta = 1.0f / beta;
  const float inv_linear_beta = clamp_up ? 1.0f / linear_beta : 0.0f;
  const scalar_t* gate_ptr = input + row * 2 * d;
  out[row * d + idx] =
      situ_elem<scalar_t>(__ldg(&gate_ptr[idx]), __ldg(&gate_ptr[d + idx]),
                          beta, linear_beta, clamp_up, inv_beta,
                          inv_linear_beta);
}

template <typename scalar_t>
void launch(torch::Tensor& out, const torch::Tensor& input, double beta,
            double linear_beta) {
  const int d = (int)(input.size(-1) / 2);
  const int64_t rows = input.numel() / input.size(-1);
  if (rows == 0) return;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto* o = reinterpret_cast<scalar_t*>(out.data_ptr());
  const auto* x = reinterpret_cast<const scalar_t*>(input.data_ptr());
  constexpr int kThreads = 256;
  const bool aligned = (reinterpret_cast<uintptr_t>(o) % 16 == 0) &&
                       (reinterpret_cast<uintptr_t>(x) % 16 == 0);
  if (sizeof(scalar_t) == 2 && d % kVec == 0 && aligned) {
    const int vpr = d / kVec;
    const int64_t total = rows * vpr;
    const int64_t blocks = (total + kThreads - 1) / kThreads;
    situ_and_mul_vec_kernel<scalar_t><<<(unsigned)blocks, kThreads, 0, stream>>>(
        o, x, total, vpr, d, (float)beta, (float)linear_beta);
  } else {
    const int64_t total = rows * d;
    const int64_t blocks = (total + kThreads - 1) / kThreads;
    situ_and_mul_scalar_kernel<scalar_t>
        <<<(unsigned)blocks, kThreads, 0, stream>>>(o, x, total, d,
                                                     (float)beta,
                                                     (float)linear_beta);
  }
}

// Same signature and contract as torch.ops._C.situ_and_mul (linear_beta <= 0
// means unset).  Like upstream it ignores strides beyond a contiguous row
// layout, so the caller guarantees contiguity (checked here, loudly).
void situ_and_mul(torch::Tensor& out, const torch::Tensor& input, double beta,
                  double linear_beta) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda(), "raon_situ: CUDA tensors");
  TORCH_CHECK(input.is_contiguous() && out.is_contiguous(),
              "raon_situ: contiguous tensors required");
  TORCH_CHECK(input.scalar_type() == out.scalar_type(), "raon_situ: dtype");
  TORCH_CHECK(out.numel() * 2 == input.numel(), "raon_situ: shape");
  const at::cuda::OptionalCUDAGuard guard(input.device());
  switch (input.scalar_type()) {
    case at::kBFloat16: launch<c10::BFloat16>(out, input, beta, linear_beta); break;
    case at::kHalf: launch<c10::Half>(out, input, beta, linear_beta); break;
    default: TORCH_CHECK(false, "raon_situ: bf16/fp16 only");
  }
}

}  // namespace raon_situ

TORCH_LIBRARY(raon_situ, m) {
  m.def("situ_and_mul(Tensor(a!) out, Tensor input, float beta, float linear_beta) -> ()");
}

TORCH_LIBRARY_IMPL(raon_situ, CUDA, m) {
  m.impl("situ_and_mul", &raon_situ::situ_and_mul);
}

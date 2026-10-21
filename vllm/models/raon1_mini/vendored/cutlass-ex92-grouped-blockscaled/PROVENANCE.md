# vendored/cutlass-ex92-grouped-blockscaled

CuTeDSL block-scaled grouped GEMM from CUTLASS example 92, used for the
swap-AB realization of the MXFP8 W8A8 MoE routed GEMM (weights as the fixed M
dimension, tokens as the ragged N dimension). This avoids padding the token
dimension to the MMA tile at small decode batch sizes.

## Source

| File | Upstream | sha256 |
|---|---|---|
| `grouped_blockscaled_gemm.py` | [NVIDIA/cutlass v4.5.0](https://github.com/NVIDIA/cutlass/tree/v4.5.0) `examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_grouped_gemm/grouped_blockscaled_gemm.py` | `4aebe0e53e6b7b6a99741372d1bfb5406ee5eb96d1b433d2372ad64f86245d7b` |

The file is byte-identical to the CUTLASS v4.5.0 tag. The same file ships in
the flashinfer-python 0.6.16.post3 wheel under
`flashinfer/data/cutlass/examples/python/CuTeDSL/blackwell/grouped_blockscaled_gemm.py`.
The C++ version of the example is
[examples/92_blackwell_moe_gemm](https://github.com/NVIDIA/cutlass/tree/main/examples/92_blackwell_moe_gemm).

License: BSD-3-Clause (see the file header, NVIDIA CORPORATION & AFFILIATES).

## Local modifications

None.

## Usage notes

- Kernel: `Sm100GroupedBlockScaledGemmKernel(sf_vec_size=32,
  mma_tiler_mn=(128,128), cluster_shape_mn=(1,1))`. MXFP8 = A/B
  `Float8E4M3FN`, scale factors `Float8E8M0FNU`, `sf_vec_size` 32.
- Math, per group: `C[m,n] = sum_k (A[m,k]*SFA[m,k//32]) * (B[n,k]*SFB[n,k//32])`
  (both A and B reduce over K). Swap-AB needs no kernel change: A = weights
  `[M=out_channels, K]`, B = tokens `[N=rows, K]`, and `C = [M, N]` is
  produced m-major and consumed as a `C^T` view.
- Per-group shapes, strides and pointers live in device arrays, so only their
  values change between steps (no recompilation; tensormaps are updated
  inside the kernel). With `host_problem_shape_available=False` the grid is
  fixed at `max_active_clusters`, which makes the call CUDA-graph friendly.
- Scale-factor layout: atom `(32,4)x(4)`; rows are stored padded to multiples
  of 128 (the padded region is never consumed).
- Constraints (`can_implement`): MMA tiler M/N in {128, 256}; 16-byte
  alignment only on each tensor's major axis, so with A/B k-major and C
  m-major the token count N needs no padding. L = 1 per group.
- The epilogue is a plain `acc_vec.to(c_dtype)` cast; activation, router
  probabilities and bias are applied separately.

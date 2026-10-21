"""Batch-invariant GEMM for raon-1-mini.

A GEMM whose per-row output is bit-identical regardless of how many rows M are
in the launch. Stock cuBLAS picks its split-K / tile schedule from the problem
shape (M in particular), so the same token's logits round differently at batch
size 1 vs N. This kernel removes that dependency: a
persistent Triton GEMM with a fixed tile schedule and a fixed BLOCK_SIZE_K
K-loop, so the reduction order over K is the same for every row no matter the
grid.

Vendored (matmul only) from Thinking Machines Lab's ``batch_invariant_ops``
(MIT). Proton timing hooks and the ATen-dispatch machinery are dropped;
callers use ``matmul_persistent`` explicitly instead of patching the dispatch
table, so other kernels are unaffected.
"""

from __future__ import annotations

from typing import Any, Callable

import os
import torch

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # CPU-only test envs
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        return pid_m, pid_n

    @triton.jit
    def matmul_kernel_persistent(
        a_ptr, b_ptr, c_ptr, bias_ptr,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
        NUM_SMS: tl.constexpr,
        A_LARGE: tl.constexpr, B_LARGE: tl.constexpr, C_LARGE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        start_pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
        num_tiles = num_pid_m * num_pid_n
        tile_id_c = start_pid - NUM_SMS
        offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n

        for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
            pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
            start_m = pid_m * BLOCK_SIZE_M
            start_n = pid_n * BLOCK_SIZE_N
            offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
            if A_LARGE:
                offs_am = offs_am.to(tl.int64)
            if B_LARGE:
                offs_bn = offs_bn.to(tl.int64)
            offs_am = tl.where(offs_am < M, offs_am, 0)
            offs_bn = tl.where(offs_bn < N, offs_bn, 0)
            offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
            offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for ki in range(k_tiles):
                if A_LARGE or B_LARGE:
                    offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
                else:
                    offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
                b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
                a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
                b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
                accumulator = tl.dot(a, b, accumulator)

            tile_id_c += NUM_SMS
            pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            if C_LARGE:
                offs_cm = offs_cm.to(tl.int64)
                offs_cn = offs_cn.to(tl.int64)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            if HAS_BIAS:
                bias_ptrs = bias_ptr + offs_cn
                bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
                accumulator += bias
            c = accumulator.to(c_ptr.dtype.element_ty)
            tl.store(c_ptrs, c, mask=c_mask)


# Fixed per-dtype tile schedule. Independent of M -> the K-loop that produces a
# given output row is identical at any batch size. Matches the TML config.
_CONFIGS = {
    torch.bfloat16: dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=3, num_warps=8),
    torch.float16: dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=3, num_warps=8),
    torch.float32: dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, num_stages=3, num_warps=8),
}

_NUM_SMS: int | None = None


def _num_sms() -> int:
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    return _NUM_SMS


def matmul_persistent(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Batch-invariant ``a @ b`` (+ optional 1-D bias). Same value per row for any M.

    ``out_dtype`` (default: ``a.dtype``) selects the STORE dtype: the kernel
    always accumulates in fp32 and casts once at the store
    (``accumulator.to(c_ptr.dtype.element_ty)``), so
    ``out_dtype=torch.float32`` emits the fp32 accumulator directly with no
    extra rounding (bf16 in, fp32 out, as used by
    RAON_MOE_ROUTER_GEMM_PIN=realized).

    Falls back to ``torch.addmm``/``torch.mm`` when Triton or CUDA is absent
    (CPU unit tests); that path is not batch-invariance-critical.
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert bias is None or bias.dim() == 1, "bias must be 1-D"
    if out_dtype is None:
        out_dtype = a.dtype
    if not (_HAVE_TRITON and a.is_cuda):
        out = torch.mm(a.to(out_dtype), b.to(out_dtype))
        return out if bias is None else out + bias.to(out_dtype)

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    num_sms = _num_sms()

    def grid(meta):
        return (min(num_sms, triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"])),)

    # RAON_KDA_PROJ_GEMM_TUNE: decode-width launch tuning, default off.
    #
    # Same tuning as vLLM upstream's batch-invariant matmul (PR #53247):
    # BLOCK_SIZE_M and num_warps are output-tiling and scheduling choices, so
    # they do not reorder the K reduction. Any other copy of this kernel whose
    # output must match bit for bit needs the same setting, and parity_contract
    # records the kernel source hash (``kda_proj_gemm_sha``), so it is opt-in.
    config = dict(_CONFIGS[a.dtype])
    if (os.environ.get("RAON_KDA_PROJ_GEMM_TUNE", "0") == "1"
            and a.dtype == torch.bfloat16 and M <= 64):
        config.update(BLOCK_SIZE_M=16, num_warps=4)
    matmul_kernel_persistent[grid](
        a, b, c, bias,
        M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        NUM_SMS=num_sms,
        A_LARGE=a.numel() > 2**31, B_LARGE=b.numel() > 2**31, C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        **config,
    )
    return c

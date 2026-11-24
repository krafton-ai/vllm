# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-K projection for the narrow Raon mHC logits."""

import torch

from vllm.triton_utils import tl, triton

PHI_SPLITS = 20
PHI_BLOCK_K = 64


@triton.jit
def _phi_reduce(part_ptr, out_ptr, R, SPLITS: tl.constexpr, RP: tl.constexpr):
    m = tl.program_id(0)
    rr = tl.arange(0, RP)
    acc = tl.zeros((RP,), dtype=tl.float32)
    for s in range(SPLITS):  # fixed order
        acc += tl.load(part_ptr + (m * SPLITS + s) * RP + rr)
    tl.store(out_ptr + m * R + rr, acc.to(out_ptr.dtype.element_ty), mask=rr < R)


@triton.jit
def _phi_partial_dot(
    v_ptr,
    p_ptr,
    part_ptr,
    T,
    K,
    R,
    SPLITS: tl.constexpr,
    KCH: tl.constexpr,
    BK: tl.constexpr,
    RP: tl.constexpr,
    BMC: tl.constexpr,
):
    s = tl.program_id(0)
    mb = tl.program_id(1)
    rm = mb * BMC + tl.arange(0, BMC)
    mmask = rm < T
    rr = tl.arange(0, RP)
    rmask = rr < R
    acc = tl.zeros((BMC, RP), dtype=tl.float32)
    k0 = s * KCH
    for kb in range(0, KCH, BK):
        rk = k0 + kb + tl.arange(0, BK)
        kmask = rk < K
        v = tl.load(
            v_ptr + rm[:, None] * K + rk[None, :],
            mask=mmask[:, None] & kmask[None, :],
            other=0.0,
        )
        w = tl.load(
            p_ptr + rr[None, :] * K + rk[:, None],
            mask=rmask[None, :] & kmask[:, None],
            other=0.0,
        )
        acc = tl.dot(v, w, acc)
    tl.store(
        part_ptr + (rm[:, None] * SPLITS + s) * RP + rr[None, :],
        acc,
        mask=mmask[:, None],
    )


def split_k_phi_projection(v, phi):
    T, K = v.shape
    R = phi.shape[0]
    v = v.contiguous()
    RP, BMC = 32, 16
    kch = triton.cdiv(triton.cdiv(K, PHI_SPLITS), PHI_BLOCK_K) * PHI_BLOCK_K
    part = torch.empty((T, PHI_SPLITS, RP), dtype=torch.float32, device=v.device)
    out = torch.empty((T, R), dtype=v.dtype, device=v.device)
    _phi_partial_dot[(PHI_SPLITS, triton.cdiv(T, BMC))](
        v,
        phi,
        part,
        T,
        K,
        R,
        SPLITS=PHI_SPLITS,
        KCH=kch,
        BK=PHI_BLOCK_K,
        RP=RP,
        BMC=BMC,
        num_warps=4,
    )
    _phi_reduce[(T,)](part, out, R, SPLITS=PHI_SPLITS, RP=RP, num_warps=1)
    return out

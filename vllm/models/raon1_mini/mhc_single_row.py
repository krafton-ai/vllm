# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels for single-token Raon mHC."""

import torch
from triton.language.extra import libdevice as tld

from vllm.triton_utils import tl, triton

from .mhc_split_k import PHI_BLOCK_K, PHI_SPLITS


@triton.jit
def _dot3(a0, a1, a2, b0, b1, b2, fuse_last: tl.constexpr):
    acc = tld.fma(a1, b1, tld.mul_rn(a0, b0))
    if fuse_last:
        return tld.fma(a2, b2, acc)
    return tld.add_rn(acc, tld.mul_rn(a2, b2))


@triton.jit
def _single_row_statistics(
    v_ptr,
    phi_ptr,
    part_ptr,
    mean_ptr,
    K,
    R,
    FACTOR: tl.constexpr,
    SPLITS: tl.constexpr,
    KCH: tl.constexpr,
    BK: tl.constexpr,
    RP: tl.constexpr,
    BMC: tl.constexpr,
    NTH: tl.constexpr,
    NIT: tl.constexpr,
):
    s = tl.program_id(0)
    if s < SPLITS:
        rm = tl.arange(0, BMC)
        mmask = rm < 1
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
                phi_ptr + rr[None, :] * K + rk[:, None],
                mask=rmask[None, :] & kmask[:, None],
                other=0.0,
            )
            acc = tl.dot(v, w, acc)
        tl.store(
            part_ptr + (rm[:, None] * SPLITS + s) * RP + rr[None, :],
            acc,
            mask=mmask[:, None],
        )
    else:
        t = tl.arange(0, NTH)
        a0 = tl.zeros([NTH], dtype=tl.float32)
        a1 = tl.zeros([NTH], dtype=tl.float32)
        a2 = tl.zeros([NTH], dtype=tl.float32)
        a3 = tl.zeros([NTH], dtype=tl.float32)
        for it in tl.static_range(NIT):
            base = it * NTH * 4 + t * 4
            x0 = tl.load(v_ptr + base + 0).to(tl.float32)
            x1 = tl.load(v_ptr + base + 1).to(tl.float32)
            x2 = tl.load(v_ptr + base + 2).to(tl.float32)
            x3 = tl.load(v_ptr + base + 3).to(tl.float32)
            a0 = tld.add_rn(a0, tld.mul_rn(x0, x0))
            a1 = tld.add_rn(a1, tld.mul_rn(x1, x1))
            a2 = tld.add_rn(a2, tld.mul_rn(x2, x2))
            a3 = tld.add_rn(a3, tld.mul_rn(x3, x3))
        val = tld.add_rn(tld.add_rn(tld.add_rn(a0, a1), a2), a3)
        val = tl.sum(tl.reshape(val, [2, 256]), axis=0)
        val = tl.sum(tl.reshape(val, [2, 128]), axis=0)
        val = tl.sum(tl.reshape(val, [2, 64]), axis=0)
        val = tl.sum(tl.reshape(val, [2, 32]), axis=0)
        val = tl.sum(tl.reshape(val, [2, 16]), axis=0)
        val = tl.sum(tl.reshape(val, [2, 8]), axis=0)
        val = tl.sum(tl.reshape(val, [2, 4]), axis=0)
        val = tl.sum(tl.reshape(val, [2, 2]), axis=0)
        tot = tl.sum(val, axis=0)
        tl.store(mean_ptr, tld.mul_rn(tot, FACTOR))


@triton.jit
def _single_row_head(
    s_ptr,
    part_ptr,
    mean_ptr,
    arow_ptr,
    brow_ptr,
    head_ptr,
    z_ptr,
    agg_ptr,
    NC: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    SPLITS: tl.constexpr,
    BC: tl.constexpr,
    EPS: tl.constexpr,
):
    t = 0
    v_base = s_ptr + t * NC
    r = tl.arange(0, 32)
    rmask = r < R
    acc = tl.zeros([32], dtype=tl.float32)
    for sp in range(SPLITS):
        acc += tl.load(part_ptr + sp * 32 + r)
    acc = tl.where(rmask, acc.to(tl.bfloat16).to(tl.float32), 0.0)
    inv = tld.rsqrt(tl.load(mean_ptr + t) + EPS)
    a = tl.load(arow_ptr + r, mask=rmask, other=0.0)
    b = tl.load(brow_ptr + r, mask=rmask, other=0.0)
    pre = tld.add_rn(tld.mul_rn(a, tld.mul_rn(acc, inv)), b)
    sig = tld.div_rn(tl.full(pre.shape, 1.0, tl.float32), 1.0 + tld.exp(-pre))
    hvec = tl.where(r < N, sig, 2.0 * sig)
    hb = head_ptr + t * (2 * N)
    tl.store(hb + r, hvec, mask=r < 2 * N)
    tl.store(z_ptr + t * (R - 2 * N) + (r - 2 * N), pre, mask=rmask & (r >= 2 * N))
    tl.debug_barrier()
    h0 = tl.load(hb + 0)
    h1 = tl.load(hb + 1)
    h2 = tl.load(hb + 2)
    h3 = tl.load(hb + 3)
    for c0 in range(0, C, BC):
        cols = c0 + tl.arange(0, BC)
        s0 = tl.load(v_base + 0 * C + cols).to(tl.float32)
        s1 = tl.load(v_base + 1 * C + cols).to(tl.float32)
        s2 = tl.load(v_base + 2 * C + cols).to(tl.float32)
        s3 = tl.load(v_base + 3 * C + cols).to(tl.float32)
        aggv = tld.fma(h3, s3, tld.fma(h2, s2, tld.fma(h1, s1, tld.mul_rn(h0, s0))))
        tl.store(agg_ptr + t * C + cols, aggv.to(agg_ptr.dtype.element_ty))


@triton.jit
def _single_row_mix(
    s_ptr,
    y_ptr,
    head_ptr,
    z_ptr,
    q_ptr,
    h0_ptr,
    negh0_ptr,
    geye_ptr,
    ggate_ptr,
    out_ptr,
    C: tl.constexpr,
    N: tl.constexpr,
    BC: tl.constexpr,
):
    t = tl.program_id(0)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    i4 = tl.arange(0, 4)
    j4 = tl.arange(0, 4)
    # Compute the two length-three contractions of Q @ Z @ Q.T.
    zb = z_ptr + t * 9
    q0 = tl.load(q_ptr + i4 * 3 + 0)
    q1 = tl.load(q_ptr + i4 * 3 + 1)
    q2 = tl.load(q_ptr + i4 * 3 + 2)
    z00 = tl.load(zb + 0)
    z01 = tl.load(zb + 1)
    z02 = tl.load(zb + 2)
    z10 = tl.load(zb + 3)
    z11 = tl.load(zb + 4)
    z12 = tl.load(zb + 5)
    z20 = tl.load(zb + 6)
    z21 = tl.load(zb + 7)
    z22 = tl.load(zb + 8)
    t1_0 = _dot3(
        q0,
        q1,
        q2,
        z00 + tl.zeros_like(q0),
        z10 + tl.zeros_like(q0),
        z20 + tl.zeros_like(q0),
        fuse_last=False,
    )
    t1_1 = _dot3(
        q0,
        q1,
        q2,
        z01 + tl.zeros_like(q0),
        z11 + tl.zeros_like(q0),
        z21 + tl.zeros_like(q0),
        fuse_last=False,
    )
    t1_2 = _dot3(
        q0,
        q1,
        q2,
        z02 + tl.zeros_like(q0),
        z12 + tl.zeros_like(q0),
        z22 + tl.zeros_like(q0),
        fuse_last=False,
    )
    qj0 = tl.load(q_ptr + j4 * 3 + 0)
    qj1 = tl.load(q_ptr + j4 * 3 + 1)
    qj2 = tl.load(q_ptr + j4 * 3 + 2)
    e = _dot3(
        t1_0[:, None] + tl.zeros([4, 4], dtype=tl.float32),
        t1_1[:, None] + tl.zeros([4, 4], dtype=tl.float32),
        t1_2[:, None] + tl.zeros([4, 4], dtype=tl.float32),
        qj0[None, :] + tl.zeros([4, 4], dtype=tl.float32),
        qj1[None, :] + tl.zeros([4, 4], dtype=tl.float32),
        qj2[None, :] + tl.zeros([4, 4], dtype=tl.float32),
        fuse_last=True,
    )
    negh0 = tl.load(negh0_ptr + i4[:, None] * N + j4[None, :])
    rho = tl.maximum(tl.max(tl.max(tld.div_rn(e, negh0), axis=1), axis=0), 0.0)
    h0 = tl.load(h0_ptr + i4[:, None] * N + j4[None, :])
    phim = h0 + tld.div_rn(e, tl.full(e.shape, 1.0, tl.float32) + rho)
    geye = tl.load(geye_ptr + i4[:, None] * N + j4[None, :])
    g = tl.load(ggate_ptr)
    hres = tld.add_rn(geye, tld.mul_rn(g, phim))
    hb = head_ptr + t * (2 * N)
    hpost = tl.load(hb + N + i4)
    y = tl.load(y_ptr + t * C + cols).to(tl.float32)
    h0r = tl.sum(tl.where(j4 == 0, hres, 0.0), axis=1)
    h1r = tl.sum(tl.where(j4 == 1, hres, 0.0), axis=1)
    h2r = tl.sum(tl.where(j4 == 2, hres, 0.0), axis=1)
    h3r = tl.sum(tl.where(j4 == 3, hres, 0.0), axis=1)
    s0 = tl.load(s_ptr + t * N * C + 0 * C + cols).to(tl.float32)
    s1 = tl.load(s_ptr + t * N * C + 1 * C + cols).to(tl.float32)
    s2 = tl.load(s_ptr + t * N * C + 2 * C + cols).to(tl.float32)
    s3 = tl.load(s_ptr + t * N * C + 3 * C + cols).to(tl.float32)
    acc = tld.fma(
        h3r[:, None],
        s3[None, :],
        tld.fma(
            h2r[:, None],
            s2[None, :],
            tld.fma(h1r[:, None], s1[None, :], tld.mul_rn(h0r[:, None], s0[None, :])),
        ),
    )
    o = tld.add_rn(acc, tld.mul_rn(hpost[:, None], y[None, :]))
    tl.store(
        out_ptr + t * N * C + i4[:, None] * C + cols[None, :],
        o.to(out_ptr.dtype.element_ty),
    )


def fused_single_row_mhc(site, s, branch, epsilon):
    _, streams, hidden = s.shape
    features = streams * hidden
    logits = 2 * streams + (streams - 1) ** 2
    row = s.view(1, features)
    chunk = triton.cdiv(triton.cdiv(features, PHI_SPLITS), PHI_BLOCK_K)
    chunk *= PHI_BLOCK_K
    partials = torch.empty((1, PHI_SPLITS, 32), dtype=torch.float32, device=s.device)
    mean = torch.empty(1, dtype=torch.float32, device=s.device)
    _single_row_statistics[(PHI_SPLITS + 1,)](
        row,
        site.phi,
        partials,
        mean,
        features,
        logits,
        FACTOR=1.0 / features,
        SPLITS=PHI_SPLITS,
        KCH=chunk,
        BK=PHI_BLOCK_K,
        RP=32,
        BMC=16,
        NTH=512,
        NIT=features // 2048,
        num_warps=4,
    )
    head = torch.empty((1, 2 * streams), dtype=torch.float32, device=s.device)
    z = torch.empty((1, streams - 1, streams - 1), dtype=torch.float32, device=s.device)
    aggregate = torch.empty((1, hidden), dtype=s.dtype, device=s.device)
    _single_row_head[(1,)](
        s,
        partials,
        mean,
        site.A_row,
        site.B_row,
        head,
        z,
        aggregate,
        NC=features,
        C=hidden,
        N=streams,
        R=logits,
        SPLITS=PHI_SPLITS,
        BC=512,
        EPS=epsilon,
        num_warps=4,
    )
    y = branch(aggregate).contiguous()
    out = torch.empty_like(s)
    _single_row_mix[(1, hidden // 512)](
        s,
        y,
        head,
        z,
        site.Q,
        site.H0,
        site.negH0,
        site.g_eye,
        site.g_gate,
        out,
        C=hidden,
        N=streams,
        BC=512,
        num_warps=4,
    )
    return out

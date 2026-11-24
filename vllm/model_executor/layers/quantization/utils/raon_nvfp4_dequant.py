# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact BF16 dequantization of a Marlin-packed NVFP4 dense weight into a
scratch buffer (Raon-1 mini on SM103).

The Marlin-resident-BF16 path keeps a permanent BF16 copy of every dense
weight. This module produces the same values on demand from the Marlin
layout (no checkpoint copy, no resident copy): the tile tables are derived
from ``gptq_marlin_repack`` itself, and the scale decode follows the
S0E5M3 format written by ``nvfp4_marlin_process_scales``. The output layout
is ``[size_k, size_n]``, identical to the resident copy, so the following
``torch.matmul`` is the same cuBLAS call.
"""

import functools
from typing import Any, cast

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _e2m1(n):
    e = (n & 7) >> 1
    mm = (n & 1).to(tl.float32)
    mag = tl.where(e == 0, mm * 0.5, (1.0 + 0.5 * mm) * tl.exp2((e - 1).to(tl.float32)))
    return tl.where((n & 8) != 0, -mag, mag)


@triton.jit
def _dequant_dense_t(
    w_ptr,
    src_ptr,
    sloc_ptr,
    sc_ptr,
    g_ptr,
    o_ptr,
    n_pad,
    num_n_tiles,
    n_real,
    k_real,
    TPP: tl.constexpr,
):
    r = tl.program_id(0)
    jg = tl.program_id(1)
    lane = tl.arange(0, 128 * TPP)
    jj = jg * TPP + lane // 128
    li = lane % 128
    nl = li // 2
    half = li % 2
    m = jj < num_n_tiles
    g = tl.load(g_ptr).to(tl.float32) * 1.1754943508222875e-38
    b = tl.load(
        sc_ptr + r * n_pad + jj * 64 + tl.load(sloc_ptr + nl), mask=m, other=0
    ).to(tl.int32)
    v = (
        tl.where(
            b == 0,
            0.0,
            (1.0 + (b & 7) * 0.125) * tl.exp2(((b >> 3) - 15).to(tl.float32)),
        )
        * g
    )
    jv = tl.arange(0, 8)
    src = tl.load(src_ptr + (nl * 16 + half * 8)[:, None] + jv[None, :])
    wbase = w_ptr + r * (2 * n_pad) + jj * 128
    word = tl.load(wbase[:, None] + (src >> 3), mask=m[:, None], other=0)
    nib = (word >> ((src & 7) * 4)) & 0xF
    val = _e2m1(nib) * v[:, None]
    n = (jj * 64 + nl).to(tl.int64)
    k = (r * 16 + half * 8)[:, None] + jv[None, :]
    keep = m[:, None] & (n[:, None] < n_real) & (k < k_real)
    tl.store(
        o_ptr + k.to(tl.int64) * n_real + n[:, None], val.to(tl.bfloat16), mask=keep
    )


@triton.jit
def _dequant_dense_t2(
    w_ptr,
    src_ptr,
    sloc_ptr,
    sc_ptr,
    g_ptr,
    o_ptr,
    n_pad,
    num_n_tiles,
    n_real,
    k_real,
    TPP: tl.constexpr,
):
    """Same values as _dequant_dense_t; each program transposes its
    TPP x (64 n x 16 k) tiles in registers so the [k, n] stores run along n
    (coalesced rows) instead of one element per row."""
    r = tl.program_id(0)
    jg = tl.program_id(1)
    lane = tl.arange(0, 128 * TPP)
    jj = jg * TPP + lane // 128
    li = lane % 128
    nl = li // 2
    half = li % 2
    m = jj < num_n_tiles
    g = tl.load(g_ptr).to(tl.float32) * 1.1754943508222875e-38
    b = tl.load(
        sc_ptr + r * n_pad + jj * 64 + tl.load(sloc_ptr + nl), mask=m, other=0
    ).to(tl.int32)
    v = (
        tl.where(
            b == 0,
            0.0,
            (1.0 + (b & 7) * 0.125) * tl.exp2(((b >> 3) - 15).to(tl.float32)),
        )
        * g
    )
    jv = tl.arange(0, 8)
    src = tl.load(src_ptr + (nl * 16 + half * 8)[:, None] + jv[None, :])
    wbase = w_ptr + r * (2 * n_pad) + jj * 128
    word = tl.load(wbase[:, None] + (src >> 3), mask=m[:, None], other=0)
    nib = (word >> ((src & 7) * 4)) & 0xF
    # [TPP*128, 8]: lane = (tile, n, half), 8 k
    val = (_e2m1(nib) * v[:, None]).to(tl.bfloat16)
    # [16 k, TPP*64 n]
    val = tl.trans(val.reshape(TPP * 64, 16))
    n = jg * (TPP * 64) + tl.arange(0, TPP * 64)
    k = r * 16 + tl.arange(0, 16)
    keep = (k[:, None] < k_real) & (n[None, :] < n_real)
    tl.store(o_ptr + k.to(tl.int64)[:, None] * n_real + n[None, :], val, mask=keep)


# The maps repeat in every 64 x 16 tile, so they are derived once per device
# on a small padded shape with several tiles in each direction. Deriving them
# on a full layer shape needs index tensors as large as the weight itself.
_TABLE_SHAPE = (256, 256)


@functools.cache
def _tile_tables(device_index: int):
    """Tile-invariant source/scale maps of the Marlin NVFP4 layout, derived
    by repacking bit-planes of the linear element index."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
    )

    dev = torch.device("cuda", device_index)
    n_pad, k_pad = _TABLE_SHAPE
    lin = (
        torch.arange(n_pad, device=dev)[:, None] * k_pad
        + torch.arange(k_pad, device=dev)[None, :]
    )
    perm = torch.empty(0, dtype=torch.int, device=dev)
    idx = None
    for j in range(int(n_pad * k_pad - 1).bit_length()):
        bits = ((lin >> j) & 1).to(torch.uint8)
        packed = (bits[:, 0::2] | (bits[:, 1::2] << 4)).contiguous()
        mw = ops.gptq_marlin_repack(
            b_q_weight=packed.view(torch.int32).T.contiguous(),
            perm=perm,
            size_k=k_pad,
            size_n=n_pad,
            num_bits=4,
            is_a_8bit=False,
        ).reshape(-1)
        plane = (
            mw.unsqueeze(-1) >> (4 * torch.arange(8, device=dev, dtype=torch.int32))
        ) & 0xF
        idx = (
            plane.to(torch.int64) << j
            if idx is None
            else idx | (plane.to(torch.int64) << j)
        )
    kg, nt = k_pad // 16, n_pad // 64
    assert idx is not None
    t4 = idx.contiguous().reshape(kg, nt, 128, 8)
    n4, k4 = t4 // k_pad, t4 % k_pad
    nloc = n4 - (torch.arange(nt, device=dev) * 64).view(1, nt, 1, 1)
    kloc = k4 - (torch.arange(kg, device=dev) * 16).view(kg, 1, 1, 1)
    if not (
        bool((nloc == nloc[0:1, 0:1]).all()) and bool((kloc == kloc[0:1, 0:1]).all())
    ):
        raise RuntimeError("Marlin NVFP4 tile table is not tile-invariant")
    loc = (nloc[0, 0] * 16 + kloc[0, 0]).to(torch.int64).reshape(-1)
    src = torch.empty(1024, dtype=torch.int32, device=dev)
    src[loc] = torch.arange(1024, device=dev, dtype=torch.int32)
    s0 = torch.arange(n_pad * kg, device=dev, dtype=torch.int64).reshape(n_pad, kg).T
    ps = marlin_permute_scales(
        s=s0, size_k=k_pad, size_n=n_pad, group_size=16, is_a_8bit=False
    )
    ps = ps.reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(ps.size(0), -1)
    flat = ps.reshape(-1)
    inv = torch.empty(n_pad * kg, dtype=torch.int64, device=dev)
    inv[flat] = torch.arange(flat.numel(), device=dev)
    sp = inv.reshape(n_pad, kg)
    row, col = sp // n_pad, sp % n_pad
    if not bool((row == torch.arange(kg, device=dev).view(1, kg)).all()):
        raise RuntimeError("Marlin NVFP4 scale row is not the k-group")
    sloc = col[0:64, 0].to(torch.int32).contiguous()
    return src, sloc


def prepare(layer: torch.nn.Module) -> None:
    """Attach the tile tables to a Marlin-prepared NVFP4 dense layer."""
    w = cast(torch.Tensor, layer.weight)
    k_pad = w.shape[0] * 16
    n_pad = w.shape[1] // 2
    cast(Any, layer)._raon_dq_shape = (n_pad, k_pad)
    if n_pad % 64 or k_pad % 16:
        raise RuntimeError(
            f"Marlin NVFP4 shape {n_pad} x {k_pad} is not a whole number of tiles"
        )
    layer._raon_dq_src, layer._raon_dq_sloc = _tile_tables(w.device.index)


def dequantize_into(
    layer: torch.nn.Module, out: torch.Tensor, version: int = 2, tpp: int = 2
) -> torch.Tensor:
    """Write the layer's effective BF16 weight, [size_k, size_n], into out."""
    raon_layer = cast(Any, layer)
    n_pad, k_pad = raon_layer._raon_dq_shape
    k_real = raon_layer.input_size_per_partition
    n_real = raon_layer.output_size_per_partition
    nt = n_pad // 64
    kernel = _dequant_dense_t2 if version == 2 else _dequant_dense_t
    kernel[(k_pad // 16, triton.cdiv(nt, tpp))](
        raon_layer.weight.view(torch.int32),
        layer._raon_dq_src,
        layer._raon_dq_sloc,
        raon_layer.weight_scale.view(torch.uint8),
        layer.weight_global_scale,
        out,
        n_pad,
        nt,
        n_real,
        k_real,
        TPP=tpp,
        num_warps=4,
    )
    return out

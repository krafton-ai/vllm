# RAON_KDA_PREFILL_FRONT_FUSED=1: KDA prefill front (causal conv1d + SiLU per
# q/k/v section, and the RDA v scale) in one Triton kernel per section.
#
# WHAT: the stock prefill front is 3x vLLM `causal_conv1d_fn` (one per q/k/v
# section; `_causal_conv1d_fwd_kernel`) + 1x `rda_scale_v_` (RAON_RDA_FUSED=1,
# `_rda_scale_v_kernel`) that re-reads and re-writes the conv'd v block.  Here
#   * the conv is re-tiled: each program loads its [BLOCK_M=8 tokens, NH
#     heads, head_dim] tile with one 3D load per tap (the stock kernel walks
#     the 8 tokens with a serial loop of 256-wide 1D loads); default
#     BLOCK_N = 256 features, 4 warps;
#   * the v launch applies the RDA ratio (computed once per token x head) in
#     its epilogue, so the separate v read+write pass disappears.
# On B300 (one layer, T=14208, H=40, D=128): stock 3x conv + rda = 755 us,
# fused 3 launches = 646 us; T=1024: 158 -> 109 us.
#
# BIT CONTRACT (bitwise equal to the stock path):
#   conv  : per output element acc = 0.0 (fp32); acc += bf16(x_j * w_j) for
#           taps j = 0..W-1 in the stock order; silu = acc / (1 + exp(-acc));
#           history taps come from the conv state (chunk_offset == 0, if
#           has_initial_state) or zeros, else from x -- all verbatim from
#           `_causal_conv1d_fwd_kernel` (vLLM 0.28.0).
#   rda   : y = bf16(conv_out) (the stock store rounding), then
#           bf16(fp32(y) * ratio) with ratio = exp(logsigmoid(b + d) -
#           logsigmoid(b)) written exactly as `_rda_scale_v_kernel`.
#   state : the conv state write (non prefix-caching path, the only one RAON
#           uses) is the stock chunk_offset == 0 logic, moved to the end of the
#           program so the state history it consumed is already in registers.
#           Same slot, same layout, same values -> the decode hand-off contract
#           is unchanged.
# Program -> (sequence, 8-token chunk) mapping reuses the metadata's
# nums_dict[8] / batch_ptr / token_chunk_offset_ptr (BLOCK_M is pinned to 8).
import os

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice as tld

from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

_BLOCK_M = 8  # must match compute_causal_conv1d_metadata's BLOCK_M list


@triton.jit
def _kda_front_conv_kernel(
    x_ptr,  # [T, dim] row-strided, feature stride 1
    w_ptr,  # [dim, W]
    conv_states_ptr,  # [N, dim, >= W-1]
    cache_indices_ptr,
    has_initial_states_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    o_ptr,  # [T, dim] feature stride 1
    beta_ptr,
    delta_ptr,
    dim,
    num_cache_lines,
    stride_x_token: tl.int64,
    stride_w_dim,
    stride_w_width,
    stride_cs_seq,
    stride_cs_dim,
    stride_cs_tok,
    stride_cache_indices,
    stride_o_token: tl.int64,
    stride_b_t,
    stride_b_h,
    stride_d_t,
    stride_d_h,
    pad_slot_id: tl.constexpr,
    null_block_id: tl.constexpr,
    HAS_NULL_BLOCK: tl.constexpr,
    RDA: tl.constexpr,
    HEAD_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SUB_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Tiles are [BLOCK_M tokens, NH heads, HEAD_D] (BLOCK_N = NH * HEAD_D
    # features), so the RDA ratio is computed once per (token, head) and
    # broadcast over HEAD_D.  Feature f = h * HEAD_D + d.
    KW: tl.constexpr = 4
    state_len: tl.constexpr = KW - 1
    NH: tl.constexpr = BLOCK_N // HEAD_D
    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))
    offs_h = tl.program_id(1) * NH + tl.arange(0, NH)
    offs_d = tl.arange(0, HEAD_D)
    idx_feats = (offs_h[:, None] * HEAD_D + offs_d[None, :])[None, :, :]  # [1, NH, D]
    if idx_seq == pad_slot_id:
        return
    seq_start = tl.load(query_start_loc_ptr + idx_seq)
    seq_end = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = seq_end - seq_start
    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    coord = tl.load(cache_indices_ptr + idx_seq * stride_cache_indices).to(tl.int64)
    if HAS_NULL_BLOCK:
        if coord == null_block_id:
            return
    mask_f = idx_feats < dim
    load_init = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
    cs_base = conv_states_ptr + coord * stride_cs_seq + idx_feats * stride_cs_dim
    x_seq = x_ptr + seq_start.to(tl.int64) * stride_x_token + idx_feats

    w_base = w_ptr + idx_feats * stride_w_dim
    w0 = tl.load(w_base + 0 * stride_w_width, mask=mask_f, other=0.0)
    w1 = tl.load(w_base + 1 * stride_w_width, mask=mask_f, other=0.0)
    w2 = tl.load(w_base + 2 * stride_w_width, mask=mask_f, other=0.0)
    w3 = tl.load(w_base + 3 * stride_w_width, mask=mask_f, other=0.0)
    for s0 in tl.static_range(0, BLOCK_M, SUB_M):
        rows = s0 + tl.arange(0, SUB_M)
        rows3 = rows[:, None, None]
        row_ok = rows3 < segment_len
        acc = tl.zeros((SUB_M, NH, HEAD_D), dtype=tl.float32)
        for j in tl.static_range(KW):
            # tap j reads token u = token_offset + r - (KW - 1 - j)
            u = token_offset + rows3 - (KW - 1 - j)
            mx = (u >= 0) & row_ok & mask_f
            xv = tl.load(x_seq + u.to(tl.int64) * stride_x_token, mask=mx, other=0.0)
            if j < KW - 1:
                # history before the sequence start: conv state (stock col0..col2)
                ms = (u < 0) & mask_f & load_init
                sv = tl.load(cs_base + (state_len + u) * stride_cs_tok, mask=ms, other=0.0)
                xv = tl.where(u < 0, sv, xv)
            if j == 0:
                acc += xv * w0
            elif j == 1:
                acc += xv * w1
            elif j == 2:
                acc += xv * w2
            else:
                acc += xv * w3
        acc = acc / (1 + tl.exp(-acc))
        y = acc.to(o_ptr.dtype.element_ty)
        tok = seq_start + token_offset + rows
        mo = row_ok & mask_f
        if RDA:
            mh = (rows < segment_len)[:, None] & (offs_h < dim // HEAD_D)[None, :]
            b = tl.load(beta_ptr + (tok * stride_b_t)[:, None] + (offs_h * stride_b_h)[None, :],
                        mask=mh, other=0.0).to(tl.float32)
            d = tl.load(delta_ptr + (tok * stride_d_t)[:, None] + (offs_h * stride_d_h)[None, :],
                        mask=mh, other=0.0).to(tl.float32)
            xs = b + d
            ls_bd = tl.minimum(xs, 0.0) - tld.log1p(tl.exp(-tl.abs(xs)))
            ls_b = tl.minimum(b, 0.0) - tld.log1p(tl.exp(-tl.abs(b)))
            ratio = tl.exp(ls_bd - ls_b)  # [SUB_M, NH]
            y = (y.to(tl.float32) * ratio[:, :, None]).to(o_ptr.dtype.element_ty)
        tl.store(o_ptr + (tok.to(tl.int64) * stride_o_token)[:, None, None] + idx_feats,
                 y, mask=mo)

    # conv state update (stock chunk_offset == 0 branch, non-APC).
    if chunk_offset == 0:
        idx_c = tl.arange(0, 4)[:, None, None]  # NP2_STATELEN for state_len 3
        tl.debug_barrier()
        if state_len <= seqlen:
            it = (seqlen - state_len) + idx_c
            cmx = (it >= 0) & (it < seqlen) & mask_f
            cnv = tl.load(x_seq + it.to(tl.int64) * stride_x_token, mask=cmx, other=0.0)
        else:
            VAL = state_len - seqlen
            cmx = ((idx_c - VAL) >= 0) & ((idx_c - VAL) < seqlen) & mask_f
            cnv = tl.load(x_seq + (idx_c - VAL).to(tl.int64) * stride_x_token,
                          mask=cmx, other=0.0)
            if load_init:
                cms = (coord < num_cache_lines) & ((idx_c + seqlen) < state_len) & mask_f
                csv = tl.load(cs_base + (idx_c + seqlen) * stride_cs_tok, mask=cms, other=0.0)
                tl.debug_barrier()
                cnv = tl.where(cms, csv, cnv)
        tl.debug_barrier()
        mw = (idx_c < state_len) & mask_f
        tl.store(cs_base + idx_c * stride_cs_tok, cnv, mask=mw)


# RAON_KDA_PREFILL_FRONT_BM: tokens per program (multiple of 8, default 8).
# 8 reuses the metadata's own program map; larger values build a coarser map
# from nums_dict[8]["nums"] on the host (ceil(ceil(L/8)/g) == ceil(L/(8g))),
# once per metadata object, H2D without a sync.  Values are unaffected (the
# arithmetic per output element does not depend on the tiling).
_DEFAULT_BLOCK_M = int(os.environ.get("RAON_KDA_PREFILL_FRONT_BM", "8"))
_MAP_CACHE = {}


def _program_map(metadata, block_m):
    nd = metadata.nums_dict[_BLOCK_M]
    if block_m == _BLOCK_M:
        bp = nd["batch_ptr"] if nd.get("batch_ptr") is not None else metadata.batch_ptr
        cp = (nd["token_chunk_offset_ptr"]
              if nd.get("token_chunk_offset_ptr") is not None
              else metadata.token_chunk_offset_ptr)
        return bp, cp, nd["tot"]
    assert block_m % _BLOCK_M == 0
    nums8 = nd["nums"]
    hit = _MAP_CACHE.get(block_m)
    if hit is not None and hit[0] is nums8:
        return hit[1:]
    g = block_m // _BLOCK_M
    nums = (-(-nums8.numpy().astype(np.int64) // g))
    mlist = np.repeat(np.arange(len(nums), dtype=np.int32), nums)
    offs = (np.arange(len(mlist), dtype=np.int32)
            - np.repeat((np.cumsum(nums) - nums).astype(np.int32), nums))
    dev = metadata.batch_ptr.device if metadata.batch_ptr is not None else nd["batch_ptr"].device
    host = torch.from_numpy(np.stack([mlist, offs])).pin_memory()
    d = host.to(dev, non_blocking=True)
    _MAP_CACHE[block_m] = (nums8, d[0], d[1], int(len(mlist)))
    return d[0], d[1], int(len(mlist))


def kda_front_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    has_initial_state: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    metadata,
    beta: torch.Tensor | None = None,
    delta: torch.Tensor | None = None,
    head_dim: int = 128,
    block_n: int = 256,
    num_warps: int = 4,
    block_m: int | None = None,
    sub_m: int = 8,
) -> torch.Tensor:
    """conv1d+SiLU of a [T, dim] section (+ RDA v scale if beta/delta given).

    Drop-in for ``causal_conv1d_fn(x.T, weight, None, activation="silu", ...).T``
    (followed by ``rda_scale_v_(out.view(T, H, D), beta, delta)`` when RDA).
    Returns a dense [T, dim] tensor; updates ``conv_states`` in place.
    """
    T, dim = x.shape
    assert x.stride(1) == 1 and weight.shape == (dim, 4)
    assert dim % head_dim == 0 and block_n % head_dim == 0
    assert x.dtype == conv_states.dtype
    assert conv_states.shape[1] == dim and conv_states.shape[2] >= 3
    out = torch.empty((T, dim), dtype=x.dtype, device=x.device)
    if block_m is None:
        block_m = _DEFAULT_BLOCK_M
    batch_ptr, chunk_ptr, tot = _program_map(metadata, block_m)
    rda = beta is not None
    if rda:
        assert delta is not None and beta.shape == delta.shape == (T, dim // head_dim)
        b_st, b_sh = beta.stride()
        d_st, d_sh = delta.stride()
    else:
        beta = delta = x
        b_st = b_sh = d_st = d_sh = 0
    grid = (tot, triton.cdiv(dim, block_n))
    _kda_front_conv_kernel[grid](
        x, weight, conv_states, cache_indices, has_initial_state,
        query_start_loc, batch_ptr, chunk_ptr, out, beta, delta,
        dim, conv_states.size(0),
        x.stride(0), weight.stride(0), weight.stride(1),
        conv_states.stride(0), conv_states.stride(1), conv_states.stride(2),
        cache_indices.stride(0), out.stride(0),
        b_st, b_sh, d_st, d_sh,
        pad_slot_id=PAD_SLOT_ID,
        null_block_id=NULL_BLOCK_ID,
        HAS_NULL_BLOCK=NULL_BLOCK_ID is not None,
        RDA=rda,
        HEAD_D=head_dim,
        BLOCK_M=block_m,
        SUB_M=min(sub_m, block_m),
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out


_USED = {"calls": 0}


def note_used() -> None:
    """Announce the first prefill that takes this path, then every 2**k-th
    call (one call = one KDA layer prefill)."""
    _USED["calls"] += 1
    n = _USED["calls"]
    if n & (n - 1) == 0:
        print(f"[raon-kda] PREFILL_FRONT_FUSED active: layer-prefill calls={n}",
              flush=True)

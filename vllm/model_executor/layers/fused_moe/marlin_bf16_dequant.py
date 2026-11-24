# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in BF16 grouped-GEMM MoE for large Raon prefill batches.

For batches of at least ``VLLM_RAON_MOE_DEQUANT_BF16_MIN_TOKENS`` tokens (or the
default threshold when ``VLLM_RAON_OPTIMIZATIONS`` selects it) the
NVFP4 Marlin expert GEMMs are replaced by a layer-local dequantization of the
Marlin-packed expert weights into two shared BF16 workspace slots followed by
two BF16 grouped GEMMs. The next layer's dequantization is issued on a side
CUDA stream so it overlaps the current layer's compute. There is no resident
BF16 copy of the experts: the two slots are reused by every layer.

The slots are allocated while the model weights are post-processed, before the
KV cache is sized, so memory profiling accounts for them. Every other case
(small batches, CUDA graph capture, expert parallelism, LoRA, biases, router
weight on input, non-BF16 activations, failed self-check) uses Marlin.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers import raon_optimizations
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

logger = init_logger(__name__)


def _resolve_min_tokens() -> int:
    min_tokens = envs.VLLM_RAON_MOE_DEQUANT_BF16_MIN_TOKENS
    if (
        min_tokens == 0
        and current_platform.is_cuda()
        and raon_optimizations.enabled("raon_moe_dequant_bf16", 0)
    ):
        min_tokens = raon_optimizations.MOE_DEQUANT_BF16_MIN_TOKENS
    return min_tokens


# Read once at import (the device profile is evaluated for device 0); 0
# disables the path entirely.
MIN_TOKENS: int = _resolve_min_tokens()

# Relative L2 error allowed between the BF16 chain and Marlin on the
# first-use self-check.
_SELF_CHECK_TOL = 2e-2
_SELF_CHECK_TOKENS = 512
_GROUP_SIZE = 16
_TILE_N = 64
_TPP = 4  # tiles handled per program


if HAS_TRITON:

    @triton.jit
    def _e2m1(n):
        e = (n & 7) >> 1
        m = n & 1
        mag = tl.where(
            e == 0,
            m * 0.5,
            (1.0 + 0.5 * m) * tl.exp2((e - 1).to(tl.float32)),
        )
        return tl.where((n & 8) != 0, -mag, mag)

    @triton.jit
    def _dequant_marlin_nvfp4(
        w_ptr,
        src_ptr,
        sloc_ptr,
        sc_ptr,
        g_ptr,
        o_ptr,
        NW,
        NS,
        N,
        K,
        NT,
        TPP: tl.constexpr,
    ):
        """Marlin NVFP4 experts -> BF16 [E, N, K] (out features major)."""
        e = tl.program_id(0).to(tl.int64)
        r = tl.program_id(1)
        jg = tl.program_id(2)
        lane = tl.arange(0, 128 * TPP)
        jj = jg * TPP + lane // 128
        li = lane % 128
        nl = li // 2
        half = li % 2
        m = jj < NT
        g = tl.load(g_ptr + e).to(tl.float32) * 1.1754943508222875e-38
        b = tl.load(
            sc_ptr + e * NS + r * N + jj * 64 + tl.load(sloc_ptr + nl),
            mask=m,
            other=0,
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
        wbase = w_ptr + e * NW + r * (2 * N) + jj * 128
        word = tl.load(wbase[:, None] + (src >> 3), mask=m[:, None], other=0)
        nib = (word >> ((src & 7) * 4)) & 0xF
        val = _e2m1(nib) * v[:, None]
        optr = (
            o_ptr
            + e * N * K
            + (jj * 64 + nl).to(tl.int64)[:, None] * K
            + r * 16
            + (half * 8)[:, None]
            + jv[None, :]
        )
        tl.store(optr, val.to(tl.bfloat16), mask=m[:, None])


def _derive_tile_maps(
    n: int, k: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover Marlin's tile permutation for an (n, k) expert matrix.

    Probes the repack and scale-permute ops with index-valued inputs, so the
    maps follow whatever layout the installed Marlin kernels expect. Returns
    ``(src, sloc)``: for each of the 1024 positions of a 64x16 tile the source
    (nibble) index inside the tile, and for each of its 64 rows the scale
    column inside the tile.
    """
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        _repack_marlin_experts,
    )

    bits = (n * k - 1).bit_length()
    if bits > 30:
        raise ValueError("expert matrix too large for the index probe")
    lin = (
        torch.arange(n, device=device)[:, None] * k
        + torch.arange(k, device=device)[None, :]
    )
    perm = torch.empty(0, dtype=torch.int, device=device)
    shifts = 4 * torch.arange(8, device=device, dtype=torch.int32)
    idx = None
    for j in range(bits):
        plane = ((lin >> j) & 1).to(torch.uint8)
        packed = (plane[:, 0::2] | (plane[:, 1::2] << 4)).contiguous().unsqueeze(0)
        words = _repack_marlin_experts(packed, n, k, perm, False).reshape(-1)
        nib = ((words.unsqueeze(-1) >> shifts) & 0xF).to(torch.int32) << j
        idx = nib if idx is None else idx | nib
    assert idx is not None

    kg = k // _GROUP_SIZE
    nt = n // _TILE_N
    table = idx.contiguous().reshape(kg, nt, 128, 8).long()
    n_pos = table // k
    k_pos = table % k
    n_loc = n_pos - (torch.arange(nt, device=device) * _TILE_N).view(1, nt, 1, 1)
    k_loc = k_pos - (torch.arange(kg, device=device) * _GROUP_SIZE).view(kg, 1, 1, 1)
    if not (
        bool((n_loc == n_loc[0:1, 0:1]).all())
        and bool((k_loc == k_loc[0:1, 0:1]).all())
        and int(n_loc.max()) == _TILE_N - 1
        and int(k_loc.max()) == _GROUP_SIZE - 1
    ):
        raise ValueError("Marlin weight tile layout is not tile-invariant")
    loc = (n_loc[0, 0] * _GROUP_SIZE + k_loc[0, 0]).to(torch.int32).reshape(-1)
    src = torch.empty(1024, dtype=torch.int32, device=device)
    src[loc.long()] = torch.arange(1024, device=device, dtype=torch.int32)

    s0 = torch.arange(n * kg, device=device, dtype=torch.int64).reshape(n, kg).T
    ps = marlin_permute_scales(
        s=s0, size_k=k, size_n=n, group_size=_GROUP_SIZE, is_a_8bit=False
    )
    # Same byte shuffle that nvfp4_marlin_process_scales applies.
    ps = ps.reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(ps.size(0), -1)
    flat = ps.reshape(-1)
    inv = torch.empty(n * kg, dtype=torch.int64, device=device)
    inv[flat] = torch.arange(flat.numel(), device=device)
    pos = inv.reshape(n, kg)
    row = pos // n
    col = pos % n
    if not bool((row == torch.arange(kg, device=device).view(1, kg)).all()):
        raise ValueError("Marlin scale row is not the scale group")
    sloc = col[0:_TILE_N, 0].to(torch.int32).contiguous()
    expect = sloc.long().view(1, _TILE_N, 1)
    got = col.view(nt, _TILE_N, kg) - (torch.arange(nt, device=device) * _TILE_N).view(
        nt, 1, 1
    )
    if not bool((got == expect).all()):
        raise ValueError("Marlin scale layout is not tile-invariant")
    return src.contiguous(), sloc


@dataclass
class _Layer:
    w1: torch.Tensor
    w2: torch.Tensor
    s1: torch.Tensor
    s2: torch.Tensor
    g1: torch.Tensor
    g2: torch.Tensor
    ok: bool | None = None


@dataclass
class _Workspace:
    device: torch.device
    dtype: torch.dtype
    num_experts: int
    hidden: int
    n1: int
    inter: int
    slots: list[tuple[torch.Tensor, torch.Tensor]]
    maps: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]
    layers: dict[tuple[int, int], _Layer] = field(default_factory=dict)
    order: list[tuple[int, int]] = field(default_factory=list)
    order_done: bool = False
    slot_key: list[tuple[int, int] | None] = field(default_factory=lambda: [None, None])
    counter: int = 0
    # Created lazily on first use so import/reserve never touches streams.
    side: torch.cuda.Stream | None = None
    ready: list[torch.cuda.Event] | None = None
    done: list[torch.cuda.Event] | None = None


_lock = threading.Lock()
_workspace: _Workspace | None = None


def clear() -> None:
    """Drop per-layer records and slot contents (weights were rewritten)."""
    ws = _workspace
    if ws is None:
        return
    ws.layers.clear()
    ws.order.clear()
    ws.order_done = False
    ws.slot_key[:] = [None, None]
    ws.counter = 0
    ws.ready = None
    ws.done = None
    ws.side = None


def reserve(
    layer: torch.nn.Module,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
) -> bool:
    """Allocate the shared BF16 slots for a Marlin-packed NVFP4 MoE layer.

    Called after the layer's weights are repacked for Marlin and before KV
    cache sizing. Returns whether the BF16 path can serve this layer's shape.
    """
    global _workspace
    if not MIN_TOKENS:
        return False
    config = cast("FusedMoEConfig", layer.moe_config)
    reason = None
    if not (current_platform.is_cuda() and HAS_TRITON and w13.is_cuda):
        reason = "requires CUDA and Triton"
    elif getattr(layer, "params_dtype", None) != torch.bfloat16:
        reason = "requires BF16 activations"
    elif config.use_ep or config.ep_size != 1:
        reason = "does not support expert parallelism"
    elif config.activation_situ_beta is None:
        reason = "applies only to SiTU experts"
    elif w13_scale.element_size() != 1 or w2_scale.element_size() != 1:
        reason = "requires one-byte NVFP4 scales"
    if reason is None:
        num_experts = w13.shape[0]
        n1 = w13_scale.shape[-1]
        hidden = w13.shape[1] * _GROUP_SIZE
        inter = w2.shape[1] * _GROUP_SIZE
        if (
            n1 != 2 * inter
            or w2_scale.shape[-1] != hidden
            or n1 % _TILE_N
            or hidden % _TILE_N
            or hidden % _GROUP_SIZE
            or inter % _GROUP_SIZE
        ):
            reason = "unsupported expert shape"
    if reason is not None:
        logger.warning_once(
            "The BF16 dequant MoE path is requested (by "
            "VLLM_RAON_MOE_DEQUANT_BF16_MIN_TOKENS or VLLM_RAON_OPTIMIZATIONS) "
            "but %s; using Marlin.",
            reason,
        )
        return False

    with _lock:
        ws = _workspace
        if ws is not None:
            return ws.device == w13.device and (
                ws.num_experts,
                ws.n1,
                ws.hidden,
                ws.inter,
            ) == (num_experts, n1, hidden, inter)
        device = w13.device
        try:
            maps = {
                (n1, hidden): _derive_tile_maps(n1, hidden, device),
                (hidden, inter): _derive_tile_maps(hidden, inter, device),
            }
        except (ValueError, RuntimeError) as exc:
            logger.warning_once(
                "BF16 dequant MoE path disabled: cannot derive the Marlin "
                "layout (%s); using Marlin.",
                exc,
            )
            return False
        slots = [
            (
                torch.empty(
                    num_experts, n1, hidden, dtype=torch.bfloat16, device=device
                ),
                torch.empty(
                    num_experts, hidden, inter, dtype=torch.bfloat16, device=device
                ),
            )
            for _ in range(2)
        ]
        _workspace = _Workspace(
            device=device,
            dtype=torch.bfloat16,
            num_experts=num_experts,
            hidden=hidden,
            n1=n1,
            inter=inter,
            slots=slots,
            maps=maps,
        )
        nbytes = sum(t.numel() * t.element_size() for pair in slots for t in pair)
        logger.info_once(
            "Allocated two BF16 expert workspace slots (%.2f GiB) for large "
            "prefill batches.",
            nbytes / 2**30,
        )
    return True


def _make_layer(
    ws: _Workspace,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
) -> _Layer | None:
    e = ws.num_experts
    if (
        tuple(w1_scale.shape) != (e, ws.hidden // _GROUP_SIZE, ws.n1)
        or tuple(w2_scale.shape) != (e, ws.inter // _GROUP_SIZE, ws.hidden)
        or w1.shape[0] != e
        or w2.shape[0] != e
        or g1.numel() != e
        or g2.numel() != e
        or w1.element_size() * w1[0].numel() != ws.n1 * ws.hidden // 2
        or w2.element_size() * w2[0].numel() != ws.hidden * ws.inter // 2
    ):
        return None
    return _Layer(
        w1=w1.contiguous().view(torch.int32).reshape(e, -1),
        w2=w2.contiguous().view(torch.int32).reshape(e, -1),
        s1=w1_scale.contiguous().view(torch.uint8).reshape(e, -1),
        s2=w2_scale.contiguous().view(torch.uint8).reshape(e, -1),
        g1=g1.reshape(e).float().contiguous(),
        g2=g2.reshape(e).float().contiguous(),
    )


def _dequant_into(ws: _Workspace, layer: _Layer, slot: int) -> None:
    pairs = (
        (layer.w1, layer.s1, layer.g1, ws.n1, ws.hidden, ws.slots[slot][0]),
        (layer.w2, layer.s2, layer.g2, ws.hidden, ws.inter, ws.slots[slot][1]),
    )
    for weight, scale, gscale, n, k, out in pairs:
        src, sloc = ws.maps[(n, k)]
        grid = (ws.num_experts, k // _GROUP_SIZE, triton.cdiv(n // _TILE_N, _TPP))
        _dequant_marlin_nvfp4[grid](
            weight,
            src,
            sloc,
            scale,
            gscale,
            out,
            weight.shape[1],
            scale.shape[1],
            n,
            k,
            n // _TILE_N,
            TPP=_TPP,
            num_warps=4,
        )


def _grouped_chain(
    ws: _Workspace,
    slot: int,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    beta: float,
    linear_beta: float | None,
    out: torch.Tensor | None,
) -> torch.Tensor:
    from vllm import _custom_ops as ops

    w13, w2 = ws.slots[slot]
    tokens, topk = topk_ids.shape
    experts = w13.shape[0]
    hidden = hidden_states.shape[1]
    flat = topk_ids.reshape(-1)
    counts = torch.zeros(experts, dtype=torch.int64, device=flat.device)
    counts.scatter_add_(0, flat.long(), torch.ones_like(flat, dtype=torch.int64))
    offsets = counts.cumsum(0).to(torch.int32)
    order = torch.argsort(flat, stable=True)
    token_of_row = torch.div(order, topk, rounding_mode="floor")
    rows = hidden_states.index_select(0, token_of_row)
    probs = topk_weights.reshape(-1).index_select(0, order).to(hidden_states.dtype)
    gate_up = torch._grouped_mm(rows, w13.transpose(1, 2), offs=offsets)
    act = torch.empty(
        gate_up.shape[0], ws.inter, dtype=gate_up.dtype, device=gate_up.device
    )
    torch.ops._C.situ_and_mul(
        act, gate_up, beta, -1.0 if linear_beta is None else linear_beta
    )
    del gate_up
    act.mul_(probs[:, None])
    fc2 = torch._grouped_mm(act, w2.transpose(1, 2), offs=offsets)
    del act
    unsorted = torch.empty_like(fc2)
    unsorted.index_copy_(0, order, fc2)
    del fc2
    if out is None:
        out = torch.empty(
            tokens, hidden, dtype=hidden_states.dtype, device=hidden_states.device
        )
    ops.moe_sum(unsorted.view(tokens, topk, hidden), out)
    return out


def _self_check(
    ws: _Workspace,
    slot: int,
    layer: _Layer,
    beta: float,
    linear_beta: float | None,
    reference: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> bool:
    """Compare the BF16 chain with Marlin on a fixed synthetic batch."""
    n = _SELF_CHECK_TOKENS
    device = ws.device
    gen = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(n, ws.hidden, generator=gen, device=device).to(torch.bfloat16)
    topk = 6
    ids = (
        torch.arange(n * topk, device=device, dtype=torch.int64) % ws.num_experts
    ).view(n, topk)
    ids = ids.to(torch.int32)
    weights = torch.full((n, topk), 1.0 / topk, dtype=torch.float32, device=device)
    expected = reference(x, weights, ids).float()
    actual = _grouped_chain(ws, slot, x, ids, weights, beta, linear_beta, None).float()
    norm = float(expected.norm())
    if not norm > 0.0:
        return False
    return float((actual - expected).norm()) / norm < _SELF_CHECK_TOL


def _events(ws: _Workspace) -> tuple[list, list]:
    if ws.ready is None or ws.done is None:
        ws.ready = [torch.cuda.Event(), torch.cuda.Event()]
        ws.done = [torch.cuda.Event(), torch.cuda.Event()]
    return ws.ready, ws.done


def run(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    global_scale1: torch.Tensor | None,
    global_scale2: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    bias1: torch.Tensor | None,
    bias2: torch.Tensor | None,
    apply_router_weight_on_input: bool,
    activation_config,
    input_dtype: torch.dtype | None,
    reference: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> bool:
    """Run the BF16 path into ``output``; return False to use Marlin instead.

    ``reference(x, weights, ids)`` evaluates the Marlin path for the layer and
    is only used by the one-time self-check.
    """
    ws = _workspace
    if (
        not MIN_TOKENS
        or ws is None
        or hidden_states.shape[0] < MIN_TOKENS
        or hidden_states.dtype != ws.dtype
        or not hidden_states.is_cuda
        or hidden_states.device != ws.device
        or hidden_states.shape[1] != ws.hidden
        or input_dtype is not None
        or expert_map is not None
        or bias1 is not None
        or bias2 is not None
        or apply_router_weight_on_input
        or global_scale1 is None
        or global_scale2 is None
        or activation_config is None
        or activation_config.activation_situ_beta is None
        or torch.cuda.is_current_stream_capturing()
    ):
        return False
    key = (w1.data_ptr(), w2.data_ptr())
    layer = ws.layers.get(key)
    if layer is None:
        layer = _make_layer(
            ws, w1, w2, w1_scale, w2_scale, global_scale1, global_scale2
        )
        if layer is None:
            return False
        ws.layers[key] = layer
        if not ws.order_done and key not in ws.order:
            ws.order.append(key)
    elif not ws.order_done and ws.order and key == ws.order[0] and len(ws.order) > 1:
        ws.order_done = True
    if layer.ok is False:
        return False

    beta = activation_config.activation_situ_beta
    linear_beta = activation_config.activation_situ_linear_beta
    ready, done = _events(ws)
    main = torch.cuda.current_stream(ws.device)

    if key in ws.slot_key:
        slot = ws.slot_key.index(key)
        main.wait_event(ready[slot])
    else:
        slot = ws.counter % 2
        if ws.slot_key[slot] is not None:
            # An unconsumed prefetch may still be writing this slot.
            main.wait_event(ready[slot])
        _dequant_into(ws, layer, slot)
        ws.slot_key[slot] = key
    ws.counter += 1

    # Prefetch the next MoE layer on the side stream once the layer order of a
    # full forward pass is known.
    if ws.order_done and key in ws.order:
        nxt = ws.order[(ws.order.index(key) + 1) % len(ws.order)]
        other = 1 - slot
        if nxt in ws.layers and ws.slot_key[other] != nxt:
            if ws.side is None:
                ws.side = torch.cuda.Stream(ws.device)
            ws.side.wait_event(done[other])
            with torch.cuda.stream(ws.side):
                _dequant_into(ws, ws.layers[nxt], other)
                ready[other].record(ws.side)
            ws.slot_key[other] = nxt

    if layer.ok is None:
        layer.ok = _self_check(ws, slot, layer, beta, linear_beta, reference)
        if not layer.ok:
            logger.warning_once(
                "BF16 dequant MoE self-check failed for a layer; using Marlin for it."
            )
            done[slot].record(main)
            return False

    _grouped_chain(
        ws, slot, hidden_states, topk_ids, topk_weights, beta, linear_beta, output
    )
    done[slot].record(main)
    return True

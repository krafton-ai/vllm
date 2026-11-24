# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused NVFP4 Marlin routing for Raon's single-row decode."""

from contextvars import ContextVar
from dataclasses import dataclass

import torch

from vllm import envs
from vllm.model_executor.layers import raon_optimizations
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

ENABLED = (
    HAS_TRITON
    and current_platform.is_cuda()
    and (
        envs.VLLM_MARLIN_SINGLE_ROW_ROUTING
        or raon_optimizations.enabled("marlin_single_row_routing")
    )
)
if ENABLED:
    from triton.language.extra import libdevice


@dataclass
class SingleRowRoute:
    topk_ids: torch.Tensor | None = None
    alignment: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None


# The runner owns this scope for one synchronous routed-expert invocation.
# The scope ends before shared-expert overlap and never crosses requests.
active_route: ContextVar[SingleRowRoute | None] = ContextVar(
    "raon_marlin_single_row_route", default=None
)


@triton.jit
def _route(
    logits,
    bias,
    weights,
    ids,
    source,
    sorted_ids,
    expert_ids,
    padded,
    scale,
    RENORMALIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IDS64: tl.constexpr,
):
    offsets = tl.arange(0, 128)
    raw = tl.load(logits + offsets).to(tl.float32)
    exp = tl.inline_asm_elementwise(
        "ex2.approx.ftz.f32 $0, $1;",
        "=f,f",
        [-raw * 1.4426950408889634],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    scores = libdevice.div_rn(1.0, 1.0 + exp)
    scores = tl.where(
        (scores == scores) & (tl.abs(scores) != float("inf")), scores, 0.0
    )
    choices = scores
    if HAS_BIAS:
        choices = scores + tl.load(bias + offsets)
    lanes = tl.arange(0, 8)
    selected_ids = tl.full((8,), 128, tl.int32)
    selected_scores = tl.full((8,), 0, tl.float32)
    total = 0.0
    for rank in tl.static_range(6):
        maximum = tl.max(choices, 0)
        choice = tl.min(tl.where(choices == maximum, offsets, 128), 0)
        score = tl.sum(tl.where(offsets == choice, scores, 0.0), 0)
        selected_ids = tl.where(lanes == rank, choice, selected_ids)
        selected_scores = tl.where(lanes == rank, score, selected_scores)
        total += score
        choices = tl.where(offsets == choice, -10000.0, choices)
    if RENORMALIZE:
        scale = libdevice.div_rn(scale, tl.where(total > 0.0, total, 1.0))
    valid = lanes < 6
    tl.store(weights + lanes, selected_scores * scale, mask=valid)
    if IDS64:
        tl.store(ids + lanes, selected_ids.to(tl.int64), mask=valid)
    else:
        tl.store(ids + lanes, selected_ids, mask=valid)
    tl.store(source + lanes, lanes, mask=valid)

    # A single token contributes one routed row to each selected expert.
    # Marlin visits blocks in ascending expert order and skips sentinel rows.
    rank_by_id = tl.sum(
        ((selected_ids[None, :] < selected_ids[:, None]) & valid[None, :]).to(tl.int32),
        axis=1,
    )
    rows = tl.arange(0, 64)
    row_match = (rows[:, None] == 8 * rank_by_id[None, :]) & valid[None, :]
    row_value = tl.sum(tl.where(row_match, lanes[None, :], 0), axis=1)
    has_value = tl.sum(row_match.to(tl.int32), axis=1) > 0
    tl.store(sorted_ids + rows, tl.where(has_value, row_value, 6), mask=rows < 48)
    expert_blocks = (rank_by_id[None, :] == lanes[:, None]) & valid[None, :]
    experts = tl.sum(tl.where(expert_blocks, selected_ids[None, :], 0), axis=1)
    tl.store(expert_ids + lanes, experts, mask=valid)
    tl.store(padded, 48)


def try_route(
    weights: torch.Tensor,
    ids: torch.Tensor,
    source: torch.Tensor,
    logits: torch.Tensor,
    renormalize: bool,
    bias: torch.Tensor | None,
    scale: float,
    padding: torch.Tensor | None,
) -> bool:
    route = active_route.get()
    if (
        route is None
        or padding is not None
        or not logits.is_cuda
        or logits.shape != (1, 128)
        or logits.dtype not in (torch.float32, torch.bfloat16)
        or not logits.is_contiguous()
        or weights.shape != (1, 6)
        or ids.shape != (1, 6)
        or source.shape != (1, 6)
        or weights.dtype != torch.float32
        or ids.dtype not in (torch.int32, torch.int64)
        or source.dtype != torch.int32
        or not (
            weights.is_contiguous() and ids.is_contiguous() and source.is_contiguous()
        )
        or (
            bias is not None
            and (
                bias.device != logits.device
                or bias.shape != (128,)
                or bias.dtype != torch.float32
                or not bias.is_contiguous()
            )
        )
    ):
        return False

    sorted_ids = torch.empty(48, device=logits.device, dtype=torch.int32)
    expert_ids = torch.empty(6, device=logits.device, dtype=torch.int32)
    padded = torch.empty(1, device=logits.device, dtype=torch.int32)
    _route[(1,)](
        logits,
        logits if bias is None else bias,
        weights,
        ids,
        source,
        sorted_ids,
        expert_ids,
        padded,
        scale,
        RENORMALIZE=renormalize,
        HAS_BIAS=bias is not None,
        IDS64=ids.dtype == torch.int64,
    )
    route.topk_ids = ids
    route.alignment = sorted_ids, expert_ids, padded
    return True


def take_alignment(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    route = active_route.get()
    if (
        route is None
        or route.topk_ids is not topk_ids
        or block_size != 8
        or num_experts != 128
        or expert_map is not None
    ):
        return None

    alignment = route.alignment
    route.topk_ids = None
    route.alignment = None
    return alignment

# SPDX-License-Identifier: Apache-2.0
"""RAON_MLA_PROJ_PERSISTENT: batch-invariant GEMMs for the MLA projections.

The MLA projections are stock vLLM linears on cuBLAS, whose split-K / tile
schedule is chosen from M, and M in a decode forward is the number of live
sequences. A row's MLA output therefore depends on how many other sequences
share the batch. The KDA projections (`RAON_KDA_PROJ_PERSISTENT`) and the KDA
core are already batch-invariant, so MLA is the first batch-dependent site.

`RAON_MLA_BATCH_INVARIANT` does not cover this: it pins only the attention's
split-KV reduction (`triton_mla._compute_num_kv_splits -> 1`).

## What it pins

The four MLA projections the forward calls:

    fused_qkv_a_proj   hidden -> q_lora | kv_lora | rope   (MergedColumnParallel)
    q_b_proj           q_c    -> heads * qk_head_dim       (ColumnParallel)
    g_proj             hidden -> heads * v_head_dim        (the output gate)
    o_proj             attn   -> hidden                    (RowParallel)

`kv_b_proj` is not wrapped: it is absorbed into the attention implementation
by the post-load hook and its `forward` is not on the decode path.

All of them use the same `matmul_persistent` as the KDA projections and the
dense FFN: a Triton kernel with a fixed tile schedule and a fixed
BLOCK_SIZE_K loop.

## Scope and refusals

Default off; when off, `apply_module` returns before touching anything.
TP > 1 is refused: `matmul_persistent` computes the raw local GEMM and carries
none of the column/row-parallel gather/reduce plumbing, as in `mlp_align.py`.
A biased or quantized projection is refused. A missing projection is refused,
since wrapping only some of them would leave the batch dependence in place.

Forward only: `matmul_persistent` records no autograd.
"""

from __future__ import annotations

import os
import types

import torch

_ENV = "RAON_MLA_PROJ_PERSISTENT"

# The projections the MLA forward calls, in call order.
# kv_b_proj is absent on purpose -- see the module docstring.
_PROJECTIONS = ("fused_qkv_a_proj", "q_b_proj", "g_proj", "o_proj")

_ANNOUNCED = False


def enabled() -> bool:
    """Whether the MLA persistent-projection GEMMs are enabled (default off)."""
    return os.environ.get(_ENV, "0") == "1"


def _persistent_linear_forward(self, inp, *args, **kwargs):
    """Instance-level replacement forward for one MLA projection.

    Same body as `mlp_align._persistent_linear_forward`, so both execute the
    identical GEMM reduction order. Keeps vLLM's `(output, bias)`
    return convention with bias None; bias-free is asserted at install.
    """
    from .batch_invariant import matmul_persistent

    weight = self.weight
    x2d = inp.reshape(-1, inp.shape[-1])
    out = matmul_persistent(x2d, weight.t())
    return out.view(*inp.shape[:-1], weight.shape[0]), None


def apply_module(self_attn) -> dict:
    """Install on one `MultiHeadLatentAttention`. No-op when disabled.

    Returns a dict naming the wrapped projections (empty when disabled) so the
    caller can announce what is actually active.
    """
    if not enabled():
        return {}

    from vllm.distributed import get_tensor_model_parallel_world_size

    if get_tensor_model_parallel_world_size() != 1:
        raise RuntimeError(
            f"{_ENV} requires tensor parallel size 1 (it replaces the MLA "
            "projections' local GEMM without their gather/reduce plumbing)"
        )

    wrapped = []
    for name in _PROJECTIONS:
        sub = getattr(self_attn, name, None)
        if sub is None:
            raise RuntimeError(
                f"{_ENV}: this MLA layer has no {name}. This MLA is not "
                "the one this adapter was written against; wrapping a "
                "subset would pin some of the "
                "batch-shape dependence and leave the rest, which is worse "
                "than not pinning it."
            )
        if not hasattr(sub, "weight"):
            raise RuntimeError(f"{_ENV}: {name} carries no weight to pin")
        bias = getattr(sub, "bias", None)
        if bias is not None and getattr(bias, "numel", lambda: 1)() != 0:
            raise RuntimeError(
                f"{_ENV} expects bias-free MLA projections, but {name} "
                "carries a bias"
            )
        if getattr(sub, "quant_method", None) is not None and not isinstance(
            sub.weight, torch.nn.Parameter
        ):
            raise RuntimeError(
                f"{_ENV} expects an unquantized {name} weight parameter"
            )
        sub.forward = types.MethodType(_persistent_linear_forward, sub)
        wrapped.append(name)

    _announce(self_attn, wrapped)
    return {"mla_proj_persistent": wrapped}


def _announce(self_attn, wrapped) -> None:
    """Say once, with the shapes, which GEMMs left cuBLAS.

    Also names what is deliberately NOT pinned, so a reader of the log can tell
    a complete pin from a partial one without opening this file."""
    global _ANNOUNCED
    if _ANNOUNCED:
        return
    _ANNOUNCED = True
    shapes = ", ".join(
        f"{n}{tuple(getattr(self_attn, n).weight.shape)}" for n in wrapped
    )
    kvb = getattr(self_attn, "kv_b_proj", None)
    print(
        f"[parity] {_ENV}=1: MLA projections routed through "
        f"batch_invariant.matmul_persistent -- {shapes}. These were stock "
        "cuBLAS, whose split-K is chosen from M, and M in a decode forward is "
        "the number of live sequences; RAON_MLA_BATCH_INVARIANT pins only the "
        "attention's split-KV reduction, not these. NOT pinned: kv_b_proj"
        f"{tuple(kvb.weight.shape) if kvb is not None else ''} (absorbed into "
        "the attention impl, its forward is off the decode path).",
        flush=True,
    )

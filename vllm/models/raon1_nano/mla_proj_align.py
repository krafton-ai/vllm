# SPDX-License-Identifier: Apache-2.0
"""Optional persistent GEMMs for the upstream vLLM MLA projections.

The fused input, query, output-gate and output projections keep their vLLM
interfaces. This adapter requires unquantized weights and preserves output
shapes and dtypes."""

from __future__ import annotations

import os
import types

import torch

_ENV = "RAON_MLA_PROJ_PERSISTENT"

# The projections the Raon MLA forward actually calls, in call order.
# kv_b_proj is absent on purpose -- see the module docstring.
_PROJECTIONS = ("fused_qkv_a_proj", "q_b_proj", "g_proj", "o_proj")

_ANNOUNCED = False


def enabled() -> bool:
    """Is the MLA persistent-projection realization lever on? (default OFF)"""
    return os.environ.get(_ENV, "0") == "1"


def _persistent_linear_forward(self, inp, *args, **kwargs):
    """Instance-level replacement forward for one MLA projection.

    Byte-for-byte the same body as `mlp_align._persistent_linear_forward` and
    megatron's `kda_parity._persistent_linear_forward`, so all three runtimes
    execute the identical GEMM reduction order. Keeps vLLM's `(output, bias)`
    return convention with bias None; bias-free is asserted at install.
    """
    from .batch_invariant import matmul_persistent

    weight = self.weight
    x2d = inp.reshape(-1, inp.shape[-1])
    out = matmul_persistent(x2d, weight.t())
    return out.view(*inp.shape[:-1], weight.shape[0]), None


def apply_module(self_attn) -> dict:
    """Install the lever on one `MultiHeadLatentAttention`. OFF -> no-op.

    Returns a dict naming the wrapped projections (empty when OFF) so the
    caller can announce what is really live. Labels have lied nine times in
    this study; the return value is what the announce is built from.
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
                f"{_ENV}: this MLA layer has no {name}. The port's MLA is not "
                "the one this adapter was written against (Raon nvidia "
                "mla.py:203-267); wrapping a subset would pin some of the "
                "batch-shape dependence and leave the rest, which is worse "
                "than not pinning it."
            )
        if not hasattr(sub, "weight"):
            raise RuntimeError(f"{_ENV}: {name} carries no weight to pin")
        bias = getattr(sub, "bias", None)
        if bias is not None and getattr(bias, "numel", lambda: 1)() != 0:
            raise RuntimeError(
                f"{_ENV} expects bias-free MLA projections, but {name} carries a bias"
            )
        if getattr(sub, "quant_method", None) is not None and not isinstance(
            sub.weight, torch.nn.Parameter
        ):
            raise RuntimeError(f"{_ENV} expects an unquantized {name} weight parameter")
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

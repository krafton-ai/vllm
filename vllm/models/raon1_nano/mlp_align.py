"""Optional persistent GEMMs for the dense SiLU MLP.

The adapter preserves the packed gate/up and down projection interfaces.
It is disabled unless RAON_MLP_PROJ_PERSISTENT is selected. Only unquantized
weights without sequence parallelism are supported."""

from __future__ import annotations

import os
import types

import torch

_ENV = "RAON_MLP_PROJ_PERSISTENT"


def enabled() -> bool:
    """Is the dense-FFN persistent-GEMM realization lever on? (default OFF)"""
    return os.environ.get(_ENV, "0") == "1"


def _persistent_linear_forward(self, inp, *args, **kwargs):
    """Instance-level replacement forward for one dense-FFN projection.

    Mirrors megatron's `kda_parity._persistent_linear_forward` so the two
    runtimes execute the identical GEMM reduction order, and keeps vLLM's
    ``(output, bias)`` return convention with bias None (both dense-FFN
    projections are built ``bias=False``; asserted at install).
    """
    from .batch_invariant import matmul_persistent

    weight = self.weight
    x2d = inp.reshape(-1, inp.shape[-1])
    out = matmul_persistent(x2d, weight.t())
    return out.view(*inp.shape[:-1], weight.shape[0]), None


def apply_module(mlp) -> dict:
    """Install the lever on one dense ``RaonMLP``. Default OFF -> no-op.

    Returns a dict naming the wrapped projections (empty when OFF) so callers
    can log/verify that the lever is really live — labels must not lie.
    """
    if not enabled():
        return {}

    from vllm.distributed import get_tensor_model_parallel_world_size

    if get_tensor_model_parallel_world_size() != 1:
        raise RuntimeError(
            f"{_ENV} requires tensor parallel size 1 (it replaces the dense "
            "FFN projections' local GEMM without their gather/reduce plumbing)"
        )
    if getattr(mlp, "shard_sequence_parallel", False):
        raise RuntimeError(
            f"{_ENV} does not support the sharded sequence-parallel dense MLP "
            "(its all-gather/reduce-scatter would be bypassed)"
        )

    wrapped = []
    for name in ("gate_up_proj", "down_proj"):
        sub = getattr(mlp, name, None)
        if sub is None or not hasattr(sub, "weight"):
            raise RuntimeError(f"{_ENV}: dense MLP has no {name} with a weight")
        bias = getattr(sub, "bias", None)
        if bias is not None and getattr(bias, "numel", lambda: 1)() != 0:
            raise RuntimeError(
                f"{_ENV} expects bias-free dense FFN projections, but {name} "
                "carries a bias"
            )
        if getattr(sub, "quant_method", None) is not None and not isinstance(
            sub.weight, torch.nn.Parameter
        ):
            raise RuntimeError(f"{_ENV} expects an unquantized {name} weight parameter")
        sub.forward = types.MethodType(_persistent_linear_forward, sub)
        wrapped.append(name)
    return {"mlp_proj_persistent": wrapped}

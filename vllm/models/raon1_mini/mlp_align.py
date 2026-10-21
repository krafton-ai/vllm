"""RAON_MLP_PROJ_PERSISTENT: batch-invariant dense-FFN GEMMs (default off).

The dense FFN's `MergedColumnParallelLinear` / `RowParallelLinear` run cuBLAS
`F.linear` at TP1, whose split-K / tile schedule depends on M. This module
replaces both projections with the Triton `matmul_persistent` from
`batch_invariant.py` (fixed tile schedule and K loop), the same realization
the KDA projections use under `RAON_KDA_PROJ_PERSISTENT`. A reference
implementation that uses the same kernel then reproduces the dense FFN output
bitwise.

The SiTU activation is unchanged here: the fused CUDA `situ_and_mul` and the
fp32 reference formula (`SituAndMul.forward_native`) differ in at most a
handful of elements; see `RAON_MLP_SITU_NATIVE` below to align it as well.

## Scope

Dense FFN only (`KimiMLP`), which in raon-1-mini is layer 0 alone; every other
layer is MoE and is covered by `RAON_MOE_FP32_COMBINE` /
`RAON_MOE_TRAINER_REALIZED` in `moe_align.py`. TP1 only: the replacement
computes the raw local GEMM and carries none of the column/row-parallel
gather/reduce plumbing, so it refuses to install when TP > 1.

Forward only: `matmul_persistent` records no autograd.
"""

from __future__ import annotations

import os
import types

import torch

_ENV = "RAON_MLP_PROJ_PERSISTENT"


def enabled() -> bool:
    """Whether the dense-FFN persistent GEMMs are enabled (default off)."""
    return os.environ.get(_ENV, "0") == "1"


def _persistent_linear_forward(self, inp, *args, **kwargs):
    """Instance-level replacement forward for one dense-FFN projection.

    Runs `matmul_persistent` for a fixed GEMM reduction order and keeps vLLM's
    ``(output, bias)`` return convention with bias None (both dense-FFN
    projections are built ``bias=False``; asserted at install).
    """
    from .batch_invariant import matmul_persistent

    weight = self.weight
    x2d = inp.reshape(-1, inp.shape[-1])
    out = matmul_persistent(x2d, weight.t())
    return out.view(*inp.shape[:-1], weight.shape[0]), None


def apply_module(mlp) -> dict:
    """Install on one dense ``KimiMLP``. No-op when disabled.

    Returns a dict naming the wrapped projections (empty when disabled) so
    callers can verify what is active.
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
            raise RuntimeError(
                f"{_ENV} expects an unquantized {name} weight parameter"
            )
        sub.forward = types.MethodType(_persistent_linear_forward, sub)
        wrapped.append(name)
    return {"mlp_proj_persistent": wrapped}


# ---------------------------------------------------------------------------
# RAON_MLP_SITU_NATIVE: dense-FFN SiTU activation realization (default off).
#
# With RAON_MLP_PROJ_PERSISTENT pinning the GEMMs, the SiTU activation is the
# only op in the dense MLP whose realization can still differ from an fp32
# reference: the fused CUDA kernel `torch.ops._C.situ_and_mul` differs from
# the fp32 formula (`SituAndMul.forward_native`) in rare elements by one bf16
# ulp. This routes the dense KimiMLP's act_fn through
# moe_align.situ_reference, the fp32 reference formula with a single cast
# back. Scope: dense FFN only (layer 0); the MoE experts have their own SiTU
# options.

_SITU_ENV = "RAON_MLP_SITU_NATIVE"


def situ_native_enabled() -> bool:
    """Whether the dense-FFN fp32-reference SiTU is enabled (default off)."""
    return os.environ.get(_SITU_ENV, "0") == "1"


def _native_situ_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Instance-level replacement forward for the dense SituAndMul: the fp32
    reference formula (single cast back)."""
    from .moe_align import situ_reference

    return situ_reference(x, self.beta, self.linear_beta)


def apply_situ_native(mlp) -> dict:
    """Install the fp32 SiTU on one dense ``KimiMLP``. No-op when disabled.

    Returns a dict naming the wrapped activation (empty when disabled) so
    callers can verify what is active.
    """
    if not situ_native_enabled():
        return {}

    act = getattr(mlp, "act_fn", None)
    beta = getattr(act, "beta", None)
    linear_beta = getattr(act, "linear_beta", None)
    if act is None or beta is None or linear_beta is None:
        raise RuntimeError(
            f"{_SITU_ENV}: dense MLP act_fn is not a SituAndMul with both "
            f"betas (act={type(act).__name__ if act is not None else None}, "
            f"beta={beta}, linear_beta={linear_beta})"
        )
    act.forward = types.MethodType(_native_situ_forward, act)
    print(
        f"[raon-mlp] {_SITU_ENV}=1: dense FFN SiTU -> fp32 reference "
        f"(moe_align.situ_reference, beta={float(beta)}, "
        f"linear_beta={float(linear_beta)}) "
        "-- fp32 formula with a single cast back",
        flush=True,
    )
    return {"mlp_situ_native": [type(act).__name__, float(beta), float(linear_beta)]}

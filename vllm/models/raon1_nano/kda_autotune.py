# SPDX-License-Identifier: Apache-2.0
"""Pin upstream Triton KDA autotuners to a repeatable launch configuration.

A fixed configuration avoids changes in reduction order caused by
per-process autotuning. The upstream module paths below are version-specific;
missing kernels are reported instead of being treated as pinned."""

from __future__ import annotations

import importlib
import logging
import os
from contextlib import suppress

logger = logging.getLogger(__name__)

# Every autotuned Triton kernel reachable from `chunk_kda_with_fused_gate`
# (the RAON KDA prefill / chunk-replay path). Two families:
#   * the Nano-vendored FLA KDA ops                (ops.third_party.kda.*)
#   * the shared flash_linear_attention ops they call
# Missing entries are tolerated across vLLM versions but reported.
_KDA_KERNELS: tuple[tuple[str, str], ...] = (
    # --- Nano-vendored FLA KDA chunk kernels ---
    (
        "vllm.models.raon1_nano.ops.kda.chunk",
        "recompute_w_u_fwd_kernel",
    ),
    (
        "vllm.models.raon1_nano.ops.kda.chunk",
        "chunk_gla_fwd_kernel_o",
    ),
    (
        "vllm.models.raon1_nano.ops.kda.chunk",
        "kda_gate_chunk_cumsum_vector_kernel",
    ),
    # Elementwise gate transform: BT here does not change numerics, but its
    # config is still autotuned, so pin it for a clean, fully-deterministic set.
    (
        "vllm.models.raon1_nano.ops.kda.chunk",
        "kda_gate_fwd_kernel",
    ),
    (
        "vllm.models.raon1_nano.ops.kda.chunk_intra",
        "chunk_kda_fwd_kernel_inter_solve_fused",
    ),
    (
        "vllm.models.raon1_nano.ops.kda.chunk_intra",
        "chunk_kda_fwd_kernel_intra_sub_chunk",
    ),
    # --- shared flash_linear_attention ops on the KDA path ---
    (
        "vllm.third_party.flash_linear_attention.ops.chunk_delta_h",
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    ),
    (
        "vllm.third_party.flash_linear_attention.ops.cumsum",
        "chunk_local_cumsum_scalar_kernel",
    ),
    (
        "vllm.third_party.flash_linear_attention.ops.cumsum",
        "chunk_local_cumsum_vector_kernel",
    ),
    (
        "vllm.third_party.flash_linear_attention.ops.l2norm",
        "l2norm_fwd_kernel1",
    ),
    (
        "vllm.third_party.flash_linear_attention.ops.l2norm",
        "l2norm_fwd_kernel",
    ),
)

_pinned = False


def kda_pin_enabled() -> bool:
    return os.environ.get("RAON_KDA_AUTOTUNE_PIN", "1") == "1"


def _config_index() -> int:
    return int(os.environ.get("RAON_KDA_AUTOTUNE_CONFIG_INDEX", "0"))


def _autotuner(module_name: str, kernel_name: str):
    """Return the ``triton.Autotuner`` behind a (possibly Heuristics-wrapped)
    Triton kernel, or ``None`` if the symbol/module is absent."""
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:  # pragma: no cover - vLLM without these vendored ops
        logger.info("[raon-kda] no module %s (%s)", module_name, e)
        return None
    kernel = getattr(mod, kernel_name, None)
    if kernel is None:
        return None
    # triton.heuristics wraps triton.autotune wraps JITFunction; unwrap until
    # we find the object that owns `.configs` and `.cache` (the Autotuner).
    obj = kernel
    for _ in range(6):
        if hasattr(obj, "configs") and hasattr(obj, "cache"):
            return obj
        obj = getattr(obj, "fn", None)
        if obj is None:
            return None
    return None


def pin_kda_autotune_configs() -> int:
    """Pin every FLA KDA kernel to one statically-chosen config. Idempotent;
    returns the number of kernels pinned."""
    global _pinned

    if _pinned:
        return 0
    if not kda_pin_enabled():
        logger.info(
            "raon_vllm: KDA autotune pin disabled (RAON_KDA_AUTOTUNE_PIN=0) -- "
            "KDA chunk kernels stay autotuned, so KDA is NOT process- or "
            "batch-invariant (throughput / baseline runs only)."
        )
        _pinned = True
        return 0

    idx = _config_index()
    pinned, missing = [], []
    for module_name, kernel_name in _KDA_KERNELS:
        at = _autotuner(module_name, kernel_name)
        if at is None:
            missing.append(f"{module_name}.{kernel_name}")
            continue
        configs = list(at.configs)
        if not configs:
            missing.append(f"{module_name}.{kernel_name}(no configs)")
            continue
        chosen = configs[idx] if idx < len(configs) else configs[0]
        at.configs = [chosen]
        # Drop any config a warm process already selected, so the pin -- not the
        # cache -- decides the geometry from here on.
        with suppress(Exception):  # Cache shape may differ across Triton versions.
            at.cache.clear()
        pinned.append(
            f"{kernel_name}:{chosen.kwargs}/w{chosen.num_warps}/s{chosen.num_stages}"
        )

    _pinned = True
    if missing:
        # Loud, not fatal: an unpinned kernel means per-process autotuning is
        # back for that op and KDA zero-mismatch is no longer guaranteed.
        logger.warning(
            "raon_vllm: KDA autotune pin could NOT pin %d kernel(s) -- %s; "
            "KDA process/batch invariance is NOT guaranteed for these (vLLM "
            "version drift, re-check kernel names).",
            len(missing),
            ", ".join(missing),
        )
    logger.info(
        "raon_vllm: pinned %d KDA autotune config(s) (index=%d) -> "
        "process- and batch-invariant KDA. Unpinned: %d",
        len(pinned),
        idx,
        len(missing),
    )
    logger.info("raon_vllm: KDA pinned configs: %s", "; ".join(pinned))
    return len(pinned)


def describe_kda_autotune() -> list[str]:
    """Return a human-readable line per KDA kernel describing its current
    autotune config list. For diagnostics / validation reporting; does not
    mutate anything."""
    out = []
    for module_name, kernel_name in _KDA_KERNELS:
        at = _autotuner(module_name, kernel_name)
        if at is None:
            out.append(f"{kernel_name}: <absent>")
            continue
        configs = list(at.configs)
        out.append(
            f"{kernel_name}: {len(configs)} config(s) -> "
            + ", ".join(
                f"{c.kwargs}/w{c.num_warps}/s{c.num_stages}" for c in configs[:4]
            )
            + ("" if len(configs) <= 4 else " ...")
        )
    return out

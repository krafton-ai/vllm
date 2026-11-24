# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device selection for the Raon-1 mini optimizations.

By default (``VLLM_RAON_OPTIMIZATIONS`` unset or ``1``) the optimizations below
are on for the CUDA compute capabilities where they help, except those in
``_ADDS_MEMORY``, so model memory and KV cache capacity stay those of the
unoptimized server. ``VLLM_RAON_OPTIMIZATIONS=full`` turns on the
``_ADDS_MEMORY`` entries as well. ``VLLM_RAON_OPTIMIZATIONS=0`` (or ``off``)
turns the selection off and restores the stock paths.
One serving configuration can thus be used on all supported GPUs. Each
optimization's own environment variable still enables it explicitly,
independent of this table. ``VLLM_RAON_OPTIMIZATIONS_EXCLUDE`` lists names from
the table to leave off when the switch is set. A table entry of ``None`` selects
its optimization on every CUDA device.
"""

import functools

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Compute capabilities per optimization; None means every CUDA device.
_DEVICES: dict[str, tuple[tuple[int, int], ...] | None] = {
    "marlin_single_row_routing": None,
}

# Optimizations that keep extra resident weights, workspaces or graph pools.
# Only VLLM_RAON_OPTIMIZATIONS=full selects them.
_ADDS_MEMORY: frozenset[str] = frozenset()


@functools.cache
def enabled(name: str, device_id: int = 0) -> bool:
    capabilities = _DEVICES[name]
    mode = envs.VLLM_RAON_OPTIMIZATIONS
    if mode in ("0", "off") or not current_platform.is_cuda():
        return False
    if mode == "1" and name in _ADDS_MEMORY:
        logger.info(
            "Raon optimization %s adds memory; it needs VLLM_RAON_OPTIMIZATIONS=full",
            name,
        )
        return False
    excluded = {
        item.strip()
        for item in envs.VLLM_RAON_OPTIMIZATIONS_EXCLUDE.split(",")
        if item.strip()
    }
    unknown = excluded - _DEVICES.keys()
    if unknown:
        raise ValueError(
            f"Unknown VLLM_RAON_OPTIMIZATIONS_EXCLUDE entries: {sorted(unknown)}"
        )
    if name in excluded:
        logger.info("Raon optimization %s excluded", name)
        return False
    selected = capabilities is None or any(
        current_platform.is_device_capability(capability, device_id=device_id)
        for capability in capabilities
    )
    logger.info("Raon optimization %s on device %d: %s", name, device_id, selected)
    return selected

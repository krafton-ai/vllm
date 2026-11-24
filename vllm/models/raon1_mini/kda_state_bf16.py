# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 recurrent state for the Raon-1 mini KDA layers.

The recurrent (mamba-style) state pool is FP32 by default. With this option
the pool is BF16: the fused decode step reads and writes it directly, and the
chunked prefill reads the pool state as is and rounds the final state once on
store, so state pages are half as large and decode moves half the bytes.
It is not bit-identical to the FP32 state.

The option is chosen when the cache layout is configured. It is selected by
``VLLM_RAON_KDA_STATE_BF16=1`` or by the device table in
``raon_optimizations`` (``kda_state_bf16``). It needs the fused KDA decode step
without speculative decoding and the standard Triton prefill. A table
selection that cannot be honored falls back to the FP32 state; the explicit
variable raises instead.
"""

import functools
import os

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers import raon_optimizations
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator
from vllm.platforms import current_platform

logger = init_logger(__name__)

_NAME = "kda_state_bf16"


def num_speculative_tokens(vllm_config) -> int:
    spec_config = getattr(vllm_config, "speculative_config", None)
    if spec_config is None:
        return 0
    return getattr(spec_config, "num_speculative_tokens", 0) or 0


def _unmet_requirement(num_spec: int) -> str | None:
    if os.environ.get("RAON_KDA_SGL_FUSED", "0") != "1":
        return "the fused KDA decode step (RAON_KDA_SGL_FUSED=1) is off"
    if num_spec > 0:
        return "speculative decoding is configured"
    if os.environ.get("RAON_KDA_DECODE", "fused") != "fused":
        return "RAON_KDA_DECODE is not the fused backend"
    if os.environ.get("RAON_KDA_PREFILL", "auto") not in ("", "auto", "triton"):
        return "the KDA prefill backend is not the Triton one"
    for name in (
        "RAON_KDA_INKERNEL_FLA",
        "RAON_KDA_CPR",
        "RAON_KDA_CHUNK_REPLAY",
    ):
        if os.environ.get(name, "0") == "1":
            return f"{name}=1 owns the KDA prefill/decode path"
    return None


def _kernel_unavailable() -> str | None:
    from . import kda_sgl_fused

    capability = current_platform.get_device_capability()
    arch = None if capability is None else (capability.major, capability.minor)
    if kda_sgl_fused._load_module(bf16=True, capability=arch) is None:
        return "the BF16-state fused decode kernel could not be built"
    return None


@functools.cache
def use_bf16_state(num_spec: int = 0) -> bool:
    explicit = envs.VLLM_RAON_KDA_STATE_BF16
    if not (explicit or raon_optimizations.enabled(_NAME)):
        return False
    problem = _unmet_requirement(num_spec) or _kernel_unavailable()
    if problem is not None:
        if explicit:
            raise ValueError(
                f"VLLM_RAON_KDA_STATE_BF16=1 cannot take effect: {problem}."
            )
        logger.info("Raon KDA BF16 state not used, keeping FP32: %s.", problem)
        return False
    logger.info("Raon KDA recurrent state is stored in BF16.")
    return True


def raon_kda_state_dtype(
    model_dtype, cache_dtype, num_spec: int = 0
) -> tuple[torch.dtype, torch.dtype]:
    conv, state = MambaStateDtypeCalculator.kda_state_dtype(model_dtype, cache_dtype)
    if use_bf16_state(num_spec or 0):
        state = torch.bfloat16
    return conv, state

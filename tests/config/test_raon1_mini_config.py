# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace as NS
from typing import cast

import pytest

from vllm.config import VllmConfig
from vllm.config.attention import AttentionConfig
from vllm.model_executor.models.config import Raon1MiniForCausalLMConfig
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum


@pytest.mark.parametrize("capability", [100, 103, 120])
@pytest.mark.parametrize("explicit", [False, True])
def test_raon_attention_defaults_match_device_and_preserve_overrides(
    monkeypatch, capability, explicit
):
    import vllm.models.raon1_mini as raon

    monkeypatch.setenv("RAON_PRECISION_PROFILE", "nvfp4")
    raon.apply_precision_profile("nvfp4")
    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda cap: cap == capability
    )
    attention = AttentionConfig()
    if explicit:
        attention.backend_per_kind["mla_attention"] = AttentionBackendEnum.TRITON_MLA
        attention.mla_prefill_backend = MLAPrefillBackendEnum.FLASH_ATTN
    config = NS(
        scheduler_config=NS(max_num_batched_tokens=2048, max_num_seqs=8),
        attention_config=attention,
        kernel_config=NS(
            ir_op_priority=NS(rms_norm=None), moe_backend="auto", linear_backend="auto"
        ),
        parallel_config=NS(tensor_parallel_size=1, data_parallel_size=1),
        compilation_config=NS(custom_ops=[], max_cudagraph_capture_size=None),
        speculative_config=None,
    )
    Raon1MiniForCausalLMConfig.verify_and_update_config(cast(VllmConfig, config))
    portable = explicit or capability == 120
    assert attention.backend_per_kind["mla_attention"] == (
        AttentionBackendEnum.TRITON_MLA
        if portable
        else AttentionBackendEnum.FLASHINFER_MLA
    )
    assert attention.mla_prefill_backend == (
        MLAPrefillBackendEnum.FLASH_ATTN
        if portable
        else MLAPrefillBackendEnum.TRTLLM_RAGGED
    )
    if capability == 120:
        assert not attention.use_prefill_query_quantization


@pytest.mark.parametrize(
    "capability,profile,warns",
    [
        ((10, 3), "mxfp8", False),
        ((10, 3), "nvfp4", False),
        ((12, 0), "nvfp4", False),
        ((12, 0), "mxfp8", True),
        ((12, 1), "nvfp4", True),
    ],
)
def test_raon_gpu_warning_distinguishes_checkpoint_profile(
    monkeypatch, capability, profile, warns
):
    from unittest.mock import Mock

    import vllm.model_executor.models.config as config_module

    monkeypatch.setenv("RAON_PRECISION_PROFILE", profile)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda: NS(major=capability[0], minor=capability[1]),
    )
    monkeypatch.setattr(current_platform, "get_device_name", lambda: "test GPU")
    warning = Mock()
    monkeypatch.setattr(config_module.logger, "warning", warning)
    Raon1MiniForCausalLMConfig.check_gpu()
    assert warning.call_count == int(warns)

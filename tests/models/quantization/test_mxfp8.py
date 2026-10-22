# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E tests for online MXFP8 quantization.

Loads a BF16 model with ``--quantization mxfp8`` (online quantization) and
compares log-probabilities against the same model served in BF16 without
quantization.  This exercises the full pipeline: config parsing,
``Mxfp8OnlineLinearMethod``, ``Mxfp8OnlineMoEMethod``, weight loading,
online quantization / shuffling, and inference through ``apply_monolithic``.

Layer skipping (``modules_to_not_convert``) is configured in the model's
``config.json`` under ``quantization_config`` and is not tested here.

``example_prompts`` is a pytest fixture (from conftest.py) that loads 8
diverse prompts from ``tests/prompts/example.txt``.
"""

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm.platforms import current_platform

from ..utils import check_logprobs_close

# A small MoE model that fits on a single GPU and has both linear + MoE layers.
MOE_MODEL = "allenai/OLMoE-1B-7B-0125-Instruct"
# A small dense model (no MoE) to validate the linear-only path.
DENSE_MODEL = "Qwen/Qwen3-0.6B"

MAX_MODEL_LEN = 1024
MAX_TOKENS = 4
NUM_LOG_PROBS = 8


def test_raon_compact_mxfp8_portable_cache_preserves_checkpoint_bytes(monkeypatch):
    from vllm.models.raon1_mini import mxfp8_w8a8_triton
    from vllm.models.raon1_mini.mxfp8_loader import _register_compact_weights
    from vllm.models.raon1_mini.mxfp8_sharding import pack_scales

    monkeypatch.setenv("RAON_MOE_MXFP8_W8A8_GEMM_MODE", "3")
    monkeypatch.setattr(mxfp8_w8a8_triton, "_WQ_CACHE", {})
    layer = torch.nn.Module()
    shapes = ((2, 192, 160), (2, 160, 96))
    layer.w13_weight = torch.nn.Parameter(torch.empty(shapes[0], device="meta"))
    layer.w2_weight = torch.nn.Parameter(torch.empty(shapes[1], device="meta"))
    q13, q2 = [torch.randn(shape).to(torch.float8_e4m3fn) for shape in shapes]
    s13 = torch.randint(120, 130, (2, 192, 5), dtype=torch.uint8)
    s2 = torch.randint(120, 130, (2, 160, 3), dtype=torch.uint8)
    tensors = {
        "q13": q13,
        "q2": q2,
        "s13_mma": pack_scales(s13, fill=0),
        "s2_mma": pack_scales(s2, fill=0),
    }
    _register_compact_weights(layer, tensors)
    cached = mxfp8_w8a8_triton._quantized_weights(layer.w13_weight, layer.w2_weight)
    assert cached[0] is q13 and cached[2] is q2
    assert torch.equal(cached[1], s13) and torch.equal(cached[3], s2)
    assert cached[1].is_contiguous() and cached[3].is_contiguous()
    assert layer.w13_weight.is_meta and layer.w2_weight.is_meta
    assert layer.raon_mxfp8_q13 is q13
    assert layer.raon_mxfp8_s2 is cached[3]
    with pytest.raises(RuntimeError, match="cannot be refitted"):
        _register_compact_weights(layer, tensors)


@pytest.mark.skipif(
    not is_quant_method_supported("mxfp8"),
    reason="mxfp8 is not supported on this GPU type (requires sm_100+).",
)
@pytest.mark.quant_model
@pytest.mark.parametrize("model", [DENSE_MODEL, MOE_MODEL], ids=["dense", "moe"])
def test_mxfp8_logprobs(
    vllm_runner,
    example_prompts,
    model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare BF16 baseline logprobs against online MXFP8-quantized model.

    Runs the same model twice -- once in BF16 (baseline) and once with
    online MXFP8 quantization -- then checks that the top log-probabilities
    are close.  Only 4 tokens are generated to keep the test fast while
    still catching numerical divergence.
    """
    with monkeypatch.context() as m:
        m.setenv("TOKENIZERS_PARALLELISM", "true")

        with vllm_runner(
            model,
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=True,
        ) as vllm_model:
            baseline_outputs = vllm_model.generate_greedy_logprobs(
                example_prompts, MAX_TOKENS, NUM_LOG_PROBS
            )

        with vllm_runner(
            model,
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=True,
            quantization="mxfp8",
        ) as vllm_model:
            test_outputs = vllm_model.generate_greedy_logprobs(
                example_prompts, MAX_TOKENS, NUM_LOG_PROBS
            )

        check_logprobs_close(
            outputs_0_lst=baseline_outputs,
            outputs_1_lst=test_outputs,
            name_0="bf16",
            name_1="mxfp8",
        )


@pytest.mark.skipif(
    not is_quant_method_supported("mxfp8"),
    reason="mxfp8 is not supported on this GPU type (requires sm_100+).",
)
@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="AITER MXFP8 MoE backend is ROCm-only.",
)
@pytest.mark.quant_model
def test_mxfp8_aiter_requires_swigluoai_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.experts import aiter_mxfp8_moe
    from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (
        select_mxfp8_moe_backend,
    )

    monkeypatch.setattr(
        aiter_mxfp8_moe.AiterMxfp8Experts,
        "_supports_current_device",
        staticmethod(lambda: True),
    )
    monkeypatch.setattr(
        aiter_mxfp8_moe,
        "is_aiter_mxfp8_moe_available",
        lambda: True,
    )

    config = FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=256,
        num_local_experts=8,
        num_logical_experts=8,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
        moe_backend="aiter",
    )

    with pytest.raises(ValueError, match="requires activation=swigluoai_uninterleave"):
        select_mxfp8_moe_backend(config)


@pytest.mark.skipif(
    not is_quant_method_supported("mxfp8"),
    reason="mxfp8 is not supported on this GPU type (requires sm_100+).",
)
@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="AITER MXFP8 MoE backend is ROCm-only.",
)
@pytest.mark.quant_model
def test_mxfp8_aiter_requires_swigluoai_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.experts import aiter_mxfp8_moe
    from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (
        select_mxfp8_moe_backend,
    )

    monkeypatch.setattr(
        aiter_mxfp8_moe.AiterMxfp8Experts,
        "_supports_current_device",
        staticmethod(lambda: True),
    )
    monkeypatch.setattr(
        aiter_mxfp8_moe,
        "is_aiter_mxfp8_moe_available",
        lambda: True,
    )

    config = FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=256,
        num_local_experts=8,
        num_logical_experts=8,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
        moe_backend="aiter",
    )

    with pytest.raises(ValueError, match="hardcodes SwiGLU-OAI"):
        select_mxfp8_moe_backend(config)


@pytest.mark.skipif(
    not is_quant_method_supported("mxfp8"),
    reason="mxfp8 is not supported on this GPU type (requires sm_100+).",
)
@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="AITER MXFP8 MoE backend is ROCm-only.",
)
@pytest.mark.quant_model
def test_mxfp8_aiter_accepts_swigluoai_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.experts import aiter_mxfp8_moe
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
    from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (
        select_mxfp8_moe_backend,
    )

    monkeypatch.setattr(
        aiter_mxfp8_moe.AiterMxfp8Experts,
        "_supports_current_device",
        staticmethod(lambda: True),
    )
    monkeypatch.setattr(
        aiter_mxfp8_moe,
        "is_aiter_mxfp8_moe_available",
        lambda: True,
    )

    config = FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=256,
        num_local_experts=8,
        num_logical_experts=8,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
        moe_backend="aiter",
        swiglu_alpha=aiter_mxfp8_moe._AITER_SWIGLU_ALPHA,
        swiglu_beta=aiter_mxfp8_moe._AITER_SWIGLU_BETA,
    )

    backend, experts_cls = select_mxfp8_moe_backend(config)

    assert backend == Fp8MoeBackend.AITER_MXFP8
    assert experts_cls is aiter_mxfp8_moe.AiterMxfp8Experts


@pytest.mark.skipif(
    not is_quant_method_supported("mxfp8"),
    reason="mxfp8 is not supported on this GPU type (requires sm_100+).",
)
@pytest.mark.quant_model
@pytest.mark.parametrize("model", [DENSE_MODEL, MOE_MODEL], ids=["dense", "moe"])
def test_mxfp8_generation(vllm_runner, model: str) -> None:
    """Smoke test: verify online MXFP8 model generates coherent text."""
    prompt = "1 2 3 4 5"
    with vllm_runner(
        model,
        enforce_eager=True,
        quantization="mxfp8",
        max_model_len=MAX_MODEL_LEN,
    ) as vllm_model:
        output = vllm_model.generate_greedy([prompt], max_tokens=5)

    generated = output[0][1]
    assert len(generated) > len(prompt), (
        f"MXFP8 model produced no new tokens. Output: {generated!r}"
    )

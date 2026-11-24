# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the opt-in BF16 dequantized Raon MoE path."""

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe import marlin_bf16_dequant as dq
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    nvfp4_marlin_process_scales,
)

E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def e2m1_decode(nibble):
    # The arithmetic used by the dequantization kernel.
    e = (nibble & 7) >> 1
    m = nibble & 1
    mag = torch.where(
        e == 0, m * 0.5, (1.0 + 0.5 * m) * torch.exp2((e - 1).to(torch.float32))
    )
    return torch.where((nibble & 8) != 0, -mag, mag)


def scale_decode(byte):
    # The arithmetic used by the dequantization kernel for Marlin scales.
    b = byte.to(torch.int32)
    value = (1.0 + (b & 7) * 0.125) * torch.exp2(((b >> 3) - 15).to(torch.float32))
    return torch.where(b == 0, torch.zeros_like(value), value)


def fake_layer(**overrides):
    config = SimpleNamespace(
        use_ep=False, ep_size=1, activation_situ_beta=1.0, **overrides
    )
    return SimpleNamespace(moe_config=config, params_dtype=torch.bfloat16)


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    monkeypatch.setattr(dq, "_workspace", None)
    yield
    monkeypatch.setattr(dq, "_workspace", None)


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_RAON_MOE_DEQUANT_BF16_MIN_TOKENS", raising=False)
    assert envs.VLLM_RAON_MOE_DEQUANT_BF16_MIN_TOKENS == 0
    monkeypatch.setattr(dq, "MIN_TOKENS", 0)
    w = fake_weight()
    assert dq.reserve(fake_layer(), w, w, w, w) is False
    assert dq._workspace is None
    assert dq.run(**run_kwargs(fake_hidden())) is False


def test_env_parsing(monkeypatch):
    monkeypatch.setenv("VLLM_RAON_MOE_DEQUANT_BF16_MIN_TOKENS", "4096")
    assert envs.VLLM_RAON_MOE_DEQUANT_BF16_MIN_TOKENS == 4096


def fake_weight(is_cuda=True):
    return SimpleNamespace(is_cuda=is_cuda, shape=(1,), element_size=lambda: 1)


@pytest.mark.parametrize(
    "layer, is_cuda",
    [
        (fake_layer(), False),
        (
            SimpleNamespace(
                moe_config=SimpleNamespace(
                    use_ep=True, ep_size=2, activation_situ_beta=1.0
                ),
                params_dtype=torch.bfloat16,
            ),
            True,
        ),
        (
            SimpleNamespace(
                moe_config=SimpleNamespace(
                    use_ep=False, ep_size=1, activation_situ_beta=None
                ),
                params_dtype=torch.bfloat16,
            ),
            True,
        ),
        (
            SimpleNamespace(
                moe_config=SimpleNamespace(
                    use_ep=False, ep_size=1, activation_situ_beta=1.0
                ),
                params_dtype=torch.float16,
            ),
            True,
        ),
    ],
)
def test_reserve_rejects_unsupported(monkeypatch, layer, is_cuda):
    monkeypatch.setattr(dq, "MIN_TOKENS", 4096)
    monkeypatch.setattr(dq, "HAS_TRITON", True)
    monkeypatch.setattr(dq, "current_platform", SimpleNamespace(is_cuda=lambda: True))
    w = fake_weight(is_cuda)
    assert dq.reserve(layer, w, w, w, w) is False
    assert dq._workspace is None


def run_kwargs(hidden_states, **overrides):
    kwargs = dict(
        output=None,
        hidden_states=hidden_states,
        w1=None,
        w2=None,
        w1_scale=None,
        w2_scale=None,
        global_scale1=torch.ones(1),
        global_scale2=torch.ones(1),
        topk_weights=None,
        topk_ids=None,
        expert_map=None,
        bias1=None,
        bias2=None,
        apply_router_weight_on_input=False,
        activation_config=SimpleNamespace(activation_situ_beta=1.0),
        input_dtype=None,
        reference=None,
    )
    kwargs.update(overrides)
    return kwargs


def fake_hidden(tokens=8192, **overrides):
    attrs = dict(
        shape=(tokens, 2560),
        dtype=torch.bfloat16,
        is_cuda=True,
        device=torch.device("cpu"),
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


@pytest.mark.parametrize(
    "hidden, overrides",
    [
        (fake_hidden(4095), {}),  # below the threshold
        (fake_hidden(dtype=torch.float16), {}),
        (fake_hidden(is_cuda=False), {}),
        (fake_hidden(), {"expert_map": object()}),
        (fake_hidden(), {"bias1": object()}),
        (fake_hidden(), {"apply_router_weight_on_input": True}),
        (fake_hidden(), {"input_dtype": torch.float8_e4m3fn}),
        (
            fake_hidden(),
            {"activation_config": SimpleNamespace(activation_situ_beta=None)},
        ),
    ],
)
def test_run_guards_fall_back_to_marlin(monkeypatch, hidden, overrides):
    monkeypatch.setattr(dq, "MIN_TOKENS", 4096)
    monkeypatch.setattr(
        dq,
        "_workspace",
        SimpleNamespace(device=torch.device("cpu"), dtype=torch.bfloat16, hidden=2560),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert dq.run(**run_kwargs(hidden, **overrides)) is False


def test_run_never_used_during_graph_capture(monkeypatch):
    monkeypatch.setattr(dq, "MIN_TOKENS", 4096)
    monkeypatch.setattr(
        dq,
        "_workspace",
        SimpleNamespace(device=torch.device("cpu"), dtype=torch.bfloat16, hidden=2560),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert dq.run(**run_kwargs(fake_hidden())) is False


def test_e2m1_arithmetic_matches_table():
    nibbles = torch.arange(16, dtype=torch.int32)
    expected = torch.tensor(E2M1_VALUES + [-v for v in E2M1_VALUES])
    assert torch.equal(e2m1_decode(nibbles), expected)


@pytest.mark.parametrize("scale_factor", [1.0, 4.0])
def test_scale_decode_matches_marlin_scale_processing(scale_factor):
    # Positive E4M3 scales as produced by NVFP4 checkpoints.
    exponent = torch.arange(-6, 3)
    mantissa = torch.arange(8)
    scales = (
        (1.0 + mantissa[None, :] / 8.0) * torch.exp2(exponent[:, None].float())
    ).reshape(-1)
    scales = scales.to(torch.float8_e4m3fn).to(torch.half)
    scales = scales[: scales.numel() // 4 * 4].reshape(-1, 4).clone()
    processed, factor = nvfp4_marlin_process_scales(
        scales.clone(), scale_factor=scale_factor, a_dtype=torch.bfloat16
    )
    assert factor == scale_factor
    # Marlin swaps the two middle bytes of every group of four.
    unshuffled = processed.view(torch.uint8).view(-1, 4)[:, [0, 2, 1, 3]]
    expected = scales.float() * scale_factor * 2**7
    expected = torch.where(expected < 2, torch.zeros_like(expected), expected)
    assert torch.equal(scale_decode(unshuffled), expected)

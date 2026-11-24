# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.models.raon1_mini import kda_vsplit, rda_base


def set_device(monkeypatch, major, minor=0, *, cuda=True):
    platform = SimpleNamespace(
        is_cuda=lambda: cuda,
        get_device_capability=lambda: SimpleNamespace(major=major, minor=minor),
    )
    monkeypatch.setattr(rda_base, "current_platform", platform)
    monkeypatch.setattr(kda_vsplit, "current_platform", platform)


@pytest.mark.parametrize(
    ("major", "minor", "backend", "vsplit"),
    [
        (8, 9, "triton", False),
        (9, 0, "flashkda", False),
        (10, 3, "flashkda", True),
        (11, 0, "triton", False),
        (12, 0, "flashkda", True),
        (12, 1, "flashkda", True),
    ],
)
def test_auto_prefill_backend(monkeypatch, major, minor, backend, vsplit):
    set_device(monkeypatch, major, minor)
    monkeypatch.setenv("RAON_KDA_PREFILL_VSPLIT", "adaptive")

    assert (
        rda_base.resolve_rda_prefill_backend(
            "auto", 128, torch.bfloat16, -5.0, torch.float32
        )
        == backend
    )
    assert kda_vsplit.enabled() is vsplit


def test_explicit_flashkda_requires_a_supported_device(monkeypatch):
    set_device(monkeypatch, 8, 9)

    with pytest.raises(RuntimeError, match="FlashKDA requires CUDA"):
        rda_base.resolve_rda_prefill_backend(
            "flashkda", 128, torch.bfloat16, -5.0, torch.float32
        )


@pytest.mark.parametrize(
    ("cuda", "head_dim", "input_dtype", "state_dtype"),
    [
        (False, 128, torch.bfloat16, torch.float32),
        (True, 64, torch.bfloat16, torch.float32),
        (True, 128, torch.float16, torch.float32),
        (True, 128, torch.bfloat16, torch.bfloat16),
    ],
)
def test_triton_prefill_contract(monkeypatch, cuda, head_dim, input_dtype, state_dtype):
    set_device(monkeypatch, 8, 9, cuda=cuda)

    with pytest.raises(RuntimeError, match="Triton RDA prefill requires"):
        rda_base.resolve_rda_prefill_backend(
            "auto", head_dim, input_dtype, -5.0, state_dtype
        )


def test_flashkda_accepts_bfloat16_state(monkeypatch):
    set_device(monkeypatch, 10, 3)

    assert (
        rda_base.resolve_rda_prefill_backend(
            "auto", 128, torch.bfloat16, -5.0, torch.bfloat16
        )
        == "flashkda"
    )

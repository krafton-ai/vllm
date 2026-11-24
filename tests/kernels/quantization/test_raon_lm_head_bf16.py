# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guards for the opt-in resident BF16 Raon LM head (MXFP8 Marlin)."""

import pytest
import torch

from vllm import envs
from vllm.model_executor.kernels.linear.mxfp8.marlin import _raon_lm_head_min_rows


def _cpu_layer() -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros((8, 32), dtype=torch.bfloat16), requires_grad=False
    )
    return layer


def test_lm_head_bf16_is_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_RAON_LM_HEAD_BF16_MIN_ROWS", raising=False)
    assert envs.VLLM_RAON_LM_HEAD_BF16_MIN_ROWS == 0
    assert _raon_lm_head_min_rows(_cpu_layer()) == 0


def test_lm_head_bf16_rejects_negative_min_rows(monkeypatch):
    monkeypatch.setenv("VLLM_RAON_LM_HEAD_BF16_MIN_ROWS", "-1")
    with pytest.raises(ValueError, match="non-negative"):
        _raon_lm_head_min_rows(_cpu_layer())


def test_lm_head_bf16_requires_cuda_weight(monkeypatch):
    monkeypatch.setenv("VLLM_RAON_LM_HEAD_BF16_MIN_ROWS", "33")
    assert _raon_lm_head_min_rows(_cpu_layer()) == 0

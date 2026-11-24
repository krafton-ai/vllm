# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import envs
from vllm.model_executor.layers.fused_moe.experts import marlin_moe


@pytest.fixture
def counted(monkeypatch):
    calls = []

    def make(device, max_blocks_per_sm=1, existing=None):
        calls.append(device)
        return torch.zeros(4, dtype=torch.int)

    monkeypatch.setattr(marlin_moe, "marlin_make_workspace_new", make)
    monkeypatch.setattr(marlin_moe, "_scratch_cache", {})
    return calls


def _enable(monkeypatch, value):
    monkeypatch.setattr(envs, "VLLM_MARLIN_REUSE_SINGLE_ROW_SCRATCH", value)


def test_disabled_allocates_every_call(monkeypatch, counted):
    _enable(monkeypatch, False)
    cpu = torch.device("cpu")
    a = marlin_moe._lock_workspace(cpu, 1)
    b = marlin_moe._lock_workspace(cpu, 1)
    assert a is not b
    assert len(counted) == 2


def test_single_row_reuses_and_multi_row_does_not(monkeypatch, counted):
    _enable(monkeypatch, True)
    cpu = torch.device("cpu")
    a = marlin_moe._lock_workspace(cpu, 1)
    assert marlin_moe._lock_workspace(cpu, 1) is a
    assert marlin_moe._lock_workspace(cpu, 2) is not a
    assert len(counted) == 2


def test_cache_is_bounded(monkeypatch, counted):
    _enable(monkeypatch, True)
    monkeypatch.setattr(marlin_moe, "_SCRATCH_MAX_ENTRIES", 0)
    cpu = torch.device("cpu")
    assert marlin_moe._lock_workspace(cpu, 1) is not marlin_moe._lock_workspace(cpu, 1)
    assert not marlin_moe._scratch_cache

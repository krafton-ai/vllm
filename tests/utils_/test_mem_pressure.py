# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only allocator-pressure tests, independent of CUDA profiling fixtures."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.utils import mem_utils


@pytest.mark.parametrize(
    "integrated,wsl,releasable,free,expected",
    [
        (False, True, 2, 1, True),
        (False, True, 2, 8, False),
        (False, True, 0.25, 1, False),
        (False, False, 2, 1, False),
        (True, False, 2, 1, True),
        (True, False, 2, 8, False),
    ],
)
def test_release_only_unused_pages_under_supported_memory_pressure(
    monkeypatch, integrated, wsl, releasable, free, expected
):
    gib = 1024**3
    platform = SimpleNamespace(
        is_integrated_gpu=lambda _: integrated, is_cuda=lambda: True
    )
    monkeypatch.setattr(mem_utils, "current_platform", platform)
    monkeypatch.setattr(mem_utils, "in_wsl", lambda: wsl)
    monkeypatch.setattr(torch.accelerator, "memory_reserved", lambda _: 4 * gib)
    monkeypatch.setattr(
        torch.accelerator, "memory_allocated", lambda _: (4 - releasable) * gib
    )
    get_info = Mock(return_value=(free * gib, 10 * gib))
    monkeypatch.setattr(torch.accelerator, "get_memory_info", get_info)
    monkeypatch.setattr(
        mem_utils.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=free * gib, total=10 * gib),
    )
    sync, release = Mock(), Mock()
    monkeypatch.setattr(torch.accelerator, "synchronize", sync)
    monkeypatch.setattr(torch.accelerator, "empty_cache", release)
    assert (
        mem_utils.release_device_memory_under_pressure(torch.device("cuda:0"))
        is expected
    )
    assert sync.call_count == release.call_count == int(expected)
    if integrated or not wsl or releasable < 0.5:
        get_info.assert_not_called()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from unittest.mock import Mock

import torch
import torch.utils.cpp_extension

from vllm.models.raon1_mini import moe_situ_fast


def test_build_failure_falls_back_to_stock_kernel(monkeypatch, tmp_path):
    monkeypatch.setenv("RAON_MOE_SITU_FAST", "1")
    monkeypatch.setenv("RAON_MOE_SITU_FAST_BUILD_DIR", str(tmp_path))
    monkeypatch.setattr(moe_situ_fast, "_loaded", False)
    monkeypatch.setattr(moe_situ_fast, "_load_failed", False)
    monkeypatch.setattr(moe_situ_fast, "_eligible", lambda output, input: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    arch_lists: list[str | None] = []

    def fail_build(**kwargs):
        arch_lists.append(os.environ.get("TORCH_CUDA_ARCH_LIST"))
        raise RuntimeError("nvcc not found")

    build = Mock(side_effect=fail_build)
    monkeypatch.setattr(torch.utils.cpp_extension, "load", build)
    warning = Mock()
    monkeypatch.setattr(moe_situ_fast.logger, "warning", warning)
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)

    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    out = torch.empty(2, 4, dtype=torch.bfloat16)
    assert not moe_situ_fast.try_situ_and_mul(out, x, 1.0, -1.0)
    assert not moe_situ_fast.try_situ_and_mul(out, x, 1.0, -1.0)

    # One build attempt for the current GPU only, one warning.
    assert arch_lists == ["9.0"]
    assert warning.call_count == 1
    assert "TORCH_CUDA_ARCH_LIST" not in os.environ


def test_lever_off_skips_the_build(monkeypatch):
    monkeypatch.setenv("RAON_MOE_SITU_FAST", "0")
    build = Mock()
    monkeypatch.setattr(torch.utils.cpp_extension, "load", build)

    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    out = torch.empty(2, 4, dtype=torch.bfloat16)
    assert not moe_situ_fast.try_situ_and_mul(out, x, 1.0, -1.0)
    assert build.call_count == 0

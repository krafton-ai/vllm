# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scratch groups of the Raon NVFP4 per-call dequant path.

Layers that can run concurrently on different CUDA streams must not share a
scratch buffer: the MLA output gate runs on the model aux stream next to the
attention projections, and the shared experts run next to the routed experts.
"""

from types import SimpleNamespace

import pytest

from vllm.model_executor.kernels.linear.nvfp4.marlin import _dequant_group


@pytest.mark.parametrize(
    "prefix, group",
    [
        ("model.layers.3.self_attn.fused_qkv_a_proj", 0),
        ("model.layers.3.self_attn.q_b_proj", 0),
        ("model.layers.3.self_attn.o_proj", 0),
        ("model.layers.3.self_attn.g_proj", 2),
        ("model.layers.4.mlp.shared_experts.gate_up_proj", 1),
        ("model.layers.4.mlp.shared_experts.down_proj", 1),
        ("model.layers.5.self_attn.in_proj_qkvgfab", 0),
        ("", 0),
    ],
)
def test_dequant_group(prefix, group):
    assert _dequant_group(SimpleNamespace(prefix=prefix)) == group


def test_concurrent_layers_never_share_a_group():
    main = _dequant_group(SimpleNamespace(prefix="model.layers.3.self_attn.q_b_proj"))
    gate = _dequant_group(SimpleNamespace(prefix="model.layers.3.self_attn.g_proj"))
    shared = _dequant_group(
        SimpleNamespace(prefix="model.layers.4.mlp.shared_experts.down_proj")
    )
    assert len({main, gate, shared}) == 3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.marlin_single_row_route import (
    SingleRowRoute,
    active_route,
    take_alignment,
)


def test_precomputed_alignment_is_consumed_only_by_its_route():
    ids = torch.empty((1, 6), dtype=torch.int32)
    alignment = tuple(torch.empty(size, dtype=torch.int32) for size in (48, 6, 1))
    token = active_route.set(SingleRowRoute(topk_ids=ids, alignment=alignment))
    try:
        assert take_alignment(ids.clone(), 8, 128, None) is None
        assert take_alignment(ids, 16, 128, None) is None
        assert take_alignment(ids, 8, 128, torch.arange(128)) is None
        assert take_alignment(ids, 8, 128, None) is alignment
        assert take_alignment(ids, 8, 128, None) is None
    finally:
        active_route.reset(token)
    assert take_alignment(ids, 8, 128, None) is None

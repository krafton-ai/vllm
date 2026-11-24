# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace as NS

import pytest
import torch

import vllm.model_executor.layers.attention.mla_attention as mla
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadataBuilder,
)

GIB = 2**30
BLOCK_SIZE = 2432
MAX_MODEL_LEN = 262144
DEFAULT_ROWS = 65664  # min(8 * max_model_len, 64K) aligned up to BLOCK_SIZE


def _config(gmu: float, max_rows: int | None = None):
    return NS(
        scheduler_config=NS(max_num_seqs=8),
        cache_config=NS(block_size=BLOCK_SIZE, gpu_memory_utilization=gmu),
        model_config=NS(
            max_model_len=MAX_MODEL_LEN,
            dtype=torch.bfloat16,
            hf_text_config=NS(
                q_lora_rank=576,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
            ),
            get_num_attention_heads=lambda parallel_config: 40,
        ),
        parallel_config=NS(
            decode_context_parallel_size=1, cp_kv_cache_interleave_size=1
        ),
        attention_config=NS(mla_prefill_workspace_max_rows=max_rows),
    )


def _rows(monkeypatch, total_gib: float | None, gmu: float, max_rows=None) -> int:
    monkeypatch.setattr(
        mla,
        "_device_total_memory_bytes",
        lambda: None if total_gib is None else int(total_gib * GIB),
    )
    return MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(
        _config(gmu, max_rows)
    )


def test_small_gpu_limits_workspace(monkeypatch):
    rows = _rows(monkeypatch, 24, 0.85)
    # 40 * (2 * 128 + 64 + 128) * 2 = 35840 bytes per row, times 12; the
    # 3.6 GiB headroom allows 8987 rows, aligned down to 3 blocks.
    assert rows == 3 * BLOCK_SIZE
    assert rows < DEFAULT_ROWS


@pytest.mark.parametrize(("total_gib", "gmu"), [(268, 0.9), (122, 0.35)])
def test_large_gpu_keeps_default(monkeypatch, total_gib, gmu):
    assert _rows(monkeypatch, total_gib, gmu) == DEFAULT_ROWS


def test_no_cuda_keeps_default(monkeypatch):
    assert _rows(monkeypatch, None, 0.85) == DEFAULT_ROWS


def test_floor_applies_when_headroom_is_tiny(monkeypatch):
    assert _rows(monkeypatch, 24, 0.99) == 3 * BLOCK_SIZE  # 8192 rounded down


def test_explicit_limit_wins(monkeypatch):
    assert _rows(monkeypatch, 268, 0.9, max_rows=8192) == 3 * BLOCK_SIZE

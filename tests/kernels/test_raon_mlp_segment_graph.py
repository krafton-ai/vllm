# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.layers import raon_optimizations
from vllm.models.raon1_mini import mlp_segment_graph as graphs


@pytest.fixture(autouse=True)
def _clear_caches():
    raon_optimizations.enabled.cache_clear()
    graphs._buckets_cache.clear()
    yield
    raon_optimizations.enabled.cache_clear()
    graphs._buckets_cache.clear()


def test_parse_buckets():
    assert graphs.parse_buckets("") == ()
    assert graphs.parse_buckets("3328, 1280,2304,1280") == (1280, 2304, 3328)


@pytest.mark.parametrize("spec", ["abc", "1024", "4096", "0", "-5", "1280,x"])
def test_parse_buckets_rejects_invalid(spec):
    with pytest.raises(ValueError):
        graphs.parse_buckets(spec)


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (1, None),
        (1024, None),
        (1025, 1280),
        (1280, 1280),
        (1281, 2304),
        (3328, 3328),
        (3329, None),
        (8192, None),
    ],
)
def test_select_bucket(tokens, expected):
    assert graphs.select_bucket(tokens, (1280, 2304, 3328)) == expected


def _select_device(monkeypatch, capability):
    platform = raon_optimizations.current_platform
    monkeypatch.setattr(platform, "is_cuda", lambda: True)
    monkeypatch.setattr(graphs.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        platform,
        "is_device_capability",
        lambda value, device_id=0: tuple(value) == capability,
    )


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_RAON_OPTIMIZATIONS", raising=False)
    monkeypatch.delenv("VLLM_RAON_MLP_SEGMENT_GRAPH_BUCKETS", raising=False)
    _select_device(monkeypatch, (10, 3))
    assert graphs.configured_buckets(0) == ()


def test_profile_selects_default_buckets_on_sm103_only(monkeypatch):
    monkeypatch.setenv("VLLM_RAON_OPTIMIZATIONS", "full")
    monkeypatch.delenv("VLLM_RAON_MLP_SEGMENT_GRAPH_BUCKETS", raising=False)
    _select_device(monkeypatch, (10, 3))
    assert graphs.configured_buckets(0) == (1280, 2304, 3328)
    raon_optimizations.enabled.cache_clear()
    graphs._buckets_cache.clear()
    _select_device(monkeypatch, (9, 0))
    assert graphs.configured_buckets(0) == ()


def test_explicit_buckets_win(monkeypatch):
    monkeypatch.setenv("VLLM_RAON_MLP_SEGMENT_GRAPH_BUCKETS", "2048")
    _select_device(monkeypatch, (9, 0))
    assert graphs.configured_buckets(0) == (2048,)


def test_precapture_requires_buckets(monkeypatch):
    monkeypatch.setenv("VLLM_RAON_MLP_SEGMENT_GRAPH_PRECAPTURE", "1")
    monkeypatch.delenv("VLLM_RAON_OPTIMIZATIONS", raising=False)
    monkeypatch.delenv("VLLM_RAON_MLP_SEGMENT_GRAPH_BUCKETS", raising=False)
    _select_device(monkeypatch, (10, 3))
    assert not graphs.precapture_enabled(0)
    monkeypatch.setenv("VLLM_RAON_MLP_SEGMENT_GRAPH_BUCKETS", "1280")
    graphs._buckets_cache.clear()
    assert graphs.precapture_enabled(0)


def test_precapture_profile_on_sm103_only(monkeypatch):
    monkeypatch.setenv("VLLM_RAON_OPTIMIZATIONS", "full")
    monkeypatch.delenv("VLLM_RAON_MLP_SEGMENT_GRAPH_PRECAPTURE", raising=False)
    monkeypatch.delenv("VLLM_RAON_MLP_SEGMENT_GRAPH_BUCKETS", raising=False)
    _select_device(monkeypatch, (10, 3))
    assert graphs.precapture_enabled(0)
    raon_optimizations.enabled.cache_clear()
    graphs._buckets_cache.clear()
    _select_device(monkeypatch, (8, 9))
    assert not graphs.precapture_enabled(0)

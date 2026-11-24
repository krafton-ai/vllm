# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.layers import raon_optimizations


@pytest.fixture(autouse=True)
def _clear_cache():
    raon_optimizations.enabled.cache_clear()
    yield
    raon_optimizations.enabled.cache_clear()


@pytest.fixture
def fake_table(monkeypatch):
    monkeypatch.setattr(
        raon_optimizations,
        "_DEVICES",
        {"opt_a": ((8, 9),), "opt_b": ((8, 9),), "opt_mem": ((8, 9),)},
    )
    monkeypatch.setattr(raon_optimizations, "_ADDS_MEMORY", frozenset({"opt_mem"}))


def _select_device(monkeypatch, capability):
    platform = raon_optimizations.current_platform
    monkeypatch.setattr(platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        platform,
        "is_device_capability",
        lambda value, device_id=0: tuple(value) == capability,
    )


def _selected():
    return {
        name for name in raon_optimizations._DEVICES if raon_optimizations.enabled(name)
    }


def _selected_with(monkeypatch, mode):
    raon_optimizations.enabled.cache_clear()
    if mode is None:
        monkeypatch.delenv("VLLM_RAON_OPTIMIZATIONS", raising=False)
    else:
        monkeypatch.setenv("VLLM_RAON_OPTIMIZATIONS", mode)
    return _selected()


_CAPABILITIES = [(8, 9), (10, 3), (11, 0), (9, 0), (12, 0)]


@pytest.mark.parametrize("capability", _CAPABILITIES)
def test_unset_selects_mode_1(monkeypatch, capability):
    monkeypatch.delenv("VLLM_RAON_OPTIMIZATIONS_EXCLUDE", raising=False)
    _select_device(monkeypatch, capability)
    assert _selected_with(monkeypatch, None) == _selected_with(monkeypatch, "1")


@pytest.mark.parametrize("mode", ["0", "off"])
@pytest.mark.parametrize("capability", _CAPABILITIES)
def test_off_selects_nothing(monkeypatch, capability, mode):
    monkeypatch.delenv("VLLM_RAON_OPTIMIZATIONS_EXCLUDE", raising=False)
    _select_device(monkeypatch, capability)
    assert _selected_with(monkeypatch, mode) == set()


# Per device: the default set ("1") and the full set ("full").
@pytest.mark.parametrize(
    ("capability", "default", "full"),
    [
        (
            (8, 9),
            {"marlin_single_row_routing", "mhc_split_k", "mhc_single_row"},
            {"marlin_single_row_routing", "mhc_split_k", "mhc_single_row"},
        ),
        (
            (10, 3),
            {"marlin_single_row_routing", "marlin_reuse_single_row_scratch"},
            {
                "marlin_single_row_routing",
                "marlin_resident_bf16",
                "marlin_reuse_single_row_scratch",
            },
        ),
        (
            (11, 0),
            {"marlin_single_row_routing", "mhc_single_row"},
            {
                "marlin_single_row_routing",
                "marlin_resident_bf16",
                "mhc_single_row",
            },
        ),
        (
            (9, 0),
            {"marlin_single_row_routing"},
            {"marlin_single_row_routing"},
        ),
        (
            (12, 0),
            {"marlin_single_row_routing"},
            {"marlin_single_row_routing"},
        ),
    ],
)
def test_device_selection(monkeypatch, capability, default, full):
    monkeypatch.delenv("VLLM_RAON_OPTIMIZATIONS_EXCLUDE", raising=False)
    _select_device(monkeypatch, capability)
    monkeypatch.setenv("VLLM_RAON_OPTIMIZATIONS", "1")
    assert _selected() == default
    raon_optimizations.enabled.cache_clear()
    monkeypatch.setenv("VLLM_RAON_OPTIMIZATIONS", "full")
    assert _selected() == full


@pytest.mark.parametrize("capability", [(9, 0), (12, 0)])
def test_single_row_routing_on_any_cuda_device(monkeypatch, capability):
    monkeypatch.delenv("VLLM_RAON_OPTIMIZATIONS_EXCLUDE", raising=False)
    _select_device(monkeypatch, capability)
    for mode in (None, "1"):
        assert _selected_with(monkeypatch, mode) == {"marlin_single_row_routing"}
    for mode in ("0", "off"):
        assert _selected_with(monkeypatch, mode) == set()


def test_any_device_entry(monkeypatch):
    monkeypatch.setattr(
        raon_optimizations, "_DEVICES", {"opt_any": None, "opt_a": ((8, 9),)}
    )
    monkeypatch.setattr(raon_optimizations, "_ADDS_MEMORY", frozenset())
    monkeypatch.delenv("VLLM_RAON_OPTIMIZATIONS_EXCLUDE", raising=False)
    _select_device(monkeypatch, (12, 0))
    assert _selected_with(monkeypatch, None) == {"opt_any"}
    assert _selected_with(monkeypatch, "off") == set()
    monkeypatch.setattr(raon_optimizations.current_platform, "is_cuda", lambda: False)
    assert _selected_with(monkeypatch, "1") == set()


def test_exclude(monkeypatch):
    monkeypatch.setenv("VLLM_RAON_OPTIMIZATIONS", "1")
    monkeypatch.setenv("VLLM_RAON_OPTIMIZATIONS_EXCLUDE", "mhc_split_k")
    _select_device(monkeypatch, (8, 9))
    assert not raon_optimizations.enabled("mhc_split_k")
    assert raon_optimizations.enabled("mhc_single_row")


def test_exclude_unknown(monkeypatch):
    monkeypatch.setenv("VLLM_RAON_OPTIMIZATIONS", "1")
    monkeypatch.setenv("VLLM_RAON_OPTIMIZATIONS_EXCLUDE", "no_such_feature")
    monkeypatch.setattr(raon_optimizations.current_platform, "is_cuda", lambda: True)
    with pytest.raises(ValueError):
        raon_optimizations.enabled("mhc_split_k")


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (None, {"opt_a", "opt_b"}),
        ("1", {"opt_a", "opt_b"}),
        ("full", {"opt_a", "opt_b", "opt_mem"}),
        ("0", set()),
        ("off", set()),
    ],
)
def test_memory_entries_need_full_mode(monkeypatch, fake_table, mode, expected):
    monkeypatch.delenv("VLLM_RAON_OPTIMIZATIONS_EXCLUDE", raising=False)
    _select_device(monkeypatch, (8, 9))
    assert _selected_with(monkeypatch, mode) == expected


def test_unknown_mode_rejected(monkeypatch, fake_table):
    monkeypatch.setenv("VLLM_RAON_OPTIMIZATIONS", "fast")
    monkeypatch.setattr(raon_optimizations.current_platform, "is_cuda", lambda: True)
    with pytest.raises(ValueError):
        raon_optimizations.enabled("opt_a")


def test_resident_threshold_per_device(monkeypatch):
    _select_device(monkeypatch, (10, 3))
    assert raon_optimizations.resident_bf16_min_rows() == 512
    _select_device(monkeypatch, (11, 0))
    assert raon_optimizations.resident_bf16_min_rows() == 256


@pytest.mark.parametrize("capability", [(8, 9), (9, 0), (12, 0)])
def test_resident_threshold_off_table(monkeypatch, capability):
    _select_device(monkeypatch, capability)
    assert raon_optimizations.resident_bf16_min_rows() is None


def test_resident_devices_follow_table():
    devices = raon_optimizations._DEVICES["marlin_resident_bf16"]
    assert devices == tuple(raon_optimizations._RESIDENT_BF16_MIN_ROWS)
    assert devices == ((10, 3), (11, 0))

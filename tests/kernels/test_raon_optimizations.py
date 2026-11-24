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
            set(),
            set(),
        ),
        (
            (10, 3),
            set(),
            set(),
        ),
        (
            (11, 0),
            set(),
            set(),
        ),
        (
            (9, 0),
            set(),
            set(),
        ),
        (
            (12, 0),
            set(),
            set(),
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


def test_exclude(monkeypatch, fake_table):
    monkeypatch.setenv("VLLM_RAON_OPTIMIZATIONS", "1")
    monkeypatch.setenv("VLLM_RAON_OPTIMIZATIONS_EXCLUDE", "opt_a")
    _select_device(monkeypatch, (8, 9))
    assert not raon_optimizations.enabled("opt_a")
    assert raon_optimizations.enabled("opt_b")


def test_exclude_unknown(monkeypatch, fake_table):
    monkeypatch.setenv("VLLM_RAON_OPTIMIZATIONS", "1")
    monkeypatch.setenv("VLLM_RAON_OPTIMIZATIONS_EXCLUDE", "no_such_feature")
    monkeypatch.setattr(raon_optimizations.current_platform, "is_cuda", lambda: True)
    with pytest.raises(ValueError):
        raon_optimizations.enabled("opt_a")


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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.models.raon1_mini import effort_steering

SIX_LEVELS = {
    "alpha": {"low": 0.3, "medium": 0.1, "high": 0.05, "xhigh": 0.02, "max": 0},
    "default": "high",
    "layers": [12, 14, 16, 18, 20],
}


@pytest.fixture
def configure(monkeypatch):
    monkeypatch.delenv(effort_steering.ENV, raising=False)
    monkeypatch.setattr(effort_steering, "CONFIG", None)

    def run(steering):
        effort_steering.configure(
            SimpleNamespace(reasoning_effort_steering=steering), "model"
        )

    return run


def test_six_levels(configure):
    configure(SIX_LEVELS)
    alpha_for = effort_steering.alpha_for
    assert alpha_for("low") == 0.3
    assert alpha_for("medium") == 0.1
    assert alpha_for("high") == 0.05
    assert alpha_for("xhigh") == 0.02
    assert alpha_for("max") == 0.0
    assert alpha_for(None) == 0.05  # the default, high
    assert alpha_for("none") == 0.0  # thinking off, never steered
    for effort in ("mid", "minimal"):
        with pytest.raises(ValueError, match="reasoning_effort must be one of"):
            alpha_for(effort)


def test_default_can_be_unsteered(configure):
    configure({**SIX_LEVELS, "default": "max"})
    assert effort_steering.alpha_for(None) == 0.0


def test_three_level_config_still_loads(configure):
    configure({"alpha": {"low": 0.2, "medium": 0.1, "high": 0}, "layers": [12]})
    assert effort_steering.alpha_for("medium") == 0.1
    assert effort_steering.alpha_for(None) == 0.0
    with pytest.raises(ValueError):
        effort_steering.alpha_for("mid")


@pytest.mark.parametrize(
    "steering",
    [
        {**SIX_LEVELS, "default": "mid"},
        {**SIX_LEVELS, "alpha": {**SIX_LEVELS["alpha"], "none": 0.5}},
        {**SIX_LEVELS, "alpha": {**SIX_LEVELS["alpha"], "none": 0}, "default": "none"},
        {**SIX_LEVELS, "alpha": {"low": -0.1}},
        {**SIX_LEVELS, "alpha": {}},
    ],
)
def test_invalid_config(configure, steering):
    with pytest.raises(ValueError):
        configure(steering)


def test_none_entry_with_zero_alpha_loads(configure):
    configure({**SIX_LEVELS, "alpha": {"none": 0, **SIX_LEVELS["alpha"]}})
    assert effort_steering.alpha_for("none") == 0.0
    assert effort_steering.alpha_for(None) == 0.05
    with pytest.raises(ValueError, match="one of none, ") as error:
        effort_steering.alpha_for("mid")
    assert str(error.value).count("none") == 1


def test_none_turns_thinking_off():
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    def thinking(effort):
        request = ChatCompletionRequest(
            model="raon",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort=effort,
        )
        params = request.build_chat_params(None, "auto")
        return params.chat_template_kwargs.get("enable_thinking")

    assert thinking("none") is False
    assert thinking("high") is True
    assert thinking(None) is None

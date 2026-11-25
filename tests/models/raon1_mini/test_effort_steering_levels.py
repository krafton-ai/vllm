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
    assert alpha_for("none") == 0.1  # thinking off: NO_THINKING_ALPHA
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
        {**SIX_LEVELS, "alpha": {**SIX_LEVELS["alpha"], "none": 0}, "default": "none"},
        {**SIX_LEVELS, "alpha": {"low": -0.1}},
        {**SIX_LEVELS, "alpha": {}},
    ],
)
def test_invalid_config(configure, steering):
    with pytest.raises(ValueError):
        configure(steering)


def test_none_entry_with_zero_alpha_loads(configure):
    # the released configs list none with 0, written when thinking off was never steered
    configure({**SIX_LEVELS, "alpha": {"none": 0, **SIX_LEVELS["alpha"]}})
    assert effort_steering.alpha_for("none") == 0.1
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


def test_none_entry_above_zero_is_the_no_thinking_alpha(configure):
    configure({**SIX_LEVELS, "alpha": {"none": 0.5, **SIX_LEVELS["alpha"]}})
    assert effort_steering.alpha_for("none") == 0.5
    assert effort_steering.alpha_for(None) == 0.05


def test_negative_none_entry_is_refused(configure):
    with pytest.raises(ValueError):
        configure({**SIX_LEVELS, "alpha": {"none": -0.1, **SIX_LEVELS["alpha"]}})


@pytest.mark.parametrize(
    "server_kwargs, request_kwargs, expected",
    [
        # thinking on: the effort selects alpha, as before
        ({}, {}, 0.05),
        ({}, {"reasoning_effort": "low"}, 0.3),
        ({}, {"reasoning_effort": "max"}, 0.0),
        # thinking off by any route: 0.1 whatever the effort
        ({}, {"reasoning_effort": "none"}, 0.1),
        ({}, {"chat_template_kwargs": {"enable_thinking": False}}, 0.1),
        ({}, {"reasoning_effort": "max",
              "chat_template_kwargs": {"enable_thinking": False}}, 0.1),
        ({}, {"chat_template_kwargs": {"reasoning_effort": "none"}}, 0.1),
        ({"enable_thinking": False}, {}, 0.1),  # --default-chat-template-kwargs
        ({"enable_thinking": False}, {"reasoning_effort": "low"}, 0.3),  # effort turns thinking on
        ({"enable_thinking": False},
         {"chat_template_kwargs": {"enable_thinking": True}}, 0.05),
    ],
)
def test_thinking_off_is_steered(configure, server_kwargs, request_kwargs, expected):
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    configure({**SIX_LEVELS, "alpha": {"none": 0, **SIX_LEVELS["alpha"]}})
    serving = SimpleNamespace(
        chat_template=None,
        chat_template_content_format="auto",
        default_chat_template_kwargs=server_kwargs,
    )
    serving._effective_chat_template_kwargs = (
        lambda request: OpenAIServingChat._effective_chat_template_kwargs(serving, request)
    )
    request = ChatCompletionRequest(
        model="raon", messages=[{"role": "user", "content": "hi"}], **request_kwargs
    )
    assert effort_steering._request_effort_alpha(serving, request) == expected


@pytest.mark.parametrize(
    "request_kwargs, expected",
    [
        ({}, 0.05),  # no effort: the default, high
        ({"reasoning_effort": "none"}, 0.1),
        ({"chat_template_kwargs": {"enable_thinking": False}}, 0.1),
        ({"reasoning_effort": "low"}, 0.3),
        ({"reasoning_effort": "max"}, 0.0),
    ],
)
def test_batch_conversations_are_tagged(configure, request_kwargs, expected):
    from vllm.entrypoints.openai.chat_completion.protocol import (
        BatchChatCompletionRequest,
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    configure({**SIX_LEVELS, "alpha": {"none": 0, **SIX_LEVELS["alpha"]}})
    serving = SimpleNamespace(
        chat_template=None,
        chat_template_content_format="auto",
        default_chat_template_kwargs={},
    )
    serving._effective_chat_template_kwargs = (
        lambda request: OpenAIServingChat._effective_chat_template_kwargs(serving, request)
    )
    conversations = [[{"role": "user", "content": "hi"}], [{"role": "user", "content": "1+1?"}]]
    batch = BatchChatCompletionRequest(model="raon", messages=conversations, **request_kwargs)
    singles = [batch.to_chat_completion_request(m) for m in batch.messages]
    assert effort_steering.batch_request_error(serving, singles) is None
    plain = ChatCompletionRequest(model="raon", messages=conversations[0], **request_kwargs)
    assert effort_steering._request_effort_alpha(serving, plain) == expected
    for single in singles:
        params = single.to_sampling_params(16, {})
        assert effort_steering.request_alpha(params) == expected
        request = SimpleNamespace(sampling_params=params, cache_salt=None)
        effort_steering._engine_request(request)  # the engine keeps the tagged alpha
        assert effort_steering.request_alpha(request.sampling_params) == expected


def test_batch_unknown_effort_is_refused(configure):
    from vllm.entrypoints.openai.chat_completion.protocol import (
        BatchChatCompletionRequest,
    )
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    configure(SIX_LEVELS)
    serving = SimpleNamespace(
        chat_template=None, chat_template_content_format="auto", default_chat_template_kwargs={}
    )
    serving._effective_chat_template_kwargs = (
        lambda request: OpenAIServingChat._effective_chat_template_kwargs(serving, request)
    )
    batch = BatchChatCompletionRequest(
        model="raon", messages=[[{"role": "user", "content": "hi"}]],
        chat_template_kwargs={"reasoning_effort": "mid"},
    )
    singles = [batch.to_chat_completion_request(m) for m in batch.messages]
    assert "reasoning_effort must be one of" in effort_steering.batch_request_error(serving, singles)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load both public checkpoints, including NVFP4's padded RDA projection."""

from typing import Any

import pytest

from vllm import SamplingParams


@pytest.mark.parametrize(
    ("model", "revision"),
    [
        pytest.param(
            "KRAFTON/Raon-1-mini-preview",
            "8c9b802acacdbda9df16606689fcf437cf7262fb",
            id="mxfp8",
        ),
        pytest.param(
            "KRAFTON/Raon-1-mini-preview-nvfp4",
            "d106d914c10bb2dda45fc5dc8ac369b368a88c4b",
            id="nvfp4",
        ),
    ],
)
@pytest.mark.parametrize("with_dflash", [False, True])
def test_public_checkpoint_completes_prompts(
    vllm_runner, monkeypatch, model, revision, with_dflash
):
    # Exercise the default backend selection without raon_fast/attention overrides.
    kwargs: dict[str, Any] = {}
    if with_dflash:
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "BLNHC")
        kwargs.update(
            kv_cache_dtype="fp8_e4m3",
            speculative_config={
                "method": "dflash",
                "model": "KRAFTON/Raon-1-mini-preview-dflash2",
                "revision": "164b3dda020f145a4650b3c050841e15250c2f77",
                "num_speculative_tokens": 7,
                "draft_sample_method": "probabilistic",
                "kv_cache_dtype": "auto",
            },
        )
    with vllm_runner(
        model,
        revision=revision,
        dtype="bfloat16",
        load_format="safetensors",
        trust_remote_code=False,
        max_model_len=8192 if with_dflash else 128,
        max_num_seqs=2,
        enable_prefix_caching=with_dflash,
        enable_chunked_prefill=True,
        enforce_eager=not with_dflash,
        disable_log_stats=False,
        **kwargs,
    ) as runner:
        outputs = runner.llm.generate(
            ["The capital of France is", "1 + 1 ="],
            SamplingParams(temperature=0, max_tokens=16, seed=0),
        )

        if with_dflash:
            counters = {
                metric.name: metric.value
                for metric in runner.llm.get_metrics()
                if hasattr(metric, "value")
            }
            assert counters["vllm:spec_decode_num_drafts"] > 0
            assert counters["vllm:spec_decode_num_accepted_tokens"] > 0

    for output, expected in zip(outputs, ["Paris", "2"], strict=True):
        completion = output.outputs[0]
        assert completion.token_ids
        assert completion.text.strip().startswith(expected), completion.text

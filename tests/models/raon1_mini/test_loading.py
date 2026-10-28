# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load both public checkpoints, including NVFP4's padded RDA projection."""

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
def test_public_checkpoint_completes_prompts(vllm_runner, model, revision):
    with vllm_runner(
        model,
        revision=revision,
        dtype="bfloat16",
        load_format="safetensors",
        trust_remote_code=False,
        max_model_len=128,
        max_num_seqs=2,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        enforce_eager=True,
    ) as runner:
        outputs = runner.llm.generate(
            ["The capital of France is", "1 + 1 ="],
            SamplingParams(temperature=0, max_tokens=16),
        )

    for output, expected in zip(outputs, ["Paris", "2"], strict=True):
        completion = output.outputs[0]
        assert completion.token_ids
        assert completion.text.strip().startswith(expected), completion.text

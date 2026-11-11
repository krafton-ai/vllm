# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint contract for the separately distributed Nano MTP head."""

import os


def validate_speculation(vllm_config) -> None:
    spec = vllm_config.speculative_config
    if spec.method != "mtp" or spec.num_speculative_tokens != 1:
        raise ValueError("Raon Nano currently supports MTP with one draft token only")
    if str(vllm_config.model_config.dtype) != "torch.bfloat16":
        raise ValueError("Raon Nano MTP requires BF16 model weights")
    parallel = vllm_config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        raise ValueError("Raon Nano MTP requires TP=PP=1")
    if not vllm_config.model_config.enforce_eager:
        raise ValueError("Raon Nano MTP currently requires enforce_eager=True")
    if os.environ.get("RAON_KDA_CPR", "0") != "0":
        raise ValueError("Raon Nano MTP is incompatible with RAON_KDA_CPR")


def validate_mtp_metadata(metadata: dict, config) -> None:
    required = {
        "num_nextn_predict_layers": 1,
        "weight_prefix": "model.layers.24.",
        "shared_embeddings_and_lm_head": True,
        "attention_type": "mla",
        "per_layer_embedding_dim": 0,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Unsupported MTP metadata: {key}={metadata.get(key)!r}")
    target = metadata["target_config"]
    for key in (
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "vocab_size",
        "num_attention_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "raon_mhc_streams",
        "rms_norm_eps",
        "mla_use_nope",
        "mla_use_output_gate",
    ):
        if target[key] != getattr(config, key):
            raise ValueError(f"MTP target architecture mismatch: {key}")
    if config.num_hidden_layers != 24:
        raise ValueError("Released Nano MTP requires a 24-layer target")


def map_mtp_weight(name: str) -> str:
    prefix = "model.layers.24."
    if not name.startswith(prefix):
        raise ValueError(f"Unexpected MTP tensor: {name}")
    suffix = name.removeprefix(prefix)
    if suffix.startswith(("enorm.", "hnorm.", "eh_proj.")):
        return "model." + suffix
    if suffix.startswith("shared_head.norm."):
        return "model.norm." + suffix.removeprefix("shared_head.norm.")
    return (prefix + suffix).replace("_raon_mhc.", "_mhc.")


def load_mtp_tensors(path):
    """Read safetensors through its keys API (safe_open is not iterable)."""
    from safetensors import safe_open

    with safe_open(path, framework="pt") as checkpoint:
        names = checkpoint.keys()
        return [(map_mtp_weight(name), checkpoint.get_tensor(name)) for name in names]

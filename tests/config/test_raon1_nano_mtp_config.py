# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU contract tests; these do not claim CUDA generation correctness."""

import copy
from types import SimpleNamespace

import pytest

from vllm.models.raon1_nano import mtp_weights as MODULE
from vllm.models.raon1_nano.config import Raon1NanoConfig

METADATA = {
    "num_nextn_predict_layers": 1,
    "weight_prefix": "model.layers.24.",
    "shared_embeddings_and_lm_head": True,
    "attention_type": "mla",
    "per_layer_embedding_dim": 0,
    "target_config": Raon1NanoConfig().to_dict(),
}


def settings(**changes):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=1),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        model_config=SimpleNamespace(enforce_eager=True, dtype="torch.bfloat16"),
    )
    for obj, attr, value in changes.get("fields", []):
        setattr(getattr(config, obj), attr, value)
    return config


def test_one_token_eager_mtp_is_allowed(monkeypatch):
    monkeypatch.delenv("RAON_KDA_CPR", raising=False)
    MODULE.validate_speculation(settings())


@pytest.mark.parametrize(
    "obj,attr,value",
    [
        ("speculative_config", "method", "eagle"),
        ("speculative_config", "num_speculative_tokens", 2),
        ("parallel_config", "tensor_parallel_size", 2),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("model_config", "enforce_eager", False),
        ("model_config", "dtype", "torch.float16"),
    ],
)
def test_unsupported_modes_fail_before_state_allocation(obj, attr, value):
    with pytest.raises(ValueError):
        MODULE.validate_speculation(settings(fields=[(obj, attr, value)]))


def test_cpr_rejected_before_runtime(monkeypatch):
    monkeypatch.setenv("RAON_KDA_CPR", "1")
    with pytest.raises(ValueError, match="CPR"):
        MODULE.validate_speculation(settings())


def test_released_metadata_accepted():
    MODULE.validate_mtp_metadata(METADATA, SimpleNamespace(**METADATA["target_config"]))


@pytest.mark.parametrize(
    "key",
    [
        "hidden_size",
        "num_hidden_layers",
        "vocab_size",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "raon_mhc_streams",
        "num_attention_heads",
        "intermediate_size",
    ],
)
def test_incompatible_auxiliary_head_rejected(key):
    target = copy.deepcopy(METADATA["target_config"])
    target[key] += 1
    with pytest.raises(ValueError, match="mismatch"):
        MODULE.validate_mtp_metadata(METADATA, SimpleNamespace(**target))


@pytest.mark.parametrize(
    "source,target",
    [
        ("enorm.weight", "model.enorm.weight"),
        ("hnorm.weight", "model.hnorm.weight"),
        ("eh_proj.weight", "model.eh_proj.weight"),
        ("shared_head.norm.weight", "model.norm.weight"),
        ("self_attn.q_a_proj.weight", "model.layers.24.self_attn.q_a_proj.weight"),
        ("self_attn_raon_mhc.res.phi_pre", "model.layers.24.self_attn_mhc.res.phi_pre"),
    ],
)
def test_checkpoint_tensor_mapping(source, target):
    assert MODULE.map_mtp_weight("model.layers.24." + source) == target


def test_target_tensor_cannot_be_loaded_as_mtp():
    with pytest.raises(ValueError, match="Unexpected"):
        MODULE.map_mtp_weight("model.layers.0.self_attn.q_proj.weight")


def test_real_safetensors_reader_uses_keys_api(tmp_path):
    import torch
    from safetensors.torch import save_file

    tensor = torch.arange(4, dtype=torch.bfloat16)
    path = tmp_path / "mtp.safetensors"
    save_file({"model.layers.24.hnorm.weight": tensor}, str(path))
    loaded = MODULE.load_mtp_tensors(path)
    assert len(loaded) == 1
    assert loaded[0][0] == "model.hnorm.weight"
    assert torch.equal(loaded[0][1], tensor)


def test_safetensors_reader_rejects_target_weights(tmp_path):
    import torch
    from safetensors.torch import save_file

    path = tmp_path / "target.safetensors"
    save_file({"model.layers.0.hnorm.weight": torch.ones(4)}, str(path))
    with pytest.raises(ValueError, match="Unexpected MTP tensor"):
        MODULE.load_mtp_tensors(path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_nextn_predict_layers", 2),
        ("weight_prefix", "model.layers.0."),
        ("shared_embeddings_and_lm_head", False),
        ("attention_type", "rda"),
        ("per_layer_embedding_dim", 256),
    ],
)
def test_unsupported_mtp_checkpoint_metadata_rejected(key, value):
    metadata = copy.deepcopy(METADATA)
    metadata[key] = value
    with pytest.raises(ValueError, match="Unsupported MTP metadata"):
        MODULE.validate_mtp_metadata(
            metadata, SimpleNamespace(**metadata["target_config"])
        )


def test_non_release_layer_count_rejected_even_if_metadata_matches():
    metadata = copy.deepcopy(METADATA)
    metadata["target_config"]["num_hidden_layers"] = 25
    with pytest.raises(ValueError, match="24-layer"):
        MODULE.validate_mtp_metadata(
            metadata, SimpleNamespace(**metadata["target_config"])
        )

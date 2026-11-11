# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native one-token MTP proposer; auxiliary weights remain under ``mtp/``."""

import copy
import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.transformers_utils.repo_utils import hf_api

from .mhc import MHCSite
from .model import QuackRMSNorm, RaonDecoderLayer, RaonModel
from .mtp_weights import load_mtp_tensors, validate_mtp_metadata, validate_speculation


class RaonNanoPredictor(RaonModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str):
        nn.Module.__init__(self)
        config = copy.deepcopy(vllm_config.model_config.hf_config)
        config.per_layer_embedding_dim = 0
        self.config = config
        self.streams = config.mhc_streams
        self._phi_seen = {}
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.enorm = QuackRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = QuackRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.layers = nn.ModuleDict(
            {
                "24": RaonDecoderLayer(
                    config, vllm_config, f"{prefix}.layers.24", torch.cuda.Stream()
                )
            }
        )
        self.norm = QuackRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions, inputs_embeds=None, hidden_states=None):
        assert hidden_states is not None
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = self.eh_proj(
            torch.cat([self.enorm(inputs_embeds), self.hnorm(hidden_states)], dim=-1)
        )
        streams = hidden_states.unsqueeze(1).expand(-1, self.streams, -1).contiguous()
        streams = self.layers["24"](positions, streams)
        return self.norm(streams.float().mean(1).to(hidden_states.dtype))


class Raon1NanoMTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        validate_speculation(vllm_config)
        if vllm_config.quant_config is not None:
            raise ValueError("Raon Nano MTP supports unquantized BF16 only")
        assert vllm_config.speculative_config is not None
        draft_config = vllm_config.speculative_config.draft_model_config
        assert draft_config is not None
        draft_config = copy.copy(draft_config)
        if draft_config.hf_token is None:
            draft_config.hf_token = vllm_config.model_config.hf_token
        vllm_config = replace(vllm_config, model_config=draft_config)
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.model = RaonNanoPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(self.config.vocab_size, self.config.hidden_size)
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids,
        positions,
        hidden_states,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        return self.model(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
            hidden_states=hidden_states,
        )

    def compute_logits(self, hidden_states, spec_step_idx=0):
        return self.logits_processor(self.lm_head, hidden_states)

    def _auxiliary_file(self, filename: str) -> Path:
        model_config = self.vllm_config.model_config
        local = Path(model_config.model)
        if local.is_dir():
            path = local / "mtp" / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing separate Nano MTP checkpoint: {path}")
            return path
        return Path(
            hf_api().hf_hub_download(
                model_config.model,
                f"mtp/{filename}",
                revision=model_config.revision,
                token=model_config.hf_token,
                cache_dir=self.vllm_config.load_config.download_dir,
            )
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Do not consume the target's root index: the MTP has its own checkpoint.
        metadata = json.loads(self._auxiliary_file("mtp_config.json").read_text())
        validate_mtp_metadata(metadata, self.config)
        mapped = load_mtp_tensors(self._auxiliary_file("model.safetensors"))
        loaded = AutoWeightsLoader(self).load_weights(mapped)
        shared = {"model.embed_tokens.weight", "lm_head.weight"}
        missing = set(dict(self.named_parameters())) - loaded - shared
        if missing:
            raise ValueError(f"Missing MTP weights: {sorted(missing)}")
        for module in self.modules():
            if isinstance(module, MHCSite):
                module.finalize()
        # The native MTP proposer replaces these modules with the target's
        # embedding/head in load_model(), before any forward or graph capture.
        return loaded | shared

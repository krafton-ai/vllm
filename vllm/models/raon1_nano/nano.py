# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raon-1 nano PLE layout 2, tied embeddings and RDA output normalization.

The dense decoder lives in model.py. Only TP=PP=1 is supported. Attention
primitives are derived from Raon-1 mini; see UPSTREAM.md.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import RaonNanoModelBase


class PleRMSNorm(nn.Module):
    """RMSNorm with training BF16 boundaries and QuACK FP32 affine products."""

    def __init__(self, width: int, eps: float, backend: str) -> None:
        super().__init__()
        if backend not in ("quack", "eager"):
            raise ValueError(f"Unsupported training PLE norm: {backend}")
        self.weight = nn.Parameter(torch.empty(width))
        self.eps, self.backend = eps, backend

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        x = value.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        if self.backend == "quack":
            return (x * self.weight.float()).to(value.dtype)
        return self.weight * x.to(value.dtype)


class PerLayerInputs(nn.Module):
    """Token table and embedding projection, with one input per layer."""

    def __init__(self, config) -> None:
        super().__init__()
        self.layers = config.num_hidden_layers
        self.width = config.per_layer_embedding_dim
        self.table = nn.Parameter(
            torch.empty(config.vocab_size, self.layers * self.width)
        )
        self.embedding_projection = nn.Parameter(
            torch.empty(self.layers * self.width, config.hidden_size)
        )
        self.projection_norm = PleRMSNorm(
            self.width,
            config.rms_norm_eps,
            getattr(config, "ple_norm_backend", "quack"),
        )

    def forward(
        self, input_ids: torch.Tensor, embeddings: torch.Tensor
    ) -> torch.Tensor:
        projected = self.projection_norm(
            F.linear(embeddings, self.embedding_projection).reshape(
                -1, self.layers, self.width
            )
        )
        rows = F.embedding(input_ids, self.table).reshape(-1, self.layers, self.width)
        rows = rows * math.sqrt(self.width)
        return (projected + rows) * (1.0 / math.sqrt(2.0))


class PerLayerInputGate(nn.Module):
    """One layer's gate: reads the post-FFN streams, adds the gated per-layer input."""

    def __init__(self, config) -> None:
        super().__init__()
        width = config.per_layer_embedding_dim
        self.gate = nn.Parameter(torch.empty(width, config.hidden_size))
        self.projection = nn.Parameter(torch.empty(config.hidden_size, width))
        self.output_norm = PleRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            getattr(config, "ple_norm_backend", "quack"),
        )

    def forward(self, streams: torch.Tensor, layer_input: torch.Tensor) -> torch.Tensor:
        hidden = streams.mean(dim=1, dtype=torch.float32).to(self.gate.dtype)
        gated = F.gelu(F.linear(hidden, self.gate), approximate="tanh") * layer_input
        delta = self.output_norm(F.linear(gated, self.projection))
        return streams + delta.to(streams.dtype).unsqueeze(1)


class Raon1NanoForCausalLM(RaonNanoModelBase):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = self.config
        # The release uses rms_norm_eps for RDA output normalization.
        # Override the upstream attention primitive's fixed 1e-5 default.
        for layer in self.model.layers:
            if hasattr(layer.self_attn, "o_norm"):
                layer.self_attn.o_norm.eps = config.rms_norm_eps
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.ple_width = getattr(config, "per_layer_embedding_dim", 0) or 0
        if self.ple_width:
            parallel = vllm_config.parallel_config
            if (
                parallel.tensor_parallel_size != 1
                or parallel.pipeline_parallel_size != 1
            ):
                raise ValueError(
                    "Raon-1-nano PLE requires TP = PP = 1; use one replica per GPU"
                )
            if getattr(config, "per_layer_embedding_layout_version", None) != 2:
                raise ValueError("Only the post-FFN PLE layout version 2 is supported")
            self.model.per_layer_inputs = PerLayerInputs(config)
            for layer in self.model.layers:
                layer.per_layer_input_gate = PerLayerInputGate(config)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if not self.ple_width:
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
            )
        if input_ids is None:
            raise ValueError("PLE needs token ids for the learned per-layer table")
        model = self.model
        if inputs_embeds is None:
            inputs_embeds = model.embed_input_ids(input_ids)
        per_layer = model.per_layer_inputs(input_ids, inputs_embeds)
        streams = inputs_embeds.unsqueeze(1).expand(-1, model.streams, -1).contiguous()
        for index, layer in enumerate(model.layers):
            streams = layer(positions, streams)
            streams = layer.per_layer_input_gate(streams, per_layer[:, index, :])
        return model.norm(streams.mean(1))

    def load_weights(self, weights):
        # The release names the Raon_mHC sites ``self_attn_raon_mhc`` / ``mlp_raon_mhc``
        # in HF; the engine modules are ``self_attn_mhc`` / ``mlp_mhc``.
        weights = (
            (name.replace("_raon_mhc.", "_mhc.", 1), *rest) for name, *rest in weights
        )
        if getattr(self.config, "tie_word_embeddings", False):
            # lm_head is embed_tokens; a materialized copy in a bundle is the same
            # tensor.
            weights = (
                (name, *rest) for name, *rest in weights if name != "lm_head.weight"
            )
        return super().load_weights(weights)

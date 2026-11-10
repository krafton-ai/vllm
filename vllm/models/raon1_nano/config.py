# SPDX-License-Identifier: Apache-2.0
"""Nano release configuration adapted to vLLM's hybrid attention interface.

Architecture defaults mirror the released Raon-1 nano HF configuration.
Only the engine-facing compatibility views below use upstream field names;
the model retains rda_config and raon_mhc_* names. See UPSTREAM.md.
"""

from __future__ import annotations

from typing import ClassVar

from transformers.configuration_utils import PretrainedConfig

_RELEASED_RDA_LAYERS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23]
_RELEASED_MLA_LAYERS = [4, 8, 12, 16, 20, 24]


class Raon1NanoConfig(PretrainedConfig):
    """Configuration of a raon-1-nano model.

    Layer indices inside ``rda_config`` (``rda_layers``: Raon Delta Attention,
    ``mla_layers``:
    MLA) are 1-based; ``layer_types`` converts.
    """

    model_type = "raon1_nano"
    keys_to_ignore_at_inference: ClassVar[list[str]] = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 259840,
        hidden_size: int = 1536,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 32,
        head_dim: int | None = 128,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        mla_latent_norm_eps: float = 1e-6,
        raon_mhc_selector_eps: float | None = None,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        max_position_embeddings: int = 65536,
        # MLA
        q_lora_rank: int = 512,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        mla_use_nope: bool = True,
        mla_use_output_gate: bool = True,
        # RDA
        rda_config: dict | None = None,
        # Raon_mHC
        raon_mhc_streams: int | None = None,
        raon_mhc_post_factor: float | None = None,
        # Per-layer embedding
        per_layer_embedding_dim: int = 256,
        per_layer_embedding_layout_version: int = 2,
        rope_parameters: dict | None = None,
        pad_token_id: int = 259200,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        **kwargs,
    ) -> None:
        if hidden_act != "silu":
            raise ValueError(
                f"raon-1-nano uses a SiLU-gated MLP; got hidden_act={hidden_act!r}"
            )
        if per_layer_embedding_dim and per_layer_embedding_layout_version != 2:
            raise ValueError("Only per-layer embedding layout version 2 is supported")
        if "linear_attn_config" in kwargs or "mhc_streams" in kwargs:
            raise ValueError("Use the Nano HF rda_config and raon_mhc_streams fields")
        if kwargs.pop("num_experts", None) is not None:
            raise ValueError("Raon-1 nano is a dense model")
        self.num_experts = None
        self.num_key_value_heads = (
            kwargs.pop("num_key_value_heads", None) or num_attention_heads
        )
        self.raon_mhc_streams = 4 if raon_mhc_streams is None else raon_mhc_streams
        self.raon_mhc_post_factor = (
            2.0 if raon_mhc_post_factor is None else raon_mhc_post_factor
        )
        self.raon_mhc_selector_eps = (
            1e-5 if raon_mhc_selector_eps is None else raon_mhc_selector_eps
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.mla_latent_norm_eps = mla_latent_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope
        self.mla_use_output_gate = mla_use_output_gate
        if rda_config is None:
            rda_config = {
                "num_heads": 16,
                "head_dim": self.head_dim,
                "conv_kernel_size": 4,
                "full_rank_gate": True,
                "gate_lower_bound": -5.0,
                "rda_layers": list(_RELEASED_RDA_LAYERS),
                "mla_layers": list(_RELEASED_MLA_LAYERS),
            }
        self.rda_config = dict(rda_config)
        self.per_layer_embedding_dim = per_layer_embedding_dim
        self.per_layer_embedding_layout_version = per_layer_embedding_layout_version
        # NoPE: nothing reads this, kept so tools that expect a mapping do not trip.
        # NoPE still needs a mapping for vLLM's generic context validation.
        self.rope_parameters = dict(
            rope_parameters or {"rope_type": "default", "rope_theta": 10000.0}
        )
        # Decay rank is fixed by the released checkpoint (f_a_proj is 128 wide); it is
        # only read to build shapes.
        self.rda_decay_rank = kwargs.pop("rda_decay_rank", 128)
        if eos_token_id is None:
            eos_token_id = [259202, 259200]
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def mhc_streams(self) -> int:
        return self.raon_mhc_streams

    @property
    def linear_attn_config(self) -> dict:
        # These keys are read by the upstream attention and state allocators.
        rda = self.rda_config
        return {
            "num_heads": rda["num_heads"],
            "head_dim": rda["head_dim"],
            "short_conv_kernel_size": rda["conv_kernel_size"],
            "use_full_rank_gate": rda["full_rank_gate"],
            "gate_lower_bound": rda["gate_lower_bound"],
            "kda_layers": list(rda["rda_layers"]),
            "full_attn_layers": list(rda["mla_layers"]),
        }

    @property
    def is_mla(self) -> bool:
        return True

    @property
    def is_moe(self) -> bool:
        return False

    @property
    def is_linear_attn(self) -> bool:
        return bool(self.rda_config["rda_layers"])

    # ---- derived views
    # -------------------------------------------------------------------
    @property
    def rda_num_heads(self) -> int:
        return int(self.rda_config.get("num_heads", self.num_attention_heads))

    @property
    def rda_head_dim(self) -> int:
        return int(self.rda_config.get("head_dim", self.head_dim))

    @property
    def rda_conv_kernel_size(self) -> int:
        return int(self.rda_config.get("conv_kernel_size", 4))

    @property
    def rda_gate_lower_bound(self) -> float | None:
        return self.rda_config.get("gate_lower_bound", -5.0)

    def is_rda_layer(self, layer_idx: int) -> bool:
        """0-based layer index -> True for an RDA (linear attention) layer."""
        return (layer_idx + 1) in self.rda_config["rda_layers"]

    @property
    def layer_types(self) -> list[str]:
        return [
            "linear_attention" if self.is_rda_layer(i) else "full_attention"
            for i in range(self.num_hidden_layers)
        ]


__all__ = ["Raon1NanoConfig"]

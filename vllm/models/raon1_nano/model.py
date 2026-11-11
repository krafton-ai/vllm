# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense Raon-1 nano decoder and vLLM weight-loading adapter.

RDA/MLA branches transform an mHC stream carrier; each MHCSite owns its
residual mixing. PLE and tied embeddings are attached in nano.py. The MLA
and dense MLP primitives are derived from Raon-1 mini; see UPSTREAM.md for their
source and compatibility constraints.
"""

import os
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    make_layers,
    maybe_prefix,
)

from . import raon_phase
from .config import Raon1NanoConfig
from .mhc import _AGGNORM_FUSED_ENV as _MHC_AGGNORM_FUSED_ENV
from .mhc import MHCSite
from .mla_align import install_mla_output_gate_two_round
from .mla_attention import MultiHeadLatentAttention
from .mla_proj_align import apply_module as apply_mla_proj_persistent
from .mla_refit import StableMLAScaleBuffers
from .mlp import RaonMLP
from .mlp_align import apply_module as apply_dense_mlp_persistent
from .rda import RaonRDAAttention

install_mla_output_gate_two_round()


class RaonRefitMLA(StableMLAScaleBuffers, MultiHeadLatentAttention):
    """Upstream MLA math with graph-safe reciprocal-scale registration."""


_STACKED_PARAMS_MAPPING = [
    (".in_proj_qkvgfab", ".q_proj", 0),
    (".in_proj_qkvgfab", ".k_proj", 1),
    (".in_proj_qkvgfab", ".v_proj", 2),
    (".in_proj_qkvgfab", ".g_proj", 3),
    (".in_proj_qkvgfab", ".f_a_proj", 4),
    (".in_proj_qkvgfab", ".b_proj", 5),
    (".conv1d", ".q_conv1d", 0),
    (".conv1d", ".k_conv1d", 1),
    (".conv1d", ".v_conv1d", 2),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
    (".fused_qkv_a_proj", ".q_a_proj", 0),
    (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
]
_PHI_BLOCKS = frozenset({"phi_pre", "phi_post", "phi_z"})
MLA_QK_LAYERNORM_EPS = 1e-06
_QUACK_RMSNORM_ENV = "RAON_VLLM_QUACK_RMSNORM"


class QuackRMSNorm(RMSNorm):
    """RMSNorm with FP32 normalization and affine products on BF16 CUDA inputs."""

    def _quack_eligible(self, x: torch.Tensor) -> bool:
        return (
            x.is_cuda
            and x.dtype == torch.bfloat16
            and (self.weight.dtype == torch.bfloat16)
            and (x.stride(-1) == 1)
            and (
                os.environ.get(_QUACK_RMSNORM_ENV, "1") == "1"
                or raon_phase.aligned("norm.quack")
            )
        )

    def forward(self, x, residual=None):
        h = x if residual is None else x + residual
        if self._quack_eligible(h):
            try:
                from quack import rmsnorm

                out = rmsnorm(h, self.weight, eps=self.variance_epsilon)
                return out if residual is None else (out, h)
            except Exception:
                pass
        return super().forward(x, residual)


class RaonDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Raon1NanoConfig,
        vllm_config: VllmConfig,
        prefix: str,
        aux_stream: torch.cuda.Stream,
    ) -> None:
        super().__init__()
        layer_idx = int(prefix.rsplit(".", 1)[1])
        quant_config = vllm_config.quant_config
        if config.is_rda_layer(layer_idx):
            self.self_attn = RaonRDAAttention(
                config, vllm_config, prefix=f"{prefix}.self_attn"
            )
        else:
            assert config.mla_use_nope, "RAON MLA is NoPE-only"
            self.self_attn = RaonRefitMLA(
                config=config,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                use_output_gate=bool(config.mla_use_output_gate),
                cache_config=vllm_config.cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                aux_stream=aux_stream,
            )
            self.self_attn.rms_norm_eps = MLA_QK_LAYERNORM_EPS
            self.self_attn.q_a_layernorm.variance_epsilon = MLA_QK_LAYERNORM_EPS
            self.self_attn.kv_a_layernorm.variance_epsilon = MLA_QK_LAYERNORM_EPS
            self._mla_proj_parity = apply_mla_proj_persistent(self.self_attn)
        self.mlp = RaonMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self._mlp_proj_parity = apply_dense_mlp_persistent(self.mlp)
        self.input_layernorm = QuackRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = QuackRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        n = config.mhc_streams
        self.self_attn_mhc = MHCSite(n, config.hidden_size, 2 * layer_idx)
        self.mlp_mhc = MHCSite(n, config.hidden_size, 2 * layer_idx + 1)

    def forward(self, positions: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """s: [T, streams, hidden] in, same out."""
        if _MHC_AGGNORM_FUSED_ENV or raon_phase.aligned("mhc.aggnorm"):
            s = self.self_attn_mhc(
                s,
                lambda a: self.self_attn(hidden_states=a, positions=positions),
                branch_norm_weight=self.input_layernorm.weight,
                branch_norm_eps=self.input_layernorm.variance_epsilon,
            )
            return self.mlp_mhc(
                s,
                lambda a: self.mlp(a),
                branch_norm_weight=self.post_attention_layernorm.weight,
                branch_norm_eps=self.post_attention_layernorm.variance_epsilon,
            )
        s = self.self_attn_mhc(
            s,
            lambda a: self.self_attn(
                hidden_states=self.input_layernorm(a), positions=positions
            ),
        )
        return self.mlp_mhc(s, lambda a: self.mlp(self.post_attention_layernorm(a)))


class RaonModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Raon1NanoConfig = vllm_config.model_config.hf_text_config
        self.config = config
        self.streams = config.mhc_streams
        self.vocab_size = config.vocab_size
        self._phi_seen: dict[str, set[str]] = {}
        parallel_config = vllm_config.parallel_config
        assert parallel_config.pipeline_parallel_size == 1, (
            "RAON's inter-layer carrier is a [T, streams, hidden] activation; PP "
            "would need it in IntermediateTensors. The model fits on one B300 at "
            "TP1."
        )
        assert not (
            parallel_config.enable_expert_parallel
            and parallel_config.tensor_parallel_size > 1
            and (parallel_config.data_parallel_size > 1)
        ), "RAON has no sequence-parallel path for the mHC stream carrier"
        assert config.num_attention_heads % get_tensor_model_parallel_world_size() == 0
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        aux_stream = torch.cuda.Stream()

        def get_layer(prefix: str):
            return RaonDecoderLayer(config, vllm_config, prefix, aux_stream)

        _, _, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.norm = QuackRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        s = inputs_embeds.unsqueeze(1).expand(-1, self.streams, -1).contiguous()
        for layer in self.layers:
            s = layer(positions, s)
        hidden_states = self.norm(s.mean(1))
        return hidden_states

    def load_weights(
        self,
        weights: Iterable[
            tuple[str, torch.Tensor] | tuple[str, torch.Tensor, dict[str, Any]]
        ],
    ) -> set[str]:
        """Load packed projections and mHC blocks; reject unknown tensor names."""
        params_dict = dict(self.named_parameters())
        mhc_sites = {n: m for n, m in self.named_modules() if isinstance(m, MHCSite)}
        phi_seen = self._phi_seen
        loaded_params: set[str] = set()
        for args in weights:
            name, loaded_weight = (args[0], args[1])
            kwargs: dict[str, Any] = args[2] if len(args) > 2 else {}
            if name.endswith("_extra_state"):
                continue
            if "_mhc." in name:
                site_name, suffix = name.split("_mhc.", 1)
                site_name += "_mhc"
                mhc_sites[site_name].load_weight(suffix, loaded_weight)
                attr = suffix.removeprefix("res.")
                if attr in _PHI_BLOCKS:
                    seen = phi_seen.setdefault(site_name, set())
                    seen.add(attr)
                    if seen != _PHI_BLOCKS:
                        continue
                    attr = "phi"
                loaded_params.add(f"{site_name}.{attr}")
                continue
            for param_name, weight_name, shard_id in _STACKED_PARAMS_MAPPING:
                if weight_name not in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped not in params_dict:
                    continue
                param = params_dict[name_mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name_mapped)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, **kwargs)
                loaded_params.add(name)
        return loaded_params


_CUDAGRAPH_READBACK_DONE = False
_GRAPH_REPLAYS = {"n": 0}


def _cudagraph_readback(vllm_config) -> None:
    """Log the resolved CUDA graph mode and check an explicitly requested mode."""
    global _CUDAGRAPH_READBACK_DONE
    if _CUDAGRAPH_READBACK_DONE:
        return
    _CUDAGRAPH_READBACK_DONE = True
    cc = getattr(vllm_config, "compilation_config", None)
    if cc is None:
        if raon_phase.PREFILL_ALIGNED:
            raise RuntimeError(
                "RAON_PREFILL_ALIGNED=1 but vllm_config has no compilation_config"
                " to read the resolved cudagraph mode from; refusing to run the "
                "phase gate against an unknown capture composition."
            )
        print(
            "[parity] cudagraph readback: no compilation_config on vllm_config; "
            "cannot judge",
            flush=True,
        )
        return
    raw_mode = getattr(cc, "cudagraph_mode", None)
    resolved = str(raw_mode)
    comp_mode = str(getattr(cc, "mode", None))
    print(
        f"[parity] cudagraph readback: cudagraph_mode={resolved} "
        f"compilation mode={comp_mode}",
        flush=True,
    )
    if raon_phase.PREFILL_ALIGNED:
        if raw_mode is None:
            raise RuntimeError(
                "RAON_PREFILL_ALIGNED=1 but compilation_config has no "
                "cudagraph_mode attribute; cannot verify the capture composition "
                "the gate requires (FULL_DECODE_ONLY or NONE). Refusing rather "
                "than risk freezing the phase verdict into a captured graph."
            )
        _r = resolved.upper()
        if "FULL_DECODE_ONLY" not in _r and (not _r.endswith("NONE")):
            raise RuntimeError(
                f"RAON_PREFILL_ALIGNED=1 requires FULL_DECODE_ONLY or NONE; "
                f"got cudagraph_mode={resolved}."
            )
    if "vllm_compile" in comp_mode.lower() or "inductor" in comp_mode.lower():
        raise RuntimeError(
            f"Raon-1 nano alignment requires unfused operations; "
            f"compilation mode {comp_mode!r} is unsupported. Use mode: none."
        )
    want = os.environ.get("RAON_EXPECT_CUDAGRAPH_MODE")
    if not want:
        return
    if want.upper() not in resolved.upper():
        raise RuntimeError(
            f"RAON_EXPECT_CUDAGRAPH_MODE={want}, but resolved {resolved} "
            f"with compilation mode={comp_mode}."
        )
    print(
        f"[parity] cudagraph readback OK: pinned {want}, resolved {resolved}",
        flush=True,
    )
    _install_graph_replay_counter()


def _install_graph_replay_counter() -> None:
    """Count graph replays for runtime diagnostics."""
    if os.environ.get("RAON_CUDAGRAPH_REPLAY_COUNTER", "1") != "1":
        return
    g = torch.cuda.CUDAGraph
    if getattr(g.replay, "_raon_counted", False):
        return
    orig = g.replay

    def replay(self, *a, **k):
        _GRAPH_REPLAYS["n"] += 1
        return orig(self, *a, **k)

    replay._raon_counted = True  # type: ignore[attr-defined]
    g.replay = replay
    import atexit

    atexit.register(
        lambda: print(
            f"[parity] cudagraph replays this process: {_GRAPH_REPLAYS['n']}",
            flush=True,
        )
    )


class RaonNanoModelBase(nn.Module, HasInnerState, IsHybrid):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        _cudagraph_readback(vllm_config)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.config = self.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        if self.quant_config is not None:
            raise ValueError("Nano recipe supports unquantized BF16 weights only")
        if vllm_config.speculative_config is not None:
            from .mtp_weights import validate_speculation

            validate_speculation(vllm_config)
        self.model = RaonModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        from .head_cache import FP32HeadWeightCache

        self._fp32_head_weight_cache = FP32HeadWeightCache()
        self._deferred_refit_active = False
        self._deferred_refit_batches = 0
        try:
            from .kda_frontend_fused import prewarm_frontend_fused

            prewarm_frontend_fused(device=torch.accelerator.current_device_index())
        except Exception as _e:
            print(f"[raon-kda] FUSED_FLOW prewarm skipped: {_e}", flush=True)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if os.environ.get("VLLM_KPAMO_FP32_LM_HEAD", "0") == "1":
            if not getattr(self, "_fp32_head_announced", False):
                self._fp32_head_announced = True
                print(
                    "[parity] VLLM_KPAMO_FP32_LM_HEAD LIVE: fp32 head GEMM "
                    "engaged (hidden+weight widened to fp32, single fp32 matmul —"
                    " not a post-cast)",
                    flush=True,
                )
            weight = self.lm_head.weight
            hidden_fp32 = hidden_states.to(torch.float32)
            weight_fp32 = self._fp32_head_weight_cache.get(weight)
            if os.environ.get("RAON_HEAD_GEMM_PERSISTENT", "0") == "1":
                from .batch_invariant import matmul_persistent

                logits = matmul_persistent(
                    hidden_fp32, weight_fp32.t(), out_dtype=torch.float32
                )
            else:
                logits = torch.matmul(hidden_fp32, weight_fp32.t())
            return logits[..., : self.config.vocab_size]
        if os.environ.get("RAON_HEAD_GEMM_PERSISTENT", "0") == "1":
            raise RuntimeError(
                "RAON_HEAD_GEMM_PERSISTENT=1 requires VLLM_KPAMO_FP32_LM_HEAD=1"
            )
        return self.logits_processor(self.lm_head, hidden_states)

    def begin_deferred_refit(self) -> None:
        """Defer derived-weight refresh until a complete streamed transaction."""
        if self._deferred_refit_active:
            raise RuntimeError("a deferred Raon refit is already active")
        self._deferred_refit_active = True
        self._deferred_refit_batches = 0

    def finish_deferred_refit(self) -> int:
        """Refresh graph-stable derived buffers after the last source copy."""
        if not self._deferred_refit_active or not self._deferred_refit_batches:
            raise RuntimeError("no streamed Raon weights were loaded")
        self._invalidate_refit_derived()
        self._finalize_mhc_refit()
        batches = self._deferred_refit_batches
        self._deferred_refit_batches = 0
        self._deferred_refit_active = False
        return batches

    def _invalidate_refit_derived(self) -> None:
        self._fp32_head_weight_cache.clear()
        for module in self.modules():
            for name in ("_raon_kda_decode_graph", "_raon_kda_decode_graph_stamp"):
                if hasattr(module, name):
                    delattr(module, name)

    def _finalize_mhc_refit(self) -> None:
        for module in self.modules():
            if isinstance(module, MHCSite):
                module.finalize()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if not self._deferred_refit_active:
            self._invalidate_refit_derived()
        loaded = AutoWeightsLoader(self).load_weights(weights)
        if self._deferred_refit_active:
            self._deferred_refit_batches += 1
        else:
            self._finalize_mhc_refit()
        return loaded

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        hf_config = vllm_config.model_config.hf_config
        return MambaStateShapeCalculator.kda_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            hf_config.linear_attn_config["num_heads"],
            hf_config.linear_attn_config["head_dim"],
            conv_kernel_size=hf_config.linear_attn_config["short_conv_kernel_size"],
            num_spec=(
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config is not None
                else 0
            ),
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

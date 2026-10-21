# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""raon-1-mini dense MLP and MoE block on vLLM's FusedMoE.

The MoE routes with fp32 sigmoid gate logits and an expert score-correction
bias, and hands its shared experts to the MoE runner so their reduction is
fused with the routed one. raon-1-mini has no latent expert projection, no
sequence-parallel sharding and no expert-group routing.
"""

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul, SituAndMul
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig


class RaonMLP(nn.Module):
    """Dense FFN and the MoE block's shared experts."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        activation_situ_beta: float | None = None,
        activation_situ_linear_beta: float | None = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act == "silu":
            self.act_fn = SiluAndMul()
        elif hidden_act == "situ":
            self.act_fn = SituAndMul(
                beta=activation_situ_beta or 1.0,
                linear_beta=activation_situ_linear_beta,
            )
        else:
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu and situ are supported."
            )

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class RaonMoE(nn.Module):
    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        layer_idx: int = 0,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        moe_intermediate_size = config.moe_intermediate_size
        num_experts = config.num_experts
        num_experts_per_token = config.num_experts_per_token
        assert moe_intermediate_size is not None
        assert num_experts is not None
        assert num_experts_per_token is not None
        if getattr(config, "routed_expert_hidden_size", None) is not None:
            raise ValueError("raon-1-mini has no latent routed-expert projection")
        if vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe":
            raise ValueError("raon-1-mini does not support the MegaMoE backend")
        self.moe_hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_renormalize = config.moe_renormalize
        self.use_grouped_topk = config.use_grouped_topk
        self.num_expert_group = config.num_expert_group
        self.topk_group = config.topk_group
        self.moe_router_activation_func = config.moe_router_activation_func
        self.num_shared_experts = config.num_shared_experts
        self.layer_idx = layer_idx
        self.padded_moe_intermediate_size = moe_intermediate_size
        min_moe_intermediate_per_partition = getattr(
            config, "min_moe_intermediate_per_partition", 256
        )
        if self.tp_size > 1 and not vllm_config.parallel_config.enable_expert_parallel:
            moe_intermediate_per_partition = moe_intermediate_size // self.tp_size
            if moe_intermediate_per_partition < min_moe_intermediate_per_partition:
                self.padded_moe_intermediate_size = (
                    min_moe_intermediate_per_partition * self.tp_size
                )
        activation_situ_beta = (
            config.activation_situ_beta if config.hidden_act == "situ" else None
        )
        activation_situ_linear_beta = (
            config.activation_situ_linear_beta if config.hidden_act == "situ" else None
        )

        # Route with fp32 logits for numerically stable expert selection.
        self.gate = GateLinear(
            input_size=hidden_size,
            output_size=num_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32)
        )

        if self.num_shared_experts is not None:
            shared_intermediate_size = moe_intermediate_size * self.num_shared_experts
            self.shared_experts = RaonMLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                activation_situ_beta=activation_situ_beta,
                activation_situ_linear_beta=activation_situ_linear_beta,
            )
        else:
            self.shared_experts = None

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            num_experts=num_experts,
            top_k=num_experts_per_token,
            hidden_size=self.moe_hidden_size,
            intermediate_size=self.padded_moe_intermediate_size,
            activation=config.hidden_act,
            activation_situ_beta=activation_situ_beta,
            activation_situ_linear_beta=activation_situ_linear_beta,
            renormalize=self.moe_renormalize,
            quant_config=quant_config,
            use_grouped_topk=config.use_grouped_topk,
            num_expert_group=config.num_expert_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.moe_router_activation_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            routed_input_transform=None,
            routed_output_transform=None,
            is_sequence_parallel=False,
            runner_cls=None,
        )
        if self.padded_moe_intermediate_size != moe_intermediate_size:
            w13_weight = getattr(self.experts, "w13_weight", None)
            if w13_weight is None:
                w13_weight = getattr(self.experts, "w13_weight_packed", None)
            w2_weight = getattr(self.experts, "w2_weight", None)
            if w2_weight is None:
                w2_weight = getattr(self.experts, "w2_weight_packed", None)
            if w13_weight is not None:
                w13_weight.data.zero_()
            if w2_weight is not None:
                w2_weight.data.zero_()
            self.experts.moe_config.intermediate_size_per_partition_unpadded = (
                moe_intermediate_size // self.tp_size
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        router_logits, _ = self.gate(hidden_states)
        # Shared experts run inside the MoE runner on shared_experts_input.
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            shared_experts_input=hidden_states,
        )
        return final_hidden_states.view(num_tokens, hidden_size)

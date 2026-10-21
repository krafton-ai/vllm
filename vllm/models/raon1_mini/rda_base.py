# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""raon-1-mini RDA layer: projections, recurrent state layout and prefill backend.

The forward pass lives in kda.py (``RaonKDAAttention``). Prefill runs FlashKDA
(``vllm._flashkda_C``); speculative and fallback decode run rda_recurrent.py.
"""

from collections.abc import Callable

import torch
from torch import nn
from torch.nn.parameter import Parameter

from vllm.config import VllmConfig
from vllm.distributed import divide, get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm.v1.attention.backend import AttentionBackend

from .rda_metadata import RaonRDAAttentionBackend

logger = init_logger(__name__)

_RDA_GATE_LOGBOUND_MIN = -5.0


def a_log_weight_loader(
    shard_axis: int,
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """Load A_log stored as either old 4D or current 1D weights."""

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = param.data.shape[shard_axis]
        start_idx = tp_rank * shard_size

        if loaded_weight.dim() == 4:
            assert loaded_weight.shape[:2] == (1, 1), (
                f"Expected old A_log shape (1, 1, H, 1), got {loaded_weight.shape}"
            )
            assert loaded_weight.shape[-1] == 1, (
                f"Expected old A_log last dim to be 1, got {loaded_weight.shape}"
            )
            loaded_weight = loaded_weight.view(loaded_weight.shape[2])

        loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)
        return default_weight_loader(param, loaded_weight)

    return loader


class _ReplicatedShardMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Merged projection with one output replicated across TP ranks."""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        replicated_shard_id: int,
        tp_size: int,
        **kwargs,
    ) -> None:
        self.replicated_shard_id = replicated_shard_id
        output_sizes = output_sizes.copy()
        output_sizes[replicated_shard_id] *= tp_size
        super().__init__(input_size, output_sizes, **kwargs)

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id == self.replicated_shard_id:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id == self.replicated_shard_id:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank


def is_fused_rda_decode_supported(
    num_heads: int,
    head_dim: int,
    conv_width: int,
    num_spec: int,
    input_dtype: torch.dtype,
    conv_state_dtype: torch.dtype,
) -> bool:
    if (
        num_heads not in (12, 24, 48, 96)
        or head_dim != 128
        or conv_width != 4
        or num_spec != 0
        or input_dtype != torch.bfloat16
        or conv_state_dtype != torch.bfloat16
        or is_conv_state_dim_first()
        or not hasattr(torch.ops._C, "fused_kda_decode")
    ):
        return False
    # SM90 is architecture-specific; SM10x and SM12x use family binaries.
    return (
        current_platform.is_device_capability(90)
        or current_platform.is_device_capability_family(100)
        or current_platform.is_device_capability_family(120)
    )


def is_flashkda_supported(
    head_dim: int,
    dtype: torch.dtype,
    lower_bound: float | None,
) -> bool:
    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    return (
        capability is not None
        and capability.major in (9, 10, 12)
        and head_dim == 128
        and dtype == torch.bfloat16
        and lower_bound is not None
    )


def _flashkda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    final_state: torch.Tensor,
    workspace: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    import vllm._flashkda_C  # noqa: F401

    # FlashKDA hardcodes dense Q/K/V/G strides. Beta may be row-strided because
    # FlashKDA materializes its transposed [H, T] layout internally.
    # TODO: Teach FlashKDA to consume beta in [T, H] layout directly instead
    # of transposing it to contiguous [H, T] storage internally.
    torch.ops._flashkda_C.fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta,
        q.shape[-1] ** -0.5,
        out,
        workspace,
        A_log.contiguous(),
        dt_bias.view(-1, q.shape[-1]).contiguous(),
        lower_bound,
        initial_state.contiguous(),
        final_state,
        cu_seqlens.contiguous(),
    )
    return out, final_state


def resolve_rda_prefill_backend(
    backend: str,
    head_dim: int,
    dtype: torch.dtype,
    lower_bound: float | None,
) -> str:
    if backend not in ("auto", "triton", "flashkda"):
        raise ValueError(f"Unsupported RDA prefill backend: {backend}")
    supported = is_flashkda_supported(head_dim, dtype, lower_bound)
    if backend == "flashkda" and not supported:
        raise RuntimeError(
            "FlashKDA requires CUDA SM90/SM10x/SM12x, bfloat16, "
            "head_dim=128, and a bounded RDA gate."
        )
    if supported and backend != "triton":
        logger.info_once("Using the FlashKDA RDA prefill backend.")
        return "flashkda"
    return "triton"


def _make_decode_conv1d_weight_loader(
    dims: list[int],
    tp_size: int,
    tp_rank: int,
    decode_conv1d_weight: torch.Tensor | None,
) -> Callable[..., None]:
    sharded_dims = [dim // tp_size for dim in dims]

    def weight_loader(
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ) -> None:
        if loaded_weight.dim() == 2:
            loaded_weight = loaded_weight.unsqueeze(1)
        shard_size = sharded_dims[loaded_shard_id]
        source_start = tp_rank * shard_size
        target_start = sum(sharded_dims[:loaded_shard_id])
        loaded_shard = loaded_weight[source_start : source_start + shard_size]
        param.data[target_start : target_start + shard_size].copy_(loaded_shard)
        if decode_conv1d_weight is not None and not param.is_meta:
            decode_conv1d_weight[loaded_shard_id].copy_(
                loaded_shard.squeeze(1).transpose(0, 1)
            )

    return weight_loader


def _make_decode_norm_weight_loader(
    decode_norm_weight: torch.Tensor,
) -> Callable[..., None]:
    def weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        default_weight_loader(param, loaded_weight)
        if not param.is_meta:
            decode_norm_weight.copy_(param.data)

    return weight_loader


class RaonRDABase(GatedDeltaNetAttention):
    """Projections, state cache layout and FlashKDA buffers of an RDA layer.

    ``RaonKDAAttention`` (kda.py) supplies the forward pass.
    """

    def get_attn_backend(self) -> type[AttentionBackend]:
        return RaonRDAAttentionBackend

    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=self.num_spec,
        )

    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(config, vllm_config, prefix)

        rda_config = config.linear_attn_config  # type: ignore[attr-defined]
        assert rda_config is not None, "linear_attn_config must be set"
        self.head_dim = rda_config["head_dim"]
        self.num_heads = rda_config["num_heads"]
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)
        self.projection_size = self.head_dim * self.num_heads
        self.local_projection_size = divide(self.projection_size, self.tp_size)
        self.conv_size = rda_config["short_conv_kernel_size"]
        assert rda_config.get("use_full_rank_gate", False), (
            "RDA requires a full-rank gate"
        )

        # Keep f_a before the narrow beta shard, then pad each TP-local row
        # to select the aligned BF16 GEMM path.
        qkvg_output_sizes = [self.projection_size] * 4
        in_proj_output_sizes = qkvg_output_sizes + [
            self.head_dim,
            self.num_heads,
        ]
        local_output_size = (
            4 * self.local_projection_size + self.head_dim + self.local_num_heads
        )
        self.in_proj_padding = -local_output_size % 16
        if self.in_proj_padding:
            in_proj_output_sizes.append(self.in_proj_padding * self.tp_size)
        self.in_proj_qkvgfab = _ReplicatedShardMergedColumnParallelLinear(
            self.hidden_size,
            in_proj_output_sizes,
            replicated_shard_id=4,
            tp_size=self.tp_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvgfab",
        )
        if self.in_proj_padding:
            self.in_proj_qkvgfab.weight.data[-self.in_proj_padding :].zero_()

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.local_projection_size, dtype=torch.float32)
        )
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        # One packed parameter and cache let decode run a single conv update.
        # Prefill slices them back into Q/K/V to obtain dense outputs cheaply.
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=3 * self.projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        # Keep a width-major copy for fused decode without changing the layout
        # consumed by the prefill and fallback decode kernels.
        conv_state_dtype, _ = self.get_state_dtype()
        decode_conv1d_weight = None
        if is_fused_rda_decode_supported(
            self.local_num_heads,
            self.head_dim,
            self.conv_size,
            self.num_spec,
            vllm_config.model_config.dtype,
            conv_state_dtype,
        ):
            logger.info_once("Fused RDA decode kernel (conv+recurrence+norm) is enabled.")
            decode_conv1d_weight = torch.empty(
                3,
                self.conv_size,
                self.local_projection_size,
                dtype=self.conv1d.weight.dtype,
                device=self.conv1d.weight.device,
            )
        self.register_buffer(
            "decode_conv1d_weight", decode_conv1d_weight, persistent=False
        )
        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": _make_decode_conv1d_weight_loader(
                    [self.projection_size] * 3,
                    self.tp_size,
                    self.tp_rank,
                    decode_conv1d_weight,
                )
            },
        )

        self.A_log = nn.Parameter(
            torch.empty(self.local_num_heads, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": a_log_weight_loader(0)})

        self.gate_lower_bound: float | None = rda_config.get("gate_lower_bound", None)
        if self.gate_lower_bound is not None:
            assert _RDA_GATE_LOGBOUND_MIN <= self.gate_lower_bound < 0, (
                "RDA gate lower bound must be in "
                f"[{_RDA_GATE_LOGBOUND_MIN}, 0). "
                f"Got {self.gate_lower_bound}."
            )

        additional_config = vllm_config.additional_config
        backend = (
            additional_config.get("kda_prefill_backend", "auto")
            if isinstance(additional_config, dict)
            else "auto"
        )
        self.kda_prefill_backend = resolve_rda_prefill_backend(
            backend,
            self.head_dim,
            vllm_config.model_config.dtype,
            self.gate_lower_bound,
        )
        self._flashkda_buffer_specs: (
            tuple[tuple[tuple[int, ...], torch.dtype], ...] | None
        ) = None
        if self.kda_prefill_backend == "flashkda":
            T = vllm_config.scheduler_config.max_num_batched_tokens
            N = vllm_config.scheduler_config.max_num_seqs
            H, D = self.local_num_heads, self.head_dim
            import vllm._flashkda_C  # noqa: F401

            workspace_size = torch.ops._flashkda_C.get_workspace_size(T, H, N)
            self._flashkda_buffer_specs = (
                ((1, T, H, D), self.model_config.dtype),
                ((N, H, D, D), self.get_state_dtype()[1]),
                ((workspace_size,), torch.uint8),
            )

        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        decode_norm_weight = None
        if decode_conv1d_weight is not None:
            decode_norm_weight = torch.empty(
                self.head_dim,
                dtype=torch.float32,
                device=self.o_norm.weight.device,
            )
        self.register_buffer("decode_norm_weight", decode_norm_weight, persistent=False)
        if decode_norm_weight is not None:
            # Upcast once while loading; direct BF16 norm weights slow the
            # fully fused decode kernel.
            if hasattr(self.o_norm.weight, "weight_loader"):
                delattr(self.o_norm.weight, "weight_loader")
            set_weight_attrs(
                self.o_norm.weight,
                {"weight_loader": _make_decode_norm_weight_loader(decode_norm_weight)},
            )
        self.o_proj = RowParallelLinear(
            self.projection_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self


@torch.inference_mode()
def warm_recurrent_rda(
    layer: "RaonRDABase",
    input_dtype: torch.dtype,
) -> None:
    """JIT the speculative-decode recurrence's launch variants before serving."""
    from .rda_recurrent import (
        fused_recurrent_rda,
        get_fused_recurrent_rda_fwd_warmup_profiles,
    )

    num_speculative_tokens = int(layer.num_spec)
    # fused_recurrent_rda_fwd_kernel is only used by speculative decode.
    if num_speculative_tokens <= 0:
        return

    kv_cache = layer.kv_cache
    if not isinstance(kv_cache, (list, tuple)) or len(kv_cache) < 2:
        return
    state = kv_cache[1]
    if not isinstance(state, torch.Tensor) or not state.numel():
        return

    logger.info("Warming up the speculative RDA kernels.")
    h = int(layer.local_num_heads)
    d = int(layer.head_dim)
    tokens_per_sequence = num_speculative_tokens + 1
    for num_sequences in get_fused_recurrent_rda_fwd_warmup_profiles(h):
        num_tokens = num_sequences * tokens_per_sequence
        packed_qkv = torch.empty(
            (num_tokens, 3 * h * d),
            dtype=input_dtype,
            device=state.device,
        )
        q, k, v = (
            tensor.view(1, num_tokens, h, d)
            for tensor in packed_qkv.split(h * d, dim=-1)
        )
        fused_recurrent_rda(
            q=q,
            k=k,
            v=v,
            raw_g=torch.empty(
                (1, num_tokens, h, d),
                dtype=input_dtype,
                device=state.device,
            ),
            raw_beta=torch.empty(
                (1, num_tokens, h),
                dtype=input_dtype,
                device=state.device,
            ),
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            lower_bound=layer.gate_lower_bound,
            initial_state=state[:1],
            cu_seqlens=torch.arange(
                0,
                num_tokens + 1,
                tokens_per_sequence,
                dtype=torch.int32,
                device=state.device,
            ),
            ssm_state_indices=torch.zeros(
                (num_sequences, tokens_per_sequence),
                dtype=torch.int32,
                device=state.device,
            ),
            num_accepted_tokens=torch.ones(
                num_sequences,
                dtype=torch.int32,
                device=state.device,
            ),
            out=torch.empty(
                (1, num_tokens, h, d),
                dtype=input_dtype,
                device=state.device,
            ),
        )


def rda_kernel_warmup(worker) -> None:
    """JIT the RDA Triton kernels of a loaded worker; vLLM's kernel_warmup slot."""
    if not current_platform.is_cuda():
        return
    static_context = getattr(
        getattr(worker.model_runner, "compilation_config", None),
        "static_forward_context",
        None,
    )
    if not isinstance(static_context, dict):
        return
    layer = next(
        (layer for layer in static_context.values() if isinstance(layer, RaonRDABase)),
        None,
    )
    if layer is not None:
        warm_recurrent_rda(layer, worker.model_config.dtype)

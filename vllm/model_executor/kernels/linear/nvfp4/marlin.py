# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, cast

import torch

from vllm import envs
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers import raon_optimizations
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
)
from vllm.platforms import current_platform

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)


def _raon_resident_min_rows(layer: torch.nn.Module) -> int:
    min_rows = envs.VLLM_MARLIN_RESIDENT_BF16_MIN_ROWS
    if min_rows < 0:
        raise ValueError("VLLM_MARLIN_RESIDENT_BF16_MIN_ROWS must be non-negative")
    weight = cast(torch.Tensor, layer.weight)
    if (
        min_rows == 0
        and weight.is_cuda
        and raon_optimizations.enabled("marlin_resident_bf16", weight.device.index)
    ):
        min_rows = raon_optimizations.resident_bf16_min_rows(weight.device.index) or 0
    if min_rows == 0 or not weight.is_cuda or getattr(layer, "bias", None) is not None:
        return 0

    config = get_current_vllm_config_or_none()
    if config is None or config.model_config is None:
        return 0
    model_config = config.model_config
    offload = config.offload_config
    if (
        model_config.hf_config.model_type != "raon1_mini"
        or model_config.dtype != torch.bfloat16
        or model_config.enable_sleep_mode
        or offload.uva.cpu_offload_gb
        or offload.prefetch.offload_group_size
        or not any(
            current_platform.is_device_capability(
                capability, device_id=weight.device.index
            )
            for capability in ((10, 3), (11, 0))
        )
    ):
        return 0
    return min_rows


class MarlinNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 weight-only GEMM via Marlin (W4A16)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if is_fp4_marlin_supported():
            return True, None
        return False, "Marlin FP4 not available"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        logger.warning_once(
            "Your GPU does not have native support for FP4 computation but "
            "FP4 quantization is being used. Weight-only FP4 compression "
            "will be used leveraging the Marlin kernel. This may degrade "
            "performance for compute-heavy workloads."
        )
        prepare_fp4_layer_for_marlin(layer)
        min_rows = _raon_resident_min_rows(layer)
        if min_rows:
            # Build before graph capture and account for the permanent copy
            # before the engine sizes the KV cache.
            cast(Any, layer)._raon_resident_min_rows = min_rows
            layer.register_buffer(
                "_raon_resident_bf16", self._materialize_bf16(layer), persistent=False
            )

    @torch.no_grad()
    def _materialize_bf16(self, layer: torch.nn.Module) -> torch.Tensor:
        size_k = cast(int, layer.input_size_per_partition)
        size_n = cast(int, layer.output_size_per_partition)
        device = cast(torch.Tensor, layer.weight).device
        result = torch.empty((size_k, size_n), dtype=torch.bfloat16, device=device)
        for begin in range(0, size_k, 2048):
            rows = min(2048, size_k - begin)
            basis = torch.zeros((rows, size_k), dtype=torch.bfloat16, device=device)
            offset = torch.arange(rows, device=device)
            basis[offset, begin + offset] = 1
            result[begin : begin + rows].copy_(self._apply_marlin(layer, basis, None))
        return result

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        resident = getattr(layer, "_raon_resident_bf16", None)
        if resident is not None and bias is None and x.dtype == resident.dtype:
            rows = x.numel() // x.shape[-1]
            if rows >= cast(int, layer._raon_resident_min_rows):
                out = torch.matmul(x.reshape(rows, x.shape[-1]), resident)
                return out.reshape(x.shape[:-1] + (resident.shape[-1],))
        return self._apply_marlin(layer, x, bias)

    def _apply_marlin(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )

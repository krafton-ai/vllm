# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graphs for the MLP-side mHC segment of Raon-1 mini in mixed steps.

The segment is ``RaonMHC.forward`` of a decoder layer's MLP site: the phi
projection, head, aggregate, norm, dense MLP or MoE, and mix. It maps a
``[T, streams, hidden]`` carrier to the same shape and is independent across
rows. Steps with more than ``FIRST_BUCKET_MIN_TOKENS`` tokens are mixed or
prefill steps that vLLM does not graph. For those, the token count is rounded
up to a configured bucket, the rows are copied into a static input buffer
shared by all layers, the layer's graph for that bucket is replayed, and the
first ``T`` rows of the shared static output are returned.

Graphs are captured manually with a shared memory pool, after eager warmup.
Any capture or replay error makes that layer use the eager path from then on.
Only the NVFP4 (Marlin) profile on one GPU is eligible; every other
configuration keeps the eager path.
"""

import functools
import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers import raon_optimizations
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from torch.cuda import _POOL_HANDLE

logger = init_logger(__name__)

# Rows at or below this count are decode-sized and never use these graphs.
FIRST_BUCKET_MIN_TOKENS = 1024
# The mHC mix kernel changes at 4096 rows; larger buckets are not supported.
MAX_BUCKET_TOKENS = 4095
_WARMUP_RUNS = 2


def parse_buckets(spec: str) -> tuple[int, ...]:
    """Parse a comma-separated bucket list into sorted unique token counts."""
    buckets: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            bucket = int(item)
        except ValueError:
            raise ValueError(
                f"Invalid VLLM_RAON_MLP_SEGMENT_GRAPH_BUCKETS entry {item!r}"
            ) from None
        if not FIRST_BUCKET_MIN_TOKENS < bucket <= MAX_BUCKET_TOKENS:
            raise ValueError(
                "VLLM_RAON_MLP_SEGMENT_GRAPH_BUCKETS entries must be in "
                f"({FIRST_BUCKET_MIN_TOKENS}, {MAX_BUCKET_TOKENS}], got {bucket}"
            )
        buckets.add(bucket)
    return tuple(sorted(buckets))


def select_bucket(tokens: int, buckets: tuple[int, ...]) -> int | None:
    """Smallest bucket that holds ``tokens``, or None for the eager path."""
    if tokens <= FIRST_BUCKET_MIN_TOKENS:
        return None
    for bucket in buckets:
        if tokens <= bucket:
            return bucket
    return None


_buckets_cache: dict[int, tuple[int, ...]] = {}


def configured_buckets(device_index: int) -> tuple[int, ...]:
    """Buckets selected by the explicit variable or the device profile."""
    cached = _buckets_cache.get(device_index)
    if cached is not None:
        return cached
    buckets: tuple[int, ...] = ()
    if current_platform.is_cuda():
        buckets = parse_buckets(envs.VLLM_RAON_MLP_SEGMENT_GRAPH_BUCKETS)
        if not buckets and raon_optimizations.enabled(
            "mlp_segment_graph", device_index
        ):
            buckets = raon_optimizations.MLP_SEGMENT_GRAPH_BUCKETS
    if buckets:
        logger.info(
            "Raon MLP-side mHC segment CUDA graphs enabled for token buckets %s",
            buckets,
        )
    _buckets_cache[device_index] = buckets
    return buckets


@functools.cache
def process_supported() -> bool:
    """Settings of this process whose per-step state a graph would freeze."""
    from . import PROFILE_KEY, raon_phase

    return (
        os.environ.get(PROFILE_KEY) == "nvfp4"
        and not raon_phase.PREFILL_ALIGNED
        and not envs.VLLM_MOE_SKIP_PADDING
        and os.environ.get("RAON_MHC_CHECK", "0") != "1"
        and os.environ.get("RAON_MHC_TRAINER_FUSED", "0") != "1"
    )


class _Buffers:
    """Static buffers and the memory pool shared by every layer's graphs."""

    def __init__(self) -> None:
        self.pool: _POOL_HANDLE | None = None
        self.captured = 0
        self.shared: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def get(self, bucket: int, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = (bucket, tuple(like.shape[1:]), like.dtype, like.device)
        buffers = self.shared.get(key)
        if buffers is None:
            shape = (bucket, *like.shape[1:])
            buffers = (
                torch.zeros(shape, dtype=like.dtype, device=like.device),
                torch.zeros(shape, dtype=like.dtype, device=like.device),
            )
            self.shared[key] = buffers
        return buffers

    def reset(self) -> None:
        for inputs, outputs in self.shared.values():
            inputs.zero_()
            outputs.zero_()


_state = _Buffers()


def _capture(
    site: torch.nn.Module,
    branch: Callable[[torch.Tensor], torch.Tensor],
    bucket: int,
    like: torch.Tensor,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]:
    inputs, outputs = _state.get(bucket, like)
    inputs[: like.shape[0]].copy_(like)
    for _ in range(_WARMUP_RUNS):
        site(inputs, branch)
    torch.cuda.synchronize()
    if _state.pool is None:
        _state.pool = torch.cuda.graph_pool_handle()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=_state.pool):
        result = site(inputs, branch)
        outputs.copy_(result)
    _state.captured += 1
    return graph, inputs, outputs


def layer_supported(layer: torch.nn.Module) -> bool:
    """Whether the layer's MLP-side segment is the stock graph-safe path."""
    mlp = layer.mlp
    return (
        getattr(layer, "_mlp_segment_graph_world", False)
        and getattr(mlp, "_raon_stock_forward", None) is None
        and not getattr(layer, "_moe_shared_mxfp8", None)
        and not getattr(layer, "_mlp_proj_parity", None)
        and not getattr(layer, "_mlp_situ_parity", None)
    )


def try_run(
    layer: torch.nn.Module,
    s: torch.Tensor,
    branch: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor | None:
    """Run the layer's MLP-side segment through its graph for ``s``.

    Returns None when the eager path must run instead.
    """
    if s.dim() != 3 or not s.is_cuda or s.shape[0] <= FIRST_BUCKET_MIN_TOKENS:
        return None
    buckets = configured_buckets(s.device.index)
    if not buckets:
        return None
    bucket = select_bucket(s.shape[0], buckets)
    if (
        bucket is None
        or getattr(layer, "_mlp_segment_graph_bad", False)
        or not s.is_contiguous()
        or not process_supported()
        or not layer_supported(layer)
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    site = cast(torch.nn.Module, layer.mlp_raon_mhc)
    graphs = layer.__dict__.setdefault("_mlp_segment_graphs", {})
    try:
        entry = graphs.get(bucket)
        if entry is None:
            entry = _capture(site, branch, bucket, s)
            graphs[bucket] = entry
        graph, inputs, outputs = entry
        inputs[: s.shape[0]].copy_(s)
        graph.replay()
        return outputs[: s.shape[0]].clone()
    except Exception:
        cast(Any, layer)._mlp_segment_graph_bad = True
        graphs.clear()
        logger.warning_once(
            "Raon MLP-side mHC segment graph failed; using the eager path"
        )
        logger.debug("Raon MLP-side mHC segment graph failure", exc_info=True)
        torch.cuda.synchronize()
        return None


def precapture_enabled(device_index: int) -> bool:
    """Whether graphs are captured at server start (needs buckets)."""
    return bool(configured_buckets(device_index)) and (
        envs.VLLM_RAON_MLP_SEGMENT_GRAPH_PRECAPTURE
        or raon_optimizations.enabled("mlp_segment_graph_precapture", device_index)
    )


def precapture(runner) -> int:
    """Capture every layer's graph for every bucket with dummy mixed steps.

    Runs after the regular CUDA graph capture, so the workspaces the segments
    use are already sized. Returns the GPU memory in bytes that the graphs and
    their static buffers hold. A failure leaves lazy capture in place.
    """
    if not hasattr(runner, "_dummy_run"):
        return 0
    device = runner.device
    if not precapture_enabled(device.index) or not process_supported():
        return 0
    sizes = [b for b in configured_buckets(device.index) if b <= runner.max_num_tokens]
    if not sizes:
        return 0
    from vllm.config import CUDAGraphMode

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free_before = torch.cuda.mem_get_info(device)[0]
    start = time.perf_counter()
    # The static buffers are created under inference mode, so they must also be
    # cleared there.
    with torch.inference_mode():
        try:
            for size in sizes:
                runner._dummy_run(
                    size,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    create_mixed_batch=True,
                    skip_eplb=True,
                )
        except Exception:
            logger.warning(
                "Raon MLP-side mHC segment graph pre-capture failed; "
                "graphs will be captured on first use",
                exc_info=True,
            )
        # The padding rows of the static buffers hold dummy activations.
        _state.reset()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    held = max(free_before - torch.cuda.mem_get_info(device)[0], 0)
    logger.info(
        "Pre-captured %d Raon MLP-side mHC segment graphs in %.1f s, holding %.2f GiB",
        _state.captured,
        time.perf_counter() - start,
        held / (1 << 30),
    )
    return held

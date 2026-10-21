"""Replay routing, routed experts and shared experts without changing arithmetic."""

import torch
from . import moe_grouped


_CAPTURE_ANNOUNCED = False
_REPLAY_ANNOUNCED = False


def try_forward(module, x, compute):
    from . import model

    # The local graph may be entered during a real prefill, whereas the
    # phase gate treats graph capture as decode. Never capture/replay this
    # phase-dependent graph for an aligned forward; its cache is shape-only.
    if model.raon_phase.prefill_now():
        return None

    routed = getattr(module.experts, "routed_experts", None)
    router = getattr(module.experts, "router", None)
    # The dynamic MXFP8 chain has three host reads (bincount().tolist(), the
    # data-dependent xs slice, Fp8Padding's python m_splits) that cannot live
    # inside a captured graph. RAON_MOE_MXFP8_STATIC_PAD removes all three
    # (see mxfp8_align.fused_expert_rows_static), so only then is the chain
    # capturable. Capturing the dynamic path would record one routing's
    # extents and replay them for every later step: silently wrong output.
    mxfp8 = model.moe_expert_mxfp8_realized()
    static_pad = model.os.environ.get("RAON_MOE_MXFP8_STATIC_PAD", "0") == "1"
    if mxfp8 and not static_pad:
        return None
    # Under static padding the capture window is bounded by what the static
    # path can hold, not by this function's own token ceiling. The static path
    # gives each expert ALIGN=256 rows and the MoE sees x.shape[0] * top_k
    # rows, so it refuses above 256 rows -- 42 tokens at top_k=6 -- and falls
    # back to the dynamic path, whose counts_t.tolist() is a host read and
    # fails inside capture
    #   Cannot copy between CPU and CUDA tensors during CUDA graph capture
    # Refuse those shapes here so they run eager on the dynamic path.
    if static_pad:
        _top_k = getattr(router, "top_k", None)
        if not isinstance(_top_k, int) or x.shape[0] * _top_k > 256:
            return None
    # _MOE_CHAIN_FUSED gates the BF16 launch-fused chain, which does not run
    # under MXFP8. Requiring it for MXFP8 would refuse the path static padding
    # makes capturable, so it applies to the BF16 path only.
    if not mxfp8 and not model._MOE_CHAIN_FUSED:
        return None
    if (
        not model.moe_trainer_realized()
        or model._MOE_ROUTE_CAPTURE
        or model.os.environ.get("RAON_MOE_GROUPED_GRAPH", "1") != "1"
        or routed is None
        or getattr(router, "top_k", None) != 6
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or not 1 <= x.shape[0] <= 64
        or x.shape[1] != 2560
    ):
        return None
    w13, w2 = routed.w13_weight, routed.w2_weight
    if (
        w13.dtype != torch.bfloat16
        or w2.dtype != torch.bfloat16
        or tuple(w13.shape) != (128, 2176, 2560)
        or tuple(w2.shape) != (128, 2560, 1088)
    ):
        return None
    device = x.device.index
    if device not in moe_grouped._capabilities:
        moe_grouped._capabilities[device] = torch.cuda.get_device_capability(x.device)
    if moe_grouped._capabilities[device] != (10, 3):
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    # Include router/shared weights and buffers, including storage replacements.
    # The stamp describes the parameters the graph will be captured against,
    # so it can only be taken AFTER the warmups below. RAON_MOE_EXPERT_PREQUANT
    # replaces every weight{g} parameter object during the first warmup, so a
    # stamp taken before it describes tensors the capture never used: the next
    # call recomputes a different stamp, clears the cache, and re-captures
    # every forward, allocating a fresh CUDAGraph each time.
    def _stamp():
        ts = tuple(module.parameters()) + tuple(module.buffers())
        return ts, tuple((id(t), t._version, t.data_ptr(), tuple(t.shape),
                          tuple(t.stride())) for t in ts)
    try:
        tensors, stamp = _stamp()
    except RuntimeError:
        return None
    key = (tuple(x.shape), x.dtype, torch.cuda.current_stream(x.device).cuda_stream)
    cache = getattr(module, "_raon_moe_outer_cache", None)
    if cache is None:
        cache = {}
        module._raon_moe_outer_cache = cache
    entry = cache.get(key)
    if entry is not None and getattr(module, "_raon_moe_outer_stamp", None) != stamp:
        cache.clear()          # a real refit changed the weights
        entry = None
    if entry is None:
        sx = torch.empty_like(x)
        sx.copy_(x)
        # Twice, not once. RAON_MOE_EXPERT_PREQUANT (default ON) quantizes the
        # expert weights on the first call and sets seq._module_groups = None
        # so TE re-groups around the swapped parameters. With a single warmup
        # that re-grouping happens on the captured call, putting one-time work
        # -- reset_recipe_state, pre_first_fuser_forward, the fusion selection
        # -- inside the graph. Call 1 prequantizes, call 2 re-groups, and the
        # captured call runs steady state.
        compute(module, sx)
        compute(module, sx)
        tensors, stamp = _stamp()      # settled: prequantized and re-grouped
        module._raon_moe_outer_stamp = stamp
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = compute(module, sx)
        # Announce here, not only from the atexit tally: vLLM stops EngineCore
        # with SIGTERM, and atexit does not run on SIGTERM.
        global _CAPTURE_ANNOUNCED
        if not _CAPTURE_ANNOUNCED:
            _CAPTURE_ANNOUNCED = True
            print(f"[parity] MoE outer graph CAPTURED: shape={tuple(x.shape)} "
                  f"dtype={x.dtype} layer={getattr(module, 'layer_idx', '?')}",
                  flush=True)
        entry = (graph, sx, result, tensors)
        cache[key] = entry
    graph, sx, result, _ = entry
    global _REPLAY_ANNOUNCED
    if not _REPLAY_ANNOUNCED:
        _REPLAY_ANNOUNCED = True
        print("[parity] MoE outer graph REPLAYED (first replay)", flush=True)
    sx.copy_(x)
    graph.replay()
    # The existing eager-break wrapper copies this borrowed output immediately.
    return result

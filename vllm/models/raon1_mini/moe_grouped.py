"""GPU-counted MoE for validated B300 Raon decode shapes.

Outputs borrow graph storage until the next call. Unsupported shapes use the
original host chain; weight writes invalidate captured pointers.
"""

import torch


def _compute(x, topk_weights, topk_ids, w13, w2, beta, linear_beta, layer_idx):
    from . import model

    t, k = topk_ids.shape
    e = w13.shape[0]
    flat = topk_ids.reshape(-1)
    counts = torch.zeros(e, dtype=torch.int64, device=flat.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    offsets = counts.cumsum(0).to(torch.int32)
    order = torch.argsort(flat, stable=True)
    tokens = torch.div(order, k, rounding_mode="floor")
    xs = x.index_select(0, tokens)
    probs = topk_weights.reshape(-1).float().index_select(0, order)
    gu = torch._grouped_mm(xs, w13.transpose(1, 2), offs=offsets)
    inter = model.moe_fold_probs(model.moe_situ_reference(gu, beta, linear_beta), probs)
    fc2 = torch._grouped_mm(inter, w2.transpose(1, 2), offs=offsets)
    out = torch.empty_like(fc2)
    out.index_copy_(0, order, fc2)
    return model._moe_combine_realized_tail(
        out.view(t, k, x.shape[1]),
        topk_ids,
        topk_weights,
        x.dtype,
        e,
        layer_idx,
        True,
        model.moe_combine_ep_partial,
    )


_cache = {}
_stamps = {}
_capabilities = {}


def clear():
    _cache.clear()
    _stamps.clear()


def supported(x, ids, w13, w2):
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or w13.dtype != torch.bfloat16
        or w2.dtype != torch.bfloat16
    ):
        return False
    if not (
        1 <= x.shape[0] <= 64
        and x.shape[1] == 2560
        and ids.shape[1] == 6
        and tuple(w13.shape) == (128, 2176, 2560)
        and tuple(w2.shape) == (128, 2560, 1088)
    ):
        return False
    dev = x.device.index
    if dev not in _capabilities:
        _capabilities[dev] = torch.cuda.get_device_capability(x.device)
    return _capabilities[dev] == (10, 3)


def try_forward(x, weights, ids, w13, w2, beta, linear_beta, layer_idx):
    if not supported(x, ids, w13, w2):
        return None
    from . import raon_phase

    # EP-combine reads the phase. A local capture would freeze decode even
    # when this call belongs to aligned prefill (including a mixed batch).
    if raon_phase.prefill_now():
        return _compute(x, weights, ids, w13, w2, beta, linear_beta, layer_idx)
    if torch.cuda.is_current_stream_capturing():
        return _compute(x, weights, ids, w13, w2, beta, linear_beta, layer_idx)
    try:
        versions = (w13._version, w2._version)
    except RuntimeError:
        return _compute(x, weights, ids, w13, w2, beta, linear_beta, layer_idx)
    owner = (id(w13), id(w2))
    stamp = (
        versions,
        w13.data_ptr(),
        w2.data_ptr(),
        tuple(w13.stride()),
        tuple(w2.stride()),
    )
    if _stamps.get(owner) != stamp:
        for stale in [key for key in _cache if key[:2] == owner]:
            del _cache[stale]
        _stamps[owner] = stamp
    stream = torch.cuda.current_stream(x.device)
    key = (
        id(w13),
        id(w2),
        versions,
        tuple(x.shape),
        tuple(ids.shape),
        x.dtype,
        weights.dtype,
        stream.cuda_stream,
        float(beta),
        float(linear_beta),
    )
    cached = _cache.get(key)
    if cached is None:
        sx = torch.empty_like(x)
        sw = torch.empty_like(weights)
        si = torch.empty_like(ids)
        sx.copy_(x)
        sw.copy_(weights)
        si.copy_(ids)
        _compute(sx, sw, si, w13, w2, beta, linear_beta, layer_idx)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = _compute(sx, sw, si, w13, w2, beta, linear_beta, layer_idx)
        cached = (graph, sx, sw, si, result, w13, w2)
        _cache[key] = cached
    graph, sx, sw, si, result, _, _ = cached
    sx.copy_(x)
    sw.copy_(weights)
    si.copy_(ids)
    graph.replay()
    return result

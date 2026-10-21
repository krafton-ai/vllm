# Copyright (c) 2026, KRAFTON.
"""RAON_MOE_EXPERT_MXFP8_REALIZED: engine-side MXFP8 realization of the
trainer's routed-expert block.

The trainer runs the routed experts as one fused TE op-fuser Sequential per
EP rank: GroupedLinear(fc1, 64 groups) -> ScaledSituGLU (interleave 32,
gate/value betas, router prob folded in the epilogue) -> GroupedLinear(fc2),
quantized MXFP8 (E4M3, E8M0 1x32 block scales), inputs Fp8Padding'ed to 256
rows/group, realized as GroupedMLP_CuTeGEMMGLU. That realization is bitwise
portable across environments, invariant to the row count per expert and to
batch composition, and deterministic. This module makes the engine run the
same realization, so engine outputs can match a trainer that uses the fused
MXFP8 experts bitwise.

Wiring (model.py expert-major dispatch):
  * rows arrive expert-major over the 128 GLOBAL experts; the trainer runs
    EP=2, i.e. one 64-group fused call per contiguous expert block
    (HybridEP's contiguous expert->rank block map). We make `ep` fused calls
    over blocks of `E/ep` experts (env RAON_MOE_EXPERT_MXFP8_EP, default 2).
  * the engine's resident w13 rows are CONTIGUOUS [gate; value]; megatron's
    fc1 rows are GLU-INTERLEAVED (blocks of 32). fc1 weights are therefore
    re-ordered into megatron's layout at build time (row permutation commutes
    with MXFP8's per-row quantization; done explicitly anyway so the fused
    epilogue sees the trainer's exact layout). fc2 weights bind as views.
  * probs are passed fp32 (trainer moe_router_dtype fp32) and folded by
    the fused epilogue, exactly where the trainer folds them.

Requires transformer_engine and megatron-lm importable for the fuser.
Fail-closed everywhere: if the fused kernel does not engage, this raises.
An unfused mxfp8 chain is a different realization and must never run under
this label.
"""
import math
import os
import types
import weakref

import torch

__all__ = [
    "expert_mxfp8_realized",
    "fused_expert_rows",
    "fused_expert_rows_static",
    "mla_mxfp8_realized",
    "shared_mxfp8_realized",
    "apply_mla_mxfp8",
    "apply_shared_mxfp8",
    "PAD_TARGETS",
    "refit_invalidate",
]

_ANNOUNCED: set = set()


# RAON_MXFP8_MIN_M (default 0 = off): a per-phase precision policy for
# configurations that do not require bitwise parity. A quantized launch whose
# real row count is below this threshold runs its own bf16 realization instead
# (same weights, no pad, no quantize/swizzle launches). At decode M=32 the
# MXFP8 dense GEMMs pay the legality pad (>=32, mla_out_proj pad 160), the
# per-call quantization launches and the scale swizzle for GEMMs that are
# launch-bound anyway, while prefill (M in the thousands) is where quantization
# wins. Never enable this when bitwise parity is required: the band check and
# the trainer comparison assume every launch is quantized.
#
# Sites: the dense wrappers (_DenseMXFP8.__call__, real m before padding) and
# the routed-expert forward (model.py, rows = tokens * top_k; below threshold
# it falls back to the stock fused MoE forward, captured in the same piecewise
# graph).
MIN_M = int(os.environ.get("RAON_MXFP8_MIN_M", "0") or "0")
# RAON_DENSE_MXFP8_MIN_M: dense-op-only threshold. A dense launch with fewer
# rows than this runs the resident BF16 weight (same path MIN_M takes); routed
# experts are untouched. Default 0 = off. The W8A8 dense chain (quant + GEMM)
# is slower than BF16 cuBLAS at every M up to 2048 and faster only around
# M=8192, while MXFP8 wins in the routed experts.
DENSE_MIN_M = int(os.environ.get("RAON_DENSE_MXFP8_MIN_M", "0") or "0")

# RAON_DENSE_MXFP8_SHADOW=1: also run the TE path for every cutedsl output and
# compare bits (diagnostics only, 2x compute, default off, skipped during
# capture).
_DENSE_SHADOW = os.environ.get("RAON_DENSE_MXFP8_SHADOW", "0") == "1"
# cutedsl launches with the same M-island padding as TE (default on). Without
# it, mla_out_proj (pad 160) diverges from TE on real activations.
_DENSE_PAD_MATCH = os.environ.get("RAON_DENSE_MXFP8_PAD_MATCH", "1") == "1"
_SHADOW_STATS = {"match": 0, "mismatch": 0}
if _DENSE_SHADOW:
    import atexit

    # Always print the match count at exit so that 0 mismatches cannot be
    # mistaken for 0 comparisons.
    atexit.register(lambda: print(
        f"[shadow] SUMMARY match={_SHADOW_STATS['match']} "
        f"mismatch={_SHADOW_STATS['mismatch']} "
        f"nan_bitwise_equal={_SHADOW_STATS.get('nan_bitwise_equal', 0)}",
        flush=True))


def rows_reach_min_m(rows: int, site: str, layer=None) -> bool:
    """True = quantize this launch. Announces ONCE per (site, verdict) so a
    run log proves both phases: which sites ran bf16 below the threshold and
    which ran MXFP8 at or above it. MIN_M=0 never declines (lever off)."""
    if MIN_M <= 0:
        return True
    ok = rows >= MIN_M
    key = ("min-m", site, ok)
    if key not in _ANNOUNCED:
        _ANNOUNCED.add(key)
        if ok:
            print(f"[parity] RAON_MXFP8_MIN_M={MIN_M} LIVE: {site} at "
                  f"rows={rows} (layer {layer}) reaches the threshold -- "
                  "this and every launch this size runs MXFP8", flush=True)
        else:
            print(f"[parity] RAON_MXFP8_MIN_M={MIN_M} LIVE: {site} at "
                  f"rows={rows} (layer {layer}) is below the threshold -- "
                  "this and every launch this size runs bf16 (no pad, no "
                  "quantize launches); per-phase precision policy", flush=True)
    return ok


def expert_mxfp8_realized() -> bool:
    return os.environ.get("RAON_MOE_EXPERT_MXFP8_REALIZED", "0") == "1"


def _ep() -> int:
    return int(os.environ.get("RAON_MOE_EXPERT_MXFP8_EP", "2"))


def _interleave_perm(ffn: int, interleave: int, device) -> torch.Tensor:
    """Row index map: megatron interleaved fc1 row j <- contiguous row perm[j].

    megatron layout = [g_blk0, v_blk0, g_blk1, v_blk1, ...] (blocks of
    `interleave`); engine w13 = [g_all, v_all]. Inverse of
    remove_glu_interleaving on the output dim.
    """
    assert ffn % interleave == 0, (ffn, interleave)
    idx = []
    for b in range(ffn // interleave):
        idx.extend(range(b * interleave, (b + 1) * interleave))
        idx.extend(range(ffn + b * interleave, ffn + (b + 1) * interleave))
    return torch.tensor(idx, dtype=torch.long, device=device)


class _FusedState:
    def __init__(self, layer_idx, w13, w2, gate_beta, value_beta, interleave):
        os.environ.setdefault("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
        import transformer_engine.pytorch as tep
        from transformer_engine.pytorch import ops as ops_mod
        from transformer_engine.common.recipe import Format, MXFP8BlockScaling
        from megatron.core.fusions.situ_te_op_fuser import (
            ScaledSituGLU, enable_situ_op_fuser)
        from megatron.core.extensions.transformer_engine import (
            Fp8Padding, Fp8Unpadding)

        enable_situ_op_fuser()
        self.autocast = tep.fp8_autocast
        self.recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)

        E, i2, h = w13.shape
        ffn = i2 // 2
        assert w2.shape == (E, h, ffn), (w13.shape, w2.shape)
        ep = _ep()
        assert E % ep == 0, (E, ep)
        self.local = E // ep
        assert self.local <= 64, (
            f"TE A_list cap is 64 groups; E={E} ep={ep} gives {self.local}")
        self.ep = ep
        dev = w13.device
        perm = _interleave_perm(ffn, interleave, dev)
        self.blocks = []
        for r in range(ep):
            fc1 = ops_mod.GroupedLinear(self.local, h, i2, bias=False,
                                        device="meta", dtype=torch.bfloat16)
            fc2 = ops_mod.GroupedLinear(self.local, ffn, h, bias=False,
                                        device="meta", dtype=torch.bfloat16)
            for g in range(self.local):
                eid = r * self.local + g
                # fc1: explicit megatron-interleaved copy (the one real
                # allocation this lever makes); fc2: zero-copy view.
                fc1.register_parameter(
                    f"weight{g}",
                    torch.nn.Parameter(
                        w13[eid].index_select(0, perm).contiguous(),
                        requires_grad=False))
                fc2.register_parameter(
                    f"weight{g}",
                    torch.nn.Parameter(w2[eid], requires_grad=False))
            act = ScaledSituGLU(interleave, gate_beta=gate_beta,
                                value_beta=value_beta)
            self.blocks.append(ops_mod.Sequential(fc1, act, fc2))
        self.pad = Fp8Padding(self.local, align_size=256)
        self.unpad = Fp8Unpadding(self.local, align_size=256)
        self.layer_idx = layer_idx
        self.fusion_checked = False
        # Ops the fuser selected, captured before prequantization drops the
        # snapshot they live in. See prequantize_weights.
        self.fusion_evidence = None
        # fc1 is a contiguous COPY (interleave re-order) and fc2 a shared
        # view: after an in-place refit a stale state runs dummy fc1 against
        # live fc2, and its own band check cannot see it (latched during
        # profiling). The versions recorded here are what _get_state compares.
        self.wver = (w13._version, w2._version)
        _record_scope(layer_idx, ("routed_fc1", "routed_fc2"))
        # live-proof state, same shape as the dense path's (_DenseMXFP8.
        # verified / band_dmax): retried on later calls until an input is
        # non-degenerate enough for _band_check to return a verdict.
        self.band_verified, self.band_dmax = False, None
        # RAON_MOE_EXPERT_PREQUANT: swapped after the first real forward
        # (the quantizers only exist inside the recipe's autocast).
        self.prequantized = False
        # RAON_MOE_PRE_FUSER_LATCH: see latch_pre_fuser_forward.
        self.pre_fuser_latched = False


    def latch_pre_fuser_forward(self) -> int:
        """Stop re-writing quantizer usage flags that never change.

        fuser.py:537-538 runs op.pre_fuser_forward(requires_grad=...) on every
        basic op on EVERY forward, unconditionally. The base implementation
        (ops/op.py:70-75) has an empty body, so the whole cost is
        GroupedLinear's override (ops/basic/grouped_linear.py:660-678), which
        per group does 3 get_quantizer() + 3 set_usage() = 6 python calls,
        i.e. 4 GroupedLinears (ep 2 blocks x fc1/fc2) x 64 groups x 6 per
        layer per step. About a third of them are
        grad_output_quantizer.set_usage, which is dead work in inference.

        Every input is constant on this path. requires_grad comes from
        `idx >= seq.first_op_requiring_backward` and the weights are built
        requires_grad=False under torch.no_grad; the recipe is fixed. So the
        flags written on every later call are the flags written on call 1.

        Shadows the bound method with an instance attribute that re-checks the
        one input and falls back to the real method if it ever differs -- 1
        python call and 1 comparison instead of 1 + 6, and fail-open rather
        than fail-silent if an assumption breaks.

        Bit-neutral: the quantizer usage flags end up with the same values,
        written by the same first call. Nothing device-side runs.

        Re-armed automatically across a refit: refit_invalidate() drops the
        whole _FusedState, and these ops are rebuilt with it.
        """
        latched = 0
        for seq in self.blocks:
            # first_op_requiring_backward and _basic_ops live on the
            # OperationFuser, not on Sequential -- Sequential only holds
            # _module_groups, a list of (OperationFuser | Module)
            # (ops/sequential.py:37, ops/fuser.py:341,353). And that list is
            # rebuilt lazily: prequantize_weights() sets it to None on purpose
            # so TE re-groups around the swapped parameters. So this must run
            # on a LATER forward than the prequantize, once the fusers exist
            # again -- otherwise the latch is installed on objects that are
            # about to be thrown away.
            groups = getattr(seq, "_module_groups", None)
            if not groups:
                return 0
            for fuser in groups:
                ops = getattr(fuser, "_basic_ops", None)
                if ops is None:
                    continue          # a plain Module group, nothing to latch
                first_bwd = getattr(fuser, "first_op_requiring_backward", None)
                if first_bwd is None:
                    return 0
                for idx, op in enumerate(ops):
                    if "pre_fuser_forward" in op.__dict__:
                        continue      # already latched
                    bound = op.pre_fuser_forward
                    if getattr(bound, "__func__", None) is _OP_PRE_FUSER_NOOP:
                        continue      # base no-op; shadowing it buys nothing
                    rg = idx >= first_bwd

                    def _latched(*, requires_grad, _rg=rg, _orig=bound):
                        if requires_grad is not _rg:
                            return _orig(requires_grad=requires_grad)
                        return None

                    op.pre_fuser_forward = _latched
                    latched += 1
        self.pre_fuser_latched = latched > 0
        return latched

    def prequantize_weights(self) -> int:
        """Quantize the expert weights once instead of on every forward.

        TE re-quantizes every expert weight on every call. The site is
        ``ops/basic/grouped_linear.py`` -- the ops API, which is what
        ``_FusedState`` builds (``from transformer_engine.pytorch import ops``),
        not ``module/grouped_mlp.py``. ``_get_weight_tensors()`` returns
        ``[getattr(self, f"weight{idx}") for idx in range(num_groups)]`` when
        ``single_grouped_weight=False`` (our case), and the prep loop beside it
        already takes an already-quantized weight straight through at zero
        kernels:

            for w, quantizer in zip(weight_params, weight_quantizers):
                if not with_quantized_compute:
                    w = maybe_dequantize(w, dtype)
                elif with_quantized_compute and not is_quantized_tensor(w):
                    quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)
                    w = quantizer(w)              # <- the branch we are on
                out.append(w)                     # <- where we want to land

        That branch launches two quantize kernels per group per layer per
        step (layers x steps x 64 groups x 2 GEMMs), independent of routing
        because every group is materialised by the padding. Most of the
        quantize time is weights, and weights do not change between steps.

        Bit-neutral. MXFP8 block quantization is stateless and elementwise,
        so the same bf16 bytes produce the same rowwise data and the same
        E8M0 scale_inv whether the quantizer runs once or every step.
        We reuse the op's own quantizer at its own index, with the same
        usage flags, so the bytes are the ones TE would have produced.

        No new staleness class, but note the following.

        fc2's weight is a zero-copy view of live w2, so an in-place refit
        reaches it for free. Snapshotting it here destroys that property.
        This adds no new risk because fc1 is already a build-time frozen copy
        sitting in the same Sequential, so a refit that this state survives
        already serves stale fc1 -- the engine is broken in that scenario
        with or without prequantization.

        The guard that covers it is refit_invalidate(), not the version
        pair: refit writes with param.data.copy_ and .data bypasses the
        version counter, so self.wver can be unchanged across a real refit.
        Treat (w13._version, w2._version) as a tripwire for non-.data
        writers only.
        refit_invalidate() deletes _mxfp8_fused_state and clears _STATES, so
        the whole state -- these swapped parameters and the prequantized flag
        with it -- is rebuilt from live weights. That hook is the load-bearing
        one.

        Call AFTER a real forward: get_quantizer only returns a configured
        quantizer once the recipe's autocast has built the fp8 metadata.
        Returns the number of parameters swapped (0 if unavailable, which
        leaves the engine on the original path).
        """
        from transformer_engine.pytorch.ops._common import is_quantized_tensor

        swapped = 0
        for seq in self.blocks:
            ops = [o for o in seq if hasattr(o, "get_quantizer")]
            for op in ops:
                n = getattr(op, "num_gemms", None) or self.local
                for g in range(n):
                    name = f"weight{g}"
                    w = getattr(op, name, None)
                    if w is None or is_quantized_tensor(w):
                        continue
                    q = op.get_quantizer("forward", 2 * g + 1)
                    if q is None:
                        return 0
                    # requires_grad is constant False on this path
                    # (torch.no_grad), which is what TE passes as
                    # input_requires_grad -- match it rather than guess.
                    q.set_usage(rowwise=True, columnwise=False)
                    # TE's base Quantizer returns the Tensor subclass
                    # (MXFP8Tensor) only when both conditions hold
                    # (quantized_tensor.py:291):
                    #     if (not self.internal) and torch.is_grad_enabled():
                    #         result = _QuantizeFunc.apply(...)   -> MXFP8Tensor
                    #     else:
                    #         result = _QuantizeFunc.forward(...) -> ...Storage
                    # The bare storage (MXFP8TensorStorage) is not a
                    # torch.Tensor: nn.Parameter cannot wrap it, and TE's
                    # fuser (fuser.maybe_fuse_ops) and cross-group check
                    # (grouped_linear.py:575-602) read .dtype, .device,
                    # .requires_grad and type(.data) off every weight. An op's
                    # forward weight quantizer carries internal=True, so both
                    # internal=False and enable_grad are needed, and only
                    # around the quantize call.
                    #
                    # internal is restored immediately: it belongs to TE's
                    # quantizer and governs its own later use, not ours.
                    # requires_grad=False on these weights, so grad mode builds
                    # no graph -- it only selects the return type.
                    _internal = q.internal
                    try:
                        q.internal = False
                        with torch.enable_grad():
                            qw = q(w)
                    finally:
                        q.internal = _internal
                    if not isinstance(qw, torch.Tensor):
                        raise RuntimeError(
                            f"{name}: quantizer returned {type(qw).__name__}, "
                            "not a torch.Tensor subclass -- TE's fuser reads "
                            ".dtype off this and would fail on the next "
                            "re-grouping pass")
                    # Do not wrap qw in torch.nn.Parameter: a storage object
                    # has no detach(), and the caller's except is fail-open, so
                    # the swap would be skipped silently and TE would keep
                    # re-quantizing every forward.
                    #
                    # TE's own prep loop keeps the quantizer's output as a
                    # plain value (ops/basic/grouped_linear.py, quoted above),
                    # and _get_weight_tensors reads it back with getattr, so a
                    # plain attribute is what that path expects. Delete the
                    # nn.Parameter entry first: _parameters is consulted before
                    # __dict__, so leaving it would keep handing TE the
                    # original bf16 weight.
                    if isinstance(qw, torch.nn.Parameter):
                        op.register_parameter(name, qw)
                    else:
                        if name in op._parameters:
                            del op._parameters[name]
                        setattr(op, name, qw)
                    if not is_quantized_tensor(getattr(op, name)):
                        raise RuntimeError(
                            f"{name}: swap did not take -- getattr returns a "
                            "non-quantized tensor, so TE would re-quantize "
                            "anyway and the lever would be a no-op")
                    swapped += 1
            # The fuser snapshots basic-op parameters into _module_groups
            # (fuser.py:358); drop the snapshot so the next forward re-reads
            # the parameters we just replaced.
            # Take the fusion evidence first. The fail-closed check in the
            # forward reads this same snapshot to prove GroupedMLP_CuTeGEMMGLU
            # was selected; without the recorded names it would see an empty
            # op list and raise. Recording them keeps the check meaningful:
            # it still asserts on the ops the fuser actually chose, read at
            # the last moment they were observable.
            names = []
            for group in getattr(seq, "_module_groups", None) or []:
                for entry in getattr(group, "_forward_ops", None) or []:
                    o = entry[0] if isinstance(entry, tuple) else entry
                    names.append(type(o).__name__)
            if self.fusion_evidence is None:
                self.fusion_evidence = []
            self.fusion_evidence.append(names)
            seq._module_groups = None
        self.prequantized = swapped > 0
        return swapped


# Per-layer state registry. The expert chain is a free function (no module
# to hang state on), so state is keyed by the
# identity of the layer's resident w13 weight; the state holds a reference to
# it, which pins the id for the process lifetime.
_STATES: dict = {}


def _get_state(module, w13, w2, gate_beta, value_beta, interleave=32,
               cache_on=None, layer_idx=None):
    if module is not None:
        st = getattr(module, "_mxfp8_fused_state", None)
        if st is not None and st.wver != (w13._version, w2._version):
            _announce_rebound(f"routed experts layer {st.layer_idx}",
                              st.wver, (w13._version, w2._version))
            st = None
        if st is None:
            st = _FusedState(getattr(module, "layer_idx", None), w13, w2,
                             float(gate_beta), float(value_beta), interleave)
            module._mxfp8_fused_state = st
        return st
    anchor = cache_on if cache_on is not None else w13
    key = id(anchor)
    st = _STATES.get(key)
    if st is not None and st.wver != (w13._version, w2._version):
        _announce_rebound(f"routed experts layer {st.layer_idx}",
                          st.wver, (w13._version, w2._version))
        st = None
    if st is None:
        # Prefer the caller's real decoder layer index over the anchor label.
        # A non-int label silently disables _record_scope, which would leave
        # the routed experts out of the engine's te_precision_scope and make
        # the startup gate refuse the pair.
        label = layer_idx if isinstance(layer_idx, int) else f"w13@{key:#x}"
        st = _FusedState(label, w13, w2,
                         float(gate_beta), float(value_beta), interleave)
        st.anchor = anchor  # pin the keyed object
        _STATES[key] = st
    return st


def fused_expert_rows_static(module, xs, counts_t, ids_sorted, probs_rows,
                             w13, w2, beta, linear_beta, *, cache_on, layer_idx):
    """Capture-safe variant of fused_expert_rows: constant shapes, no host reads.

    moe_outer_graph.try_forward captures the whole MoE forward as a CUDA
    graph, but refuses the MXFP8 chain (``model.moe_expert_mxfp8_realized()``
    in its gate). The obstacle is not the arithmetic: fused_expert_rows reads
    three things off the device on every call and a graph cannot contain a
    host read:

      1. ``counts`` arrives as a python list (model.py's bincount().tolist())
      2. ``n = int(sum(local_counts))`` then slices ``xs[cursor:cursor + n]``
         -- a data-dependent extent, so the tensor shape moves per step
      3. ``Fp8Padding`` takes m_splits / padded_m_splits as python lists

    The fix extends the trainer's own padding. The trainer Fp8Pads to 256
    rows/group: a group with k>0 rows becomes ceil(k/256)*256 and an empty
    group becomes 0, so the total moves with the routing. Padding every group
    to exactly ALIGN rows makes the total E * ALIGN, a constant, and every
    index below is derivable on the device from ``counts_t`` alone.

    Bit-neutral: this realization is invariant to the row count per expert.
    Padding rows are zeros that occupy their own 1x32 MXFP8 blocks along the
    hidden dim, so they cannot move any real row's E8M0 scale; and each output
    row depends only on its own input row and its expert's weights.

    The cost is shape-dependent. At width 8 the padded extent goes from
    roughly 5,100 rows to 16,384 (~3.2x more GEMM); at width 32 it is roughly
    12,800 -> 16,384 (~1.3x). This trades against the CUDA graph speedup;
    profile both widths.
    """
    # Keywords, not position. _get_state is
    #   (module, w13, w2, gate_beta, value_beta, interleave=32,
    #    cache_on=None, layer_idx=None)
    # and interleave is pure arithmetic: it drives _interleave_perm (the fc1
    # row permutation) and ScaledSituGLU's grouping. The trainer uses 32. A
    # layer_idx bound positionally to interleave could pass
    # `ffn % interleave == 0` and silently build a different permutation.
    st = _get_state(module, w13, w2, beta, linear_beta,
                    cache_on=cache_on, layer_idx=layer_idx)
    dev = xs.device
    align = int(st.pad.align_size) if hasattr(st.pad, "align_size") else 256
    e_total = int(w13.shape[0])
    n_rows = int(xs.shape[0])          # T*K -- constant for a captured shape
    # Every expert gets exactly `align` rows, so a routing that sends more than
    # `align` rows to one expert would make `within` exceed the slot and `dest`
    # run into the next expert -- or off the end of `padded`. n_rows <= align
    # makes that impossible for ANY routing. Shapes are python ints here, so
    # this costs no device sync and is safe inside capture.
    if n_rows > align:
        raise RuntimeError(
            f"fused_expert_rows_static: {n_rows} rows with align={align}. "
            "One expert could receive more rows than its slot holds, and the "
            "index_copy_ would write out of bounds (device-side assert). The "
            "caller must fall back to the dynamic path above this size.")

    # Destination row for every real row, computed entirely on device.
    #   starts[e]  exclusive prefix sum of counts -> where expert e's rows begin
    #   within     each row's rank inside its own expert
    #   dest       expert's fixed slot base + that rank
    starts = counts_t.cumsum(0) - counts_t
    within = (torch.arange(n_rows, device=dev, dtype=starts.dtype)
              - starts.index_select(0, ids_sorted))
    dest = ids_sorted.to(starts.dtype) * align + within

    # Zeros, where TE's Fp8Padding allocates with torch.empty and lets
    # fused_multi_row_padding decide what the pad rows hold. That difference
    # cannot reach a real row's bits: MXFP8 quantizes in 1x32 blocks ALONG THE
    # HIDDEN DIM within a row, so a pad row's contents set only its own E8M0
    # scale, and the GEMM gives each output row a function of its own input row
    # and its expert's weights alone. Their outputs are then dropped by the
    # gather at the end. Zeros additionally keep a NaN or Inf out of the padded
    # region, which garbage memory does not guarantee.
    padded = xs.new_zeros(e_total * align, xs.shape[1])
    padded.index_copy_(0, dest, xs)
    # Pad probabilities are zero, so the epilogue's prob fold sends those rows
    # to exactly 0 rather than to whatever a stale buffer held.
    pp_all = probs_rows.new_zeros(e_total * align)
    pp_all.index_copy_(0, dest, probs_rows)

    # One constant split vector, built once per (device, local, align).
    tpe_pt = _static_splits(st.local, align, dev)

    outs = []
    for r, seq in enumerate(st.blocks):
        lo, hi = r * st.local * align, (r + 1) * st.local * align   # CONSTANT
        with torch.no_grad(), st.autocast(enabled=True, fp8_recipe=st.recipe):
            outs.append(seq(padded[lo:hi], tpe_pt, pp_all[lo:hi], tpe_pt))
    _announce_graph_safe_path(st)
    y_all = torch.cat(outs) if len(outs) > 1 else outs[0]
    return y_all.index_select(0, dest)


_GRAPH_SAFE_ANNOUNCED = False


def _announce_graph_safe_path(st) -> None:
    """Say ONCE which TE flow the static-pad path actually reached.

    The static-pad work is pointless if TE still takes
    ``_fuser_forward_split_quantize``, whose own comment reads "``m_splits`` is
    needed on CPU here, so this flow is NOT cuda-graphable". The graph-safe twin
    ``_fuser_forward_grouped_tensor`` has no host read anywhere in its 133
    lines. Which one runs is decided by
    ``GroupedLinear._is_graph_safe_path_supported``: on CC 10.x-11.0 it returns
    True as soon as every input quantizer is an MXFP8Quantizer.

    Asking the predicate directly is the only reliable way to know; reading
    the env var and assuming can report graphs on while running eager.
    """
    global _GRAPH_SAFE_ANNOUNCED
    if _GRAPH_SAFE_ANNOUNCED:
        return
    _GRAPH_SAFE_ANNOUNCED = True
    try:
        op = next(o for seq in st.blocks for o in seq
                  if hasattr(o, "_is_graph_safe_path_supported"))
        qs = [op.get_quantizer("forward", 2 * g) for g in range(op.num_groups)]
        safe = op._is_graph_safe_path_supported(
            with_quantized_compute=True,
            input_quantizers=qs,
            dtype=torch.bfloat16,
            single_grouped_weight=op.single_grouped_weight,
        )
    except Exception as exc:
        print(f"[parity] RAON_MOE_MXFP8_STATIC_PAD: cannot determine TE flow "
              f"({type(exc).__name__}: {exc})", flush=True)
        return
    if safe:
        print("[parity] RAON_MOE_MXFP8_STATIC_PAD: TE is on the GRAPH-SAFE "
              "grouped-tensor flow (no host read); capture can proceed",
              flush=True)
    else:
        print("[parity] RAON_MOE_MXFP8_STATIC_PAD WARNING: TE fell back to the "
              "legacy split-quantize flow, which its own source calls NOT "
              "cuda-graphable. The static padding is then pure overhead -- "
              "check compute capability and quantizer types before trusting "
              "any number from this configuration.", flush=True)


_STATIC_SPLITS: dict = {}


def _static_splits(local: int, align: int, device):
    """``[align] * local`` as an int32 tensor, cached per (device, local, align).

    Rebuilding it per call would be harmless arithmetically but would add a
    host->device copy inside the captured region.
    """
    key = (str(device), local, align)
    t = _STATIC_SPLITS.get(key)
    if t is None:
        t = torch.full((local,), align, dtype=torch.int32, device=device)
        _STATIC_SPLITS[key] = t
    return t


def fused_expert_rows(module, xs, counts, probs_rows, w13, w2,
                      gate_beta, value_beta, cache_on=None, layer_idx=None):
    """Run the trainer's fused MXFP8 expert block on expert-major rows.

    xs         [sum(counts), H] rows sorted by ascending GLOBAL expert id
    counts     python list, len E (host)
    probs_rows [sum(counts)] fp32 router weights, same order
    returns    [sum(counts), H] fc2 rows (probs folded), same order
    """
    st = _get_state(module, w13, w2, gate_beta, value_beta,
                    cache_on=cache_on, layer_idx=layer_idx)
    dev = xs.device
    outs = []
    cursor = 0
    for r, seq in enumerate(st.blocks):
        local_counts = counts[r * st.local:(r + 1) * st.local]
        n = int(sum(local_counts))
        seg = xs[cursor:cursor + n]
        pr = probs_rows[cursor:cursor + n]
        cursor += n
        xp, tpe_p = st.pad(seg, local_counts)
        pp, _ = st.pad(pr.unsqueeze(-1), local_counts)
        tpe_pt = torch.tensor(tpe_p, dtype=torch.int32, device=dev)
        with torch.no_grad(), st.autocast(enabled=True, fp8_recipe=st.recipe):
            y = seq(xp, tpe_pt, pp.squeeze(-1), tpe_pt)
        outs.append(st.unpad(y, local_counts))
    assert cursor == xs.shape[0], (cursor, xs.shape)
    out = torch.cat(outs) if len(outs) > 1 else outs[0]

    # RAON_MOE_EXPERT_PREQUANT: after the first real forward the recipe's
    # quantizers exist, so the weights can be quantized ONCE and handed to
    # TE already-quantized on every later step. See _FusedState.
    # prequantize_weights for the mechanism and why it is bit-neutral.
    # Default ON: the swap removes two quantize launches per group per step
    # and the branch it targets is TE's own. Set 0 to disable.
    if (not st.prequantized
            and os.environ.get("RAON_MOE_EXPERT_PREQUANT", "1") == "1"):
        try:
            # inside the SAME autocast the forward above used: get_quantizer
            # only hands back a configured quantizer while the recipe's fp8
            # metadata is live.
            with torch.no_grad(), st.autocast(enabled=True,
                                              fp8_recipe=st.recipe):
                n = st.prequantize_weights()
        except Exception as exc:          # never fail a forward for a perf lever
            st.prequantized = True        # do not retry every step
            print(f"[parity] RAON_MOE_EXPERT_PREQUANT SKIPPED on layer "
                  f"{st.layer_idx}: {type(exc).__name__}: {exc}; TE keeps "
                  f"re-quantizing every forward", flush=True)
        else:
            if n:
                print(f"[parity] RAON_MOE_EXPERT_PREQUANT LIVE: swapped {n} "
                      f"expert weight tensors on layer {st.layer_idx}; TE's "
                      f"is_quantized_tensor branch now passes them through",
                      flush=True)
            else:
                print(f"[parity] RAON_MOE_EXPERT_PREQUANT INERT on layer "
                      f"{st.layer_idx}: no quantizer found, TE's per-forward "
                      f"path stays armed", flush=True)

    # Strictly after the prequantize: that call nulls seq._module_groups so
    # TE re-groups around the swapped parameters, which discards the fusers the
    # latch would attach to. Gating on st.prequantized means the latch lands on
    # the first forward that runs with the rebuilt groups.
    if (st.prequantized and not st.pre_fuser_latched
            and os.environ.get("RAON_MOE_PRE_FUSER_LATCH", "1") == "1"):
        try:
            n = st.latch_pre_fuser_forward()
        except Exception as exc:
            st.pre_fuser_latched = True
            print(f"[parity] RAON_MOE_PRE_FUSER_LATCH SKIPPED on layer "
                  f"{st.layer_idx}: {type(exc).__name__}: {exc}", flush=True)
        else:
            if n:
                print(f"[parity] RAON_MOE_PRE_FUSER_LATCH LIVE: shadowed "
                      f"pre_fuser_forward on {n} ops on layer {st.layer_idx}; "
                      f"6 python calls per group per forward become 1",
                      flush=True)

    # ---- fail-closed fusion evidence + one-time live-proof announce ---------
    # Not require_situ_fusion_engaged: that helper verifies once and then
    # returns "already verified" via a module-global flag, which skips the
    # second EP block. Inspect each Sequential's selected ops directly -- the
    # same introspection the helper does, per block.
    if not st.fusion_checked:
        st.fusion_checked = True

        def _selected_ops(sq):
            names = []
            for group in getattr(sq, "_module_groups", None) or []:
                for entry in getattr(group, "_forward_ops", None) or []:
                    o = entry[0] if isinstance(entry, tuple) else entry
                    names.append(type(o).__name__)
            return names

        # Live ops when they are there; otherwise the snapshot taken by
        # prequantization just before dropping them. An empty list must
        # never stand in for "checked".
        evidence = [_selected_ops(sq) for sq in st.blocks]
        if not any(evidence) and getattr(st, "fusion_evidence", None):
            evidence = st.fusion_evidence
        for ev in evidence:
            if "GroupedMLP_CuTeGEMMGLU" not in ev:
                raise RuntimeError(
                    "RAON_MOE_EXPERT_MXFP8_REALIZED=1: fused CuTe "
                    f"blockscaled kernel did NOT engage (forward ops {ev}); "
                    "an unfused mxfp8 chain is a different realization -- "
                    "refusing to run under this label")
        key = ("announce", st.layer_idx)
        if key not in _ANNOUNCED:
            _ANNOUNCED.add(key)
            print(f"[parity] RAON_MOE_EXPERT_MXFP8_REALIZED=1: fused CuTe "
                  f"blockscaled grouped MLP INSTALLED on "
                  f"layer_idx={st.layer_idx} (ep={st.ep}, {st.local} "
                  f"groups/call); fusion evidence {evidence}. Whether it is "
                  f"correctly WIRED is stated by the LIVE line, not by this "
                  f"one.", flush=True)

    # ---- live-proof, through the SHARED _band_check -------------------------
    # Quantization delta vs the bf16 chain on these rows. A wrong fc1
    # interleave permutation or a mis-folded probs epilogue shows up as O(10+).
    #
    # Use the shared _band_check rather than an inline `dmax > 2.0` test,
    # which is false for nan: _band_check tests `not (d < band)` so a nan
    # fails instead of passing, declines to judge vLLM's degenerate warmup
    # activations (all-zero or saturated) instead of raising on them, and
    # refuses `d == 0` (a path that quantizes nothing). It lives outside the
    # once-only fusion block above so a declined verdict is retried on the
    # next call.
    if not st.band_verified and _band_check_active():
        with torch.no_grad():
            ref = torch.empty_like(out)
            c0 = 0
            for eid, cnt in enumerate(counts):
                if cnt == 0:
                    continue
                segr = xs[c0:c0 + cnt]
                gu = segr @ w13[eid].t()
                half = gu.shape[-1] // 2
                g = gu[..., :half].float()
                v = gu[..., half:].float()
                ga = (float(gate_beta) * torch.tanh(g / float(gate_beta))
                      * torch.sigmoid(g))
                va = float(value_beta) * torch.tanh(v / float(value_beta))
                situ = (ga * va).to(gu.dtype)
                situ = situ * probs_rows[c0:c0 + cnt].to(
                    situ.dtype).unsqueeze(-1)
                ref[c0:c0 + cnt] = situ @ w2[eid].t()
                c0 += cnt
        # tag carries the layer so the LIVE line is per MoE layer, not once
        # for the whole model (_band_check keys its announce by tag).
        band_d = _band_check("RAON_MOE_EXPERT_MXFP8_REALIZED",
                             f"routed_experts_l{st.layer_idx}",
                             out, ref, st.layer_idx)
        if band_d is not None:        # None = degenerate input, retry later
            st.band_verified = True
            st.band_dmax = band_d
    return out


# ==========================================================================
# RAON_MLA_MXFP8_REALIZED / RAON_MOE_SHARED_MXFP8_REALIZED
# --------------------------------------------------------------------------
# The other two quantized groups, realized on the engine. The trainer runs
# these as ordinary TE Linear modules under MXFP8BlockScaling:
#
#   MLA (7 layers)      engine module          trainer module        K -> N
#     q  down           fused_qkv_a_proj[:qL]  linear_q_down_proj    2560 -> 576
#     kv down           fused_qkv_a_proj[qL:]  linear_kv_down_proj   2560 -> 576
#     q  up             q_b_proj               linear_q_up_proj       576 -> 7680
#     kv up             kv_b_proj              linear_kv_up_proj      512 -> 10240
#     output gate       g_proj                 linear_gate           2560 -> 5120
#     out proj          o_proj                 linear_proj           5120 -> 2560
#   shared expert (24 layers)
#     fc1               gate_up_proj           linear_fc1            2560 -> 2176
#     fc2               down_proj              linear_fc2            1088 -> 2560
#
# Two structural facts this wiring has to respect, both of them places where a
# "wrap every projection in te.Linear" version would silently be a different
# realization than the trainer's:
#
#  1. `fused_qkv_a_proj` is ONE MergedColumnParallelLinear in the engine and
#     TWO separate GEMMs in the trainer. It is split here into two te.Linear
#     calls at the trainer's boundary, because the K-reduction is identical but
#     cuBLASLt picks its algorithm from the full problem shape (N included), so
#     a fused N=1152 launch is not guaranteed to match the N=576 launch's bits.
#  2. The MXFP8 dense GEMM's bits are a function of the row count M at some
#     shapes (e.g. mla out_proj K=5120 differs between M=128 and M=24576).
#     The engine's M is its token count, the trainer's is its packed
#     footprint, so the short side is zero-padded up onto the trainer's
#     realization. PAD_TARGETS below holds the measured floor per op; 0 means
#     the op is M-invariant at the real shape and needs no pad.
#
# Fail-closed: TP>1, a bias, a missing module, a quantized checkpoint weight,
# or an output outside the quantization band vs the module's own bf16 result
# all raise rather than run a mislabelled configuration.
# ==========================================================================

_MLA_ENV = "RAON_MLA_MXFP8_REALIZED"
_SHARED_ENV = "RAON_MOE_SHARED_MXFP8_REALIZED"

# Measured M-island floors per op. Overridable per op with
# RAON_MXFP8_PAD_<OP>=<int> and globally disabled with
# RAON_MXFP8_PAD_DISABLE=1.
#
# mla_out_proj (K=5120 -> N=2560) is M-dependent at exactly M in
# {32, 64, 96, 128} and bitwise identical to the trainer's launch at every
# M >= 160, with no pockets above that. So 160 is the floor: launching at
# >= 160 rows puts the engine on the trainer's own realization. Pad content
# cannot perturb real rows: MXFP8's E8M0 scale is per 1x32 block within a
# row. The other seven ops are M-invariant at their real shapes for every
# legal M, so they need no pad beyond the legality floor of 32.
#
# Step 32 is the finest grid MXFP8 permits: TE's C++ quantizer refuses any
# tensor dim not divisible by 32.
#
# The pad costs about 30% on mla_out_proj at small decode batches (M <= 128)
# and is free from M >= 256.
#
# 0 means "M-invariant at this shape, no pad beyond the legality floor of
# 32". Do not add pads to the zero entries: there is nothing to cover, and
# _pad_for already floors every launch at 32.
PAD_TARGETS = {
    "mla_q_down": 0,
    "mla_kv_down": 0,
    "mla_q_up": 0,
    "mla_kv_up": 0,
    "mla_linear_gate": 0,
    "mla_out_proj": 160,
    "shared_fc1": 0,
    "shared_fc2": 0,
}


# ---- te_precision_scope recording -------------------------------------------
# The startup parity gate compares the trainer's and the engine's
# composition_sha256 by byte equality, and te_precision_scope flows into that
# hash verbatim on both sides. The trainer records its map decisions through
# record_te_precision (megatron's RecipeConfig.match_to_config_key calls it);
# without recording here, the engine would emit "none" and the gate would
# refuse the pair.
#
# So the engine records the same decisions with the trainer-side module
# names. Serialization cannot diverge: trainer and engine run the identical
# parity_contract.py, so the same _te_precision_scope() (same ".N."
# normalization, same sort, same separators) aggregates both registries.
#
# Env-gated, default off, on purpose. Some trainer configurations record the
# routed experts as bf16 (the fused path never consults the map) while this
# engine runs them quantized, so an unconditional engine record would
# mismatch those configurations. Set RAON_MXFP8_SCOPE_RECORD=1 where the
# trainer records them as quantized.
#
# Timing: record_te_precision re-emits the side manifest when a decision
# arrives after the first write, so recording at wrap/install time is safe.
_SCOPE_ENV = "RAON_MXFP8_SCOPE_RECORD"
_SCOPE_TRAINER_NAMES = {
    "mla_q_down": "self_attention.linear_q_down_proj",
    "mla_kv_down": "self_attention.linear_kv_down_proj",
    "mla_q_up": "self_attention.linear_q_up_proj",
    "mla_kv_up": "self_attention.linear_kv_up_proj",
    "mla_linear_gate": "self_attention.linear_gate",
    "mla_out_proj": "self_attention.linear_proj",
    "shared_fc1": "mlp.shared_experts.linear_fc1",
    "shared_fc2": "mlp.shared_experts.linear_fc2",
    "routed_fc1": "mlp.experts.linear_fc1",
    "routed_fc2": "mlp.experts.linear_fc2",
}


def _requant_inplace(op, src) -> None:
    """Requantize a cutedsl prequant into its existing buffers.

    Reapplies the same quantizer (mxfp8_quantize, swizzled) to the live
    weight and copy_'s into the existing buffers, so the result is bitwise
    identical to an independent rebuild while keeping the pointers captured
    CUDA graphs hold. _cutedsl[2] (transposed view) shares storage with [0]
    and updates automatically.
    """
    from flashinfer import mxfp8_quantize as _q
    q1, s1 = _q(src.detach().contiguous(), is_sf_swizzled_layout=True)
    w_q, w_sf = op._cutedsl[0], op._cutedsl[1]
    p0, p1 = w_q.data_ptr(), w_sf.data_ptr()
    # The _cutedsl buffers are inference tensors created inside the vLLM v1
    # EngineCore's inference_mode forward. Refit is called outside
    # InferenceMode (from a worker extension), where a bare copy_ fails with
    # "Inplace update to inference tensor outside InferenceMode". In-place
    # updates inside InferenceMode are legal; arithmetic and pointers are
    # unchanged.
    with torch.inference_mode():
        w_q.copy_(q1)
        w_sf.copy_(s1.view(w_sf.dtype))
    assert (w_q.data_ptr() == p0 and w_sf.data_ptr() == p1), \
        "requant: prequant buffer pointer moved (contract)"
    op.verified = False   # re-arm the band check (same as the drop path)


def refit_invalidate(model) -> int:
    """Drop every cached MXFP8 wrapper: the next forward rebuilds from live weights.

    Called from the RL framework's post-refit weight reprocessing hook (the
    same hook that refreshes kv_b_proj absorption). Three caches go stale
    across a refit and none are observable from outside, because refit writes
    with param.data.copy_ and .data bypasses the version counter:

      * TE is_first_microbatch weight workspaces  (covered belt-and-braces:
        __call__ no longer passes the kwarg, so TE re-quantizes per call)
      * the q/kv_down .contiguous() halves of fused_qkv_a_proj
      * the routed experts' interleave-reordered fc1 copies

    Dropping the wrapper state re-cuts every copy from the post-refit weights
    and re-runs the band verification against them. With load_format=dummy
    this is load-bearing, not defensive: the priming forward always sees
    random weights, so without this hook the engine would keep serving
    arithmetic based on random weights.
    """
    n_dense = n_split = 0
    n_inplace = 0
    for m in model.modules():
        if hasattr(m, "_raon_mxfp8_ops"):
            st = m._raon_mxfp8_ops
            op = st.get("op") if isinstance(st, dict) else None
            if op is not None and getattr(op, "_cutedsl", None) is not None:
                # cutedsl prequants are requantized in place instead of
                # dropped and rebuilt: captured decode graphs hold the
                # _cutedsl buffer pointers, so a rebuild (new buffers) would
                # leave the graph reading stale or freed-and-reused memory.
                # Reapplying the same quantizer (mxfp8_quantize, swizzled)
                # to the live weight and copy_'ing into the same buffers is
                # bitwise identical to an independent rebuild.
                w = m.weight
                sl = st.get("sl")
                _requant_inplace(op, w if sl is None
                                 else w[sl[0]:sl[1]].contiguous())
                st["ptr"], st["ver"] = w.data_ptr(), w._version
                n_inplace += 1
            else:
                del m._raon_mxfp8_ops
                n_dense += 1
        if hasattr(m, "_raon_mxfp8_split"):
            # The q/kv_down .contiguous() halves are cutedsl prequants too,
            # and captured graphs hold their buffer pointers the same way.
            # Requantize in place only when both halves are cutedsl (the
            # backend is chosen by env + K, so the halves cannot differ);
            # otherwise drop as before.
            st = m._raon_mxfp8_split
            qL = getattr(m, "_raon_mxfp8_qlora", None)
            halves = ()
            if isinstance(st, dict) and qL is not None:
                halves = tuple((st.get(k) for k in ("q_down", "kv_down")))
            if (len(halves) == 2 and all(
                    o is not None and getattr(o, "_cutedsl", None)
                    is not None for o in halves)):
                w = m.weight
                _requant_inplace(halves[0], w[:qL].contiguous())
                _requant_inplace(halves[1], w[qL:].contiguous())
                st["ptr"], st["ver"] = w.data_ptr(), w._version
                n_inplace += 2
            else:
                del m._raon_mxfp8_split
                n_split += 1
        if hasattr(m, "_mxfp8_fused_state"):
            del m._mxfp8_fused_state
    n_routed = len(_STATES)
    _STATES.clear()
    total = n_dense + n_split + n_routed + n_inplace
    if total:
        print(f"[parity] mxfp8 refit-invalidate: dropped {n_dense} dense ops, "
              f"{n_split} qkv splits, {n_routed} fused expert states; "
              f"{n_inplace} cutedsl prequants requantized IN PLACE "
              "(graph pointer contract); copies re-cut and "
              "verification re-armed on post-refit weights",
              flush=True)
    return total


def _announce_rebound(what, v_old, v_new) -> None:
    """One line per weight-change rebuild. Refit copies weights in place
    (model.load_weights -> param copy_), so data_ptr() does not move, and
    without a rebuild TE's is_first_microbatch quantized-weight cache and
    every .contiguous() split/copy keep serving the dummy weights the
    profiling forward primed them with. The sliced q/kv_down copies and the
    routed fc1 copy are stale on both sides of their own band check, so such
    staleness can be silent.
    """
    print(f"[parity] mxfp8 REBOUND {what}: weight storage changed in place "
          f"(version {v_old} -> {v_new}; refit); rebuilt, re-primed, "
          "re-verifying against live weights", flush=True)


def _record_scope(layer_idx, tags) -> None:
    """Record realized MXFP8 decisions under the trainer's module names.

    Skips silently when layer_idx is not an int (the keyed routed-expert
    fallback labels a state "w13@0x..."): a skipped record makes the engine's
    scope FALL SHORT of the trainer's and the gate refuses -- the failure
    lands on the correct side.
    """
    if os.environ.get(_SCOPE_ENV, "0") != "1":
        return
    if not isinstance(layer_idx, int):
        return
    from vllm.models.raon1_mini import parity_contract
    for tag in tags:
        parity_contract.record_te_precision(
            f"decoder.layers.{layer_idx}.{_SCOPE_TRAINER_NAMES[tag]}", None)


def mla_mxfp8_realized() -> bool:
    """Are the MLA projections running the trainer's MXFP8? (default OFF)"""
    return os.environ.get(_MLA_ENV, "0") == "1"


def shared_mxfp8_realized() -> bool:
    """Is the shared expert running the trainer's MXFP8? (default OFF)"""
    return os.environ.get(_SHARED_ENV, "0") == "1"


# TE's C++ MXFP8 quantizer refuses any tensor whose dims are not multiples of
# 32 --
#   quantizer.cpp:1492 create_tensor: "MXFP8 requires tensor dims that are
#   divisible by 32 (got shape=(8,5120))"
# -- which is stricter than the python-level assert_dim_for_fp8_exec (leading
# dims by 8, last dim by 16) that TE advertises. So a decode batch of fewer
# than 32 rows cannot execute the dense MXFP8 GEMMs at all, and every pad
# target must be a multiple of 32. This is a hard requirement, not a
# bit-parity choice.
FP8_M_MULTIPLE = 32


def _pad_for(tag: str) -> int:
    if os.environ.get("RAON_MXFP8_PAD_DISABLE", "0") == "1":
        return FP8_M_MULTIPLE          # never below TE's own legality floor
    want = int(os.environ.get(f"RAON_MXFP8_PAD_{tag.upper()}",
                              PAD_TARGETS.get(tag, 0)))
    want = max(want, FP8_M_MULTIPLE)
    return want + (-want) % FP8_M_MULTIPLE


_DENSE_BACKEND_FALLBACKS = 0


# Boot-time autotune on real shapes (shares the key with mxfp8_w8a8_trtllm):
# the first mm_mxfp8 call per shape runs inside the autotune context, avoiding
# the "No tuned config covers mxfp8_gemm" default-tactic fallback. Default off.
_DENSE_BOOT_AUTOTUNE = os.environ.get(
    "RAON_MXFP8_BOOT_AUTOTUNE", "0") == "1"


def _dense_backend_fallback(tag, layer, why) -> None:
    """RAON_DENSE_MXFP8_BACKEND=cutedsl asked for a shape it cannot serve.

    The op stays on the TE realization -- but never silently: every fallback
    is counted and announced, so the configuration cannot quietly become a
    mixture. Unlike mode 2 this does not raise: the TE path is a verified
    realization of the same E4M3+UE8M0 1x32 W8A8 format, so a fallback is a
    performance regression, not a precision mislabel. Check the log for
    "FALLBACK #" (absent = clean)."""
    global _DENSE_BACKEND_FALLBACKS
    _DENSE_BACKEND_FALLBACKS += 1
    print(f"[parity] RAON_DENSE_MXFP8_BACKEND=cutedsl FALLBACK "
          f"#{_DENSE_BACKEND_FALLBACKS} -> TE on {tag} layer {layer}: {why}",
          flush=True)


class _DenseMXFP8:
    """One trainer-quantized dense GEMM, executed the trainer's way.

    Binds the resident BF16 engine weight without copying it. The FlashInfer
    backend prequantizes supported shapes without requiring Transformer Engine;
    the TE backend keeps a te.Linear and MXFP8BlockScaling recipe. Both retain
    the reference BF16 weight and the op's measured pad target. `is_first_microbatch=True` on the first
    call primes TE's quantized-weight cache; every later call reuses it, which
    is what the trainer does inside a microbatch. That reuse is bitwise
    irrelevant ONLY while the bf16 weight does not change: the RL refit copies
    new weights in place every step, so the callers guard on weight._version
    and rebuild this op (fresh prime, fresh verification) when it moves.
    """

    def __init__(self, tag, weight, env="", layer=None):
        os.environ.setdefault("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
        n, k = weight.shape
        self.tag, self.k, self.n = tag, k, n
        self.env, self.layer = env, layer
        self.verified, self.band_dmax = False, None
        self.pad = _pad_for(tag)
        self.backend = os.environ.get("RAON_DENSE_MXFP8_BACKEND", "te")
        if self.backend not in {"te", "cutedsl"}:
            raise ValueError(f"Unknown dense MXFP8 backend: {self.backend!r}")
        # FlashInfer executes supported shapes independently of Transformer Engine.
        # Keep TE only for its explicit backend, shape fallback, or diagnostic shadow.
        needs_te = self.backend != "cutedsl" or k % 32 != 0 or _DENSE_SHADOW
        if needs_te and os.environ.get("RAON_MXFP8_WEIGHTS"):
            raise RuntimeError("Preconverted MXFP8 requires supported FlashInfer shapes without TE shadow")
        if needs_te:
            import transformer_engine.pytorch as tep
            from transformer_engine.common.recipe import Format, MXFP8BlockScaling
            self.autocast = tep.fp8_autocast
            self.recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)
            self.lin = tep.Linear(k, n, bias=False, params_dtype=weight.dtype, device="meta")
        else:
            self.lin = torch.nn.Linear(k, n, bias=False, device="meta", dtype=weight.dtype)
        # .detach() first: vLLM's resident weights are `ModelWeightParameter`,
        # a Parameter subclass whose detach() returns a plain Tensor, and
        # torch.nn.Parameter(<that subclass>) refuses to construct because the
        # round trip changes type. detach() shares storage, so the engine's
        # weight is still bound in place with no copy.
        # register_parameter, not attribute assignment: the same call the
        # routed-expert path uses on GroupedLinear, so TE's own parameter
        # bookkeeping stays intact.
        self.lin.register_parameter(
            "weight", torch.nn.Parameter(weight.detach(), requires_grad=False))
        # RAON_DENSE_MXFP8_BACKEND=cutedsl (default "te"): run this op's
        # GEMM+quant chain through flashinfer mm_mxfp8 backend="cute-dsl"
        # instead of te.Linear. It is faster on all 8 dense shapes
        # (4.8-8.4us/call vs TE 10.3-19.0) and bit-identical to the TE path
        # at the same M. The weight is prequantized once per op build instead
        # of TE's per-call workspace re-quantization (~436MB bf16 re-read per
        # decode step across the 83 dense ops). Refit safety is inherited:
        # refit_invalidate requantizes the prequant in place (or deletes
        # _raon_mxfp8_ops / _raon_mxfp8_split so the op is rebuilt from the
        # post-refit weight). M-padding follows _DENSE_PAD_MATCH (see
        # __call__). Unsupported shapes fall back to the TE path with a
        # counted, announced fallback (never silent); unknown backend values
        # are fail-closed.
        self.backend = os.environ.get("RAON_DENSE_MXFP8_BACKEND", "te")
        self._cutedsl = None
        if self.backend == "cutedsl":
            if k % 32 != 0:
                _dense_backend_fallback(
                    tag, layer, f"K={k} is not a multiple of 32 "
                    "(mxfp8 1x32 scale rows do not tile)")
            else:
                from flashinfer import mm_mxfp8, mxfp8_quantize
                if os.environ.get("RAON_MXFP8_WEIGHTS"):
                    from .mxfp8_artifact import load
                    tensors = load(f"layer-{layer}-{tag}", [weight.shape], weight.device)
                    w_q, w_sf = tensors["q"], tensors["scale"]
                else:
                    w_q, w_sf = mxfp8_quantize(
                        weight.detach().contiguous(), is_sf_swizzled_layout=True)
                # (q (N,K) fp8, sf u8 1D swizzled 128x4, (K,N) col-major view,
                #  cached fn refs -- import cost off the hot path)
                self._cutedsl = (w_q, w_sf, w_q.t(), mm_mxfp8, mxfp8_quantize)
                # RAON_MXFP8_BOOT_AUTOTUNE: wrap the first call per shape in
                # flashinfer autotune to avoid the "No tuned config covers
                # mxfp8_gemm" default-tactic fallback (autotuner.py:1534).
                # Consumed in __call__.
                self._tuned_ms = set()
                key = ("dense-backend", "cutedsl")
                if key not in _ANNOUNCED:
                    _ANNOUNCED.add(key)
                    print("[parity] RAON_DENSE_MXFP8_BACKEND=cutedsl: dense "
                          "MXFP8 ops run flashinfer mm_mxfp8 (cute-dsl) with "
                          "per-op PREQUANT weights -- same E4M3+UE8M0 1x32 "
                          "W8A8 format, bit-identical to the TE realization "
                          "on all 8 shapes; refit invalidation via "
                          "refit_invalidate; "
                          "unsupported shapes fall back to TE, counted",
                          flush=True)
        elif self.backend != "te":
            raise RuntimeError(
                f"RAON_DENSE_MXFP8_BACKEND={self.backend!r}: unknown backend "
                "(expected 'te' or 'cutedsl') -- refusing to run a "
                "mislabelled configuration")
        # RAON_TE_LINEAR_FASTPATH=1: call te.Linear's own forward
        # directly, skipping torch.nn.Module.__call__'s hook dispatch on the
        # hot path. ONLY that empty dispatch layer is bypassed -- the
        # load-bearing wrappers all stay: fp8_autocast (TE reads the recipe
        # from its global state), and TE's forward itself (prepare_forward +
        # the per-call weight re-quantization that the refit-correctness
        # note at the call site depends on). Engaged only when
        # NOTHING is hooked, module-level or global, at build time; the
        # callers rebuild this op whenever the weight moves, so the check
        # re-runs across the refit. Default OFF: self.lin_call IS self.lin.
        self.lin_call = self.lin
        if os.environ.get("RAON_TE_LINEAR_FASTPATH", "0") == "1":
            import torch.nn.modules.module as _nnm
            hooked = any((
                self.lin._forward_hooks, self.lin._forward_pre_hooks,
                self.lin._backward_hooks, self.lin._backward_pre_hooks,
                _nnm._global_forward_hooks, _nnm._global_forward_pre_hooks,
                _nnm._global_backward_hooks,
                _nnm._global_backward_pre_hooks,
            ))
            if not hooked:
                self.lin_call = self.lin.forward
            key = ("te-fastpath", bool(hooked))
            if key not in _ANNOUNCED:
                _ANNOUNCED.add(key)
                print("[parity] RAON_TE_LINEAR_FASTPATH=1: "
                      + ("dense MXFP8 ops call te.Linear.forward directly "
                         "(nn.Module hook dispatch bypassed; no hooks "
                         "registered)" if not hooked else
                         "DECLINED -- hooks are registered on te.Linear or "
                         "globally, keeping the full nn.Module __call__ "
                         "wrapper"), flush=True)
        _DENSE_OPS.add(self)

    def __call__(self, x2d):
        m = x2d.shape[0]
        # Phase policy: below-threshold launches stay on this op's own
        # bf16 realization (same resident weight, no pad, no quantization).
        if MIN_M > 0 and not rows_reach_min_m(m, f"dense:{self.tag}",
                                              self.layer):
            return self.bf16(x2d)
        if DENSE_MIN_M > 0 and m < DENSE_MIN_M:
            key = ("dense-min-m", self.tag)
            if key not in _ANNOUNCED:
                _ANNOUNCED.add(key)
                print(f"[parity] RAON_DENSE_MXFP8_MIN_M={DENSE_MIN_M}: dense:{self.tag} rows={m} "
                      f"(layer {self.layer}) runs the resident BF16 weight", flush=True)
            return self.bf16(x2d)
        if self._cutedsl is not None:
            # cute-dsl chain: activation quant (1x32, swizzled 128x4 scales)
            # + mm_mxfp8. dtype and format are identical to the TE branch.
            # The band-check tail below still runs on this output.
            w_q, w_sf, w_qt, mm_mxfp8, mxfp8_quantize = self._cutedsl
            # Pad to the same M-island as TE. cute-dsl and TE are
            # bit-identical at the same M, but on real decode activations
            # unpadded cute-dsl diverges from padded TE by 1-8 ulp on
            # mla_out_proj (the only op with pad 160). Default on;
            # RAON_DENSE_MXFP8_PAD_MATCH=0 disables it.
            x_launch = x2d
            fused_pad = (os.environ.get("RAON_DENSE_MXFP8_PAD_FUSED", "0") == "1"
                         and _DENSE_PAD_MATCH and m <= 8
                         and x2d.dtype == torch.bfloat16 and x2d.is_contiguous())
            want = m
            if _DENSE_PAD_MATCH:
                want = max(self.pad, m + (-m) % FP8_M_MULTIPLE)
                if want != m and not fused_pad:
                    x_launch = torch.zeros(want, self.k, dtype=x2d.dtype,
                                           device=x2d.device)
                    x_launch[:m] = x2d
            # Boot-time autotune (RAON_MXFP8_BOOT_AUTOTUNE=1, default off):
            # warm up the first call for this op and M three times in the
            # autotune context so the tactic lands in the process cache and
            # later calls and capture use the tuned tactic. An untuned shape
            # during capture raises (fail-closed).
            if _DENSE_BOOT_AUTOTUNE and m not in self._tuned_ms:
                if fused_pad and want != m:
                    x_launch = torch.zeros(want, self.k, dtype=x2d.dtype, device=x2d.device)
                    x_launch[:m] = x2d
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "RAON_MXFP8_BOOT_AUTOTUNE=1: first cutedsl call for "
                        f"dense:{self.tag} M={m} inside graph capture — "
                        "warmup must cover this shape pre-capture")
                from flashinfer.autotuner import autotune as _fi_autotune
                with torch.no_grad(), _fi_autotune(True):
                    for _ in range(3):
                        # warm up with the actual launch shape (x_launch)
                        _aq, _asf = mxfp8_quantize(
                            x_launch.contiguous(), is_sf_swizzled_layout=True)
                        mm_mxfp8(_aq, w_qt, _asf, w_sf,
                                 out_dtype=torch.bfloat16,
                                 backend="cute-dsl")
                torch.cuda.synchronize()
                self._tuned_ms.add(m)
                print(f"[parity] RAON_MXFP8_BOOT_AUTOTUNE=1: dense "
                      f"{self.tag} L{self.layer} M={m} mm_mxfp8 tactics "
                      f"tuned (pre-capture)", flush=True)
            with torch.no_grad():
                if fused_pad and want != m:
                    from .mxfp8_pad_quant import quantize
                    a_q, a_sf = quantize(x2d, want)
                else:
                    a_q, a_sf = mxfp8_quantize(x_launch.contiguous(),
                                               is_sf_swizzled_layout=True)
                out = mm_mxfp8(a_q, w_qt, a_sf, w_sf,
                               out_dtype=torch.bfloat16, backend="cute-dsl")
                if out.shape[0] != m:
                    out = out[:m]
            # Shadow comparison (diagnostics only): run the TE path for every
            # cutedsl output and compare bits, reporting mismatches by
            # tag/layer/M/call number. Finds divergence on real activations
            # that isolated benchmarks miss. Default off, skipped during
            # capture, 2x compute.
            if _DENSE_SHADOW and not torch.cuda.is_current_stream_capturing():
                ref = self._te_out(x2d)
                self._shadow_n = getattr(self, "_shadow_n", 0) + 1
                # torch.equal treats NaN != NaN, so identical-bit NaNs on
                # both sides (profile-pass activations) would count as
                # mismatches. Compare int16 bit views instead, and count
                # NaN-carrying matches separately.
                du = (out.view(torch.int16).to(torch.int32)
                      - ref.view(torch.int16).to(torch.int32)).abs()
                if int(du.max()) != 0:
                    _SHADOW_STATS["mismatch"] += 1
                    if _SHADOW_STATS["mismatch"] <= 20:
                        dd = (out.float() - ref.float()).abs()
                        print(f"[shadow] MISMATCH dense {self.tag} "
                              f"L{self.layer} M={m} call#{self._shadow_n} "
                              f"frac={float((du != 0).float().mean()):.4f} "
                              f"max_ulp={int(du.max())} "
                              f"max_abs={float(dd.max()):.3e}", flush=True)
                else:
                    _SHADOW_STATS["match"] += 1
                    if bool(torch.isnan(out).any()):
                        _SHADOW_STATS["nan_bitwise_equal"] = (
                            _SHADOW_STATS.get("nan_bitwise_equal", 0) + 1)
            if (not self.verified and m >= FP8_M_MULTIPLE
                    and _band_check_active()
                    and not torch.cuda.is_current_stream_capturing()):
                d = _band_check(self.env, self.tag, out, self.bf16(x2d),
                                self.layer)
                if d is not None:
                    self.verified = True
                    self.band_dmax = d
            return out
        out = self._te_out(x2d)
        # A BF16 reference GEMM plus .item() is deliberately forbidden inside
        # capture: the host sync invalidates CUDA graphs and capturing the
        # reference GEMM would also contaminate every throughput replay. The
        # serving harness calls validate_dense_mxfp8_ops() immediately after
        # vLLM finishes capture, before correctness or throughput.
        # _band_check_active() is the kill switch (RAON_MXFP8_BAND_CHECK) and
        # the capture guard is the graph-safety condition.
        if (not self.verified and m >= FP8_M_MULTIPLE
                and _band_check_active()
                and not torch.cuda.is_current_stream_capturing()):
            d = _band_check(self.env, self.tag, out, self.bf16(x2d), self.layer)
            if d is not None:            # None = degenerate input, retry later
                self.verified = True
                self.band_dmax = d
        return out

    def _te_out(self, x2d):
        # TE path, factored out of __call__ so the shadow comparison runs
        # the same code.
        m = x2d.shape[0]
        xin = x2d
        # Launch at max(pad target, next multiple of 8) so the row count is
        # both LEGAL for FP8 and on the trainer's measured M-island.
        want = max(self.pad, m + (-m) % FP8_M_MULTIPLE)
        if os.environ.get("RAON_MXFP8_LOG_M") == "1":
            # Per-step launch-M log: only transitions are printed, so a
            # stable batch is one line per tag and a membership change is
            # one line per affected tag.
            _lm = getattr(self, "_log_m_last", None)
            if _lm != (m, want):
                self._log_m_last = (m, want)
                print(f"[mxfp8-M] {self.tag} L{self.layer} m={m} launch={want}",
                      flush=True)
        if want != m:
            xin = torch.zeros(want, self.k, dtype=x2d.dtype,
                              device=x2d.device)
            xin[:m] = x2d
        with torch.no_grad(), self.autocast(enabled=True,
                                            fp8_recipe=self.recipe):
            # NO is_first_microbatch kwarg, ON PURPOSE. TE's cache condition
            # is `update_ws = is_first_microbatch is None or
            # is_first_microbatch` (linear.py:435): None re-quantizes the
            # weight workspace from the LIVE tensor on every call
            # (base.py:830-841 workspace.quantize_), while a prime-then-False
            # pattern would serve the cached quantization forever, including
            # across a refit: refit writes via param.data.copy_ (vllm
            # weight_utils), and .data bypasses the version counter, so no
            # wrapper-side guard can see the change. Weight quantization is
            # deterministic in the bf16 weight, so while the weight is static
            # this is bitwise identical to the cache; when refit moves the
            # weight, this follows and the cache would not.
            y = self.lin_call(xin)
        return y if y.shape[0] == m else y[:m]

    def bf16(self, x2d):
        with torch.no_grad():
            return self.lin(x2d)


_DENSE_OPS = weakref.WeakSet()


def validate_dense_mxfp8_ops() -> list[dict]:
    """Fail-closed live validation outside CUDA graph capture.

    Every lazy wrapper actually constructed during model profiling/capture is
    exercised against its own BF16 weight. Inputs are deterministic without
    touching the model sampling RNG. An unverified/degenerate result is fatal.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("dense MXFP8 validation must run outside CUDA capture")
    ops = sorted(list(_DENSE_OPS), key=lambda op: (str(op.env), str(op.tag), str(op.layer)))
    if not ops:
        raise RuntimeError("dense MXFP8 enabled but no lazy wrappers were constructed")
    results = []
    for op in ops:
        m = max(FP8_M_MULTIPLE, _pad_for(op.tag))
        # Column-varying values prevent the cancellation/all-zero dummy-input
        # case while keeping activation magnitudes representative and finite.
        columns = ((torch.arange(op.k, device=op.lin.weight.device) % 17) - 8)
        x = columns.to(dtype=op.lin.weight.dtype).mul_(0.015625).expand(m, -1).contiguous()
        op(x)
        if not op.verified or op.band_dmax is None:
            raise RuntimeError(f"dense MXFP8 validation inconclusive: {op.tag} layer {op.layer}")
        results.append(dict(env=op.env, tag=op.tag, layer=op.layer,
                            m=m, max_abs_difference=op.band_dmax))
    print(f"RAON_DENSE_MXFP8_VALIDATED ops={len(results)} outside_capture=1", flush=True)
    return results


def _check_wrappable(env, owner, name, sub):
    if sub is None or not hasattr(sub, "weight"):
        raise RuntimeError(
            f"{env}: {owner} has no {name} carrying a weight -- this port's "
            "module set is not the one this adapter was written against; "
            "quantizing a subset would leave part of the group on a different "
            "realization, which is worse than not arming the lever")
    bias = getattr(sub, "bias", None)
    if bias is not None and getattr(bias, "numel", lambda: 1)() != 0:
        raise RuntimeError(f"{env}: {name} carries a bias; the trainer's "
                           "quantized TE Linears are bias-free")
    if not isinstance(sub.weight, torch.nn.Parameter):
        raise RuntimeError(f"{env}: {name} weight is not a plain parameter "
                           "(already quantized checkpoint?)")
    if sub.weight.dtype != torch.bfloat16:
        raise RuntimeError(f"{env}: {name} weight dtype {sub.weight.dtype} "
                           "!= bfloat16")


def _te_base_pre_fuser_noop():
    """The unbound base ops.op.BasicOperation.pre_fuser_forward, or None.

    Used to skip ops that only inherit the empty base implementation -- there
    is nothing to latch there and shadowing it would only add a python frame.
    """
    try:
        from transformer_engine.pytorch.ops.op import BasicOperation
    except Exception:
        return None
    return getattr(BasicOperation, "pre_fuser_forward", None)


_OP_PRE_FUSER_NOOP = _te_base_pre_fuser_noop()


def _band_check_active() -> bool:
    """RAON_MXFP8_BAND_CHECK=0 opts a benchmark run out of the live proof.

    Default "1" (or unset) keeps the check on: every
    _DenseMXFP8 op and the routed chain recompute a bf16 reference and read
    two host scalars (.item() syncs) until `verified` latches -- and the RL
    refit rebuilds the dense ops every step (fresh prime, fresh
    verification), so under RL that cost recurs per refit. "0" skips the
    reference recompute and the syncs entirely. Fail-closed spirit: the
    default keeps checking; disabling is announced ONCE so a log without
    LIVE lines is attributable to this setting and not to a silent fallback,
    and a run that opts out is knowingly unprotected against the mis-wire /
    stale-refit classes _band_check exists to catch.
    """
    if os.environ.get("RAON_MXFP8_BAND_CHECK", "1") != "0":
        return True
    key = ("band-check-disabled",)
    if key not in _ANNOUNCED:
        _ANNOUNCED.add(key)
        print("[parity] RAON_MXFP8_BAND_CHECK=0: live band verification "
              "SKIPPED -- no bf16 reference recompute, no host sync, no LIVE "
              "lines; mis-wire and stale-refit defects are NOT being checked "
              "on this run", flush=True)
    return False


def _band_check(env, tag, mx_out, bf_out, layer):
    """Live proof the quantized op RAN and is not mis-wired.

    An op that silently fell back to bf16 gives 0; a transposed/permuted weight
    gives O(10). Correct MXFP8 sits in between. `not (d < band)` rather than
    `d > band` so a nan fails instead of passing.

    Returns None when it DECLINES to judge, which the caller treats as "not
    verified yet, try the next call". vLLM runs dummy warmup/profile forwards
    whose activations are degenerate -- all-zero (both precisions then agree
    bitwise, which is not evidence of a fallback) or saturated to inf (the
    difference is inf-minus-inf = nan, which is not evidence of a mis-wire).
    For example, a layer's warmup input can be all zeros; the correct verdict
    is then "no information", not "mis-wired".

    Declines under stream capture too. Everything below this line is a host
    read of device data, which CUDA graph capture forbids
    (cudaErrorStreamCaptureUnsupported; mla_align.py has the same guard).
    Reachability is narrow: _DenseMXFP8.__call__ gates on
    m >= FP8_M_MULTIPLE (32), so only capture sizes >= 32 can reach it, and
    only through apply_mla_mxfp8, the one dense op not already wrapped by
    _raon_fp32_combine_moe_impl. Declining costs the verification nothing,
    because vLLM's memory-profiling forward runs before capture and normally
    latches `verified` there.
    """
    if torch.cuda.is_current_stream_capturing():
        return None
    scale = bf_out.float().abs().max().item()
    if not (scale > 0.0) or not math.isfinite(scale):
        return None                      # degenerate warmup input -- no verdict
    d = (mx_out.float() - bf_out.float()).abs().max().item()
    band = max(2.0, 0.5 * scale)
    if not (d < band):
        raise RuntimeError(
            f"{env}: {tag} on layer {layer} deviates from its own bf16 result "
            f"by {d:.3e} (band {band:.3e}, scale {scale:.3e}, "
            f"d/scale {d / scale:.3f}, M={mx_out.shape[0]}) -- d~scale means "
            "the two paths computed with DIFFERENT WEIGHTS (stale quantized "
            "cache or stale copy across an in-place refit); a "
            "mis-wired pad or weight binding also lands here")
    if d == 0.0:
        raise RuntimeError(
            f"{env}: {tag} on layer {layer} is BITWISE EQUAL to its bf16 "
            f"result on a NON-degenerate input (bf16 max {scale:.3e}), so the "
            "MXFP8 path cannot have run -- refusing to report a quantized path "
            "that quantizes nothing")
    # One line per tag the first time it really executes. A wrapped module the
    # forward never calls (kv_b_proj is absorbed on the engine's decode path)
    # produces no line, so the log states which of the eight quantized dense
    # kinds actually ran rather than which ones were installed.
    key = ("live", env, tag)
    if key not in _ANNOUNCED:
        _ANNOUNCED.add(key)
        print(f"[parity] {env} LIVE: {tag} executed MXFP8 on layer {layer}, "
              f"M={mx_out.shape[0]}, vs its own bf16 max|d|={d:.6e} "
              f"(scale {scale:.3e}, d/scale {d / scale:.3e}, band {band:.3e}; "
              "0 would mean the quantized path never ran; a verdict at tiny "
              "scale is a liveness statement only)",
              flush=True)
    return d


# ---- MLA + shared expert: lazy weight binding ------------------------------
# vLLM builds the modules first and loads the checkpoint into them afterwards,
# so a te.Linear bound at __init__ would hold the UNLOADED parameter -- and for
# the merged-qkv split, an unloaded COPY of it. Both ops are therefore built on
# the first forward, from the module's weight as it is then. The build is also
# re-checked against the parameter's storage on every call, because vLLM's
# loaders sometimes REPLACE a Parameter rather than copy into it; a stale
# binding would silently score the wrong weights.

def _op_for(module, spec):
    """spec: (tag, env, layer, slice-or-None). Build/rebuild on weight identity."""
    tag, env, layer, sl = spec
    w = module.weight
    st = getattr(module, "_raon_mxfp8_ops", None)
    # ptr catches a loader that REPLACES the Parameter. _version was meant to
    # catch the refit that copies INTO it and does not: refit writes via
    # param.data.copy_, and .data bypasses the version counter (a direct
    # copy_ does bump it). The guard stays as a belt for non-.data writers;
    # the authoritative invalidation is refit_invalidate, driven from the
    # post-refit weight reprocessing hook.
    if st is not None and st["ptr"] == w.data_ptr() and st["ver"] != w._version:
        _announce_rebound(f"{tag} layer {layer}", st["ver"], w._version)
        st = None
    if st is None or st["ptr"] != w.data_ptr():
        src = w if sl is None else w[sl[0]:sl[1]].contiguous()
        # Keep "sl": refit_invalidate's in-place requantization needs it to
        # re-cut the slice.
        st = {"ptr": w.data_ptr(), "ver": w._version, "sl": sl,
              "op": _DenseMXFP8(tag, src, env, layer)}
        module._raon_mxfp8_ops = st
    return st["op"]


def _mla_qkv_a_forward(self, inp, *args, **kwargs):
    """Two GEMMs, exactly where the trainer has two.

    Row split of the merged weight at q_lora_rank: rows [0:qL] are
    linear_q_down_proj, rows [qL:] are linear_kv_down_proj (kv_lora + rope).
    """
    qL = self._raon_mxfp8_qlora
    w = self.weight
    st = getattr(self, "_raon_mxfp8_split", None)
    # The split halves are COPIES, so a stale one is stale on BOTH sides of
    # its own band check -- version staleness here is SILENT, unlike the
    # shared-storage ops where it raises. Same guard, more at stake.
    if st is not None and st["ptr"] == w.data_ptr() and st["ver"] != w._version:
        _announce_rebound(f"q/kv_down layer {self._raon_mxfp8_layer}",
                          st["ver"], w._version)
        st = None
    if st is None or st["ptr"] != w.data_ptr():
        st = {"ptr": w.data_ptr(), "ver": w._version,
              "q_down": _DenseMXFP8("mla_q_down", w[:qL].contiguous(),
                                    _MLA_ENV, self._raon_mxfp8_layer),
              "kv_down": _DenseMXFP8("mla_kv_down", w[qL:].contiguous(),
                                     _MLA_ENV, self._raon_mxfp8_layer)}
        self._raon_mxfp8_split = st
    x2d = inp.reshape(-1, inp.shape[-1])
    out = torch.cat((st["q_down"](x2d), st["kv_down"](x2d)), dim=-1)
    return out.view(*inp.shape[:-1], out.shape[-1]), None


def _mla_plain_forward(self, inp, *args, **kwargs):
    op = _op_for(self, self._raon_mxfp8_spec)
    x2d = inp.reshape(-1, inp.shape[-1])
    out = op(x2d)
    return out.view(*inp.shape[:-1], out.shape[-1]), None


def apply_mla_mxfp8(self_attn, layer_idx=None, q_lora_rank=None) -> dict:
    """Run one MLA layer's projections as the trainer's MXFP8. OFF -> no-op."""
    if not mla_mxfp8_realized():
        return {}
    if os.environ.get("RAON_MLA_PROJ_PERSISTENT", "0") == "1":
        raise RuntimeError(
            f"{_MLA_ENV}=1 with RAON_MLA_PROJ_PERSISTENT=1: the persistent "
            "Triton GEMM is the BF16 realization and this option is the "
            "MXFP8 one. Both installs "
            "rebind the same four forwards, so whichever ran second would "
            "silently decide the precision. Set exactly one.")
    from vllm.distributed import get_tensor_model_parallel_world_size

    if get_tensor_model_parallel_world_size() != 1:
        raise RuntimeError(
            f"{_MLA_ENV} requires tensor parallel size 1 (it replaces the MLA "
            "projections' local GEMM without their gather/reduce plumbing)")

    qkv_a = getattr(self_attn, "fused_qkv_a_proj", None)
    _check_wrappable(_MLA_ENV, "MLA", "fused_qkv_a_proj", qkv_a)
    qL = int(q_lora_rank if q_lora_rank is not None
             else getattr(self_attn, "q_lora_rank"))
    w = qkv_a.weight
    if not 0 < qL < w.shape[0]:
        raise RuntimeError(f"{_MLA_ENV}: q_lora_rank {qL} does not split "
                           f"fused_qkv_a_proj rows {tuple(w.shape)}")
    # The split halves are .contiguous() COPIES (te.Linear needs a real matrix,
    # not a strided row view), so they are built on the first forward -- after
    # the checkpoint has been loaded -- and rebuilt if the loader swapped the
    # Parameter. The row split is exactly the trainer's two parameters.
    qkv_a._raon_mxfp8_qlora = qL
    qkv_a._raon_mxfp8_layer = layer_idx
    qkv_a.forward = types.MethodType(_mla_qkv_a_forward, qkv_a)

    wrapped = [("fused_qkv_a_proj[:qL]", "mla_q_down", w.shape[1], qL),
               ("fused_qkv_a_proj[qL:]", "mla_kv_down", w.shape[1],
                w.shape[0] - qL)]
    for name, tag in (("q_b_proj", "mla_q_up"),
                      ("kv_b_proj", "mla_kv_up"),
                      ("g_proj", "mla_linear_gate"),
                      ("o_proj", "mla_out_proj")):
        sub = getattr(self_attn, name, None)
        if sub is None and name == "g_proj":
            continue                       # output gate is config-optional
        _check_wrappable(_MLA_ENV, "MLA", name, sub)
        sub._raon_mxfp8_spec = (tag, _MLA_ENV, layer_idx, None)
        sub.forward = types.MethodType(_mla_plain_forward, sub)
        wrapped.append((name, tag, sub.weight.shape[1], sub.weight.shape[0]))

    _record_scope(layer_idx, [t for _, t, _, _ in wrapped])
    _announce_dense(_MLA_ENV, f"MLA layer_idx={layer_idx}", wrapped)
    return {"mla_mxfp8": [f"{n}->{t}" for n, t, _, _ in wrapped]}


# ---- shared expert --------------------------------------------------------

def apply_shared_mxfp8(moe_module, layer_idx=None) -> dict:
    """Run one MoE block's SHARED expert as the trainer's MXFP8. OFF -> no-op.

    The routed experts are NOT touched here: they are the fused CuTe grouped
    MLP and are owned by RAON_MOE_EXPERT_MXFP8_REALIZED above.
    """
    if not shared_mxfp8_realized():
        return {}
    from vllm.distributed import get_tensor_model_parallel_world_size

    if get_tensor_model_parallel_world_size() != 1:
        raise RuntimeError(f"{_SHARED_ENV} requires tensor parallel size 1")

    shared = getattr(moe_module, "shared_experts", None)
    if shared is None:
        raise RuntimeError(
            f"{_SHARED_ENV} is set but this MoE block has no shared expert; "
            "the lever cannot be armed against nothing")
    wrapped = []
    for name, tag in (("gate_up_proj", "shared_fc1"),
                      ("down_proj", "shared_fc2")):
        sub = getattr(shared, name, None)
        _check_wrappable(_SHARED_ENV, "shared expert", name, sub)
        sub._raon_mxfp8_spec = (tag, _SHARED_ENV, layer_idx, None)
        sub.forward = types.MethodType(_mla_plain_forward, sub)
        wrapped.append((name, tag, sub.weight.shape[1], sub.weight.shape[0]))
    _record_scope(layer_idx, [t for _, t, _, _ in wrapped])
    _announce_dense(_SHARED_ENV, f"shared expert layer_idx={layer_idx}",
                    wrapped)
    return {"moe_shared_mxfp8": [f"{n}->{t}" for n, t, _, _ in wrapped]}


def _announce_dense(env, where, wrapped) -> None:
    """Say once, with the shapes and the pad targets, what left bf16.

    `wrapped` is (engine module name, trainer-side tag, K, N) per GEMM, built
    while installing -- the ops themselves do not exist yet (they are built on
    the first forward, after the checkpoint loads). What actually RAN is a
    separate statement, made by the per-tag LIVE lines in _band_check.
    """
    key = ("dense-announce", env)
    if key in _ANNOUNCED:
        return
    _ANNOUNCED.add(key)
    shapes = ", ".join(
        f"{name}->{tag}[K={k},N={n}]"
        f"{'+pad' + str(_pad_for(tag)) if _pad_for(tag) > FP8_M_MULTIPLE else ''}"
        for name, tag, k, n in wrapped)
    print(f"[parity] {env}=1: {where} WRAPPED for MXFP8 "
          f"(E4M3 data, E8M0 1x32 block scales, TE) -- {shapes}. "
          "The trainer keeps its pretraining "
          "quantization and the engine matches it. Which of these actually "
          "execute is stated by the LIVE lines, not by this one.",
          flush=True)

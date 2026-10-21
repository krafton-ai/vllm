# SPDX-License-Identifier: Apache-2.0
"""RAON_KDA_INKERNEL_FLA: fla-MAINLINE KDA prefill ops (vLLM-free).

Factored out of ``kda.py`` so the exact code the live engine runs can be
unit-tested against the isolation in-kernel gate
(``run/kda_inkernel_xvenv.py``) WITHOUT importing vLLM — the "live == isolation"
mapping check that guards against a silent-wrong reroute.

The RAON trainer reference dump takes
``megatron/core/ssm/kimi_delta_attention.py::_hf_fla_training_forward``, which
runs the fla-MAINLINE ops:
  conv = fla.modules.conv.causal_conv1d.causal_conv1d(backend='triton'),
  core = fla.ops.kda.chunk_kda with the IN-KERNEL contract (g=decay_raw raw,
         beta=beta_raw.float() raw, A_log, dt_bias.reshape(-1),
         scale=kda_query_scale, use_qk_l2norm_in_kernel=True,
         use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
         safe_gate=True, lower_bound=-5.0),
  o_norm = fla.modules.fused_norm_gate.rms_norm_gated(activation='sigmoid').
The stock engine runs the vLLM-VENDORED equivalents (a different build), whose
last bits differ by ~1 bf16 ULP and are amplified ~86x by o_norm into the
~6.25e-2 full-model L00.kda gap.  Rerouting onto the identical fla-mainline
objects + pinning autotune makes the KDA core AND the o_norm'd site bitwise-0 vs
the trainer.
"""
import inspect
import os

import torch

_FLA_AUTOTUNE_PINNED = False

# --------------------------------------------------------------------------
# RAON_KDA_DECODE_STATE_VFIRST -- the prefill->decode recurrent-state layout
# --------------------------------------------------------------------------
#
# fla's chunk_kda and vLLM's vendored chunk_kda return their `final_state` in
# OPPOSITE index orders, and for RAON the two orders are shape-identical:
#
#   fla/ops/common/chunk_delta_h.py:702-707   `state_v_first` (default False)
#       selects [N, HV, V, K] or [N, HV, K, V]; fla.ops.kda.chunk_kda's own
#       default is False, i.e. K-major.
#   vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py:352-355
#       hard-codes [N, H, V, K], i.e. V-major, with no flag.
#   vllm/models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py:534
#       the DECODE kernel addresses the cache as
#       `p_state += i_h*V*K + o_v[:,None]*K + o_k[None,:]` -- V-major.
#
# RAON's KDA has K == V == head_dim, so the two layouts are the same shape and
# `recurrent_state[state_indices] = last_recurrent_state` (raon_vllm/kda.py)
# silently stores a TRANSPOSED state whenever the prefill core is fla's.
# Nothing in a prefill-only probe can see this: the prefill OUTPUT is correct
# and the corrupted value is only ever read by the first decode forward.
#
# Measured cost of leaving it (engine-vs-engine, row 30, P=8, iter_0010000):
# the first decode token's L00.kda is 58% relative away from the same token's
# prefill value and the error decays over ~100 tokens as the gate forgets the
# state; stock vLLM's own decode-vs-prefill difference at that site is 0.5%.
#
# The lever is fla's OWN supported flag rather than a transpose of our own,
# because `initial_state` has to be interpreted in the same order the state was
# stored in and only the flag makes both ends agree.
#
# Default OFF. With RAON_KDA_INKERNEL_FLA=0 this module's core is not on the
# path at all, so OFF is byte-identical in every arm.
_STATE_VFIRST_ENV = "RAON_KDA_DECODE_STATE_VFIRST"
_STATE_VFIRST: bool | None = None


def _state_v_first_enabled() -> bool:
    """Read the lever and FAIL CLOSED if it cannot possibly take effect.

    A silently-inert parity lever is easy to mistake for a fix, so an armed
    lever that fla would ignore raises here rather than producing a run that
    looks fixed.
    """
    global _STATE_VFIRST
    if _STATE_VFIRST is not None:
        return _STATE_VFIRST
    if os.environ.get(_STATE_VFIRST_ENV, "0") != "1":
        _STATE_VFIRST = False
        return False
    from fla.ops.kda import chunk_kda as _fla_chunk_kda

    params = inspect.signature(_fla_chunk_kda).parameters
    if "state_v_first" not in params:
        raise RuntimeError(
            f"{_STATE_VFIRST_ENV}=1 but this fla build's chunk_kda has no "
            "`state_v_first` parameter, so the recurrent state would still be "
            "written K-major into a cache the vLLM decode kernel reads V-major "
            "(fused_recurrent.py:534). Refusing to run a lever that cannot "
            "take effect."
        )
    _STATE_VFIRST = True
    return True


def pin_all_fla_autotune(config_index: int = 0, verbose: bool = True) -> int:
    """Pin every live Triton ``Autotuner`` to ``configs[config_index]``.

    fla's KDA/conv/norm kernels are autotuned; with the default
    ``FLA_CACHE_MODE=disabled`` each process timing-selects a config, so two
    processes running byte-identical source can disagree by a few bf16 ULP.  The
    kernel is bitwise-deterministic once a config is fixed, and ``configs[0]`` is
    built in the same order in every venv running the same fla source, so
    truncating to it makes fla process- and batch-invariant and matches the
    pinned trainer.  Mirrors ``run/fla_pin.py``.  Idempotent.
    """
    global _FLA_AUTOTUNE_PINNED
    if _FLA_AUTOTUNE_PINNED:
        return 0
    import gc

    from triton.runtime.autotuner import Autotuner

    pinned = 0
    for obj in gc.get_objects():
        if isinstance(obj, Autotuner) and getattr(obj, "configs", None):
            if len(obj.configs) > 1:
                idx = config_index if config_index < len(obj.configs) else 0
                obj.configs = [obj.configs[idx]]
                if hasattr(obj, "cache"):
                    obj.cache = {}
                pinned += 1
    if pinned == 0:
        raise RuntimeError(
            "RAON_KDA_INKERNEL_FLA found no FLA autotuners to pin; import "
            "the KDA kernels before pinning or qualify this FLA build "
            "without the previous bitwise-parity guarantee"
        )
    _FLA_AUTOTUNE_PINNED = True
    if verbose:
        print(f"[raon-kda] fla in-kernel: pinned {pinned} autotuners to "
              f"configs[{config_index}]")
    return pinned


def fla_inkernel_core(
    q, k, v, g1, beta_raw, A_log, dt_bias, scale, lower_bound,
    initial_state=None, cu_seqlens=None, output_final_state=True,
):
    """fla-MAINLINE KDA core with the trainer IN-KERNEL contract.

    Inputs are POST-CONV, RDA-scaled activations exactly as
    ``_hf_fla_training_forward`` hands them to ``chunk_kda``:
      q, k, v : ``[1, T, H, D]`` (v already RDA value-scaled),
      g1      : ``[1, T, H, D]`` RAW decay (== trainer ``decay_raw``),
      beta_raw: ``[1, T, H]``    RAW beta logits (sigmoid folded in-kernel),
      A_log   : ``[H]``, dt_bias ``[H, D]`` (reshaped to ``[H*D]`` for fla),
      scale   : query scale (``kda_query_scale``; None -> fla uses 1/sqrt(K)).
    Returns ``(core [1, T, H, D], final_state)``.

    ``final_state``'s index order follows ``RAON_KDA_DECODE_STATE_VFIRST``: OFF
    (default) it is fla's K-major ``[N, H, K, V]``; ON it is the V-major
    ``[N, H, V, K]`` that vLLM's KV cache and its decode kernel use. See the
    module header for why that matters and what it costs to get it wrong.
    """
    from fla.ops.kda import chunk_kda as _fla_chunk_kda

    state_v_first = _state_v_first_enabled()
    if state_v_first and not _ANNOUNCED:
        _announce_state_v_first()

    # b2 front-end fusion lever (default OFF => this block is dead and the
    # call below is byte-identical to before).  Inference-only: with grad
    # enabled we fall through to the original path (fail-open == bit-safe).
    if os.environ.get("RAON_KDA_PREFILL_FUSED_FLOW", "0") == "1" \
            and not torch.is_grad_enabled():
        from .kda_frontend_fused import chunk_kda_frontend_fused
        global _FUSED_FLOW_ANNOUNCED
        if not _FUSED_FLOW_ANNOUNCED:
            print("[raon-kda] prefill front-end fusion active "
                  "(RAON_KDA_PREFILL_FUSED_FLOW=1): l2norm(q)+l2norm(k)+"
                  "beta-sigmoid+gate-cumsum in one launch")
            _FUSED_FLOW_ANNOUNCED = True
        return chunk_kda_frontend_fused(
            q, k, v, g1, beta_raw.float(), A_log, dt_bias.reshape(-1),
            scale, initial_state=initial_state,
            output_final_state=output_final_state,
            lower_bound=lower_bound, cu_seqlens=cu_seqlens,
            state_v_first=state_v_first,
        )

    out = _fla_chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g1,
        beta=beta_raw.float(),
        A_log=A_log,
        dt_bias=dt_bias.reshape(-1),
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=False,
        safe_gate=(lower_bound is not None),
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        **({"state_v_first": True} if state_v_first else {}),
    )
    core, final_state = out if isinstance(out, (tuple, list)) else (out, None)
    return core, final_state


_ANNOUNCED = False
_FUSED_FLOW_ANNOUNCED = False


def _announce_state_v_first() -> None:
    global _ANNOUNCED
    _ANNOUNCED = True
    print(
        f"[parity] {_STATE_VFIRST_ENV}=1: fla chunk_kda returns the KDA "
        "recurrent state V-major [N, H, V, K], the order vLLM's cache and "
        "fused_recurrent_kda_packed_decode (fused_recurrent.py:534) read -- "
        "without it the fla prefill stores a silently TRANSPOSED state (K==V "
        "for RAON) and every decode token continues from it",
        flush=True,
    )


def fla_gated_rmsnorm(x, g, weight, eps, activation="sigmoid"):
    """fla-mainline gated RMSNorm — the exact op the trainer uses for o_norm."""
    from fla.modules.fused_norm_gate import rms_norm_gated

    return rms_norm_gated(x, g, weight, None, activation=activation, eps=eps)


class FlaMainlineGatedRMSNorm(torch.nn.Module):
    """Drop-in for the vendored ``FusedRMSNormGated`` dispatching to the
    fla-mainline ``rms_norm_gated`` so the KDA *site* (o_norm(core, g2)) is
    trainer-bitwise, not just the core.  Reuses the replaced module's weight/eps.
    """

    def __init__(self, weight: torch.nn.Parameter, eps: float,
                 activation: str = "sigmoid") -> None:
        super().__init__()
        self.weight = weight
        self.eps = eps
        self.activation = activation

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return fla_gated_rmsnorm(x, g, self.weight, self.eps, self.activation)


# --------------------------------------------------------------------------
# RAON_KDA_CPR -- chunk-pinned recurrence (CPR), vLLM-free core helpers
# --------------------------------------------------------------------------
#
# CPR makes the ENGINE DECODE bitwise-identical to the TRAINER FORWARD by
# realizing BOTH as the same composition:
#
#   * an fp32 ENTRY CHAIN advanced by fla chunk_kda state passes at the fixed
#     absolute-position 64-token grid (never bf16-rounded between chunks;
#     measured split-invariant at 64-boundaries, B0/B0s bitwise);
#   * per-token outputs from the RECURRENT kernel, whose running state is
#     RESYNCED at every 64-boundary to fp32(bf16(entry)) -- the bf16 boundary
#     snapshot upcast, the donor's (gdn_cpr_state.py) rounding contract.
#
# The trainer twin is megatron/core/ssm/kimi_delta_attention.py::_kda_cpr_core
# (RAON_KDA_CPR).  Prefill here = state pass + ONE segmented
# fused_recurrent_kda scan (P1/P2 bitwise); decode = the vendored packed
# recurrent kernel + CPRSlotTracker boundary resync (P1e bitwise; the packed
# kernel's fp32 running-state 1-ulp tail vs the fla scan, P2e, is DISCARDED at
# every boundary, bounding it to <64 tokens).
#
# These helpers are vLLM-free so run/kda_cpr_integrated.py can drive the exact
# code objects the live engine runs (same pattern as fla_inkernel_core).

CPR_CHUNK = 64  # fla KDA chunk size == the trainer's chunk grid


def cpr_enabled() -> bool:
    """RAON_KDA_CPR lever, read per call."""
    return os.environ.get("RAON_KDA_CPR", "0") == "1"


def _cpr_contract(A_log, dt_bias, scale, lower_bound) -> dict:
    """The trainer's in-kernel contract, shared by both CPR kernel calls."""
    return dict(
        A_log=A_log,
        dt_bias=dt_bias.reshape(-1),
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=False,
        lower_bound=lower_bound,
    )


def cpr_snapshot(entry: torch.Tensor) -> torch.Tensor:
    """bf16 boundary snapshot upcast to fp32 -- THE CPR rounding contract."""
    return entry.to(torch.bfloat16).float()


# --------------------------------------------------------------------------
# RAON_KDA_CU_CACHE -- identity-stable cu_seqlens (+ host-side twin)
# --------------------------------------------------------------------------
#
# WHAT THIS REMOVES.  A profile of a 32x8192 prefill + 48 decode steps shows
# the 8K window is 98% PREFILL, and that the prefill issues 73,728 ``chunk_kda`` calls -- 18 KDA layers x 32 sequences x
# 128 chunks of 64 tokens, walked serially.  Per call, fla recomputes its own
# integer chunk metadata, and that recomputation -- not the KDA math -- is what
# the trace is full of:
#
#   294,912 device-to-device copies   468 ms  (pad/constant_pad_nd, stack/cat
#                                              under ChunkKDAFunction)
#   147,456 DtoH host syncs           446 ms  (repeat_interleave -> .item())
#   arange / cumsum / fill / int-arith kernels ~2.4 s
#
# The cause is that fla memoizes ``prepare_chunk_indices`` / ``prepare_lens`` /
# ``prepare_chunk_offsets`` with ``@tensor_cache``, which compares arguments by
# OBJECT IDENTITY (``a is b``; fla/utils/_decorators.py).  Every CPR seam below
# built a FRESH cu_seqlens tensor on every call, so the identity test could
# never match and the memo was a guaranteed miss.  Reusing one tensor object
# per (values, device) turns that miss into a hit.
#
# Separately, fla's ``chunk_kda`` accepts ``cu_seqlens_cpu``.  Its own
# ``_segmented_arange`` docstring states the cost of not passing it: "With CUDA
# counts, repeat_interleave reads counts.sum() on the host (one device sync).
# Pass host-side counts to avoid it."  raon_vllm never passed it.
#
# BIT NEUTRALITY.  Both changes sit outside the numeric path:
#
#   1. The cu_seqlens VALUES are unchanged -- the same int32 offsets, byte for
#      byte.  Only the tensor OBJECT is reused instead of reallocated.  A
#      tensor_cache HIT returns exactly the chunk_indices fla would otherwise
#      have recomputed from the same integers; that tensor is segment/offset
#      bookkeeping selecting WHICH ROWS a kernel reads, never a value a kernel
#      arithmetically consumes.
#   2. ``cu_seqlens_cpu`` is a host copy of those same integers, used only to
#      source ``counts.sum()`` without a device read.  The chunk_indices built
#      from it are identical.
#
# No kernel receives a different operand, shape, dtype or launch geometry, and
# no fla path writes into cu_seqlens (it is read-only metadata), so reuse
# cannot alias.  This is why the lever is deliberately NOT registered in
# parity_contract._FLAGS: that table is "the bit_relevant subset", and a
# bit-neutral lever listed there would assert an arithmetic effect it does not
# have -- and would force an edit to the mirrored trainer copy for no reason.
#
# DEFAULT OFF.  With RAON_KDA_CU_CACHE unset, every call site below evaluates
# the ORIGINAL construction expression verbatim and passes no cu_seqlens_cpu,
# so the arm is byte-identical to e5a7b8b26.
_CU_CACHE: dict = {}
_CU_CACHE_MAX = int(os.environ.get("RAON_KDA_CU_CACHE_MAX", "512"))
_CU_ANNOUNCED: set = set()


def cu_cache_enabled() -> bool:
    """RAON_KDA_CU_CACHE lever, read per call (default OFF)."""
    return os.environ.get("RAON_KDA_CU_CACHE", "0") == "1"


def _chunk_kda_takes_cu_cpu() -> bool:
    """Whether the installed fla ``chunk_kda`` accepts ``cu_seqlens_cpu``.

    Feature-detected, not assumed: an older fla still gets the identity reuse
    (which is the larger half), just not the host-side twin.
    """
    cached = getattr(_chunk_kda_takes_cu_cpu, "_v", None)
    if cached is not None:
        return cached
    try:
        from fla.ops.kda import chunk_kda

        ok = "cu_seqlens_cpu" in inspect.signature(chunk_kda).parameters
    except Exception:
        ok = False
    _chunk_kda_takes_cu_cpu._v = ok
    return ok


def cu_seqlens_cached(build, values, device):
    """``(cu_device, cu_cpu)`` for one chunk_kda call.

    ``build`` is a zero-arg callable holding the ORIGINAL construction
    expression; it is what runs when the lever is off, so the OFF path is the
    untouched code.  ``values`` are the same integers, used as the cache key
    and to materialize the reused pair when the lever is on.
    """
    if not cu_cache_enabled():
        return build(), None
    vals = tuple(int(v) for v in values)
    key = (vals, str(device))
    entry = _CU_CACHE.get(key)
    if entry is None:
        if len(_CU_CACHE) >= _CU_CACHE_MAX:
            # Bounded rather than trusted: an unexpected shape distribution
            # must degrade to today's behaviour, not grow without limit.
            if ("cap",) not in _CU_ANNOUNCED:
                _CU_ANNOUNCED.add(("cap",))
                print(
                    f"[parity] RAON_KDA_CU_CACHE INERT for new shapes: "
                    f"{len(_CU_CACHE)} distinct cu_seqlens already cached "
                    f"(cap {_CU_CACHE_MAX}); further shapes build fresh "
                    f"tensors exactly as with the lever off",
                    flush=True,
                )
            return build(), None
        # torch.tensor(vals, device=cuda) is an H2D copy through non-pinned
        # host staging, which fails during CUDA graph capture with "Cannot
        # copy ... unless pinned". Stage through pinned memory instead; values,
        # dtype and result are unchanged.
        cu_cpu = torch.tensor(vals, dtype=torch.int32)
        if str(device) != "cpu" and torch.cuda.is_available():
            cu_cpu = cu_cpu.pin_memory()
            cu = cu_cpu.to(device, non_blocking=True)
        else:
            cu = torch.tensor(vals, device=device, dtype=torch.int32)
        entry = (cu, cu_cpu)
        _CU_CACHE[key] = entry
        _warn_if_memo_too_small()
        # Emitted from inside the MISS branch, so the line is evidence that the
        # cache was populated by real traffic -- not a restatement of the env
        # var.  The hit itself is silent; the whole point is that it is free.
        if ("live",) not in _CU_ANNOUNCED:
            _CU_ANNOUNCED.add(("live",))
            print(
                f"[parity] RAON_KDA_CU_CACHE LIVE: caching cu_seqlens by "
                f"object identity so fla's @tensor_cache can hit "
                f"(first key len={len(vals)} on {device}; "
                f"cu_seqlens_cpu supported={_chunk_kda_takes_cu_cpu()})",
                flush=True,
            )
    return entry


def _fla_memo_size() -> int:
    """fla's @tensor_cache depth (FLA_TENSOR_CACHE_SIZE, default 4)."""
    try:
        import fla.utils as _fu

        return int(getattr(_fu, "FLA_TENSOR_CACHE_SIZE", 4))
    except Exception:
        return 4


def _warn_if_memo_too_small() -> None:
    """Say ONCE when more distinct cu_seqlens are live than fla's memo holds.

    fla's @tensor_cache is a deque(maxlen=FLA_TENSOR_CACHE_SIZE), default 4,
    and it is scanned linearly for an identity match. Pinning the tensors is
    necessary but NOT sufficient: if more than `maxlen` distinct cu_seqlens are
    interleaved, every lookup evicts the entry the next lookup wants and the
    hit rate collapses to zero -- measured on CPU at 4 shapes 100% hits, at 5
    shapes 0%. The prefill is immune (it repeats ONE shape, [0, 64], for all
    73,728 calls), but a decode batch whose width shrinks as sequences finish
    can cross the bound and silently get nothing from this lever.
    FLA_TENSOR_CACHE_SIZE is read at DECORATION time, so it must be exported
    BEFORE fla is imported -- it belongs in the arm env, not here.
    Raising it is bit-neutral for the same reason the pinning is: a deeper memo
    returns the same chunk_indices, it just stops throwing them away.
    """
    if ("memo",) in _CU_ANNOUNCED:
        return
    depth = _fla_memo_size()
    if len(_CU_CACHE) <= depth:
        return
    _CU_ANNOUNCED.add(("memo",))
    print(
        f"[parity] RAON_KDA_CU_CACHE DEGRADED: {len(_CU_CACHE)} distinct "
        f"cu_seqlens are live but fla's @tensor_cache holds only {depth} "
        f"(FLA_TENSOR_CACHE_SIZE). Past that depth the memo thrashes and this "
        f"lever buys nothing. Export FLA_TENSOR_CACHE_SIZE>={len(_CU_CACHE)} "
        f"in the arm env, BEFORE fla is imported.",
        flush=True,
    )


def cu_cpu_kwargs(cu_cpu) -> dict:
    """``{'cu_seqlens_cpu': ...}`` only when the lever is on AND fla takes it."""
    if cu_cpu is None or not _chunk_kda_takes_cu_cpu():
        return {}
    return {"cu_seqlens_cpu": cu_cpu}


def cu_cache_stats() -> dict:
    """For the CPU gate / harness: how many distinct cu_seqlens are pinned."""
    return {
        "enabled": cu_cache_enabled(),
        "entries": len(_CU_CACHE),
        "cap": _CU_CACHE_MAX,
        "chunk_kda_takes_cu_cpu": _chunk_kda_takes_cu_cpu(),
        "fla_memo_size": _fla_memo_size(),
        "memo_can_hold_all": len(_CU_CACHE) <= _fla_memo_size(),
    }


def cpr_state_pass(
    q, k, v, g1, beta_raw, A_log, dt_bias, scale, lower_bound, entry
):
    """Advance the fp32 entry chain over one chunk of <= 64 REAL tokens.

    q/k/v/g1: [1, L, H, D] post-conv, RDA-scaled, PRE-l2norm; beta_raw [1, L, H]
    raw logits; entry fp32 [1, H, V, K] (V-major, the layout the vLLM cache and
    packed decode kernel read).  One fla-mainline chunk_kda call under the
    in-kernel contract, state_v_first=True.  Returns the fp32 final state,
    CLONED to private storage (fla reuses its output workspace).
    """
    from fla.ops.kda import chunk_kda

    L = q.shape[1]
    # cu_seqlens site 1 of 3 -- the hot one: 73,728 calls in a 32x8192
    # prefill (18 KDA layers x 32 seqs x 128 chunks).  Lever OFF evaluates the
    # original expression verbatim; see RAON_KDA_CU_CACHE above.
    cu, cu_cpu = cu_seqlens_cached(
        lambda: torch.tensor([0, L], device=q.device, dtype=torch.int32),
        (0, L),
        q.device,
    )
    _, fs = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g1,
        beta=beta_raw.float(),
        initial_state=entry,
        output_final_state=True,
        safe_gate=(lower_bound is not None),
        cu_seqlens=cu,
        state_v_first=True,
        **cu_cpu_kwargs(cu_cpu),
        **_cpr_contract(A_log, dt_bias, scale, lower_bound),
    )
    return fs.clone()


def fla_cpr_prefill_core(*args, **kwargs):
    """Batch independent CPR entry chains without changing sequence order."""
    import os
    if (_FLA_AUTOTUNE_PINNED
            and os.environ.get("RAON_KDA_AUTOTUNE_CONFIG_INDEX", "0") == "0"
            and os.environ.get("RAON_KDA_CPR_PREFILL_BATCH", "1") == "1"):
        from .kda_prefill_batch import batched
        return batched(*args, **kwargs)
    return _fla_cpr_prefill_core_serial(*args, **kwargs)


def _fla_cpr_prefill_core_serial(
    q, k, v, g1, beta_raw, A_log, dt_bias, scale, lower_bound,
    entries0=None, cu_seqlens=None, chunk=CPR_CHUNK,
):
    """CPR prefill: chained state pass + ONE segmented recurrent scan.

    Inputs exactly as ``fla_inkernel_core`` receives them (post-conv,
    RDA-scaled, pre-l2norm; raw g/beta).  ``entries0``: optional list of fp32
    [1, H, V, K] entry states, one per sequence (position 0 of each sequence
    must sit ON the 64 grid of its absolute chunking -- the caller guarantees
    this by open-chunk replay).  Returns ``(core [1, T, H, V], seq_info)``
    where seq_info[i] is a dict:
      entry     : fp32 [1, H, V, K] entry at the sequence's LAST committed
                  64-boundary (the open tail is NOT folded in),
      running   : fp32 [1, H, V, K] recurrent running state after the last
                  token -- what the decode continues from.  When the sequence
                  ends exactly ON a boundary this is already the resynced
                  fp32(bf16(entry)) snapshot,
      tail_start: index (into the packed token axis) of the open-chunk tail,
      tail_len  : number of open (uncommitted) tail tokens (0..63).
    """
    from fla.ops.kda import fused_recurrent_kda

    device = q.device
    T, H, K = q.shape[1], q.shape[2], q.shape[3]
    V = v.shape[-1]
    if cu_seqlens is None:
        bounds = [0, T]
    else:
        bounds = [int(x) for x in cu_seqlens.tolist()]

    pool_parts, cu_chunked = [], [0]
    seq_info = []
    seg_of_seq_last = []
    for si in range(len(bounds) - 1):
        s, e = bounds[si], bounds[si + 1]
        if entries0 is not None and entries0[si] is not None:
            entry = entries0[si].float()
        else:
            entry = torch.zeros(1, H, V, K, device=device, dtype=torch.float32)
        tail_start, tail_len = e, 0
        pos = s
        while pos < e:
            nxt = min(pos + chunk, e)
            pool_parts.append(cpr_snapshot(entry))
            if nxt - pos == chunk:
                entry = cpr_state_pass(
                    q[:, pos:nxt], k[:, pos:nxt], v[:, pos:nxt],
                    g1[:, pos:nxt], beta_raw[:, pos:nxt],
                    A_log, dt_bias, scale, lower_bound, entry,
                )
            else:
                tail_start, tail_len = pos, nxt - pos
            cu_chunked.append(nxt)
            pos = nxt
        seg_of_seq_last.append(len(cu_chunked) - 2)
        seq_info.append(
            dict(entry=entry, running=None, tail_start=tail_start, tail_len=tail_len)
        )

    core, seg_finals = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g1,
        beta=beta_raw.float(),
        initial_state=torch.cat(pool_parts, dim=0),
        output_final_state=True,
        state_v_first=True,
        cu_seqlens=torch.tensor(cu_chunked, device=device, dtype=torch.int32),
        **_cpr_contract(A_log, dt_bias, scale, lower_bound),
    )
    for si, info in enumerate(seq_info):
        if info["tail_len"] == 0:
            # Ended ON a boundary: the resync applies before the next token.
            info["running"] = cpr_snapshot(info["entry"])
        else:
            j = seg_of_seq_last[si]
            info["running"] = seg_finals[j : j + 1].clone()
    return core, seq_info


def fla_cpr_decode_step(
    q, k, v, g1, beta_raw, A_log, dt_bias, scale, lower_bound, initial_states
):
    """CPR exact-step decode: one-token segments on the SAME fla
    ``fused_recurrent_kda`` kernel that produces the CPR prefill scan bits.

    The vendored ``fused_recurrent_kda_packed_decode`` drifts its fp32
    running state ~1 ulp per step against the fla scan within each open
    chunk, flipping sparse bf16 core-output elements where a drifted value
    lands on a rounding boundary.  The chained one-token
    ``fused_recurrent_kda`` (state_v_first=True, fp32 state fed back) is
    bitwise equal to the one-call scan, so the decode live step joins the
    prefill kernel family and the within-chunk KDA decode-vs-prefill residual
    closes by construction (conv and handoff are already bitwise).

    q/k/v/g1: [1, n, H, D] (post-conv, RDA-scaled, PRE-l2norm), beta_raw
    [1, n, H] raw logits, initial_states fp32 [n, H, V, K] V-major (one
    gathered cache slot per decode token; each token is its own one-token
    segment — at batch 1 this is EXACTLY the 193329-proven call).  Returns
    ``(core [1, n, H, V], finals fp32 [n, H, V, K])``; the caller scatters
    finals back into the recurrent cache.
    """
    from fla.ops.kda import fused_recurrent_kda

    n = q.shape[1]
    cu = torch.arange(0, n + 1, device=q.device, dtype=torch.int32)
    core, finals = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g1,
        beta=beta_raw.float(),
        initial_state=initial_states,
        output_final_state=True,
        state_v_first=True,
        cu_seqlens=cu,
        **_cpr_contract(A_log, dt_bias, scale, lower_bound),
    )
    # Private storage (fla reuses its output workspace): the boundary
    # state pass may launch fla kernels before the caller consumes these.
    return core.clone(), finals.clone()


class CPRSlotTracker:
    """Per-slot CPR decode bookkeeping: fp32 entry chain + open-chunk buffer.

    The MATH lives in cpr_state_pass / cpr_snapshot; this class only buffers
    the open (< 64 token) chunk per KV-cache slot and fires the boundary
    update.  Same slot-keyed style as the chunk-replay ``_cr_open`` machinery.
    """

    def __init__(self, chunk: int = CPR_CHUNK) -> None:
        self.chunk = chunk
        self.entry: dict[int, torch.Tensor] = {}   # fp32 [1, H, V, K]
        self.open: dict[int, dict] = {}            # lists of [H, D] / [H] rows
        # [15] RAON_KDA_CPR_RING=1: keep each slot's open chunk in
        # PREALLOCATED [chunk, ...] ring tensors written by row copy_ instead
        # of python lists of per-token clones -- no per-append allocation and
        # no 64-way torch.stack at the boundary (the append path is host
        # serialization on the kda 12.8% forward share).  Default OFF is the
        # list path above, byte-identical; the ring feeds cpr_state_pass the
        # same contiguous [1, chunk, H, D] values, so the MATH is unchanged.
        self.ring = os.environ.get("RAON_KDA_CPR_RING", "0") == "1"

    def _empty(self) -> dict:
        return {"q": [], "k": [], "v": [], "g": [], "beta": []}

    def _ring_empty(self, q_t, k_t, v_t, g_t, beta_t) -> dict:
        """Fresh ring storage sized off one row of each field.

        FRESH at every boundary rather than rewound in place: the boundary
        state pass may still be reading the old rows asynchronously, and a
        fresh tensor once per 64
        appends is not the per-append cost this lever removes.
        """
        c = self.chunk

        def alloc(row):
            return torch.empty((c, *row.shape), dtype=row.dtype,
                               device=row.device)

        return {"q": alloc(q_t), "k": alloc(k_t), "v": alloc(v_t),
                "g": alloc(g_t), "beta": alloc(beta_t), "len": 0}

    def drop(self, slot: int) -> None:
        self.entry.pop(slot, None)
        self.open.pop(slot, None)

    def reset(self, slot: int, entry: torch.Tensor,
              q_tail=None, k_tail=None, v_tail=None, g_tail=None, beta_tail=None):
        """(Re)initialize a slot after prefill: committed entry + open tail.

        Tail tensors are [L, H, D] ([L, H] for beta) with L in 0..63.
        """
        self.entry[slot] = entry.clone()
        if self.ring:
            if q_tail is not None and q_tail.shape[0] > 0:
                rbuf = self._ring_empty(q_tail[0], k_tail[0], v_tail[0],
                                        g_tail[0], beta_tail[0])
                L = q_tail.shape[0]
                rbuf["q"][:L].copy_(q_tail)
                rbuf["k"][:L].copy_(k_tail)
                rbuf["v"][:L].copy_(v_tail)
                rbuf["g"][:L].copy_(g_tail)
                rbuf["beta"][:L].copy_(beta_tail)
                rbuf["len"] = L
                self.open[slot] = rbuf
            else:
                # No tail: drop any stale ring; append allocates lazily (row
                # shapes are only known once a row arrives).
                self.open.pop(slot, None)
            return
        buf = self._empty()
        if q_tail is not None and q_tail.shape[0] > 0:
            for t in range(q_tail.shape[0]):
                buf["q"].append(q_tail[t].clone())
                buf["k"].append(k_tail[t].clone())
                buf["v"].append(v_tail[t].clone())
                buf["g"].append(g_tail[t].clone())
                buf["beta"].append(beta_tail[t].clone())
        self.open[slot] = buf

    def append(self, slot: int, q_t, k_t, v_t, g_t, beta_t,
               A_log, dt_bias, scale, lower_bound, *, _owned=False):
        """Append one decoded token (post-conv, RDA-scaled, PRE-l2norm rows:
        q/k/v/g [H, D], beta [H]).

        At the 64th token of the open chunk: advance the fp32 entry via the
        chunk state pass over the REAL buffered tokens and return the RESYNC
        state fp32 [1, H, V, K] (bf16 snapshot upcast) for the recurrent
        cache; otherwise return None (the packed kernel's running state
        stands until the boundary).
        """
        if self.ring:
            # copy_ into the preallocated ring never retains the caller's
            # tensor, so `_owned` is moot on this path (owned or view, the
            # row's bytes land in ring-private storage either way).
            rbuf = self.open.get(slot)
            if rbuf is None:
                rbuf = self._ring_empty(q_t, k_t, v_t, g_t, beta_t)
                self.open[slot] = rbuf
            i = rbuf["len"]
            rbuf["q"][i].copy_(q_t)
            rbuf["k"][i].copy_(k_t)
            rbuf["v"][i].copy_(v_t)
            rbuf["g"][i].copy_(g_t)
            rbuf["beta"][i].copy_(beta_t)
            rbuf["len"] = i + 1
            if rbuf["len"] < self.chunk:
                return None
            H, D = q_t.shape
            entry = self.entry.get(slot)
            if entry is None:
                entry = torch.zeros(
                    1, H, D, D, device=q_t.device, dtype=torch.float32
                )
            entry = cpr_state_pass(
                rbuf["q"].unsqueeze(0),
                rbuf["k"].unsqueeze(0),
                rbuf["v"].unsqueeze(0),
                rbuf["g"].unsqueeze(0),
                rbuf["beta"].unsqueeze(0),
                A_log, dt_bias, scale, lower_bound, entry,
            )
            self.entry[slot] = entry
            self.open[slot] = self._ring_empty(q_t, k_t, v_t, g_t, beta_t)
            return cpr_snapshot(entry)
        # _owned callers transfer immutable private storage, never live views.
        buf = self.open.setdefault(slot, self._empty())
        buf["q"].append(q_t if _owned else q_t.clone())
        buf["k"].append(k_t if _owned else k_t.clone())
        buf["v"].append(v_t if _owned else v_t.clone())
        buf["g"].append(g_t if _owned else g_t.clone())
        buf["beta"].append(beta_t if _owned else beta_t.clone())
        if len(buf["q"]) < self.chunk:
            return None
        H, D = q_t.shape
        entry = self.entry.get(slot)
        if entry is None:
            entry = torch.zeros(
                1, H, D, D, device=q_t.device, dtype=torch.float32
            )
        entry = cpr_state_pass(
            torch.stack(buf["q"]).unsqueeze(0),
            torch.stack(buf["k"]).unsqueeze(0),
            torch.stack(buf["v"]).unsqueeze(0),
            torch.stack(buf["g"]).unsqueeze(0),
            torch.stack(buf["beta"]).unsqueeze(0),
            A_log, dt_bias, scale, lower_bound, entry,
        )
        self.entry[slot] = entry
        self.open[slot] = self._empty()
        return cpr_snapshot(entry)

    def open_len(self, slot: int) -> int:
        buf = self.open.get(slot)
        if self.ring:
            return buf["len"] if buf else 0
        return len(buf["q"]) if buf else 0

    def open_stacked(self, slot: int):
        """The open tail as ([1, L, H, D] x4, [1, L, H]) or None if empty."""
        buf = self.open.get(slot)
        if self.ring:
            if not buf or buf["len"] == 0:
                return None
            L = buf["len"]
            # .clone(): match the stack path's private storage -- callers
            # hold these across later appends into the (mutable) ring.
            return (
                buf["q"][:L].clone().unsqueeze(0),
                buf["k"][:L].clone().unsqueeze(0),
                buf["v"][:L].clone().unsqueeze(0),
                buf["g"][:L].clone().unsqueeze(0),
                buf["beta"][:L].clone().unsqueeze(0),
            )
        if not buf or not buf["q"]:
            return None
        return (
            torch.stack(buf["q"]).unsqueeze(0),
            torch.stack(buf["k"]).unsqueeze(0),
            torch.stack(buf["v"]).unsqueeze(0),
            torch.stack(buf["g"]).unsqueeze(0),
            torch.stack(buf["beta"]).unsqueeze(0),
        )

    def flush_state(self, slot: int, A_log, dt_bias, scale, lower_bound):
        """fp32 entry advanced over the open tail (NOT committed): equals the
        plain whole-sequence chunk_kda fp32 final state bitwise (B0s).  For
        state handoff / harness comparison only."""
        entry = self.entry.get(slot)
        stacked = self.open_stacked(slot)
        if stacked is None:
            return None if entry is None else entry.clone()
        qc, kc, vc, gc, bc = stacked
        if entry is None:
            H, D = qc.shape[2], qc.shape[3]
            entry = torch.zeros(1, H, D, D, device=qc.device, dtype=torch.float32)
        return cpr_state_pass(
            qc, kc, vc, gc, bc, A_log, dt_bias, scale, lower_bound, entry
        )

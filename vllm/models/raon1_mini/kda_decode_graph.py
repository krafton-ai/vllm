"""Capture pure single-request KDA GPU work; keep CPR bookkeeping live."""

import os
import threading
import torch
from einops import rearrange
from . import kda
from .kda_inkernel import fla_cpr_decode_step


# RAON_KDA_DECODE_GRAPH_MAX: the largest decode token count this may capture
# for. gpu() is batch-agnostic (`rearrange(t, "n (h d) -> 1 n h d")`,
# `rec[indices]`, `out[0, : x.shape[0]]`), so the bound is the only limit.
# Defaults to 1 (single-token decode only).
_MAX_TOKENS = int(os.environ.get("RAON_KDA_DECODE_GRAPH_MAX", "1"))
# One graph is captured per distinct token count. Decode batch sizes shrink as
# sequences finish, so the set is small in practice, but it is bounded rather
# than trusted: past the cap this returns False and the step runs eager.
_MAX_GRAPHS = int(os.environ.get("RAON_KDA_DECODE_GRAPH_SHAPES", "8"))

# RAON_KDA_DECODE_GRAPH_SAFE: allow this path under vLLM's breakable
# cudagraph by treating capture and replay separately (without it,
# _breakable_cudagraph_in_use below declines both).
#
#   CAPTURE is the illegal half.  ``torch.cuda.graph(g).__enter__`` runs a
#   device-wide ``torch.cuda.synchronize()``, a ``gc.collect()`` and an
#   ``empty_cache()``, then begins a stream capture with its own private pool.
#   Doing that from inside a ``BreakableCUDAGraphCapture`` session -- in the
#   eager-break window of the outer CAPTURE (breakable_cudagraph.py:195-208,
#   ``add_eager``: ``_end_segment(); fn(); _begin_segment()``) or inside the
#   outer REPLAY (:212-214, where the recorded eager callable re-runs) --
#   corrupts allocator/stream state and surfaces as an illegal access at the
#   next synchronize.
#
#   REPLAY is legal in BOTH windows.  ``graph.replay()`` is a cudaGraphLaunch:
#   it allocates nothing, captures nothing and synchronizes nothing; it
#   enqueues a fixed node set on the current stream.  During ``add_eager`` the
#   outer capture is already STOPPED (``_end_segment()`` ran before ``fn()``),
#   and during the outer replay no stream is capturing at all.
#
# So SAFE mode captures ONLY where a top-level capture is legal and replays
# everywhere.  The window that makes that useful already exists in vLLM:
# cudagraph_utils.py:333-336 runs ``forward_fn(CUDAGraphMode.NONE)`` as a
# WARMUP for every descriptor before capturing it, and at mode NONE
# BreakableCUDAGraphWrapper.__call__ (:322-323) returns the raw runnable, so no
# session is entered.  The KDA layers see a real decode-shaped batch there
# (_dummy_run builds ``num_reqs = min(num_tokens, max_num_seqs)`` one-token
# requests, model_runner.py:653, so split_decodes_and_prefills reports
# num_prefills=0, kda_metadata.py:276) with the KV caches already allocated --
# i.e. exactly the shape the served decode steps will present.  The graph
# captured in that window is the graph every later decode step replays.
#
# Default OFF: with SAFE=0 every capture under breakable graphs is declined.
_SAFE = os.environ.get("RAON_KDA_DECODE_GRAPH_SAFE", "0") == "1"

# Per-thread depth of a vLLM breakable REPLAY.  No existing predicate sees
# this window: ``BreakableCUDAGraphCapture.current()`` is None during a replay
# (``__exit__`` clears the tls slot, :167-171) and no stream is capturing, so
# ``is_current_stream_capturing()`` does not detect it.  This module
# instruments the window instead.
_TLS = threading.local()
_REPLAY_GUARD: "bool | None" = None
# Hot-path announce latches.  Membership tests against _ANNOUNCED build a tuple
# per call; these are read on every replay, so they are plain booleans.
_SAFE_REPLAY_ANNOUNCED = [False]
_SAFE_NOCAPTURE_ANNOUNCED = [False]
# Resolved once.  The replay branch consults this module while its announce is
# still pending, and that branch is the decode hot path.
_BCG = [False, None]


def _breakable_module():
    if not _BCG[0]:
        _BCG[0] = True
        try:
            from vllm.compilation import breakable_cudagraph as _bcg
            _BCG[1] = _bcg
        except Exception:
            _BCG[1] = None
    return _BCG[1]


def _install_replay_guard() -> bool:
    """Make vLLM's breakable REPLAY window observable from inside the callback.

    Wraps ``BreakableCUDAGraphCapture.replay`` so a thread-local depth counter
    is raised for exactly as long as the recorded segments -- including the
    eager callables that land back in ``kda.py::_forward`` -- are running.
    Idempotent per class, and the wrapper only counts: the call it forwards is
    the original bound method, so replay semantics are untouched.

    Fails closed.  If the symbol is missing (vLLM drift), this returns False
    and SAFE mode then declines every capture: slow, never corrupt.
    """
    global _REPLAY_GUARD
    if _REPLAY_GUARD is not None:
        return _REPLAY_GUARD
    bcg = _breakable_module()
    cls = getattr(bcg, "BreakableCUDAGraphCapture", None) if bcg else None
    if (cls is None
            or not callable(getattr(cls, "replay", None))
            or not callable(getattr(cls, "current", None))):
        _REPLAY_GUARD = False
        print("[parity] RAON_KDA_DECODE_GRAPH_SAFE INERT: could not instrument "
              "vllm.compilation.breakable_cudagraph.BreakableCUDAGraphCapture "
              "(replay/current missing), so the breakable replay window is not "
              "observable from here and every capture is declined -- the KDA "
              "decode step runs EAGER, exactly as it does without this lever",
              flush=True)
        return False
    if not getattr(cls, "_raon_replay_depth_guard", False):
        _original_replay = cls.replay

        def _guarded_replay(self):
            depth = getattr(_TLS, "breakable_replay", 0)
            _TLS.breakable_replay = depth + 1
            try:
                return _original_replay(self)
            finally:
                _TLS.breakable_replay = depth

        cls.replay = _guarded_replay
        cls._raon_replay_depth_guard = True
    _REPLAY_GUARD = True
    return True


def _capture_permitted():
    """(permitted, reason).  A capture is legal only at the TOP level.

    Every clause below names a window in which ``torch.cuda.graph()`` would
    nest inside somebody else's graph machinery.
    """
    if torch.cuda.is_current_stream_capturing():
        return False, "the current stream is itself capturing a CUDA graph"
    if not _breakable_cudagraph_in_use():
        # No breakable session can exist, so the capture is top-level by
        # construction.
        return True, "VLLM_USE_BREAKABLE_CUDAGRAPH is off, so this is top-level"
    if not _install_replay_guard():
        return False, ("vLLM's BreakableCUDAGraphCapture could not be "
                       "instrumented, so the replay window is unobservable")
    cls = _breakable_module().BreakableCUDAGraphCapture
    if cls.current() is not None:
        return False, ("a breakable capture session is open on this thread, so "
                       "this call is an eager break inside it")
    if getattr(_TLS, "breakable_replay", 0):
        return False, ("this call is a recorded eager segment running inside a "
                       "breakable graph replay")
    return True, ("no breakable capture session is open and no breakable replay "
                  "is in flight, so this is a top-level capture")

# One line per distinct verdict per process. Each announce is printed from
# inside the branch that did the work (a capture, a replay, or the decline)
# and carries its token count next to the bound in effect.
_ANNOUNCED: set = set()


def _breakable_cudagraph_in_use() -> bool:
    """True when vLLM's breakable CUDA-graph machinery is driving the model.

    This path captures its own ``torch.cuda.CUDAGraph`` around the KDA GPU
    work (``with torch.cuda.graph(graph):`` in try_forward, which also
    forces a ``torch.cuda.synchronize()``).  That is safe ONLY as a top-level
    capture.  When ``VLLM_USE_BREAKABLE_CUDAGRAPH=1`` the engine runs the whole
    model forward inside a ``BreakableCUDAGraphCapture`` and invokes each KDA
    ``_forward`` as an eager-break callback (``@eager_break_during_capture`` on
    kda.py::_forward).  Taking a fresh graph capture from inside that outer
    session -- during the eager-break window of the outer CAPTURE, or during
    the outer REPLAY where the recorded eager callback re-runs -- nests a
    capture+synchronize on the same stream with a different graph pool and
    corrupts the allocator/stream state.  The corruption surfaces as an illegal
    memory access at the next ``torch.cuda.synchronize()`` (the one inside the
    next ``torch.cuda.graph().__enter__``).

    ``torch.cuda.is_current_stream_capturing()`` (checked in try_forward and
    _announce_skipped) does NOT catch this: no stream is actively capturing
    during the outer eager break or during the outer replay, so that guard
    passes and the nested capture proceeds.

    The failure is triggered by the nested capture under breakable graphs,
    not by context length.

    Declining here is numerically neutral: try_forward's ``gpu()`` is a capture
    of the SAME ops the eager CPR decode path runs (causal_conv1d_update ->
    RDA scale -> fla_cpr_decode_step -> cache scatter -> o_norm), and the
    _cpr_decode_track boundary bookkeeping is identical on both paths.  When
    this returns True the step simply runs eager.
    """
    global _BREAKABLE
    if _BREAKABLE is None:
        try:
            import vllm.envs as _envs
            _BREAKABLE = bool(_envs.VLLM_USE_BREAKABLE_CUDAGRAPH)
        except Exception:
            _BREAKABLE = False
    return _BREAKABLE


_BREAKABLE: bool | None = None


def _announce_skipped(self, m) -> None:
    """Say ONCE that the token count alone kept the graph path unreached.

    Only fires when ``num_actual_tokens`` is over the bound, which is the one
    decline this path owns; the other ``eligible`` conditions are listed
    alongside so the line is a diagnosis and not a guess.
    """
    # Cheapest discriminator first: try_forward declines on EVERY eager decode
    # step of every KDA layer, so the steady state of this helper must be one
    # attribute read and one integer compare, not a CUDA call.
    if ("skipped",) in _ANNOUNCED or m is None:
        return
    # The breakable-cudagraph decline applies to every decode step (all n), so
    # it cannot be inferred from the token count; announce it once.  This is
    # the default configuration: the KDA decode graph is off under breakable
    # graphs and the step runs eager.
    if _breakable_cudagraph_in_use() and not _SAFE:
        _ANNOUNCED.add(("skipped",))
        print(f"[parity] RAON_KDA_DECODE_GRAPH_MAX SKIPPED: "
              f"VLLM_USE_BREAKABLE_CUDAGRAPH is on, so eligible() declined and "
              f"every KDA decode step runs EAGER -- a nested torch.cuda.graph "
              f"capture inside vLLM's breakable cudagraph takes an illegal "
              f"memory access; the bound in "
              f"effect ({_MAX_TOKENS}) is moot here", flush=True)
        return
    n = getattr(m, "num_actual_tokens", None)
    if not isinstance(n, int) or n <= _MAX_TOKENS:
        return                  # some OTHER condition declined -- not this one
    if torch.cuda.is_current_stream_capturing():
        return
    others = []
    if getattr(m, "num_prefills", 0):
        others.append("num_prefills>0")
    if getattr(m, "num_spec_decodes", 0):
        others.append("num_spec_decodes>0")
    if not getattr(self, "kda_cpr", False):
        others.append("kda_cpr off")
    if getattr(self, "kda_chunk_replay", False):
        others.append("kda_chunk_replay on")
    if getattr(self, "kda_decode_backend", None) == "flashkda":
        others.append("kda_decode_backend=flashkda")
    _ANNOUNCED.add(("skipped",))
    print(f"[parity] RAON_KDA_DECODE_GRAPH_MAX SKIPPED: a decode step with "
          f"n={n} tokens exceeds the bound in effect ({_MAX_TOKENS}), so "
          f"eligible() declined and the KDA decode step ran EAGER -- no graph "
          f"was captured or replayed for this shape (other eligible() "
          f"conditions failing: {', '.join(others) if others else 'none'})",
          flush=True)


def eligible(self, m):
    return (
        m is not None
        and m.num_prefills == 0
        and m.num_spec_decodes == 0
        and 1 <= m.num_actual_tokens <= _MAX_TOKENS
        and self.kda_cpr
        and not self.kda_chunk_replay
        and self.kda_decode_backend != "flashkda"
        # A nested torch.cuda.graph capture inside vLLM's breakable cudagraph
        # takes an illegal memory access (see _breakable_cudagraph_in_use).
        # Decline and run the eager CPR decode instead -- numerically the same.
        #
        # SAFE mode moves this decision out of `eligible` and down to the
        # capture branch of try_forward, because the two halves are not equally
        # illegal: the CAPTURE is, the REPLAY is not.  Declining here would
        # forfeit the replay as well.
        and (_SAFE or not _breakable_cudagraph_in_use())
    )


def gpu(self, x, g1, g2, beta, ratio, indices, out_shape):
    conv, rec = self.kv_cache
    if not kda.is_conv_state_dim_first():
        conv = conv.transpose(-1, -2)
    weight = self.conv1d.weight.view(
        self.conv1d.weight.size(0), self.conv1d.weight.size(2)
    )
    packed = torch.empty_like(x)
    packed = kda.causal_conv1d_update(
        x,
        conv,
        weight,
        self.conv1d.bias,
        activation="silu",
        conv_state_indices=indices,
        validate_data=True,
        out=packed,
    )
    self._scale_packed_v(packed, ratio, beta)
    q, k, v = (
        rearrange(t, "n (h d) -> 1 n h d", d=self.head_dim)
        for t in packed.split(self.local_projection_size, dim=-1)
    )
    core, final = fla_cpr_decode_step(
        q,
        k,
        v,
        g1,
        beta,
        self.A_log,
        self.dt_bias,
        self._inkernel_scale,
        self.gate_lower_bound,
        rec[indices],
    )
    rec[indices] = final
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    out[0, : x.shape[0]] = core[0, : x.shape[0]]
    out.copy_(self.o_norm(out, g2))
    return packed, out


def try_forward(self, x, g1, g2, beta, ratio, out, m):
    if not eligible(self, m) or torch.cuda.is_current_stream_capturing():
        _announce_skipped(self, m)
        return False
    n = m.num_actual_tokens
    idx = m.non_spec_state_indices_tensor[:n]
    x = x[:n]
    g1 = g1[:, :n]
    beta = beta[:, :n]
    ratio = ratio[:, :n]
    if _SAFE:
        # The stream is deliberately NOT part of the key here.  Under SAFE the
        # capture is taken in vLLM's pre-capture warmup forward, and that whole
        # phase runs inside `graph_capture(device=...)` (cudagraph_utils.py:321)
        # on a side stream, while every served decode step runs on the default
        # stream.  Keying on the stream would file the pre-captured graph under
        # an id no serving step ever presents, so the graph would never be
        # replayed.  Dropping it is sound: `graph.replay()` is
        # a cudaGraphLaunch onto the CURRENT stream and the recorded node set
        # carries its own internal dependencies, so the stream a graph was
        # recorded on does not change what it computes.
        key = (
            tuple(x.shape),
            tuple(g2.shape),
            tuple(out.shape),
            self.kv_cache[0].data_ptr(),
            self.kv_cache[1].data_ptr(),
        )
        # Replay-hotpath statics: no per-step `p._version` walk over the
        # parameters.  Invalidation is owned by the refit path: load_weights
        # in model.py deletes `_raon_kda_decode_graph` and
        # `_raon_kda_decode_graph_stamp` from every module, so a weight reload
        # drops the graphs immediately.  The ADDRESS stamp is not redundant:
        # an in-place weight update needs no invalidation at all (the graph
        # reads the same storage), while the one thing a captured graph cannot
        # survive -- a parameter reallocated somewhere else -- is exactly what
        # data_ptr() catches.
        stamp = (
            self.conv1d.weight.data_ptr(),
            self.A_log.data_ptr(),
            self.dt_bias.data_ptr(),
            self.o_norm.weight.data_ptr(),
            None if self.conv1d.bias is None else self.conv1d.bias.data_ptr(),
            self._inkernel_scale,
            self.gate_lower_bound,
            self.o_norm.eps,
            self.rda_fused,
        )
    else:
        key = (
            tuple(x.shape),
            tuple(g2.shape),
            tuple(out.shape),
            self.kv_cache[0].data_ptr(),
            self.kv_cache[1].data_ptr(),
            torch.cuda.current_stream(x.device).cuda_stream,
        )
        params = [self.conv1d.weight, self.A_log, self.dt_bias, self.o_norm.weight]
        if self.conv1d.bias is not None:
            params.append(self.conv1d.bias)
        try:
            stamp = tuple(
                (id(p), p._version, p.data_ptr(), tuple(p.shape), tuple(p.stride()))
                for p in params
            )
        except RuntimeError:
            return False
        stamp += (
            self._inkernel_scale,
            self.gate_lower_bound,
            self.o_norm.eps,
            self.rda_fused,
        )
    if getattr(self, "_raon_kda_decode_graph_stamp", None) != stamp:
        self._raon_kda_decode_graph = {}
        self._raon_kda_decode_graph_stamp = stamp
    cache = getattr(self, "_raon_kda_decode_graph", {})
    entry = cache.get(key)
    if entry is None and len(cache) >= _MAX_GRAPHS:
        if ("shapes",) not in _ANNOUNCED:
            _ANNOUNCED.add(("shapes",))
            print(f"[parity] RAON_KDA_DECODE_GRAPH_SHAPES INERT: {len(cache)} "
                  f"distinct decode shapes are already captured on this layer "
                  f"(cap {_MAX_GRAPHS}) and n={n} is a new one, so this step "
                  f"runs EAGER. Raising RAON_KDA_DECODE_GRAPH_MAX (now "
                  f"{_MAX_TOKENS}) widens the shape set, so it has to be "
                  f"raised together with this cap", flush=True)
        return False           # bounded; run eager rather than grow without limit
    if entry is None:
        if _SAFE:
            permitted, why = _capture_permitted()
            if not permitted:
                if not _SAFE_NOCAPTURE_ANNOUNCED[0]:
                    _SAFE_NOCAPTURE_ANNOUNCED[0] = True
                    print(f"[parity] RAON_KDA_DECODE_GRAPH_SAFE: declined to "
                          f"CAPTURE at n={n} because {why}; this step runs "
                          f"EAGER. Capture is taken only where a top-level "
                          f"capture is legal (vLLM's pre-capture warmup "
                          f"forward); replay of an already-captured shape is "
                          f"legal in every window and is unaffected by this "
                          f"line", flush=True)
                return False
        inputs = [torch.empty_like(t) for t in (x, g1, g2, beta, ratio, idx)]
        for dst, src in zip(inputs, (x, g1, g2, beta, ratio, idx)):
            dst.copy_(src)
        # This warm execution is the current step; capture does not execute it again.
        packed, value = gpu(self, *inputs, tuple(out.shape))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = gpu(self, *inputs, tuple(out.shape))
        entry = (graph, inputs, captured)
        cache[key] = entry
        self._raon_kda_decode_graph = cache
        # Emitted from inside the capture branch: the graph object above exists
        # only because this code ran, so the line is evidence of a capture, not
        # a restatement of the env var.
        if _SAFE and ("safe-capture", n) not in _ANNOUNCED:
            _ANNOUNCED.add(("safe-capture", n))
            print(f"[parity] RAON_KDA_DECODE_GRAPH_SAFE LIVE: captured a KDA "
                  f"decode graph at n={n} tokens in a legal window -- {why} "
                  f"(graph {len(cache)}/{_MAX_GRAPHS} on this layer, bound in "
                  f"effect {_MAX_TOKENS}); later decode steps replay it",
                  flush=True)
        if ("capture", n) not in _ANNOUNCED:
            _ANNOUNCED.add(("capture", n))
            if n > 1:
                print(f"[parity] RAON_KDA_DECODE_GRAPH_MAX LIVE: captured a "
                      f"KDA decode graph at n={n} tokens (bound in effect "
                      f"{_MAX_TOKENS}; graph {len(cache)}/{_MAX_GRAPHS} on "
                      f"this layer) -- a multi-token decode batch is now "
                      f"covered, which a bound of 1 could never capture",
                      flush=True)
            elif _MAX_TOKENS > 1:
                print(f"[parity] RAON_KDA_DECODE_GRAPH_MAX INERT: captured a "
                      f"KDA decode graph at n=1 token although the bound in "
                      f"effect is {_MAX_TOKENS} -- only the single-token "
                      f"graph was captured here", flush=True)
            else:
                print(f"[parity] RAON_KDA_DECODE_GRAPH_MAX INERT: captured a "
                      f"KDA decode graph at n=1 token and the bound in effect "
                      f"is 1 (the code default when the variable is absent) -- "
                      f"one token is all this bound admits, so every "
                      f"multi-sequence decode step runs eager", flush=True)
    else:
        graph, inputs, captured = entry
        for dst, src in zip(inputs, (x, g1, g2, beta, ratio, idx)):
            dst.copy_(src)
        graph.replay()
        packed, value = captured
        # Printed after replay() returned; it names which of the two breakable
        # windows this replay happened in, so a replay during warmup does not
        # satisfy it.
        if _SAFE and not _SAFE_REPLAY_ANNOUNCED[0]:
            _cls = getattr(_breakable_module(), "BreakableCUDAGraphCapture", None)
            if _cls is not None and _cls.current() is not None:
                _SAFE_REPLAY_ANNOUNCED[0] = True
                print(f"[parity] RAON_KDA_DECODE_GRAPH_SAFE LIVE: replayed the "
                      f"captured KDA decode graph at n={n} inside the EAGER "
                      f"BREAK of a vLLM breakable capture -- no nested capture "
                      f"was taken, so the launch-side cost of this layer is "
                      f"one cudaGraphLaunch", flush=True)
            elif getattr(_TLS, "breakable_replay", 0):
                _SAFE_REPLAY_ANNOUNCED[0] = True
                print(f"[parity] RAON_KDA_DECODE_GRAPH_SAFE LIVE: replayed the "
                      f"captured KDA decode graph at n={n} inside a vLLM "
                      f"breakable graph REPLAY -- capturing in this window "
                      f"takes an illegal memory access; it only replays here",
                      flush=True)
        # Emitted from inside the replay branch, AFTER graph.replay() returned:
        # a capture that is never replayed is pure overhead, and the capture
        # line above cannot distinguish the two.
        if ("replay", n) not in _ANNOUNCED:
            _ANNOUNCED.add(("replay", n))
            if n > 1:
                print(f"[parity] RAON_KDA_DECODE_GRAPH_MAX LIVE: replayed a "
                      f"captured KDA decode graph at n={n} tokens (bound in "
                      f"effect {_MAX_TOKENS}) -- the decode step is running "
                      f"from the graph, not from eager launches", flush=True)
            else:
                print(f"[parity] RAON_KDA_DECODE_GRAPH_MAX INERT: replayed a "
                      f"captured KDA decode graph at n=1 token (bound in "
                      f"effect {_MAX_TOKENS}) -- replay is live, but only for "
                      f"the single-token shape", flush=True)
    # The core is private storage; normalization does not read recurrent state.
    self._cpr_decode_track(packed, g1, beta, idx, self.kv_cache[1])
    out.copy_(value)
    return True

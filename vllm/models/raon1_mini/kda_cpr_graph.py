"""RAON_KDA_CPR_GRAPH_LEDGER=1 — capture-safe CPR decode-track.

WHY.  Under FULL-decode-only (FDO) CUDA graphs the whole KDA decode forward is
captured inline (breakable eager-breaks are off at mode==FULL), and the CPR
decode seam ``_cpr_decode_track`` is host-driven twice over:

  (1) ``_slots_tolist_pinned``'s synchronous D2H (``dst.copy_(...,
      non_blocking=False)`` + ``tolist()``) is ILLEGAL during stream capture —
      ``CUDA error: operation not permitted when stream is capturing``.
  (2) even if the sync were removed, the python slot loop, the
      ``CPRSlotTracker`` dict ledger and the 64-boundary branch are HOST code:
      a capture would freeze the boot-dummy batch's loop structure into the
      graph and replay it against unrelated slots.

FIX — move the ledger onto the device and the boundary maintenance out of the
captured region:

  * in-graph (capture-safe, pure device ops, zero host readback): each decoded
    token's post-conv rows are scattered into PREALLOCATED per-layer ring
    tensors ``[rows, 64, H, D]`` addressed by a device ``slot -> row`` map,
    and a per-layer device ``row_len`` counter is bumped with ``index_put_
    (accumulate=True)``.  A CUDA graph records only fixed pointers; at replay
    the gathers/scatters re-read the CURRENT values of the (persistent)
    ``decode_conv_indices`` buffer and of ``slot2row``/``row_len``, so the
    ledger keeps tracking the real batch — nothing is frozen.
  * out-of-graph (host, eager, once per engine step): an epilogue hooked in
    front of ``KimiK3KDAMetadataBuilder.build`` (which the runner calls on the
    host EVERY step, graphs or not, BEFORE the forward) reads ``row_len`` back
    (legal outside capture), and for every row that reached the 64-boundary
    runs EXACTLY the eager boundary math — one varlen ``chunk_kda`` state pass
    over the buffered 64 real tokens under ``_cpr_contract`` with
    ``state_v_first=True`` (the KDA VFIRST pairing is untouched), advances the
    fp32 entry chain, and RESYNCs ``recurrent_state[slot]`` in place to
    ``fp32(bf16(entry))``.  The resync therefore lands between the boundary
    step's replay and the next step's forward — the next live decode step
    reads the same bytes the eager path would have produced, and no _forward
    ever observes a row at 64 (the epilogue runs before every forward).

NUMERICAL CONTRACT.  The boundary pass consumes the same contiguous
``[1, R*64, H, D]`` values, the same ``cu_seqlens = arange*64``, the same
``_cpr_contract`` / ``state_v_first=True`` / ``cpr_snapshot`` as the eager
``kda_boundary_batch.append_batch`` path (bit-gated when BOUNDARY_BATCH
shipped), so per-slot entries/resyncs are bitwise the eager tracker's.  The
64-boundary resync grid and the prefill<->decode chunk-state handoff
(``reset``/``open_stacked``/``flush_state``) are unchanged — this module only
changes WHERE the open-chunk bytes live and WHEN (same step boundary) the
resync is written.

CAPTURE / PADDING SAFETY.
  * vLLM pads FDO decode batches and fills padding state indices with
    NULL_BLOCK_ID == 0 (gdn_attn.py:489); slot 0 is the reserved null block.
    ``slot2row`` maps every unbound slot to a SCRATCH row, so dummy-warmup,
    capture and padding appends land in scratch and are periodically reset by
    the epilogue — no real row is ever touched by a fake request.
  * all device tensors are allocated EAGERLY during vLLM's warmup dummy runs
    (``cudagraph_num_of_warmups >= 1`` runs the forward with
    ``cudagraph_runtime_mode=NONE`` before every capture); allocating while a
    stream is capturing raises (fail-closed) instead of leaking into a graph
    pool.
  * write positions are ``row_len % 64`` so an in-graph append can never index
    out of bounds; the epilogue raises if a BOUND row ever exceeds 64
    (a missed epilogue would otherwise silently wrap).

CAPACITY.  Rings are sized by ``2*max_num_seqs + 8`` rows (override:
RAON_KDA_CPR_GRAPH_LEDGER_ROWS), NOT by the engine's theoretical slot count
(~1024), which would need a single multi-GiB tensor per ring.  Rows are bound at prefill ``reset`` and reclaimed
by idle-based eviction (a bound row that has not appended for
RAON_KDA_CPR_GRAPH_LEDGER_EVICT_IDLE (default 16) epilogue steps belongs to a
finished/preempted request: running mamba decodes append every scheduled step
and preempted requests are fully recomputed).  Eviction also pops the slot's
entry from every layer's tracker so an impossible resume fails loudly at
kda.py's "no CPR entry-chain bookkeeping" guard instead of resuming with an
empty open chunk.

Default OFF.  ``RAON_KDA_CPR_GRAPH_LEDGER=1`` requires RAON_KDA_CPR=1 and is
mutually exclusive with RAON_KDA_CPR_RING (the ledger IS the ring).
"""

import os

import torch

from vllm.models.raon1_mini.kda_inkernel import (
    CPR_CHUNK,
    CPRSlotTracker,
    _cpr_contract,
    cpr_snapshot,
    cu_cpu_kwargs,
    cu_seqlens_cached,
)

_MAX_LAYERS = 64  # row_len is one [_MAX_LAYERS, rows] tensor: ONE D2H per step


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _capturing():
    """True iff the current stream is capturing (False on CUDA-less hosts)."""
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class _LedgerManager:
    """Process-wide shared state: slot->row map, per-layer row_len, registry."""

    def __init__(self, capacity_hint):
        self.chunk = CPR_CHUNK
        self.capacity = _env_int(
            "RAON_KDA_CPR_GRAPH_LEDGER_ROWS", 2 * capacity_hint + 8
        )
        self.evict_idle = _env_int("RAON_KDA_CPR_GRAPH_LEDGER_EVICT_IDLE", 16)
        self.layers = []          # _LayerLedger, registration order
        self.slot2row = None      # int64 [num_slots] device, unbound -> scratch
        self.row_len = None       # int32 [_MAX_LAYERS, capacity+1] device
        self.ones = None          # int32 [max_batch] device (index_put_ src)
        self.scratch = self.capacity  # scratch row index
        # host mirrors
        self.slot2row_host = {}
        self.row2slot = {}
        self.free = list(range(self.capacity))
        self.last_active = [0] * (self.capacity + 1)
        self.prev_len = None      # host int32 [capacity+1]
        self.step = 0
        self.boundaries = 0       # total boundary resyncs (freeze refutation)
        self.evictions = 0
        self.announced = False

    # ----- device state ----------------------------------------------------
    def ensure_maps(self, num_slots, device):
        """Allocate the slot->row map for the FULL slot id space.

        The slot id space is NOT max_num_seqs. vLLM allocates the mamba/KDA
        state cache by KV-cache BLOCKS, so ``recurrent_state.shape[0]`` is the
        block-id space — memory-derived and boot-variable (e.g. 3,105 at
        8K/util 0.70).  Also, the FIRST forward that allocates this ledger is
        the memory-PROFILING dummy run, whose placeholder cache is
        max_num_seqs-sized, so sizing the map from that snapshot would freeze
        it before the real cache exists.

        Hence one allocation, generously bounded (default 65,536 ids —
        int64 map = 512 KiB, negligible; RAON_KDA_CPR_GRAPH_LEDGER_SLOTS to
        raise).  The map pointer must stay stable once any graph captured it,
        so growth-by-reallocation is deliberately NOT supported: ids beyond
        the bound fail closed.  Row capacity (concurrency-bound rings) is a
        separate, unchanged model — 3,105 is an id space, not a concurrency.
        """
        if self.slot2row is not None:
            if num_slots <= self.slot2row.numel():
                return
            raise RuntimeError(
                "RAON_KDA_CPR_GRAPH_LEDGER: slot id space "
                f"{num_slots} exceeds the preallocated map "
                f"({self.slot2row.numel()}); raise "
                "RAON_KDA_CPR_GRAPH_LEDGER_SLOTS (the map cannot be "
                "reallocated once a graph captured its pointer)"
            )
        if _capturing():
            raise RuntimeError(
                "RAON_KDA_CPR_GRAPH_LEDGER: first use is inside a CUDA graph "
                "capture — the ledger must be allocated during an eager "
                "warmup (cudagraph_num_of_warmups >= 1)"
            )
        bound = max(
            num_slots, _env_int("RAON_KDA_CPR_GRAPH_LEDGER_SLOTS", 65536)
        )
        self.slot2row = torch.full(
            (bound,), self.scratch, dtype=torch.int64, device=device
        )
        self.row_len = torch.zeros(
            (_MAX_LAYERS, self.capacity + 1), dtype=torch.int32, device=device
        )
        self.ones = torch.ones(4096, dtype=torch.int32, device=device)
        self.prev_len = torch.zeros(self.capacity + 1, dtype=torch.int32)
        if not self.announced:
            self.announced = True
            print(
                "[raon-kda] RAON_KDA_CPR_GRAPH_LEDGER LIVE: CPR decode-track "
                f"is device-resident ({self.capacity} rows x {self.chunk} "
                f"tokens per layer, scratch row {self.scratch}); 64-boundary "
                "entry-chain advance + resync run in the pre-step epilogue "
                "outside any capture",
                flush=True,
            )

    # ----- row binding (host, eager only) ----------------------------------
    def bind(self, slot):
        # Slot 0 is vLLM's reserved NULL block
        # (padding fill).  Boot-time dummy prefills present it for reset; if it
        # ever BOUND a row, every later padded decode lane would append into
        # that live row (duplicate-lane accumulate) and blow the 64-boundary
        # bookkeeping.  Alias it to scratch permanently.
        if slot == 0:
            return self.scratch
        row = self.slot2row_host.get(slot)
        if row is not None:
            return row
        if not self.free:
            self._evict_one()
        row = self.free.pop()
        self.slot2row_host[slot] = row
        self.row2slot[row] = slot
        self.slot2row[slot] = row
        self.last_active[row] = self.step
        return row

    def mark_active(self, slots):
        """Refresh liveness for the slots scheduled THIS step (host ids).

        Called from the builder hook with the step's real state
        indices.  A RUNNING request is scheduled every step, so its row is
        refreshed every step by construction; rows of finished requests and of
        boot-phase dummy prefills stop being marked the moment they disappear
        from the batch and become reclaimable after ``evict_idle`` steps
        (default 4) — deterministic reclamation instead of waiting on the
        append-based idle heuristic (boot-dummy binds plus real binds can
        exhaust the rows before anything has been idle long enough to evict).
        """
        for s in slots:
            if s <= 0:
                continue
            row = self.slot2row_host.get(int(s))
            if row is not None:
                self.last_active[row] = self.step

    def _evict_one(self):
        cand = None
        for row, slot in self.row2slot.items():
            idle = self.step - self.last_active[row]
            if idle >= self.evict_idle and (
                cand is None or self.last_active[row] < self.last_active[cand]
            ):
                cand = row
        if cand is None:
            dump = ", ".join(
                f"row{r}=slot{s}@{self.last_active[r]}"
                for r, s in sorted(self.row2slot.items())
            )
            raise RuntimeError(
                "RAON_KDA_CPR_GRAPH_LEDGER: no free ledger row and no bound "
                f"row idle >= {self.evict_idle} steps (capacity "
                f"{self.capacity}, step {self.step}); raise "
                "RAON_KDA_CPR_GRAPH_LEDGER_ROWS or lower "
                f"RAON_KDA_CPR_GRAPH_LEDGER_EVICT_IDLE. ledger: [{dump}]"
            )
        slot = self.row2slot.pop(cand)
        self.slot2row_host.pop(slot, None)
        self.slot2row[slot] = self.scratch
        self.row_len[:, cand] = 0
        self.prev_len[cand] = 0
        # Fail-closed for the (impossible under mamba recompute) resume of an
        # evicted slot: without an entry, kda.py's resume guard raises instead
        # of silently continuing with an empty open chunk.
        for lay in self.layers:
            lay.tracker.entry.pop(slot, None)
        self.free.append(cand)
        self.evictions += 1
        print(
            f"[raon-kda] CPR graph ledger: evicted idle row {cand} "
            f"(slot {slot}, idle {self.step - self.last_active[cand]} steps)",
            flush=True,
        )

    # ----- epilogue (host, eager, once per engine step) ---------------------
    def epilogue(self):
        if self.row_len is None or _capturing():
            return
        self.step += 1
        nl = len(self.layers)
        # ONE small D2H sync, outside capture.  copy=True: on a CPU tensor
        # ``.cpu()`` would return the SAME storage and alias prev_len into
        # row_len (silently disabling change detection).
        lens = self.row_len[:nl].to("cpu", copy=True)
        l0 = lens[0]
        if nl > 1 and not bool((lens == l0.unsqueeze(0)).all()):
            raise RuntimeError(
                "RAON_KDA_CPR_GRAPH_LEDGER: per-layer append counts diverged "
                f"— a layer skipped the decode seam. row_len:\n{lens}"
            )
        changed = (l0 != self.prev_len).nonzero(as_tuple=False).flatten()
        for r in changed.tolist():
            self.last_active[r] = self.step
        ready, stale = [], []
        for r in changed.tolist():
            n = int(l0[r])
            if r in self.row2slot:
                if n > self.chunk:
                    raise RuntimeError(
                        f"RAON_KDA_CPR_GRAPH_LEDGER: bound row {r} (slot "
                        f"{self.row2slot[r]}) reached {n} > {self.chunk} — an "
                        "epilogue was skipped; the open chunk is corrupt"
                    )
                if n == self.chunk:
                    ready.append(r)
            elif n >= self.chunk:
                stale.append(r)  # scratch / unbound garbage: just rewind
        if ready:
            ready.sort()
            slots = [self.row2slot[r] for r in ready]
            rows_t = torch.tensor(
                ready, dtype=torch.int64, device=self.row_len.device
            )
            for lay in self.layers:
                lay.boundary_pass(rows_t, slots)
            self.boundaries += len(ready)
        for r in ready + stale:
            self.row_len[:, r] = 0
            l0[r] = 0
        self.prev_len = l0


_MANAGER = None
_HOOKED = False


def _manager(capacity_hint=32):
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = _LedgerManager(capacity_hint)
    return _MANAGER


def run_cpr_graph_epilogue():
    """Resolve pending 64-boundaries; runs before every engine step."""
    if _MANAGER is not None:
        _MANAGER.epilogue()


def _install_builder_hook():
    """Run the epilogue in front of every KDA metadata build (host, per step,
    before the forward/replay — the only per-step host seam under FDO)."""
    global _HOOKED
    if _HOOKED:
        return
    from .rda_metadata import (  # fail-closed import
        RaonRDAMetadataBuilder as _B,
    )
    orig = _B.build

    def build(self, *args, **kwargs):
        run_cpr_graph_epilogue()
        return orig(self, *args, **kwargs)

    _B.build = build
    _HOOKED = True


class _LayerLedger:
    """Per-layer device rings + the boundary state pass for this layer."""

    def __init__(self, module, mgr, layer_idx):
        self.module = module
        self.mgr = mgr
        self.layer_idx = layer_idx
        if layer_idx >= _MAX_LAYERS:
            raise RuntimeError(
                f"RAON_KDA_CPR_GRAPH_LEDGER: >{_MAX_LAYERS} KDA layers"
            )
        self.rings = None  # dict q/k/v/g: [rows, chunk, H, D]; beta: [rows, chunk, H]
        self.tracker = None  # set by CPRGraphSlotTracker

    def _ensure(self, q_row, k_row, v_row, g_row, beta_row):
        """Allocate rings from live row templates (eager warmup / prefill)."""
        if self.rings is not None:
            return
        if _capturing():
            raise RuntimeError(
                "RAON_KDA_CPR_GRAPH_LEDGER: ring allocation reached inside a "
                "capture — warmup did not exercise the CPR decode seam"
            )
        conv, rec = self.module.kv_cache
        self.mgr.ensure_maps(rec.shape[0], q_row.device)
        n = self.mgr.capacity + 1
        c = self.mgr.chunk

        def alloc(row):
            return torch.empty(
                (n, c, *row.shape[1:]), dtype=row.dtype, device=row.device
            )

        self.rings = {
            "q": alloc(q_row), "k": alloc(k_row), "v": alloc(v_row),
            "g": alloc(g_row), "beta": alloc(beta_row),
        }

    def append_graph(self, slots_dev, q, k, v, g, beta):
        """Capture-safe batched append: q/k/v/g [n, H, D], beta [n, H].

        Pure device ops (gather/scatter/add) on preallocated storage — no
        host readback, no allocation after the first (eager) call, no
        data-dependent host branch.  Padding/dummy slots resolve to the
        scratch row through ``slot2row``.
        """
        self._ensure(q, k, v, g, beta)
        mgr = self.mgr
        n = q.shape[0]
        if n > mgr.ones.numel():
            raise RuntimeError(
                f"RAON_KDA_CPR_GRAPH_LEDGER: decode batch {n} exceeds the "
                f"ones buffer ({mgr.ones.numel()})"
            )
        rows = mgr.slot2row.index_select(0, slots_dev.long())
        lens = mgr.row_len[self.layer_idx].index_select(0, rows)
        pos = (lens % mgr.chunk).long()
        idx = (rows, pos)
        self.rings["q"].index_put_(idx, q)
        self.rings["k"].index_put_(idx, k)
        self.rings["v"].index_put_(idx, v)
        self.rings["g"].index_put_(idx, g)
        self.rings["beta"].index_put_(idx, beta)
        mgr.row_len[self.layer_idx].index_put_(
            (rows,), mgr.ones[:n], accumulate=True
        )

    # ----- eager-side (prefill seam / epilogue) -----------------------------
    def write_tail(self, row, q_t, k_t, v_t, g_t, beta_t):
        self._ensure(q_t, k_t, v_t, g_t, beta_t)
        L = q_t.shape[0]
        for name, t in (
            ("q", q_t), ("k", k_t), ("v", v_t), ("g", g_t), ("beta", beta_t)
        ):
            ring = self.rings[name]
            if ring.dtype != t.dtype or ring.shape[2:] != t.shape[1:]:
                raise RuntimeError(
                    f"RAON_KDA_CPR_GRAPH_LEDGER: tail {name} "
                    f"{tuple(t.shape)}/{t.dtype} != ring "
                    f"{tuple(ring.shape)}/{ring.dtype}"
                )
            ring[row, :L].copy_(t)
        self.mgr.row_len[self.layer_idx, row] = L

    def open_len_row(self, row):
        return int(self.mgr.row_len[self.layer_idx, row].item())

    def open_rows(self, row, L):
        r = self.rings
        return (
            r["q"][row, :L].clone().unsqueeze(0),
            r["k"][row, :L].clone().unsqueeze(0),
            r["v"][row, :L].clone().unsqueeze(0),
            r["g"][row, :L].clone().unsqueeze(0),
            r["beta"][row, :L].clone().unsqueeze(0),
        )

    def boundary_pass(self, rows_t, slots):
        """Advance the fp32 entry chain + resync — bitwise the eager
        ``kda_boundary_batch.append_batch`` kernel call (same packed values,
        same cu_seqlens, same contract, ``state_v_first=True``)."""
        from fla.ops.kda import chunk_kda

        m = self.module
        r = self.rings
        R = rows_t.numel()
        c = self.mgr.chunk
        packed = [
            r[name].index_select(0, rows_t).reshape(
                1, R * c, *r[name].shape[2:]
            )
            for name in ("q", "k", "v", "g", "beta")
        ]
        h, d = packed[0].shape[-2:]
        vd = packed[2].shape[-1]
        tracker = self.tracker
        entries = [
            tracker.entry.get(slot)
            if tracker.entry.get(slot) is not None
            else torch.zeros(
                1, h, vd, d, device=rows_t.device, dtype=torch.float32
            )
            for slot in slots
        ]
        cu, cu_cpu = cu_seqlens_cached(
            lambda: torch.arange(
                R + 1, device=rows_t.device, dtype=torch.int32
            ) * c,
            range(0, (R + 1) * c, c),
            rows_t.device,
        )
        _, fs = chunk_kda(
            q=packed[0],
            k=packed[1],
            v=packed[2],
            g=packed[3],
            beta=packed[4].float(),
            initial_state=torch.cat(entries),
            output_final_state=True,
            safe_gate=m.gate_lower_bound is not None,
            cu_seqlens=cu,
            state_v_first=True,
            **cu_cpu_kwargs(cu_cpu),
            **_cpr_contract(
                m.A_log, m.dt_bias, m._inkernel_scale, m.gate_lower_bound
            ),
        )
        fs = fs.clone()
        snapshots = cpr_snapshot(fs)
        conv, rec = m.kv_cache
        for j, slot in enumerate(slots):
            tracker.entry[slot] = fs[j : j + 1]
            rec[slot] = snapshots[j]


class CPRGraphSlotTracker(CPRSlotTracker):
    """CPRSlotTracker whose open-chunk storage is the device ledger.

    ``entry`` stays the host dict of fp32 device tensors (consumed only by the
    eager prefill seam and the epilogue).  ``append`` is forbidden — the
    decode seam goes through ``ledger.append_graph`` and boundaries are
    resolved by the epilogue.
    """

    def __init__(self, ledger):
        super().__init__()
        self.ledger = ledger
        ledger.tracker = self

    def append(self, *args, **kwargs):
        raise RuntimeError(
            "RAON_KDA_CPR_GRAPH_LEDGER: host-side CPRSlotTracker.append is "
            "unreachable in ledger mode (decode-track routes to append_graph)"
        )

    def drop(self, slot):
        self.entry.pop(slot, None)
        row = self.ledger.mgr.slot2row_host.get(slot)
        if row is not None:
            # Clear only THIS layer's count: layers drop/reset the same slot
            # interleaved within one step, and another layer may already have
            # reset it.  The binding itself is reclaimed by idle eviction.
            self.ledger.mgr.row_len[self.ledger.layer_idx, row] = 0

    def reset(self, slot, entry, q_tail=None, k_tail=None, v_tail=None,
              g_tail=None, beta_tail=None):
        self.entry[slot] = entry.clone()
        mgr = self.ledger.mgr
        # Maps are shape-independent — size them from the recurrent cache so
        # a prefill reset can bind rows even before the first decode append
        # (rings themselves are allocated from the first tail/append rows).
        conv, rec = self.ledger.module.kv_cache
        mgr.ensure_maps(rec.shape[0], rec.device)
        row = mgr.bind(slot)
        if q_tail is not None and q_tail.shape[0] > 0:
            self.ledger.write_tail(
                row, q_tail, k_tail, v_tail, g_tail, beta_tail
            )
        else:
            mgr.row_len[self.ledger.layer_idx, row] = 0

    def open_len(self, slot):
        row = self.ledger.mgr.slot2row_host.get(slot)
        if row is None:
            return 0
        return self.ledger.open_len_row(row)

    def open_stacked(self, slot):
        row = self.ledger.mgr.slot2row_host.get(slot)
        if row is None:
            return None
        L = self.ledger.open_len_row(row)
        if L == 0:
            return None
        return self.ledger.open_rows(row, L)


def attach_graph_ledger(module, max_num_seqs, install_hook=True):
    """Wire a RaonKDAAttention layer into the shared graph ledger.

    Returns the layer's CPRGraphSlotTracker (drop-in for ``self._cpr``) and
    stores the layer ledger on the module as ``_cpr_graph_ledger``.
    """
    mgr = _manager(max_num_seqs)
    ledger = _LayerLedger(module, mgr, len(mgr.layers))
    mgr.layers.append(ledger)
    module._cpr_graph_ledger = ledger
    if install_hook:
        _install_builder_hook()
    return CPRGraphSlotTracker(ledger)

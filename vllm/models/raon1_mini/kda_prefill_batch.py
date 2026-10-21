"""Batch independent CPR entry chains while preserving each sequence recurrence."""

import os

import torch
import vllm.models.raon1_mini.kda_inkernel as ki

original = ki._fla_cpr_prefill_core_serial

_FUSED_ENV = "RAON_KDA_CPR_PREFILL_FUSED"
_FUSED_ANNOUNCED = False


def _fused_capable():
    """Whether the installed fla exposes intermediate states (feature detection, not assumed)."""
    cached = getattr(_fused_capable, "_v", None)
    if cached is not None:
        return cached
    try:
        import inspect

        from fla.ops.kda import chunk_kda

        ok = "return_intermediate_states" in inspect.signature(chunk_kda).parameters
    except Exception:
        ok = False
    _fused_capable._v = ok
    return ok


def _fused_require():
    """Armed lever + unmet prerequisite = raise (fail-closed).

    A silent fallback to the wave path would hide a perf regression behind an
    identical-bits output; the operator armed the lever, so an unmet
    prerequisite is an error, not a preference.
    """
    if not _fused_capable():
        raise RuntimeError(
            f"{_FUSED_ENV}=1 but the installed fla chunk_kda has no "
            "return_intermediate_states parameter (feature detection failed). "
            "Refusing the silent wave fallback -- unset the lever or fix the "
            "fla install."
        )
    if not torch.is_inference_mode_enabled():
        raise RuntimeError(
            f"{_FUSED_ENV}=1 but this prefill is not running under "
            "torch.inference_mode(); fla asserts inference mode for "
            "return_intermediate_states. Refusing the silent wave fallback."
        )


def _batched_fused(q, k, v, g1, beta_raw, A_log, dt_bias, scale, lower_bound,
                   entries0, bounds, chunk):
    """RAON_KDA_CPR_PREFILL_FUSED=1: CPR prefill without the serial wave loop.

    The wave loop issues one chunk_kda call per chunk index (128 calls at 8K
    tokens) and dominates CPR prefill time. fla ``chunk_kda`` already offers
    ``return_intermediate_states``, which exposes the intermediate 64-boundary
    states h ([1, NT, HV, V(K), K(V)], **stored in bf16**) in a single call.

    Why the bits are unchanged:
      1) The in-kernel fp32 state propagation is bit-identical to the wave
         chain (see kda_inkernel.flush_state: "equals the plain
         whole-sequence chunk_kda fp32 final state bitwise").
      2) h is the bf16 store of those fp32 boundary states, i.e. exactly the
         ``cpr_snapshot`` rounding, so the pool (scan initial states) is
         bit-identical to the wave pool.
      3) The tracker needs the **fp32** last-boundary entry, which h (bf16)
         cannot provide. Splitting each sequence's varlen boundaries into
         [main = multiple of 64][tail] subsequences (no data movement) makes
         main's final_state (fp32) exactly that entry. The tail subsequence
         runs from zero state and all its h/final/o outputs are discarded
         (at most 63 wasted tokens).
    The scan (fused_recurrent, the bitwise source of the output) is unchanged;
    its inputs (pool, cu) are bit-identical, so the core is too.
    """
    from fla.ops.kda import chunk_kda, fused_recurrent_kda

    count = len(bounds) - 1
    H, K = q.shape[2:]
    V = v.shape[-1]
    dev = q.device

    def e0(i):
        if entries0 is not None and entries0[i] is not None:
            return entries0[i].float()
        return torch.zeros(1, H, V, K, device=dev, dtype=torch.float32)

    zero_init = torch.zeros(1, H, V, K, device=dev, dtype=torch.float32)

    # ---- cu split: [main][tail] subsequences + bookkeeping -------------------
    cu_p = [0]              # split varlen boundaries (no data movement)
    init_rows = []          # subsequence initial states (fp32)
    main_sub = [None] * count   # seq -> main subsequence idx
    n_chunks = [0] * count      # main chunk count per seq
    tail_len = [0] * count
    sub_chunks = []             # chunk count per subsequence (h row offsets)
    for i in range(count):
        s, e = bounds[i], bounds[i + 1]
        L = e - s
        n = L // chunk
        t = L - n * chunk
        n_chunks[i] = n
        tail_len[i] = t
        if n > 0:
            main_sub[i] = len(init_rows)
            init_rows.append(e0(i))
            cu_p.append(s + n * chunk)
            sub_chunks.append(n)
        if t > 0:
            init_rows.append(zero_init)  # tail subsequence: all outputs discarded
            cu_p.append(e)
            sub_chunks.append(1)

    # Bit-neutral (same ordering and rounding points): both calls share
    #   beta_raw.float() / _cpr_contract (one duplicate launch fewer), and the
    #   cu tensor goes through the site-3 helper's pinned staging (with
    #   RAON_KDA_CU_CACHE OFF the original expression is used; same values
    #   and dtype).
    beta_f = beta_raw.float()
    contract = ki._cpr_contract(A_log, dt_bias, scale, lower_bound)
    cu_p_t, cu_p_cpu = ki.cu_seqlens_cached(
        lambda: torch.tensor(cu_p, device=dev, dtype=torch.int32),
        cu_p,
        dev,
    )
    o_disc, fs, h = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g1,
        beta=beta_f,
        initial_state=torch.cat(init_rows, dim=0),
        output_final_state=True,
        safe_gate=lower_bound is not None,
        cu_seqlens=cu_p_t,
        state_v_first=True,
        return_intermediate_states=True,
        **ki.cu_cpu_kwargs(cu_p_cpu),
        **contract,
    )
    del o_disc  # the scan produces the core (the wave loop also discards chunk outputs)
    hf = h[0].float()  # [NT_sub, H, V(K), K(V)]; bf16 boundary states = snapshot

    # ---- pool assembly: main uses h rows, tail uses snapshot(fs[main]) ------
    S_total = sum(n_chunks[i] + (1 if tail_len[i] else 0) for i in range(count))
    pool = torch.empty(S_total, H, V, K, device=dev, dtype=torch.float32)
    cu2 = [0]
    sub_off = []
    acc = 0
    for n in sub_chunks:
        sub_off.append(acc)
        acc += n
    pos = 0
    for i in range(count):
        s = bounds[i]
        n, t = n_chunks[i], tail_len[i]
        if n > 0:
            off = sub_off[main_sub[i]]
            pool[pos : pos + n] = hf[off : off + n]
            for c in range(1, n + 1):
                cu2.append(s + c * chunk)
            pos += n
        if t > 0:
            base = fs[main_sub[i] : main_sub[i] + 1] if n > 0 else e0(i)
            pool[pos : pos + 1] = ki.cpr_snapshot(base)
            cu2.append(s + n * chunk + t)
            pos += 1
    assert pos == S_total

    cu2_t, _ = ki.cu_seqlens_cached(
        lambda: torch.tensor(cu2, device=dev, dtype=torch.int32),
        cu2,
        dev,
    )
    core, finals = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g1,
        beta=beta_f,
        initial_state=pool,
        output_final_state=True,
        state_v_first=True,
        cu_seqlens=cu2_t,
        **contract,
    )

    info = []
    last = -1
    for i in range(count):
        segs = n_chunks[i] + (1 if tail_len[i] else 0)
        last += segs
        entry = (
            fs[main_sub[i] : main_sub[i] + 1].clone()
            if n_chunks[i] > 0
            else e0(i)
        )
        running = (
            ki.cpr_snapshot(entry)
            if tail_len[i] == 0
            else finals[last : last + 1].clone()
        )
        info.append(
            dict(
                entry=entry,
                running=running,
                tail_start=bounds[i] + n_chunks[i] * chunk,
                tail_len=tail_len[i],
            )
        )
    return core, info


def batched(
    q,
    k,
    v,
    g1,
    beta_raw,
    A_log,
    dt_bias,
    scale,
    lower_bound,
    entries0=None,
    cu_seqlens=None,
    chunk=ki.CPR_CHUNK,
):
    from fla.ops.kda import chunk_kda, fused_recurrent_kda

    bounds = [0, q.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
    count = len(bounds) - 1
    degenerate = count < 1 or any(
        bounds[i] >= bounds[i + 1] for i in range(count)
    )
    # RAON_KDA_CPR_PREFILL_FUSED=1 (default OFF): delegate to the variant
    # without the serial wave loop. If armed and its prerequisites (the fla
    # feature + inference mode) are unmet, raise: a silent wave fallback
    # would hide a perf regression. chunk != 64 or degenerate bounds are
    # outside the supported configuration rather than an environment fault,
    # so they keep the original path (h is exposed only at 64-boundaries).
    # count == 1 is handled too: _batched_fused does not depend on the
    # sequence count, and a single sequence also runs one serial call per
    # chunk index in the wave loop.
    if os.environ.get(_FUSED_ENV, "0") == "1" and chunk == 64 and not degenerate:
        _fused_require()
        global _FUSED_ANNOUNCED
        if not _FUSED_ANNOUNCED:
            _FUSED_ANNOUNCED = True
            print(
                f"[parity] {_FUSED_ENV} LIVE: CPR prefill wave loop -> single "
                "chunk_kda with return_intermediate_states (h = bf16 boundary "
                "states == cpr_snapshot; fp32 last-boundary entries via "
                "main/tail cu split); the output scan is unchanged",
                flush=True,
            )
        return _batched_fused(
            q, k, v, g1, beta_raw, A_log, dt_bias, scale, lower_bound,
            entries0, bounds, chunk,
        )

    if count < 2 or chunk != 64 or degenerate:
        return original(
            q,
            k,
            v,
            g1,
            beta_raw,
            A_log,
            dt_bias,
            scale,
            lower_bound,
            entries0,
            cu_seqlens,
            chunk,
        )

    H, K = q.shape[2:]
    V = v.shape[-1]
    entries = [
        (
            entries0[i].float()
            if entries0 is not None and entries0[i] is not None
            else torch.zeros(1, H, V, K, device=q.device, dtype=torch.float32)
        )
        for i in range(count)
    ]
    positions = list(bounds[:-1])
    pools = [[] for _ in entries]
    ends = [[] for _ in entries]
    tails = [(bounds[i + 1], 0) for i in range(count)]
    while any(positions[i] < bounds[i + 1] for i in range(count)):
        full = []
        for i in range(count):
            pos = positions[i]
            end = bounds[i + 1]
            if pos >= end:
                continue
            nxt = min(pos + chunk, end)
            pools[i].append(ki.cpr_snapshot(entries[i]))
            ends[i].append(nxt)
            if nxt - pos == chunk:
                full.append((i, pos, nxt))
            else:
                tails[i] = (pos, nxt - pos)
            positions[i] = nxt
        if full:
            packed = [
                torch.cat([x[:, a:b] for _, a, b in full], dim=1)
                for x in (q, k, v, g1, beta_raw)
            ]
            # cu_seqlens site 2 of 3.  Lever OFF evaluates the original
            # expression verbatim; see RAON_KDA_CU_CACHE in kda_inkernel.py.
            cu, cu_cpu = ki.cu_seqlens_cached(
                lambda: torch.arange(
                    len(full) + 1, device=q.device, dtype=torch.int32
                )
                * chunk,
                range(0, (len(full) + 1) * chunk, chunk),
                q.device,
            )
            _, fs = chunk_kda(
                q=packed[0],
                k=packed[1],
                v=packed[2],
                g=packed[3],
                beta=packed[4].float(),
                initial_state=torch.cat([entries[i] for i, _, _ in full], dim=0),
                output_final_state=True,
                safe_gate=lower_bound is not None,
                cu_seqlens=cu,
                state_v_first=True,
                **ki.cu_cpu_kwargs(cu_cpu),
                **ki._cpr_contract(A_log, dt_bias, scale, lower_bound),
            )
            fs = fs.clone()
            for j, (i, _, _) in enumerate(full):
                entries[i] = fs[j : j + 1]
    pool = [x for seq in pools for x in seq]
    cu = [0] + [x for seq in ends for x in seq]
    core, finals = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g1,
        beta=beta_raw.float(),
        initial_state=torch.cat(pool, dim=0),
        output_final_state=True,
        state_v_first=True,
        cu_seqlens=torch.tensor(cu, device=q.device, dtype=torch.int32),
        **ki._cpr_contract(A_log, dt_bias, scale, lower_bound),
    )
    info = []
    last = -1
    for i in range(count):
        last += len(pools[i])
        start, length = tails[i]
        running = (
            ki.cpr_snapshot(entries[i])
            if length == 0
            else finals[last : last + 1].clone()
        )
        info.append(
            dict(entry=entries[i], running=running, tail_start=start, tail_len=length)
        )
    return core, info

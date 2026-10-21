"""Batch independent CPR boundary passes under the pinned FLA contract."""

import torch
from vllm.models.raon1_mini.kda_inkernel import (
    cpr_snapshot,
    _cpr_contract,
    cu_seqlens_cached,
    cu_cpu_kwargs,
)


def append_batch(tracker, slots, q, k, v, g, beta, A_log, dt_bias, scale, lower_bound):
    from fla.ops.kda import chunk_kda

    if len(set(slots)) != len(slots) or tracker.chunk != 64:
        return [
            tracker.append(
                slot,
                q[i],
                k[i],
                v[i],
                g[i],
                beta[i],
                A_log,
                dt_bias,
                scale,
                lower_bound,
            )
            for i, slot in enumerate(slots)
        ]
    ready = []
    result = [None] * len(slots)
    ring = getattr(tracker, "ring", False)
    if ring:
        # RAON_KDA_CPR_RING=1: row copy_ into the tracker's preallocated
        # rings -- no batch clone (nothing here retains a live view) and no
        # per-slot stack at pack time; a full ring IS the [chunk, ...] block.
        for i, slot in enumerate(slots):
            buf = tracker.open.get(slot)
            if buf is None:
                buf = tracker._ring_empty(q[i], k[i], v[i], g[i], beta[i])
                tracker.open[slot] = buf
            n = buf["len"]
            buf["q"][n].copy_(q[i])
            buf["k"][n].copy_(k[i])
            buf["v"][n].copy_(v[i])
            buf["g"][n].copy_(g[i])
            buf["beta"][n].copy_(beta[i])
            buf["len"] = n + 1
            if buf["len"] == tracker.chunk:
                ready.append((i, slot))
        if not ready:
            return result
        packed = [
            torch.cat(
                [tracker.open[slot][name] for _, slot in ready], dim=0
            ).unsqueeze(0)
            for name in ("q", "k", "v", "g", "beta")
        ]
    else:
        rows = [x.clone() for x in (q, k, v, g, beta)]
        for i, slot in enumerate(slots):
            buf = tracker.open.setdefault(slot, tracker._empty())
            for name, x in zip(("q", "k", "v", "g", "beta"), rows):
                buf[name].append(x[i])
            if len(buf["q"]) == tracker.chunk:
                ready.append((i, slot))
        if not ready:
            return result
        packed = [
            torch.cat(
                [torch.stack(tracker.open[slot][name]) for _, slot in ready],
                dim=0,
            ).unsqueeze(0)
            for name in ("q", "k", "v", "g", "beta")
        ]
    h, d = q.shape[-2:]
    vd = v.shape[-1]
    entries = [tracker.entry.get(slot) for _, slot in ready]
    entries = [
        (
            x
            if x is not None
            else torch.zeros(1, h, vd, d, device=q.device, dtype=torch.float32)
        )
        for x in entries
    ]
    # cu_seqlens for the decode 64-boundary batch.  With RAON_KDA_CU_CACHE off
    # the original expression is evaluated as-is.
    cu, cu_cpu = cu_seqlens_cached(
        lambda: torch.arange(len(ready) + 1, device=q.device, dtype=torch.int32)
        * 64,
        range(0, (len(ready) + 1) * 64, 64),
        q.device,
    )
    _, fs = chunk_kda(
        q=packed[0],
        k=packed[1],
        v=packed[2],
        g=packed[3],
        beta=packed[4].float(),
        initial_state=torch.cat(entries),
        output_final_state=True,
        safe_gate=lower_bound is not None,
        cu_seqlens=cu,
        state_v_first=True,
        **cu_cpu_kwargs(cu_cpu),
        **_cpr_contract(A_log, dt_bias, scale, lower_bound),
    )
    fs = fs.clone()
    snapshots = cpr_snapshot(fs)
    for j, (i, slot) in enumerate(ready):
        tracker.entry[slot] = fs[j : j + 1]
        # Ring mode allocates fresh storage: chunk_kda may still be reading
        # the old rows.
        tracker.open[slot] = (
            tracker._ring_empty(q[i], k[i], v[i], g[i], beta[i])
            if ring
            else tracker._empty()
        )
        result[i] = snapshots[j : j + 1]
    return result


def track(module, mixed, g, beta, indices, recurrent_state):
    q, k, v = [
        x.reshape(x.shape[0], -1, module.head_dim)
        for x in mixed.split(module.local_projection_size, dim=-1)
    ]
    # Reuse the step-shared host copy (RAON_KDA_CONV_IDX_CACHE; off = the
    # per-call tolist).  Lazy import: kda.py imports this module and is always
    # loaded first, so there is no import cycle.
    from vllm.models.raon1_mini.kda import _decode_conv_slots_host
    slots = [int(x) for x in _decode_conv_slots_host(module, indices)]
    states = append_batch(
        module._cpr,
        slots,
        q,
        k,
        v,
        g[0],
        beta[0],
        module.A_log,
        module.dt_bias,
        module._inkernel_scale,
        module.gate_lower_bound,
    )
    for slot, state in zip(slots, states):
        if state is not None:
            recurrent_state[slot] = state[0]

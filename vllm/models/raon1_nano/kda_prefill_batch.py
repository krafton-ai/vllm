"""Batch independent CPR entry chains while preserving each sequence recurrence."""

import os

import torch

import vllm.models.raon1_nano.kda_inkernel as ki

original = ki._fla_cpr_prefill_core_serial

_FUSED_ENV = "RAON_KDA_CPR_PREFILL_FUSED"
_FUSED_ANNOUNCED = False


def _fused_capable():
    """설치된 fla 가 중간상태 노출을 지원하는가 (기능 탐지 — 가정 금지)."""
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
    """Armed lever + unmet prerequisite = raise (fail-closed, charter rule 8).

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


def _batched_fused(
    q, k, v, g1, beta_raw, A_log, dt_bias, scale, lower_bound, entries0, bounds, chunk
):
    """RAON_KDA_CPR_PREFILL_FUSED=1 — wave 직렬 루프 소거판 (mb45 설계 A).

    mb45 실측(347075/347079): batched 의 wave 루프(청크 인덱스당 chunk_kda
    1회, 8K=128회)가 CPR prefill 26x 감속의 67%. 수리 = fla ``chunk_kda`` 의
    상류 기성 기능 ``return_intermediate_states`` (check-vendor-wheel 교훈):
    중간 64-경계 상태 h ([1, NT, HV, V(K), K(V)], **bf16 저장**)를 한 콜로
    노출한다.

    비트 계약 근거 3장:
      1) 커널 내부의 fp32 상태 전파는 wave 사슬과 비트 동일 —
         kda_inkernel.flush_state 문서의 B0s 정리("equals the plain
         whole-sequence chunk_kda fp32 final state bitwise").
      2) h 는 그 fp32 경계 상태의 bf16 하강 저장 = ``cpr_snapshot`` 의
         반올림 그 자체. 따라서 pool(스캔 초기상태) == wave pool 비트 동일.
      3) 트래커가 요구하는 **fp32** 마지막-경계 entry 는 h(bf16)로는 복원
         불가 → **cu 절단 트릭**: 각 시퀀스를 [main=64배수][tail] 두
         부분수열로 varlen 경계만 쪼개면(데이터 재배치 0) main 의
         final_state(fp32)가 정확히 그 entry 다. tail 부분수열은 0-초기로
         돌고 h/final/o 전부 버린다(≤63토큰 낭비).
    스캔(fused_recurrent, 출력 비트 정본)은 원문 그대로 — 입력(pool·cu)이
    비트 동일하므로 core 도 비트 동일. 게이트: mb46 사다리 (h==wave pool /
    e2e core·entry·seq_info, 첫-호출 소비 후 비교 — mb45 부검 규약).
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

    # ---- cu 절단: [main][tail] 부분수열 + 장부 ------------------------------
    cu_p = [0]  # 절단 varlen 경계 (물리 위치 그대로 — 재배치 0)
    init_rows = []  # 부분수열 초기상태 (fp32)
    main_sub = [None] * count  # seq -> main 부분수열 idx
    n_chunks = [0] * count  # seq 별 main 청크 수
    tail_len = [0] * count
    sub_chunks = []  # 부분수열별 청크 수 (h 행 오프셋)
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
            init_rows.append(zero_init)  # tail 부분수열: 산출 전량 폐기
            cu_p.append(e)
            sub_chunks.append(1)

    # 저비용 수확 (비트 중립 — 순서·반올림 지점 불변):
    #   beta_raw.float() / _cpr_contract 를 두 콜이 공유 (동일 입력 동일 연산
    #   의 중복 launch 1회 제거), cu 텐서는 site-3 헬퍼의 pinned 스테이징
    #   경유 (RAON_KDA_CU_CACHE OFF 면 원문 표현식 그대로 — 값·dtype 동일).
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
    del o_disc  # core 는 스캔이 정본 (wave 루프도 chunk 출력을 버린다)
    hf = h[0].float()  # [NT_sub, H, V(K), K(V)] — bf16 경계상태 = snapshot

    # ---- pool 조립: main 은 h 행 그대로, tail 은 snapshot(fs[main]) ---------
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
        entry = fs[main_sub[i] : main_sub[i] + 1].clone() if n_chunks[i] > 0 else e0(i)
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
    degenerate = count < 1 or any(bounds[i] >= bounds[i + 1] for i in range(count))
    # RAON_KDA_CPR_PREFILL_FUSED=1 (기본 OFF): wave 직렬 루프 소거판으로 위임.
    # 무장 시 전제(fla 기능 실재 + inference mode) 미충족이면 raise —
    # fail-closed (헌장 8; 조용한 wave 폴백은 성능 회귀를 숨긴다).
    # chunk != 64 / 퇴화 bounds 는 환경 결함이 아니라 조성 범위 밖이라 원문
    # 경로 그대로 (h 는 64-경계에서만 노출된다).  count == 1 도 흡수한다 —
    # _batched_fused 는 시퀀스 수에 무관하고, 단일 시퀀스도 wave 루프에서는
    # 청크 인덱스당 1콜 직렬이었다.
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
            bounds,
            chunk,
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
            # [P1] cu_seqlens site 2 of 3.  Lever OFF evaluates the original
            # expression verbatim; see RAON_KDA_CU_CACHE in kda_inkernel.py.
            cu, cu_cpu = ki.cu_seqlens_cached(
                lambda full=full: torch.arange(
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

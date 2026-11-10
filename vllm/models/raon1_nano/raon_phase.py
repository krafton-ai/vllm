# Copyright (c) 2026, KRAFTON.
# SPDX-License-Identifier: Apache-2.0
"""Optional phase-aware normalization adapters for dense Raon execution.

The Nano recipe leaves RAON_PREFILL_ALIGNED disabled. When enabled by an
internal caller, scheduler metadata distinguishes prefill from decode;
mixed batches use the aligned path for the whole forward. Graph capture
must not freeze a prefill decision into decode replay. No MoE sites exist."""

import os
from contextlib import suppress

import torch

__all__ = [
    "PREFILL_ALIGNED",
    "PREFILL_ALIGNED_VALUES",
    "prefill_now",
    "aligned",
    "announce_installed",
    "preflight",
    "install_metadata_adapter",
    "install_graph_dispatch_adapter",
]

PREFILL_ALIGNED = os.environ.get("RAON_PREFILL_ALIGNED", "0") == "1"

PREFILL_ALIGNED_VALUES = {
    "RAON_VLLM_QUACK_RMSNORM": "1",
    "RAON_MLA_LATENT_NORM_QUACK": "1",
    "RAON_MLA_OUTPUT_GATE_TWO_ROUND": "1",
    "RAON_MHC_TRAINER_FUSED": "1",
    "RAON_MHC_AGGNORM_FUSED": "1",
}

SITE_KEYS = (
    "norm.quack",
    "mla.latent_norm",
    "mla.output_gate",
    "mhc.site",
    "mhc.aggnorm",
)

_ANNOUNCED: set = set()
_INSTALLED_ANNOUNCED = False

# 진단 전용 (기본 "" = byte 동일): RAON_PREFILL_ALIGNED_SKIP_SITES=
# "site1,site2" — 나열된 site 는 게이트 ON prefill 에서도 aligned() 가
# False 를 돌려줘 그 site 만 fastest 실현을 유지한다. 위상 게이트의 축별
# 가격 귀속(1축 ptp A/B) 측정용 — armed 상태의 비트는 해당 site 에서
# trainer 비정렬이므로 게이트/경계 판정에 쓰면 안 된다 (SHADOW 류 진단
# 부류, 측정 라벨에 명기). 오탈자 site 는 import 시 fail-closed.
SKIP_SITES = frozenset(
    s for s in os.environ.get("RAON_PREFILL_ALIGNED_SKIP_SITES", "").split(",") if s
)
if SKIP_SITES - set(SITE_KEYS):
    raise RuntimeError(
        "RAON_PREFILL_ALIGNED_SKIP_SITES names unknown site(s) "
        f"{sorted(SKIP_SITES - set(SITE_KEYS))}; known sites: {SITE_KEYS} "
        "-- refusing a lever that silently skips nothing"
    )


def prefill_now() -> bool:
    """True iff (a) the gate is on, (b) we are not inside CUDA graph capture,
    (c) we have real forward metadata, and (d) that metadata says this
    forward is (or contains) prefill work. Cached once per ForwardContext so
    two evaluations in the same forward (design doc Section 1.2's load-bearing
    invariant: model.py's wiring decision and mhc.py's path-selection
    decision must agree) always agree -- recomputing per call site is
    explicitly forbidden by the design (Section 1.2: "캐시 없는 구현은 금지").

    Resolution order fails OPEN to fastest/decode ("모르면 현행"): any
    ambiguity (no PREFILL_ALIGNED, capturing, no forward context, no attn
    metadata) returns False, never True.
    """
    if not PREFILL_ALIGNED:
        return False
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    try:
        from vllm.forward_context import get_forward_context

        ctx = get_forward_context()
    except Exception:
        return False
    if ctx is None or getattr(ctx, "attn_metadata", None) is None:
        return False  # profile/dummy run -- same guard mla_cache_only.py uses
    cached = getattr(ctx, "_raon_phase", None)
    if cached is not None:
        return cached

    verdict = _resolve_phase(ctx)

    # Slotted contexts can recompute this pure verdict instead of caching it.
    with suppress(AttributeError, TypeError):
        ctx._raon_phase = verdict
    return verdict


def _iter_metadata(attn_metadata):
    """Yield every per-layer metadata object. vLLM's ForwardContext carries
    ``attn_metadata`` as a dict in the common case but as ``list[dict]`` under
    DBO/micro-batching (vllm/forward_context.py) -- treating the list itself
    as one opaque object made every counter read 0 and the whole gate a
    silent no-op (review 2026-09-22 MEDIUM). Flatten both shapes."""
    groups = attn_metadata if isinstance(attn_metadata, list) else [attn_metadata]
    for group in groups:
        if isinstance(group, dict):
            yield from group.values()
        else:
            yield group


def install_metadata_adapter() -> None:
    """Preserve the scheduler's request phase before KDA groups spec tokens.

    The upstream KDA metadata counters describe kernel layout, not whether a request
    is still consuming prompt tokens. In particular, speculative verification
    has several query tokens but no true prefill. The original CPU flag is
    available to the builder and is deliberately reduced to two counts here;
    no GPU readback or vLLM site-package edit is required.
    """
    if not PREFILL_ALIGNED:
        return
    from contextvars import ContextVar
    from importlib import import_module

    kda_metadata = import_module("vllm.models.raon1_nano.rda_metadata")

    RaonRDAMetadataBuilder = kda_metadata.RaonRDAMetadataBuilder

    original = RaonRDAMetadataBuilder.build
    if getattr(original, "_raon_true_prefill_adapter", False):
        return
    original_split = kda_metadata.split_decodes_and_prefills
    active_scope = ContextVar("raon_aligned_kda_builder", default=False)

    def split(
        common,
        decode_threshold=1,
        require_uniform=False,
        treat_short_extends_as_decodes=True,
    ):
        # A real one-token prompt must remain prefill. The stock KDA builder
        # otherwise labels it decode while the Raon phase gate labels the
        # same forward prefill, so metadata and selected kernels disagree.
        return original_split(
            common,
            decode_threshold,
            require_uniform,
            treat_short_extends_as_decodes=False
            if active_scope.get()
            else treat_short_extends_as_decodes,
        )

    def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
        token = active_scope.set(
            getattr(self.vllm_config.model_config.hf_config, "model_type", None)
            == "raon1_nano"
        )
        try:
            result = original(
                self, common_prefix_len, common_attn_metadata, *args, **kwargs
            )
        finally:
            active_scope.reset(token)
        common = common_attn_metadata
        flags = common.is_prefilling
        query_lens = common.query_start_loc_cpu.diff()
        if flags is None or flags.device.type != "cpu" or len(flags) != len(query_lens):
            raise RuntimeError(
                "RAON_PREFILL_ALIGNED requires CPU per-request is_prefilling metadata"
            )
        active = query_lens > 0  # exclude CUDA graph padding
        result._raon_active_requests = int(active.sum().item())
        result._raon_true_prefills = int((flags & active).sum().item())
        return result

    build._raon_true_prefill_adapter = True
    RaonRDAMetadataBuilder.build = build
    kda_metadata.split_decodes_and_prefills = split


def install_graph_dispatch_adapter() -> None:
    """Keep true prefill out of V2's shape-compatible varlen decode graphs.

    A short prompt can have <= 1+num_speculative_tokens query tokens. V2's
    varlen graph matcher bounds query length but does not carry has_prefill;
    it can therefore replay a graph captured with decode-only operations.
    Carry the already-CPU scheduler flag to dispatch and request eager before
    DP synchronization. Pure decode/spec and dummy capture remain unchanged.
    """
    if not PREFILL_ALIGNED:
        return
    from vllm.v1.worker.gpu import model_runner as runner_module

    runner_cls = runner_module.GPUModelRunner
    original_gather = runner_cls.gather_batch_req_state
    if getattr(original_gather, "_raon_prefill_graph_guard", False):
        return
    original_dispatch = runner_module.dispatch_cg_and_sync_dp

    def gather(self, *args, **kwargs):
        state, uniform = original_gather(self, *args, **kwargs)
        manager = self.cudagraph_manager
        if manager is not None:
            manager._raon_has_prefill = bool(
                getattr(self.model_config.hf_config, "model_type", None) == "raon1_nano"
                and state is not None
                and state.has_prefill
            )
        return state, uniform

    def dispatch(
        manager,
        num_reqs,
        num_tokens,
        uniform_token_count,
        dp_size,
        dp_rank,
        max_query_len=None,
        need_eager=False,
        num_active_loras=0,
    ):
        true_prefill = getattr(manager, "_raon_has_prefill", False)
        if true_prefill and "prefill-graph-guard" not in _ANNOUNCED:
            _ANNOUNCED.add("prefill-graph-guard")
            print(
                "[parity] RAON true prefill excluded from V2 decode graph "
                f"dispatch (max_query_len={max_query_len})",
                flush=True,
            )
        return original_dispatch(
            manager,
            num_reqs,
            num_tokens,
            uniform_token_count,
            dp_size,
            dp_rank,
            max_query_len=max_query_len,
            need_eager=need_eager or true_prefill,
            num_active_loras=num_active_loras,
        )

    gather._raon_prefill_graph_guard = True
    runner_cls.gather_batch_req_state = gather
    runner_module.dispatch_cg_and_sync_dp = dispatch


def install_mla_phase_adapter() -> None:
    """Do not classify a short prompt as MLA decode merely by query length."""
    if not PREFILL_ALIGNED:
        return
    from contextvars import ContextVar

    from vllm.model_executor.layers.attention import mla_attention as mla

    original_build = mla.MLACommonMetadataBuilder.build
    if getattr(original_build, "_raon_true_prefill_adapter", False):
        return
    original_split = mla.split_decodes_and_prefills
    active = ContextVar("raon_aligned_mla_builder", default=False)

    def build(self, *args, **kwargs):
        token = active.set(
            getattr(self.model_config.hf_config, "model_type", None) == "raon1_nano"
        )
        try:
            return original_build(self, *args, **kwargs)
        finally:
            active.reset(token)

    def split(
        common,
        decode_threshold=1,
        require_uniform=False,
        treat_short_extends_as_decodes=True,
    ):
        return original_split(
            common,
            decode_threshold,
            require_uniform,
            treat_short_extends_as_decodes=False
            if active.get()
            else treat_short_extends_as_decodes,
        )

    build._raon_true_prefill_adapter = True
    mla.MLACommonMetadataBuilder.build = build
    mla.split_decodes_and_prefills = split


def _resolve_phase(ctx) -> bool:
    bd = getattr(ctx, "batch_descriptor", None)
    if bd is not None and getattr(bd, "uniform", False):
        return False  # uniform decode shape (vllm/v1/cudagraph_dispatcher.py)
    adapted = [
        m
        for m in _iter_metadata(ctx.attn_metadata)
        if m is not None and hasattr(m, "_raon_true_prefills")
    ]
    if not adapted:
        raise RuntimeError(
            "RAON_PREFILL_ALIGNED missing true-prefill KDA metadata adapter"
        )
    phase_counts = {
        (m._raon_true_prefills, m._raon_active_requests)
        for m in adapted
        if m._raon_active_requests > 0
    }
    if not phase_counts:
        return False  # padding/dummy forward, no active requests
    if len(phase_counts) != 1:
        raise RuntimeError(
            f"RAON_PREFILL_ALIGNED inconsistent layer phases: {phase_counts}"
        )
    true_prefills, active = phase_counts.pop()
    if true_prefills > active or true_prefills < 0:
        raise RuntimeError("RAON_PREFILL_ALIGNED invalid true-prefill count")
    if 0 < true_prefills < active and "mixed-aligned" not in _ANNOUNCED:
        _ANNOUNCED.add("mixed-aligned")
        print(
            "[parity] RAON_PREFILL_ALIGNED mixed batch: all rows use "
            "aligned norm/MHC/MoE; request/cache order preserved",
            flush=True,
        )
    return true_prefills > 0


def _phase_detail() -> str:
    try:
        from vllm.forward_context import get_forward_context

        ctx = get_forward_context()
    except Exception:
        return "forward context unavailable"
    if ctx is None or getattr(ctx, "attn_metadata", None) is None:
        return "no attn_metadata"
    metas = [m for m in _iter_metadata(ctx.attn_metadata) if m is not None]
    n_pf = sum(int(getattr(m, "_raon_true_prefills", 0) or 0) for m in metas)
    n_dec = sum(int(getattr(m, "num_decode_tokens", 0) or 0) for m in metas)
    capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    return f"true_prefills={n_pf}, num_decode_tokens={n_dec}, capturing={capturing}"


def aligned(site: str) -> bool:
    """Per-call-site phase predicate. Returns the SAME boolean as every other
    call to ``aligned(...)`` (or ``prefill_now()``) inside the current
    forward, regardless of ``site`` -- ``site`` only selects the LIVE
    announce's identity/log line, never the verdict. Announces once per
    (site, verdict) pair (rows_reach_min_m's pattern, mxfp8_align.py:95)."""
    verdict = prefill_now()
    if not PREFILL_ALIGNED:
        return verdict
    if verdict and site in SKIP_SITES:
        # 진단 skip (기본 빈 값 = 이 분기 불가침): 이 site 만 prefill 에서도
        # fastest 실현을 유지. LIVE 라벨을 별도로 찍어 관문이 armed 여부를
        # 로그 실물로 판별할 수 있게 한다 (MINE-26 env≠발화).
        skey = (site, "skipped")
        if skey not in _ANNOUNCED:
            _ANNOUNCED.add(skey)
            print(
                f"[parity] RAON_PREFILL_ALIGNED_SKIP_SITES DIAGNOSTIC: "
                f"site={site} phase=PREFILL -> fastest realization FORCED "
                f"(bits NOT trainer-aligned at this site; price-attribution "
                f"measurement only)",
                flush=True,
            )
        return False
    key = (site, verdict)
    if key not in _ANNOUNCED:
        _ANNOUNCED.add(key)
        phase = "PREFILL" if verdict else "DECODE"
        detail = _phase_detail()
        outcome = (
            "takes the zero(prod) realization for this site"
            if verdict
            else "fastest realization unchanged for this site"
        )
        print(
            f"[parity] RAON_PREFILL_ALIGNED=1 LIVE: site={site} "
            f"phase={phase} ({detail}) -> {outcome}",
            flush=True,
        )
    return verdict


_LEVER_ENV_DEFAULTS = {
    "RAON_VLLM_QUACK_RMSNORM": "1",
    "RAON_MLA_LATENT_NORM_QUACK": "0",
    "RAON_MLA_OUTPUT_GATE_TWO_ROUND": "0",
    "RAON_MHC_TRAINER_FUSED": "0",
    "RAON_MHC_AGGNORM_FUSED": "0",
}


def announce_installed() -> None:
    """Boot-time (1x) INSTALLED announce (design doc Section 3.1, insertion
    point #2). No-op when the gate is off. Fail-closed preflight
    (``preflight()``) must run before this, from ``plugin.py``'s
    ``register()``."""
    global _INSTALLED_ANNOUNCED
    if not PREFILL_ALIGNED or _INSTALLED_ANNOUNCED:
        return
    _INSTALLED_ANNOUNCED = True
    lines = []
    for name, target in PREFILL_ALIGNED_VALUES.items():
        current = os.environ.get(name, _LEVER_ENV_DEFAULTS.get(name, "0"))
        lines.append(f"    {name:<30} {current} -> {target}")
    body = "\n".join(lines)
    print(
        "[parity] RAON_PREFILL_ALIGNED=1 INSTALLED: phase-split alignment "
        f"armed for {len(PREFILL_ALIGNED_VALUES)} shared consumption sites; "
        "prefill takes the zero(prod) realization, decode keeps fastest. "
        f"Levers phase-split (env -> prefill value):\n{body}\n"
        "  phase signal = CPU CommonAttentionMetadata.is_prefilling via KDA "
        "adapter (zero-query padding excluded); mixed request phases "
        "use aligned operations for all rows; full decode capture resolves "
        "to DECODE and aligned forwards bypass local MoE graphs.",
        flush=True,
    )


def preflight() -> None:
    """Fail-closed composition legality checks (design doc Section 1.0
    insertion point #2, Section 5 R2/R4). No-op when the gate is off --
    every check here is about what happens when the gate is ON, so an OFF
    arm must never pay for it or be able to fail it."""
    if not PREFILL_ALIGNED:
        return

    # R2(i)/(iii): v1 is FULL_DECODE_ONLY-or-NONE capture only. A piecewise/
    # breakable capture region could capture a PREFILL shape too, freezing the
    # phase branch's verdict into the graph forever (capture-hazard-map's
    # "quiet value freeze" class) -- and the piecewise dispatcher RELAXES
    # uniform-decode keys onto the same entry as mixed batches
    # (cudagraph_dispatcher.py dispatch(): replace(uniform=False)), so under
    # PIECEWISE the frozen verdict is wrong for one of the two phases by
    # construction. Refuse rather than silently mismeasure.
    if os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH", "0") == "1":
        raise RuntimeError(
            "RAON_PREFILL_ALIGNED=1 with VLLM_USE_BREAKABLE_CUDAGRAPH=1: v1 "
            "of this gate is FULL_DECODE_ONLY-capture only (design doc "
            "Section 5 R2). A breakable/piecewise capture region can record "
            "a prefill shape, which would freeze the phase branch's verdict "
            "into the graph instead of re-evaluating it eagerly every "
            "prefill call. Refusing rather than risk a silent value-freeze."
        )
    cudagraph_mode = os.environ.get("RAON_EXPECT_CUDAGRAPH_MODE")
    if cudagraph_mode is not None and cudagraph_mode not in (
        "FULL_DECODE_ONLY",
        "NONE",
    ):
        raise RuntimeError(
            f"RAON_PREFILL_ALIGNED=1 with RAON_EXPECT_CUDAGRAPH_MODE="
            f"{cudagraph_mode!r}: v1 of this gate only has evidence for "
            "FULL_DECODE_ONLY and NONE capture (design doc Section 5 R2); "
            "refusing an unverified capture mode rather than silently "
            "mismeasuring."
        )

    aggnorm_raw = os.environ.get("RAON_MHC_AGGNORM_FUSED", "0") == "1"
    trainer_fused_raw = os.environ.get("RAON_MHC_TRAINER_FUSED", "0") == "1"
    if aggnorm_raw and not trainer_fused_raw:
        raise RuntimeError(
            "RAON_PREFILL_ALIGNED=1 with RAON_MHC_AGGNORM_FUSED=1 and "
            "RAON_MHC_TRAINER_FUSED=0: this raw combination is illegal on "
            "its own (mhc.py's import-time check), but the gate ORs both "
            "module constants with PREFILL_ALIGNED so that check no longer "
            "fires. Refusing here instead of letting decode-phase forwards "
            "raise deep inside MHCSite.forward."
        )

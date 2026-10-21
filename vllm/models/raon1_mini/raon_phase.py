# Copyright (c) 2026, KRAFTON.
"""RAON_PREFILL_ALIGNED -- phase-split trainer-alignment gate.

Motivation: engine/trainer divergence is much larger on the first generated
token (prefill) than on later tokens (decode), and the levers that close it
live in code shared by both phases (norm / mHC / MoE expert chain). This
module is the single place that decides, once per forward and cached on the
vLLM ForwardContext, whether "now" is prefill or decode, and what value each
of those levers should take in each phase.

Default OFF (``RAON_PREFILL_ALIGNED`` unset or "0"): ``prefill_now()``'s
first line returns False before touching CUDA or the forward context, so
every call site that reads ``phase.aligned(...)`` executes one
already-evaluated boolean and NOTHING else -- the engine stays
byte-identical to running without this gate.

Decode is not moved by this gate in the supported composition (pure-prefill
or unmixed batches, FULL_DECODE_ONLY capture, packing OFF): every call site
keeps the existing raw-env condition on one side of an ``or``/``and not`` so
a decode forward (or any forward with the gate off) reduces to exactly what
the tree already does. The predicate is per-forward, and speculative
verification can add query tokens without a true prefill. The KDA builder
adapter preserves the scheduler's per-request prefill signal; mixed
prefill/decode runs the aligned realization for the entire forward. This
keeps request/cache row order intact and costs extra aligned work on decode
rows; pure decode and spec verification keep their fast path.
Every site is edited as ``existing_condition or aligned(site)`` (lever OFF
by default, ON aligned) or ``existing_condition and not aligned(site)``
(lever ON by default, OFF aligned). Decode capture regions resolve to False
by construction: capture must be FULL_DECODE_ONLY (see ``preflight()``), so
``torch.cuda.is_current_stream_capturing()`` bakes False into the one Python
evaluation the capture pass makes.
"""
import os

import torch

__all__ = [
    "PREFILL_ALIGNED",
    "PREFILL_ALIGNED_VALUES",
    "prefill_now",
    "aligned",
    "combine_ep_value",
    "announce_installed",
    "preflight",
    "install_metadata_adapter",
    "install_graph_dispatch_adapter",
]

PREFILL_ALIGNED = os.environ.get("RAON_PREFILL_ALIGNED", "0") == "1"

# Prefill target values for the 9 shared-consumption levers. This is the ONLY
# place these targets are written down; every call site reads through
# ``aligned(site)`` (boolean levers) or ``combine_ep_value()`` (the one
# value-typed lever) rather than re-deciding the target locally, so a
# half-armed gate (some sites see the aligned value, others don't) cannot
# happen by construction.
PREFILL_ALIGNED_VALUES = {
    "RAON_VLLM_QUACK_RMSNORM": "1",
    "RAON_MLA_LATENT_NORM_QUACK": "1",
    "RAON_MLA_OUTPUT_GATE_TWO_ROUND": "1",
    "RAON_MHC_TRAINER_FUSED": "1",
    "RAON_MHC_AGGNORM_FUSED": "1",
    "RAON_MOE_RENORM_REALIZED": "1",
    "RAON_MOE_MXFP8_W8A8_TRITON": "0",
    "RAON_MOE_MXFP8_W8A8_FINALIZE": "0",
    "RAON_MOE_COMBINE_EP_REALIZED": "2",
}

# Distinct instrumented site keys. RAON_MOE_MXFP8_W8A8_TRITON and
# RAON_MOE_MXFP8_W8A8_FINALIZE share ONE site key ("moe.expert_chain") by
# design so the two branches cannot split within one phase -- both levers are
# one phase decision (enter the trtllm mxw8a8 chain or not), so there are 8
# distinct site keys behind the 9 levers. A prefill-only boot emits at most 8
# distinct "LIVE: site=... phase=PREFILL" lines.
SITE_KEYS = (
    "norm.quack",
    "mla.latent_norm",
    "mla.output_gate",
    "mhc.site",
    "mhc.aggnorm",
    "moe.renorm",
    "moe.expert_chain",
    "moe.combine_ep",
)

_ANNOUNCED: set = set()
_INSTALLED_ANNOUNCED = False

# Diagnostic only (default "" = no change): RAON_PREFILL_ALIGNED_SKIP_SITES=
# "site1,site2" makes aligned() return False for the listed sites even in
# prefill with the gate on, so those sites keep the fastest realization. Used
# to attribute the cost of the gate per site; results are not trainer-aligned
# at the skipped sites. Unknown site names fail closed at import.
SKIP_SITES = frozenset(
    s for s in os.environ.get(
        "RAON_PREFILL_ALIGNED_SKIP_SITES", "").split(",") if s)
if SKIP_SITES - set(SITE_KEYS):
    raise RuntimeError(
        "RAON_PREFILL_ALIGNED_SKIP_SITES names unknown site(s) "
        f"{sorted(SKIP_SITES - set(SITE_KEYS))}; known sites: {SITE_KEYS} "
        "-- refusing a lever that silently skips nothing")


def prefill_now() -> bool:
    """True iff (a) the gate is on, (b) we are not inside CUDA graph capture,
    (c) we have real forward metadata, and (d) that metadata says this
    forward is (or contains) prefill work. Cached once per ForwardContext so
    two evaluations in the same forward (model.py's wiring decision and
    mhc.py's path-selection decision) always agree -- do not recompute per
    call site.

    Resolution fails OPEN to fastest/decode: any ambiguity (no
    PREFILL_ALIGNED, capturing, no forward context, no attn metadata)
    returns False, never True.
    """
    if not PREFILL_ALIGNED:
        return False
    # Capture region is FULL_DECODE_ONLY (preflight() enforces it) -- the
    # branch is on a python-visible boolean evaluated once at capture time,
    # and always resolves to DECODE while capturing.
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    try:
        from vllm.forward_context import get_forward_context

        ctx = get_forward_context()
    except Exception:
        # Import-time / non-engine-process probes: no vLLM forward context
        # machinery reachable -> fail open to "not prefill" rather than
        # raising out of a hot path.
        return False
    if ctx is None or getattr(ctx, "attn_metadata", None) is None:
        return False  # profile/dummy run -- same guard mla_cache_only.py uses
    cached = getattr(ctx, "_raon_phase", None)
    if cached is not None:
        return cached

    verdict = _resolve_phase(ctx)

    try:
        ctx._raon_phase = verdict
    except (AttributeError, TypeError):
        # ForwardContext on this vLLM build refuses arbitrary attributes
        # (e.g. a slotted dataclass). Recomputing per call is a performance
        # regression, not a correctness one -- _resolve_phase is a pure
        # function of ctx's own metadata, so every call in the same forward
        # still agrees. Do not add a second cache keyed on something
        # fragile.
        pass
    return verdict


def _iter_metadata(attn_metadata):
    """Yield every per-layer metadata object. vLLM's ForwardContext carries
    ``attn_metadata`` as a dict in the common case but as ``list[dict]`` under
    DBO/micro-batching (vllm/forward_context.py) -- treating the list itself
    as one opaque object would make every counter read 0 and the whole gate a
    silent no-op. Flatten both shapes."""
    groups = attn_metadata if isinstance(attn_metadata, list) else [attn_metadata]
    for group in groups:
        if isinstance(group, dict):
            yield from group.values()
        else:
            yield group


def install_metadata_adapter() -> None:
    """Preserve the scheduler's request phase before KDA groups spec tokens.

    Kimi's KDA metadata counters describe kernel layout, not whether a request
    is still consuming prompt tokens. In particular, speculative verification
    has several query tokens but no true prefill. The original CPU flag is
    available to the builder and is deliberately reduced to two counts here;
    no GPU readback or vLLM site-package edit is required.
    """
    if not PREFILL_ALIGNED:
        return
    from contextvars import ContextVar
    from . import rda_metadata as kda_metadata

    RaonRDAMetadataBuilder = kda_metadata.RaonRDAMetadataBuilder

    original = RaonRDAMetadataBuilder.build
    if getattr(original, "_raon_true_prefill_adapter", False):
        return
    original_split = kda_metadata.split_decodes_and_prefills
    active_scope = ContextVar("raon_aligned_kda_builder", default=False)

    def split(common, decode_threshold=1, require_uniform=False,
              treat_short_extends_as_decodes=True):
        # A real one-token prompt must remain prefill. The stock KDA builder
        # otherwise labels it decode while the Raon phase gate labels the
        # same forward prefill, so metadata and selected kernels disagree.
        return original_split(
            common, decode_threshold, require_uniform,
            treat_short_extends_as_decodes=False if active_scope.get()
            else treat_short_extends_as_decodes,
        )

    def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
        token = active_scope.set(
            getattr(self.vllm_config.model_config.hf_config, "model_type", None)
            in ("raon1_mini", "raon_v1_1")
        )
        try:
            result = original(self, common_prefix_len, common_attn_metadata, *args, **kwargs)
        finally:
            active_scope.reset(token)
        common = common_attn_metadata
        flags = common.is_prefilling
        query_lens = common.query_start_loc_cpu.diff()
        if flags is None or flags.device.type != "cpu" or len(flags) != len(query_lens):
            raise RuntimeError("RAON_PREFILL_ALIGNED requires CPU per-request is_prefilling metadata")
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
                getattr(self.model_config.hf_config, "model_type", None) in ("raon1_mini", "raon_v1_1")
                and state is not None and state.has_prefill
            )
        return state, uniform

    def dispatch(manager, num_reqs, num_tokens, uniform_token_count, dp_size,
                 dp_rank, max_query_len=None, need_eager=False, num_active_loras=0):
        true_prefill = getattr(manager, "_raon_has_prefill", False)
        if true_prefill and "prefill-graph-guard" not in _ANNOUNCED:
            _ANNOUNCED.add("prefill-graph-guard")
            print("[parity] RAON true prefill excluded from V2 decode graph "
                  f"dispatch (max_query_len={max_query_len})", flush=True)
        return original_dispatch(
            manager, num_reqs, num_tokens, uniform_token_count, dp_size, dp_rank,
            max_query_len=max_query_len, need_eager=need_eager or true_prefill,
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
            getattr(self.model_config.hf_config, "model_type", None) in ("raon1_mini", "raon_v1_1")
        )
        try:
            return original_build(self, *args, **kwargs)
        finally:
            active.reset(token)

    def split(common, decode_threshold=1, require_uniform=False,
              treat_short_extends_as_decodes=True):
        return original_split(
            common, decode_threshold, require_uniform,
            treat_short_extends_as_decodes=False if active.get()
            else treat_short_extends_as_decodes,
        )

    build._raon_true_prefill_adapter = True
    mla.MLACommonMetadataBuilder.build = build
    mla.split_decodes_and_prefills = split


def _resolve_phase(ctx) -> bool:
    bd = getattr(ctx, "batch_descriptor", None)
    if bd is not None and getattr(bd, "uniform", False):
        return False  # uniform decode shape (vllm/v1/cudagraph_dispatcher.py)
    adapted = [m for m in _iter_metadata(ctx.attn_metadata)
               if m is not None and hasattr(m, "_raon_true_prefills")]
    if not adapted:
        raise RuntimeError("RAON_PREFILL_ALIGNED missing true-prefill KDA metadata adapter")
    phase_counts = {
        (m._raon_true_prefills, m._raon_active_requests)
        for m in adapted if m._raon_active_requests > 0
    }
    if not phase_counts:
        return False  # padding/dummy forward, no active requests
    if len(phase_counts) != 1:
        raise RuntimeError(f"RAON_PREFILL_ALIGNED inconsistent layer phases: {phase_counts}")
    true_prefills, active = phase_counts.pop()
    if true_prefills > active or true_prefills < 0:
        raise RuntimeError("RAON_PREFILL_ALIGNED invalid true-prefill count")
    if 0 < true_prefills < active:
        if "mixed-aligned" not in _ANNOUNCED:
            _ANNOUNCED.add("mixed-aligned")
            print("[parity] RAON_PREFILL_ALIGNED mixed batch: all rows use "
                  "aligned norm/MHC/MoE; request/cache order preserved", flush=True)
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
    (site, verdict) pair (same pattern as mxfp8_align.rows_reach_min_m)."""
    verdict = prefill_now()
    if not PREFILL_ALIGNED:
        # Byte-identical-when-OFF extends to the LOG, not just the tensors.
        # Every OR-pattern call site (``raw_env or aligned(site)``) reaches
        # this function on every OFF forward whose raw env is False, so
        # without this guard the first such call would print an extra
        # "RAON_PREFILL_ALIGNED=1 LIVE" line in a run that never turned the
        # gate on.
        return verdict
    if verdict and site in SKIP_SITES:
        # Diagnostic skip (unreachable with the default empty value): this
        # site keeps the fastest realization even in prefill. A separate LIVE
        # label is logged so the log shows which sites were actually skipped.
        skey = (site, "skipped")
        if skey not in _ANNOUNCED:
            _ANNOUNCED.add(skey)
            print(f"[parity] RAON_PREFILL_ALIGNED_SKIP_SITES DIAGNOSTIC: "
                  f"site={site} phase=PREFILL -> fastest realization FORCED "
                  f"(bits NOT trainer-aligned at this site; diagnostic "
                  f"measurement only)", flush=True)
        return False
    key = (site, verdict)
    if key not in _ANNOUNCED:
        _ANNOUNCED.add(key)
        phase = "PREFILL" if verdict else "DECODE"
        detail = _phase_detail()
        outcome = ("takes the trainer-aligned realization for this site"
                   if verdict else
                   "fastest realization unchanged for this site")
        print(f"[parity] RAON_PREFILL_ALIGNED=1 LIVE: site={site} "
              f"phase={phase} ({detail}) -> {outcome}", flush=True)
    return verdict


def combine_ep_value(env_fallback: int) -> int:
    """Value-typed counterpart of ``aligned()`` for
    RAON_MOE_COMBINE_EP_REALIZED: prefill reads the table value, decode (or
    gate OFF) keeps the caller-supplied env-resolved value unchanged.
    Announces through the shared ``aligned()`` site so the LIVE line count
    stays comparable to the boolean levers."""
    if aligned("moe.combine_ep"):
        return int(PREFILL_ALIGNED_VALUES["RAON_MOE_COMBINE_EP_REALIZED"])
    return env_fallback


_LEVER_ENV_DEFAULTS = {
    "RAON_VLLM_QUACK_RMSNORM": "1",
    "RAON_MLA_LATENT_NORM_QUACK": "0",
    "RAON_MLA_OUTPUT_GATE_TWO_ROUND": "0",
    "RAON_MHC_TRAINER_FUSED": "0",
    "RAON_MHC_AGGNORM_FUSED": "0",
    "RAON_MOE_RENORM_REALIZED": "0",
    "RAON_MOE_MXFP8_W8A8_TRITON": "0",
    "RAON_MOE_MXFP8_W8A8_FINALIZE": "0",
    "RAON_MOE_COMBINE_EP_REALIZED": "0",
}


def announce_installed() -> None:
    """Boot-time (1x) INSTALLED announce. No-op when the gate is off.
    Fail-closed preflight (``preflight()``) must run before this, from
    ``plugin.py``'s ``register()``."""
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
        "prefill takes the trainer-aligned realization, decode keeps fastest. "
        f"Levers phase-split (env -> prefill value):\n{body}\n"
        "  phase signal = CPU CommonAttentionMetadata.is_prefilling via KDA "
        "adapter (zero-query padding excluded); mixed request phases "
        "use aligned operations for all rows; full decode capture resolves "
        "to DECODE and aligned forwards bypass local MoE graphs.",
        flush=True,
    )


def preflight() -> None:
    """Fail-closed composition legality checks. No-op when the gate is off
    -- every check here is about what happens when the gate is ON, so an OFF
    run must never pay for it or be able to fail it."""
    if not PREFILL_ALIGNED:
        return

    # Capture must be FULL_DECODE_ONLY or NONE. A piecewise/breakable capture
    # region could capture a PREFILL shape too, freezing the phase branch's
    # verdict into the graph -- and the piecewise dispatcher RELAXES
    # uniform-decode keys onto the same entry as mixed batches
    # (cudagraph_dispatcher.py dispatch(): replace(uniform=False)), so under
    # PIECEWISE the frozen verdict is wrong for one of the two phases by
    # construction. Refuse rather than silently mismeasure.
    if os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH", "0") == "1":
        raise RuntimeError(
            "RAON_PREFILL_ALIGNED=1 with VLLM_USE_BREAKABLE_CUDAGRAPH=1: "
            "this gate supports FULL_DECODE_ONLY capture only. A "
            "breakable/piecewise capture region can record a prefill shape, "
            "which would freeze the phase branch's verdict into the graph "
            "instead of re-evaluating it eagerly every prefill call. Refusing "
            "rather than risk a silent value-freeze."
        )
    # NONE is as safe as FULL_DECODE_ONLY -- with no capture region at all the
    # phase predicate is evaluated eagerly on every forward, so there is no
    # graph to freeze a verdict into (consistent with _cudagraph_readback in
    # model.py).
    cudagraph_mode = os.environ.get("RAON_EXPECT_CUDAGRAPH_MODE")
    if cudagraph_mode is not None and cudagraph_mode not in (
            "FULL_DECODE_ONLY", "NONE"):
        raise RuntimeError(
            f"RAON_PREFILL_ALIGNED=1 with RAON_EXPECT_CUDAGRAPH_MODE="
            f"{cudagraph_mode!r}: this gate only supports FULL_DECODE_ONLY "
            "and NONE capture; refusing an unverified capture mode rather "
            "than silently mismeasuring."
        )

    # mhc.py's fail-closed pair (AGGNORM requires TRAINER_FUSED) is checked
    # against module constants that this gate ORs with PREFILL_ALIGNED, so
    # BOTH become True process-wide once the gate is on, regardless of their
    # raw env values -- which would silently let an illegal raw combination
    # (AGGNORM=1, TRAINER_FUSED=0) through the OR'd check. Re-check the RAW
    # envs here so that combination still fails closed under the gate.
    aggnorm_raw = os.environ.get("RAON_MHC_AGGNORM_FUSED", "0") == "1"
    trainer_fused_raw = os.environ.get("RAON_MHC_TRAINER_FUSED", "0") == "1"
    if aggnorm_raw and not trainer_fused_raw:
        raise RuntimeError(
            "RAON_PREFILL_ALIGNED=1 with RAON_MHC_AGGNORM_FUSED=1 and "
            "RAON_MHC_TRAINER_FUSED=0: this raw combination is illegal on "
            "its own (mhc.py's import-time check), but the gate ORs both "
            "module constants with PREFILL_ALIGNED so that check no longer "
            "fires. Refusing here instead of letting decode-phase forwards "
            "raise deep inside RaonMHC.forward."
        )

    # The gate makes prefill run the trainer-structured two-stage EP combine
    # (COMBINE_EP=2 via combine_ep_value()) even when the raw env is "0" --
    # so moe_align's import-time fail-closed block (`if combine_ep_realized():
    # require trainer_realized; _self_check_combine_ep()`) evaluates False at
    # import and never fires for the realization that will actually execute.
    # Re-run BOTH checks here, exactly as the import-time block would have.
    # Lazy import: moe_align imports this module at its own import time.
    from vllm.models.raon1_mini import moe_align
    if not moe_align.trainer_realized():
        raise RuntimeError(
            "RAON_PREFILL_ALIGNED=1 forces RAON_MOE_COMBINE_EP_REALIZED=2 "
            "in prefill, which replaces the combine of the trainer-"
            "structured chain and is meaningless without "
            "RAON_MOE_TRAINER_REALIZED=1 -- the same dependency moe_align "
            "enforces at import when the raw env enables the lever."
        )
    with torch.device("cpu"):
        moe_align._self_check_combine_ep()

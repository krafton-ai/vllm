"""Emit-only leaf for the engine/trainer parity manifest.

Each side (inference engine and trainer) writes a JSON manifest describing
the bit-relevant realization flags it ran with. The two manifests are
compared by a separate check() gate, which owns the full flag census and
schema. Each manifest records contract_module_sha = sha256 of the emitting
module's own source, so the engine and trainer copies of this file must be
byte-identical for the gate to pass.

Stdlib-only leaf: importing this module must succeed in every process that
loads the vLLM plugin, including ones that never build a model.

TE PRECISION SCOPE (te_precision_scope / te_precision_scope_sha256)
-------------------------------------------------------------------
The manifest records the realized module -> precision config decisions, not
the config path: the path says which file was opened, only the decisions say
which modules ran quantized. Without this, an all-bf16 trainer map and an
MXFP8 trainer map produce the same composition hash.

On the trainer, the precision recipe's config matcher calls
record_te_precision() for every TransformerEngine module it is consulted
about. The engine has no TransformerEngine modules, so its registry stays
empty and it emits "none"; a trainer whose scored-path modules are all bf16
also emits "none" and the pair matches. A trainer that leaves anything on the
global (mxfp8/nvfp4) recipe emits the module kinds and counts, which cannot
match "none", and the gate fails.

Two deliberate properties:

* FAIL CLOSED. A trainer that recorded nothing emits "UNOBSERVED", not
  "none". A trainer that never observed a single TE module has not shown it
  is bf16, and a silently uninstalled hook must never look like a clean run.
* mtp.* IS EXCLUDED, by name, in _UNSCORED_PREFIXES. The MTP block feeds no
  compared output and the engine does not implement it, so requiring the two
  sides to agree about it would always fail.

BATCH-INVARIANCE TRANSFORMS
---------------------------
The flags below are grouped by which batch transform they make invariant.
Invariance under T1-T6 needs no reference implementation: only the batch
layout moves while kernel, dtype, data and device are fixed, so the
threshold is bitwise, not a tolerance.

  T1 batch population   -> batch_invariant.matmul_persistent (fixed schedule
                           at any M); RAON_HEAD_GEMM_PERSISTENT (head GEMM)
  T2 position in batch  -> per-row kernels (logprob_invariant.py, one program
                           per row, constant block)
  T3 neighbour content  -> same per-row property as T2
  T4 padding layout     -> RAON_MOE_MXFP8_STATIC_PAD
  T5 execution schedule -> fixed tile/K-loop constexprs (matmul_persistent,
                           moe_grouped); RAON_LOGPROB_FIXED_ORDER (reduction
                           order pinned per row)
  T6 selection/packing  -> moe_route replay pins; kda CPR chunk grid

Batch invariance alone does not cover cross-backend equivalence (engine vs
trainer); the paired flags below add that on top of T1-T6.
"""
import hashlib
import json
import os
import re
from pathlib import Path

# Byte-identical on both sides: the emitting side is inferred from the import
# path (engine copy lives under raon_vllm.*, the trainer copy anywhere else).
_DEFAULT_SIDE = "engine" if "raon_vllm" in __name__ else "trainer"

# flag name -> (side, default, pair-or-None); bit_relevant subset of the
# canonical FLAG_CENSUS, kept in lockstep via contract_module_sha.
_FLAGS = {
    # kda_proj_gemm_tune pair: the decode-width launch tuning in
    # batch_invariant.py (BLOCK_SIZE_M=16, num_warps=4 for bf16 at M<=64).
    # Both sides run a byte-identical copy of that file, so kda_proj_gemm_sha
    # catches a source divergence but not a flag divergence: with the same
    # file on both sides the sha agrees no matter how the env is set, and
    # enabling it on the engine alone would change its arithmetic silently.
    # Same name on both sides because the file is the same file.
    "RAON_KDA_PROJ_GEMM_TUNE": ("both", "0", "kda_proj_gemm_tune"),
    "RAON_BATCH_INVARIANT": ("engine", "1", None),
    "RAON_MOE_FP32_COMBINE": ("engine", "0", None),
    "RAON_MOE_TRAINER_REALIZED": ("engine", "0", "moe_realized"),
    # moe_route_replay pair: the engine records the eager MoE forward's top-k
    # selection (moe_route.py) and the trainer replays it through MCore
    # RouterReplay (RAON_MOE_ROUTE_REPLAY, below).
    "RAON_MOE_ROUTE_CAPTURE": ("engine", "0", "moe_route_replay"),
    # moe_route_replay_weights pair: the engine additionally records the fp32
    # top-k weights at the same dispatch site (moe_route.py) and the trainer
    # substitutes them for its own after renormalization. Pinning the ids does
    # not pin the weights: the router gate GEMM realizations differ by a few
    # tens of fp32 ulp, which can cross a bf16 rounding boundary. Paired so a
    # half-enabled pair fails the gate.
    "RAON_MOE_ROUTE_CAPTURE_WEIGHTS": ("engine", "0", "moe_route_replay_weights"),
    "RAON_MLA_BATCH_INVARIANT": ("engine", "1", None),
    "RAON_VLLM_QUACK_RMSNORM": ("engine", "1", None),
    "RAON_KDA_CHUNK_REPLAY": ("engine", "0", None),
    # Engine-side unpaired realization flags: the trainer side is the
    # reference realization (fused mHC kernels / mla_align.py FA4 ns=1), so
    # there is nothing to cross-compare; recorded for visibility only.
    "RAON_MHC_TRAINER_FUSED": ("engine", "0", None),
    "RAON_MHC_AGGNORM_FUSED": ("engine", "0", None),
    "RAON_MLP_PROJ_PERSISTENT": ("engine", "0", None),
    # Dense-FFN SiTU onto the fp32 reference formula (mlp_align.py), matching
    # the trainer's situ_glu_reference realization. Engine-side unpaired.
    "RAON_MLP_SITU_NATIVE": ("engine", "0", None),
    # MLA latent q/kv RMSNorm onto the trainer's QuACK kernel (mla_align.py).
    # Engine-side unpaired.
    "RAON_MLA_LATENT_NORM_QUACK": ("engine", "0", None),
    # MLA q/kv_a/kv_b/o projections onto the shared matmul_persistent -- the
    # same mechanism as RAON_MLP_PROJ_PERSISTENT on a different GEMM family.
    # Makes the MLA projections invariant to the decode batch size.
    # Engine-side unpaired.
    "RAON_MLA_PROJ_PERSISTENT": ("engine", "0", None),
    "RAON_MLA_PREFILL_FA4_NS1": ("engine", "0", None),
    # vLLM ships _gate_sigmoid_mul under @torch.compile, so inductor fuses
    # sigmoid+mul and rounds once; the trainer's fused output gate rounds the
    # sigmoid to bf16 before the product. Engine-side unpaired.
    "RAON_MLA_OUTPUT_GATE_TWO_ROUND": ("engine", "0", None),
    # Path-valued (quack preload dir for the vendored FA4 CuTe import):
    # recorded as presence only ("set"/"unset"), never the machine path.
    "RAON_QUACK_COMPAT_PATH": ("engine", "", None),
    # Engine-only, unpaired: not a realization choice the trainer can mirror.
    # It fixes the index order the fla prefill core writes the KDA recurrent
    # state in so vLLM's decode kernel reads it correctly. fla returns
    # [N,H,K,V] and the decode kernel reads [N,H,V,K]; RAON has K == V, so
    # nothing errors and a prefill-only check cannot detect it.
    # Only meaningful with RAON_KDA_INKERNEL_FLA=1.
    "RAON_KDA_DECODE_STATE_VFIRST": ("engine", "0", None),
    # moe_router_gemm pair, both sides, same env name, value-typed (off |
    # cublas | cublas-pad8 | cublas-pad384 | persistent | realized). Takes
    # the MoE router gate GEMM off token-count-dependent kernel selection
    # (engine: moe_router_gemm.py over GateLinear's tier dispatch; trainer:
    # over the cuBLASLt heuristic). Paired by value so a pin that differs
    # between the sides is part of composition_sha256 and fails the gate.
    # `cublas-pad8` / `cublas-pad384` re-derive their cuBLASLt invariants per
    # environment on the first live forward (fail-closed self-checks on each
    # side). `persistent` has no trainer equivalent. `realized` is the shared
    # realization -- fp32-store matmul_persistent, byte-identical Triton
    # source on both sides (kda_proj_gemm_sha pins the source, the emitted
    # triton version pins the compiler) -- the only mode whose two sides are
    # one function at different M; padding pins do not remove the remaining
    # M-dependent differences.
    "RAON_MOE_ROUTER_GEMM_PIN": ("both", "off", "moe_router_gemm"),
    # moe_renorm pair, both sides, same env name. Engine: the eager MoE
    # forward computes its top-k gate weights with a transcription of the
    # trainer's unfused sigmoid+bias renormalization (moe_renorm.py) --
    # fp32, torch.topk gather/sum order -- instead of the fused
    # torch.ops._moe_C.topk_sigmoid, which is 1-2 fp32 ULP low on every slot
    # and can flip bf16 ties. Trainer: forces every routing call onto the
    # unfused branch, since TE's fused topk is not bitwise equal to the
    # unfused branch; the first forced call also reports the fused-vs-unfused
    # distance. Paired so a half-enabled pair (engine on the shared renorm,
    # trainer silently on something else) fails the manifest gate.
    "RAON_MOE_RENORM_REALIZED": ("both", "0", "moe_renorm"),
    # Engine-only, unpaired: un-absorbed MLA decode (mla_unabsorb.py) -- per
    # decode step, expand the cached latents through kv_b_proj exactly as
    # prefill does and attend through the same FA4 ns=1 kernel family. The
    # absorbed decode is a structurally different GEMM chain that no pin can
    # make bitwise equal to prefill. The trainer has no decode path; prefill
    # is the reference realization.
    "RAON_MLA_UNABSORBED_DECODE": ("engine", "0", None),
    # Sub-flag: pin kv_b_proj to matmul_persistent instance-wide (changes
    # the prefill realization too when on, so both sides must match). Off by
    # default.
    "RAON_MLA_UNABSORBED_KVB_PERSISTENT": ("engine", "0", None),
    # kda_cpr pair, both sides, same env name: chunk-pinned recurrence -- one
    # chunk_kda state pass to 64-token boundaries (bf16 snapshots upcast to
    # fp32) + a segmented fused_recurrent_kda scan, so engine decode
    # resynchronizes to the trainer's own chunk grid (engine
    # kda_inkernel.cpr_enabled). CPR changes the trainer's chunk_kda bits, so
    # a half-enabled pair is a different function and must fail the gate.
    # Kwargs are part of the contract: output_final_state=True on both sides
    # (STORE_FINAL_STATE is a constexpr). Engine requires
    # RAON_KDA_INKERNEL_FLA=1, enforced fail-closed at model build (kda.py).
    "RAON_KDA_CPR": ("both", "0", "kda_cpr"),
    "RAON_KDA_INKERNEL_FLA": ("engine", "0", "kda_contract"),
    "VLLM_KPAMO_FP32_LM_HEAD": ("engine", "0", "head_fp32"),
    # logprob_fixed_order pair, both sides, same env name: the gold-logprob
    # logsumexp onto the fixed-order per-row kernel (engine exact_logprob.py;
    # shared source logprob_invariant.py, vendored byte-identical -- drift is
    # caught by the logprob_lse_sha manifest extra, same mechanism as
    # kda_proj_gemm_sha). One program per row with a constant block sweep in
    # index order, so a row's bits do not depend on M or on position
    # chunking. Changes bits when enabled (fixed sequential merge differs
    # from the torch.logsumexp tree at the ~1e-6 level; see exact_logprob.py),
    # so both sides must enable it together; a half-enabled pair fails the
    # gate.
    "RAON_LOGPROB_FIXED_ORDER": ("both", "0", "logprob_fixed_order"),
    # Engine-side unpaired: the FP32 head GEMM onto the shared batch-invariant
    # matmul_persistent instead of cuBLAS, removing the schedule's M
    # dependence at the one GEMM whose M follows logprob batching. Requires
    # VLLM_KPAMO_FP32_LM_HEAD=1, fail-closed in model.py. Changes bits vs the
    # cuBLAS product; recorded for visibility.
    "RAON_HEAD_GEMM_PERSISTENT": ("engine", "0", None),
    "RAON_KDA_PIN_AUTOTUNE": ("both", "0", "kda_pin"),
    "RAON_KDA_PROJ_PERSISTENT": ("both", "0", "kda_proj"),
    # moe_combine_ep pair, value-typed (0 = off, else the expert-parallel
    # size). An expert-parallel combine reduces twice with a bf16 rounding in
    # between (per-EP-rank fp32 partial -> bf16 on the wire -> fp32 on the
    # token owner); the engine has no EP and emulates that split
    # (moe_align.combine_ep_partial_fp32), and the trainer verifies the
    # topology the emulation assumes. Paired by value so a wrong EP size
    # fails the gate.
    "RAON_MOE_COMBINE_EP_REALIZED": ("both", "0", "moe_combine_ep"),
    # moe_shared_situ pair. RAON_MOE_EXPERT_BF16_REALIZED aligns only the
    # routed experts; without this pair the shared expert can use a third
    # SiTU realization (the trainer's Triton kernel, the engine's
    # torch.ops._C.situ_and_mul, and the fp32 formula the routed experts
    # run), and no two agree everywhere. Engine:
    # moe_align.install_shared_expert_situ. Trainer: the reference SiTU
    # backend for the shared expert only.
    "RAON_MOE_SHARED_SITU_REALIZED": ("both", "0", "moe_shared_situ"),
    # moe_expert_mxfp8 pair: the engine runs the fused CuTe MXFP8 grouped MLP
    # as its routed-expert realization (mxfp8_align.py). The trainer has no
    # single env that means "my routed experts are fused mxfp8" -- that is a
    # property of its precision config -- so the trainer side of this pair
    # is the declaration env RAON_MOE_EXPERT_MXFP8_TWIN. Paired by value so a
    # half-enabled pair (engine quantized, trainer de-quantized via
    # RAON_MOE_EXPERT_BF16_REALIZED=1, or vice versa) fails the manifest
    # gate. Composition rule enforced in model.py: the engine flag replaces
    # the expert chain, so RAON_MOE_CHAIN_FUSED (a launch fusion of the bf16
    # chain) is bypassed while it is on; RAON_MOE_TRAINER_REALIZED=1 remains
    # required (dispatch + combine contract), fail-closed.
    "RAON_MOE_EXPERT_MXFP8_REALIZED": ("engine", "0", "moe_expert_mxfp8"),
    "RAON_MOE_EXPERT_MXFP8_TWIN": ("trainer", "0", "moe_expert_mxfp8"),
    # mla_mxfp8 / moe_shared_mxfp8 pairs: the other two module groups the
    # default trainer precision config leaves quantized, realized on the
    # engine (mxfp8_align.apply_mla_mxfp8 / apply_shared_mxfp8). Same
    # declaration shape as the routed pair: the trainer side is a declaration
    # env because "my MLA is mxfp8" is a property of the loaded config.
    # RAON_MLA_MXFP8_REALIZED supersedes RAON_MLA_PROJ_PERSISTENT (the bf16
    # realization); enabling both is refused in mxfp8_align, not here,
    # because it is a runtime wiring conflict rather than a cross-side
    # mismatch.
    "RAON_MLA_MXFP8_REALIZED": ("engine", "0", "mla_mxfp8"),
    "RAON_MLA_MXFP8_TWIN": ("trainer", "0", "mla_mxfp8"),
    "RAON_MOE_SHARED_MXFP8_REALIZED": ("engine", "0", "moe_shared_mxfp8"),
    "RAON_MOE_SHARED_MXFP8_TWIN": ("trainer", "0", "moe_shared_mxfp8"),
    "RAON_KDA_PARITY_CONTRACT": ("trainer", "", "kda_contract"),
    # Trainer-side unpaired: the engine has no matching env name -- its conv
    # realization follows RAON_KDA_INKERNEL_FLA (already paired above), which
    # swaps the prefill conv onto the same fla kernel this selects. Not
    # paired because the trainer's default conv realization depends on the
    # input layout, not on any flag, so a pair value derived from the env
    # alone would fail spuriously.
    "RAON_KDA_CONV_BACKEND": ("trainer", "", None),
    # moe_realized pair: trainer twin of RAON_MOE_TRAINER_REALIZED -- the
    # bf16-realized expert chain, bitwise equal to the engine flag.
    "RAON_MOE_EXPERT_BF16_REALIZED": ("trainer", "0", "moe_realized"),
    # moe_route_replay pair: trainer twin of RAON_MOE_ROUTE_CAPTURE.
    # Implies dropping --moe-router-fusion (the TE fused topk returns before
    # router_replay is consulted, so replay under fusion is a silent no-op).
    "RAON_MOE_ROUTE_REPLAY": ("trainer", "0", "moe_route_replay"),
    # moe_route_replay_weights pair: trainer twin of
    # RAON_MOE_ROUTE_CAPTURE_WEIGHTS. Requires RAON_MOE_ROUTE_REPLAY=1 -- it
    # substitutes weights on the REPLAY_FORWARD path and fails closed with no
    # staged payload.
    "RAON_MOE_ROUTE_REPLAY_WEIGHTS": ("trainer", "0", "moe_route_replay_weights"),
    "NRL_FP32_LOGPROB_HEAD": ("trainer", "0", "head_fp32"),
    # Recorded for visibility; enters the composition through the trainer's
    # 3-value head_fp32 normalization below, not as its own pair.
    "NRL_FP32_LOGPROB_HEAD_GEMM": ("trainer", "0", None),
    # Engine-only, unpaired: phase-split gate. The trainer side has no phase
    # concept to pair against -- this gate only changes which realization the
    # engine's prefill forward takes among code already covered by the flags
    # it phase-splits (RAON_VLLM_QUACK_RMSNORM, RAON_MLA_LATENT_NORM_QUACK,
    # RAON_MLA_OUTPUT_GATE_TWO_ROUND, RAON_MHC_TRAINER_FUSED,
    # RAON_MHC_AGGNORM_FUSED, RAON_MOE_RENORM_REALIZED,
    # RAON_MOE_MXFP8_W8A8_TRITON, RAON_MOE_MXFP8_W8A8_FINALIZE,
    # RAON_MOE_COMBINE_EP_REALIZED), each of which has its own entry above.
    # Recorded for visibility only.
    "RAON_PREFILL_ALIGNED": ("engine", "0", None),
}


def _env(name: str, default: str) -> str:
    import os
    v = os.environ.get(name)
    # Alias: the engine's autotune pin is read as RAON_KDA_AUTOTUNE_PIN
    # (kda_autotune.py) while the census/trainer use RAON_KDA_PIN_AUTOTUNE.
    # Read both so the manifest reflects the value the code actually honors.
    if v is None and name == "RAON_KDA_PIN_AUTOTUNE":
        v = os.environ.get("RAON_KDA_AUTOTUNE_PIN")
    v = v if v is not None else default
    # Path-valued census entries: presence only (the path is machine-specific
    # and not bit-relevant; whether a compat quack was preloaded IS).
    if name == "RAON_QUACK_COMPAT_PATH":
        return "set" if v else "unset"
    return v


def _normalize(name: str, raw: str, pair: str) -> str:
    if name == "RAON_KDA_INKERNEL_FLA":
        return "inkernel" if raw == "1" else "external"
    if name == "RAON_KDA_PARITY_CONTRACT":
        return raw or "external"
    if pair == "head_fp32":
        # 3-value trainer domain: the engine flag is a true FP32 GEMM, while
        # NRL_FP32_LOGPROB_HEAD is only a post-GEMM upcast, so those two must
        # not pair. "fp32" = FP32 GEMM realization on this side;
        # "fp32-upcast" = BF16 GEMM + fp32 upcast (engine has no such state).
        if name == "NRL_FP32_LOGPROB_HEAD":
            if _env("NRL_FP32_LOGPROB_HEAD_GEMM", "0") == "1":
                return "fp32"
            return "fp32-upcast" if raw == "1" else "bf16"
        return "fp32" if raw == "1" else "bf16"
    if pair == "kda_cpr":
        return "cpr" if raw == "1" else "off"
    if pair == "kda_pin":
        return "pinned" if raw == "1" else "off"
    if pair == "kda_proj":
        return "persistent" if raw == "1" else "native"
    if pair == "moe_realized":
        # moe_expert_mxfp8 supersede: under an enabled mxfp8 pair the expert
        # GEMMs are the fused MXFP8 kernel on both sides -- the engine keeps
        # the trainer-structured dispatch/fold/combine
        # (RAON_MOE_TRAINER_REALIZED=1) and the trainer's de-quantizing
        # bf16 path stays off, so the raw values (engine 1 / trainer 0) would
        # fail a correctly configured pair. Each side folds its own mxfp8 flag
        # in; contradictory combinations map to sentinels that can never
        # match across sides (fail closed).
        mx_flag = ("RAON_MOE_EXPERT_MXFP8_REALIZED"
                   if name == "RAON_MOE_TRAINER_REALIZED"
                   else "RAON_MOE_EXPERT_MXFP8_TWIN")
        if _env(mx_flag, "0") == "1":
            if name == "RAON_MOE_TRAINER_REALIZED" and raw != "1":
                return "INVALID-engine-mxfp8-without-realized-chain"
            if name == "RAON_MOE_EXPERT_BF16_REALIZED" and raw == "1":
                return "INVALID-trainer-bf16-and-mxfp8"
            return "mxfp8-chain"
        return "realized" if raw == "1" else "native"
    if pair in ("moe_expert_mxfp8", "mla_mxfp8", "moe_shared_mxfp8"):
        return "mxfp8" if raw == "1" else "off"
    if pair == "moe_route_replay":
        return "replay" if raw == "1" else "live"
    if pair == "moe_route_replay_weights":
        return "pinned" if raw == "1" else "local"
    if pair == "moe_shared_situ":
        # Off is "native" on both sides but means two different kernels
        # (trainer Triton, engine _C.situ_and_mul). The pair certifies the two
        # sides made the same choice; only "reference" resolves both to one
        # formula.
        return "reference" if raw == "1" else "native"
    if pair == "moe_combine_ep":
        return "single" if raw in ("0", "", "false") else f"ep{int(raw)}"
    if pair == "moe_renorm":
        return "realized" if raw == "1" else "native"
    if pair == "moe_router_gemm":
        # Value-typed; canonicalize only the spellings of off that both
        # sides' parsers already treat as one mode, so exporting "0" to one
        # side and nothing to the other still compares equal.
        return "off" if raw.strip().lower() in ("off", "0", "") else raw.strip().lower()
    if pair == "logprob_fixed_order":
        return "fixed" if raw == "1" else "native"
    return raw


def _gemm_module_sha() -> str:
    # The RAON_KDA_PROJ_PERSISTENT flag requires both runtimes to execute
    # the identical Triton matmul_persistent source (byte-identical copies
    # on each side). Record the resolved copy's sha256 in the manifest values
    # so any drift between the two copies fails the composition gate, same
    # mechanism as contract_module_sha. Side-aware resolution order so each
    # process hashes its own copy.
    import importlib.util
    mods = ["raon_vllm.batch_invariant", "megatron.core.ssm.batch_invariant"]
    if _DEFAULT_SIDE == "trainer":
        mods.reverse()
    for mod in mods:
        try:
            spec = importlib.util.find_spec(mod)
        except Exception:
            spec = None
        if spec is not None and spec.origin:
            return hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
    return "absent"


def _lse_module_sha() -> str:
    # Same mechanism for the RAON_LOGPROB_FIXED_ORDER flag's shared kernel
    # (logprob_invariant.py, byte-identical copies): record the resolved
    # copy's sha256 so drift between the two fails the composition gate. The
    # copy lives next to this module, so the sibling path resolves without
    # importing either runtime.
    sibling = Path(__file__).with_name("logprob_invariant.py")
    try:
        return hashlib.sha256(sibling.read_bytes()).hexdigest()
    except OSError:
        return "absent"


# --- TE precision scope registry (see the module docstring) -----------------
# module_path -> config key ("bf16", "nvfp4", ... or "GLOBAL" when no matcher
# matched and the module fell through to the global fp8 recipe).
_TE_DECISIONS: dict = {}
# Set on the first write_side_manifest call so late decisions can re-emit:
# the trainer may write its manifest during model build, before every module
# has been constructed. Without the re-emit the scope would be silently
# truncated.
_MANIFEST_TARGET: dict = {}
# Module prefixes outside the scored path. Named, not pattern-guessed.
_UNSCORED_PREFIXES = ("mtp.",)
# The config key that means "not quantized". A config may define other keys
# (nvfp4, ...); anything that is not exactly this is reported as quantized
# and its key is printed, so a renamed bf16 config shows up in the gate diff
# rather than passing silently.
_BF16_CONFIG_KEY = "bf16"


def record_te_precision(module_path: str, config_key) -> None:
    """Record one realized TE precision decision.

    Called by the trainer's precision recipe config matcher -- the function
    that actually chooses the config. This is deliberately not derived from
    a config file or a log: the running process is the only authority on
    what ran.
    """
    key = str(module_path)
    val = str(config_key) if config_key is not None else "GLOBAL"
    if _TE_DECISIONS.get(key) == val:
        return
    before = _te_precision_scope()
    _TE_DECISIONS[key] = val
    if _MANIFEST_TARGET and _te_precision_scope() != before:
        write_side_manifest(
            _MANIFEST_TARGET["mdir"],
            _MANIFEST_TARGET["side"],
            _MANIFEST_TARGET["extra"],
        )


def _te_precision_scope() -> str:
    """Canonical, diffable description of what ran non-bf16 on the scored path."""
    if not _TE_DECISIONS:
        # Engine: no TransformerEngine modules exist, so an empty registry is
        # the true answer. Trainer: an empty registry means the recording hook
        # never fired, which is not evidence of bf16. Fail closed.
        return "none" if _DEFAULT_SIDE == "engine" else "UNOBSERVED"
    quantized: dict = {}
    for path, cfg in _TE_DECISIONS.items():
        if path.startswith(_UNSCORED_PREFIXES):
            continue
        if cfg == _BF16_CONFIG_KEY:
            continue
        kind = re.sub(r"\.\d+\.", ".N.", path)
        quantized[f"{kind}={cfg}"] = quantized.get(f"{kind}={cfg}", 0) + 1
    if not quantized:
        return "none"
    return ",".join(f"{k}*{n}" for k, n in sorted(quantized.items()))


def _versions() -> dict:
    out = {}
    for mod in ("fla", "torch", "triton"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "unknown")
        except Exception:
            out[mod] = "absent"
    return out


def write_side_manifest(mdir, side: str = None, extra: dict = None) -> Path:
    side = side or _DEFAULT_SIDE
    values = {
        name: _env(name, default)
        for name, (owner, default, _pair) in _FLAGS.items()
        if owner in (side, "both")
    }
    values.update(_versions())
    values["contract_module_sha"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    values["kda_proj_gemm_sha"] = _gemm_module_sha()
    values["logprob_lse_sha"] = _lse_module_sha()
    # Precision is a bit-relevant property of the run (see the module
    # docstring). Both keys are non-census extras, so they flow into the
    # composition verbatim on both sides and must agree. The readable form is
    # emitted alongside the sha so a failing gate can print the offending
    # module kinds.
    _scope = _te_precision_scope()
    values["te_precision_scope"] = _scope
    values["te_precision_scope_sha256"] = hashlib.sha256(
        _scope.encode()
    ).hexdigest()
    values.update(extra or {})
    comp = {
        pair: _normalize(name, values[name], pair)
        for name, (owner, _default, pair) in _FLAGS.items()
        if pair and owner in (side, "both")
    }
    comp.update({k: v for k, v in values.items() if k not in _FLAGS})
    blob = json.dumps(comp, sort_keys=True, separators=(",", ":")).encode()
    doc = {
        "side": side,
        "values": values,
        "composition_sha256": hashlib.sha256(blob).hexdigest(),
    }
    d = Path(mdir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{side}.json"
    path.write_text(json.dumps(doc, sort_keys=True, indent=2) + "\n")
    _MANIFEST_TARGET.update({"mdir": mdir, "side": side, "extra": extra})
    return path

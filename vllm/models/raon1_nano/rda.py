# SPDX-License-Identifier: Apache-2.0
"""Raon Delta Attention using vLLM's KDA kernels.

RDA separates erase and write coefficients. Applying
sigmoid(beta + delta) / sigmoid(beta) to values after the causal convolution
and before the delta recurrence gives the required write gate. The stable
ratio is evaluated in FP32. The base layer and metadata derive from Raon-1
mini; see UPSTREAM.md for source attribution."""

import os

import torch
import torch.nn.functional as F
from einops import rearrange

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as tld

from .config import Raon1NanoConfig
from .rda_base import RaonRDABase, _flashkda_prefill
from .rda_metadata import RaonRDAMetadata

_FLASHKDA_DEDICATED: dict = {}

_CR_DECODE_LIVE_ANNOUNCED = [False]


def _flashkda_dedicated_buffers(specs, device):
    key = (str(device), tuple((tuple(sh), dt) for sh, dt in specs))
    bufs = _FLASHKDA_DEDICATED.get(key)
    if bufs is None:
        bufs = [torch.empty(sh, dtype=dt, device=device) for sh, dt in specs]
        _FLASHKDA_DEDICATED[key] = bufs
    return bufs


# vLLM 0.28.0 API change: _flashkda_prefill writes into caller-supplied
# out/final_state/workspace buffers, which stock carves out of the shared
# workspace arena (stock kda.py:49, :724-728).


def compute_rda_value_ratio(
    beta_logits: torch.Tensor, write_delta_logits: torch.Tensor
) -> torch.Tensor:
    """RDA's write/erase ratio, without forming an unstable quotient.

    fp32 throughout, and expressed as a difference of log-sigmoids so that a
    zero delta gives exactly 1.0 — a zero-init ``rda_w_proj`` is then
    bit-identical to the upstream delta recurrence.
    """
    beta_fp32 = beta_logits.float()
    delta_fp32 = write_delta_logits.float()
    return torch.exp(F.logsigmoid(beta_fp32 + delta_fp32) - F.logsigmoid(beta_fp32))


# RAON_KDA_PROJ_PERSISTENT: shared fixed-schedule Triton GEMM for the KDA
# projections (see the lever block in __init__).
from vllm.models.raon1_nano.batch_invariant import matmul_persistent  # noqa: E402

# ---------------------------------------------------------------------------
# RAON_KDA_INKERNEL_FLA: route the KDA prefill through the fla-MAINLINE ops the
# trainer scores KDA with, so full-model L00.kda reaches bitwise-0.
#
# The stock RAON prefill runs the vLLM-vendored KDA kernels
# (`vllm.models.raon1_nano.ops.kda.chunk_kda_with_fused_gate`,
# the vendored `FusedRMSNormGated`, and the vLLM mamba `causal_conv1d_fn`).
# The RAON trainer reference dump instead takes
# the training FLA forward path (see UPSTREAM.md), which
# runs the fla-MAINLINE ops:
#   conv = fla.modules.conv.causal_conv1d.causal_conv1d(backend='triton'),
#          one call per q/k/v section,
#   core = fla.ops.kda.chunk_kda with the IN-KERNEL contract
#          (g=decay_raw raw, beta=beta_raw.float() raw, A_log,
#          dt_bias.reshape(-1), scale=kda_query_scale,
#          use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
#          use_beta_sigmoid_in_kernel=True, safe_gate=True, lower_bound=-5.0),
#   o_norm = fla.modules.fused_norm_gate.rms_norm_gated(activation='sigmoid').
#
# The vendored ops are the same source lineage but a different BUILD, so their
# last bits differ from the fla-mainline ops (kda-fla-unify note: ~1 bf16 ULP in
# the core, amplified ~86x by o_norm into the ~6.25e-2 full-model L00.kda gap).
# Rerouting the engine onto the identical fla-mainline objects + pinning autotune
# makes the KDA core AND the o_norm'd site bitwise-identical to the trainer
# (isolation cross-venv gate job 173564: core+site 0.0 on n192/n130).
#
# Opt-in (default off) because it swaps the throughput kernels for the
# parity-oriented fla-mainline ones and is currently wired for the plain prefill
# path (no chunk-replay, no flashkda) used by the layer-probe dump.
# ---------------------------------------------------------------------------
# The fla-mainline helpers live in kda_inkernel.py (vLLM-free) so the exact
# code the live engine runs is unit-testable against the isolation gate.
from vllm.models.raon1_nano.kda_inkernel import (  # noqa: E402
    FlaMainlineGatedRMSNorm,
    cu_seqlens_cached,
    fla_inkernel_core,
    pin_all_fla_autotune,
)


@triton.jit
def _rda_scale_v_kernel(
    v_ptr,
    beta_ptr,
    delta_ptr,
    v_st,
    v_sh,
    b_st,
    b_sh,
    d_st,
    d_sh,
    H,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """v[t, h, :] *= exp(logsigmoid(beta+delta) - logsigmoid(beta)), fp32.

    logsigmoid replicates aten: min(x, 0) - log1p(exp(-|x|)).  The store
    rounds fp32 -> v dtype once, matching (v.float() * ratio).to(v.dtype).
    """
    pid = tl.program_id(0)
    t = pid // H
    h = pid % H
    b = tl.load(beta_ptr + t * b_st + h * b_sh).to(tl.float32)
    d = tl.load(delta_ptr + t * d_st + h * d_sh).to(tl.float32)
    x = b + d
    ls_bd = tl.minimum(x, 0.0) - tld.log1p(tl.exp(-tl.abs(x)))
    ls_b = tl.minimum(b, 0.0) - tld.log1p(tl.exp(-tl.abs(b)))
    ratio = tl.exp(ls_bd - ls_b)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    vp = v_ptr + t * v_st + h * v_sh + offs
    v = tl.load(vp, mask=mask).to(tl.float32)
    tl.store(vp, (v * ratio).to(v_ptr.dtype.element_ty), mask=mask)


def rda_scale_v_(
    v: torch.Tensor, beta_logits: torch.Tensor, write_delta_logits: torch.Tensor
) -> None:
    """Fused in-place RDA scale of a [T, H, D] value view (last dim contiguous)."""
    T, H, D = v.shape
    assert v.stride(2) == 1
    _rda_scale_v_kernel[(T * H,)](
        v,
        beta_logits,
        write_delta_logits,
        v.stride(0),
        v.stride(1),
        beta_logits.stride(0),
        beta_logits.stride(1),
        write_delta_logits.stride(0),
        write_delta_logits.stride(1),
        H,
        D=D,
        BLOCK_D=triton.next_power_of_2(D),
    )


# ---------------------------------------------------------------------------
# [18] RAON_KDA_CONV_IDX_CACHE: one decode-slot D2H sync per STEP, not per layer
# ---------------------------------------------------------------------------
_CONV_IDX_CACHE_STATE = {"verified": 0, "announced": False}
_PINNED_SLOT_STAGE = {}


def _slots_tolist_pinned(t):
    """pinned-copy 지뢰 수리 (2026-09-23): tolist() 는 비고정 D2H 라 CUDA
    graph 캡처 중 "Cannot copy ... unless pinned" 로 즉사한다. 값·순서 불변으로
    pinned 스테이징 버퍼를 경유한다 (pin 은 semantics 불변 — 코디 발주;
    비-CUDA/미가용 환경은 원문 그대로)."""
    if not (t.is_cuda and torch.cuda.is_available()):
        return t.tolist()
    n = t.numel()
    key = (str(t.device), str(t.dtype))
    buf = _PINNED_SLOT_STAGE.get(key)
    if buf is None or buf.numel() < n:
        cap = max(64, 1 << (n - 1).bit_length())
        buf = torch.empty(cap, dtype=t.dtype, pin_memory=True)
        _PINNED_SLOT_STAGE[key] = buf
    dst = buf[:n]
    dst.copy_(t.reshape(-1), non_blocking=False)
    return dst.tolist()


def _decode_conv_slots_host(module, decode_conv_indices):
    """Host slot ids for this decode step's conv/recurrent-state rows.

    Every KDA layer's decode seam runs `decode_conv_indices.tolist()` -- an
    aten::_local_scalar_dense D2H that drains the CUDA queue -- so 18 layers
    pay 18 identical syncs per step for the same integers.  All layers of a
    step share ONE RaonRDAMetadata instance: gpu_model_runner assigns the
    same built object to every layer_name in the attention group
    (`for layer_name in attn_group.layer_names: attn_metadata_dict[layer_name]
    = attn_metadata_i`), and GDNAttentionMetadataBuilder leaves
    supports_update_block_table at its False default, so that object is
    freshly built each step and never reused across steps.  Stash the host
    copy on it: the first layer to reach a seam pays the one sync, the other
    17 read the cached list.

    Bit-neutrality: the cached values are the SAME integers every other layer
    would have read from the SAME tensor -- nothing device-side changes.
    Self-checking like RAON_MLA_HOST_SEQLENS_VERIFY: the first
    RAON_KDA_CONV_IDX_CACHE_VERIFY cache hits (default 4) re-read the device
    tensor, raise on any difference, then latch.

    Default OFF (RAON_KDA_CONV_IDX_CACHE=1 to arm): the off path is exactly
    the pristine statement, one tolist per layer-call.
    """
    if os.environ.get("RAON_KDA_CONV_IDX_CACHE", "0") != "1":
        return _slots_tolist_pinned(decode_conv_indices)
    meta = None
    try:
        fc_meta = get_forward_context().attn_metadata
        if isinstance(fc_meta, dict):
            meta = fc_meta.get(module.prefix)
    except Exception:
        meta = None
    if meta is None:
        # No shared carrier reachable; behave exactly like OFF for this call.
        return _slots_tolist_pinned(decode_conv_indices)
    n = decode_conv_indices.size(0)
    cached = getattr(meta, "_raon_conv_idx_host", None)
    if cached is None or len(cached) != n:
        cached = _slots_tolist_pinned(decode_conv_indices)
        meta._raon_conv_idx_host = cached
        if not _CONV_IDX_CACHE_STATE["announced"]:
            _CONV_IDX_CACHE_STATE["announced"] = True
            print(
                "[parity] RAON_KDA_CONV_IDX_CACHE LIVE: decode conv slot "
                "ids sync once per step and are cached on the shared KDA "
                "metadata for the sibling layers",
                flush=True,
            )
        return cached
    want = int(os.environ.get("RAON_KDA_CONV_IDX_CACHE_VERIFY", "4"))
    if _CONV_IDX_CACHE_STATE["verified"] < want:
        device_side = _slots_tolist_pinned(decode_conv_indices)
        if [int(v) for v in cached] != [int(v) for v in device_side]:
            raise RuntimeError(
                "RAON_KDA_CONV_IDX_CACHE: cached host slots "
                f"{[int(v) for v in cached]} != device {device_side}; the "
                "shared-metadata precondition is broken, refusing to reuse "
                "stale slot ids."
            )
        _CONV_IDX_CACHE_STATE["verified"] += 1
        if _CONV_IDX_CACHE_STATE["verified"] == want:
            print(
                f"[parity] RAON_KDA_CONV_IDX_CACHE VERIFIED: cache hits "
                f"matched the device tensor {want}/{want} times; latching "
                "(no further per-hit checks)",
                flush=True,
            )
    return cached


class RaonRDAAttention(RaonRDABase):
    def __init__(
        self,
        config: Raon1NanoConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(config, vllm_config, prefix)

        # Opt-in experiment: route plain decode through the FlashKDA chunk
        # kernel as N one-token varlen sequences.  The fused decode kernel
        # never fires at RAON's 40 heads, so the default decode path is the
        # recurrent fallback; this measures whether FlashKDA-at-T=1 beats it.
        self.kda_decode_backend = os.environ.get("RAON_KDA_DECODE", "fused")

        _prefill_override = os.environ.get("RAON_KDA_PREFILL")
        if _prefill_override:
            if _prefill_override not in ("auto", "triton", "flashkda"):
                raise ValueError(
                    "RAON_KDA_PREFILL must be auto, triton or flashkda, got "
                    f"{_prefill_override!r}"
                )
            self.kda_prefill_backend = _prefill_override

        # RAON_RDA_FUSED=1: skip the torch ratio chain in forward() and fold
        # ratio-compute + v-scale into one Triton launch per seam.
        self.rda_fused = os.environ.get("RAON_RDA_FUSED", "0") == "1"

        self.kda_chunk_replay = os.environ.get("RAON_KDA_CHUNK_REPLAY", "0") == "1"
        self.kda_replay_chunk = 64  # FLA_CHUNK_SIZE -- the trainer's chunk boundary
        # slot(int) -> {"q":[Tensor[H,D]], "k":..., "v":..., "g":[Tensor[H,D]],
        #               "beta":[Tensor[H]]}; the open (< 64) chunk being filled.
        # The committed 64-boundary state for a slot lives in
        # recurrent_state[slot] and advances only at a boundary.
        self._cr_open: dict[int, dict] = {}
        # slot(int) -> list[Tensor[Lc, H, D]]: FINALIZED (deferred, bitwise) KDA
        # output, one completed-64-chunk block per boundary + the flushed tail.
        # The rollout worker drains this to score logprobs; cleared per slot at
        # prefill (slot reuse across requests).
        self._cr_finalized: dict[int, list] = {}
        # slot(int) -> list[Tensor[Lc, H, D]]: the o_norm gate (g2) for each
        # finalized block, kept in lockstep with self._cr_finalized so the KDA
        # SITE (o_proj(o_norm(core, g2))) can be reconstructed from the deferred
        # core.  Populated only when g2 is threaded in (real engine forward); the
        # isolated op harness drives the raw-core path and leaves this empty.
        self._cr_finalized_g2: dict[int, list] = {}
        self._cr_slot_bufs: dict[int, dict] = {}

        # vLLM 0.28.0 API change: the FlashKDA buffer specs.  super().__init__
        # above already built self._flashkda_buffer_specs (stock kda.py:433-447)
        # whenever kda_prefill_backend resolved to "flashkda", so the prefill
        # call site below inherits them for free -- nothing to add for it.  The
        # one case stock cannot know about is this file's decode experiment:
        # RAON_KDA_DECODE=flashkda with kda_prefill_backend="triton" leaves the
        # specs None.  Build them then, byte-identical to stock's block.
        if (
            self.kda_decode_backend == "flashkda"
            and self._flashkda_buffer_specs is None
        ):
            T = vllm_config.scheduler_config.max_num_batched_tokens
            N = vllm_config.scheduler_config.max_num_seqs
            H, D = self.local_num_heads, self.head_dim
            import vllm._flashkda_C  # noqa: F401

            workspace_size = torch.ops._flashkda_C.get_workspace_size(T, H, N)
            self._flashkda_buffer_specs = (
                ((1, T, H, D), self.model_config.dtype),
                ((N, H, D, D), self.get_state_dtype()[1]),
                ((workspace_size,), torch.uint8),
            )

        self.rda_w_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.rda_w_proj",
        )

        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        _in_proj_qm = getattr(self.in_proj_qkvgfab, "quant_method", None)
        if (
            self.in_proj_padding
            and _in_proj_qm is not None
            and not isinstance(_in_proj_qm, UnquantizedLinearMethod)
        ):
            _orig_process = _in_proj_qm.process_weights_after_loading
            _pad_rows = int(self.in_proj_padding)

            def _zero_pad_then_process(layer, _orig=_orig_process, _pad=_pad_rows):
                w = getattr(layer, "weight", None)
                if (
                    w is not None
                    and not w.is_meta
                    and w.dtype in (torch.bfloat16, torch.float16, torch.float32)
                ):
                    w.data[-_pad:].zero_()
                    print(
                        f"[q4] in_proj padding guard LIVE: zeroed the last "
                        f"{_pad} unloaded padding rows of "
                        f"{getattr(layer, 'prefix', '<in_proj>')} before "
                        "online quantization computed its scale",
                        flush=True,
                    )
                return _orig(layer)

            _in_proj_qm.process_weights_after_loading = _zero_pad_then_process

        # The fully fused decode kernel runs conv + recurrence + norm in one
        # launch, so there is no seam to scale v at.  It is structurally
        # unreachable for RAON (40 heads is not in (12,24,48,96), kda.py:67, and
        # no supported TP degree lands in that set either), and the vendored
        # _forward below drops its branch.  Assert rather than let a future head
        # count silently discard the RDA gate.
        assert self.decode_conv1d_weight is None, (
            "RDA gate cannot ride the fused KDA decode kernel; it has no "
            "post-conv seam to scale v at"
        )

        # RAON_KDA_INKERNEL_FLA=1: reroute the KDA prefill onto the fla-MAINLINE
        # ops the trainer reference dump scores with (see the module header), so
        # full-model L00.kda reaches bitwise-0.  Wired for the plain prefill path
        # (asserts chunk-replay / flashkda are off, which is how the parity dump
        # runs).  Import the fla ops now (so their autotuners exist), pin every
        # autotuner to configs[0] (process/batch-invariant, matches the pinned
        # trainer), and swap the vendored o_norm for the fla-mainline one so the
        # o_norm'd SITE — not just the core — is trainer-bitwise.
        self.kda_inkernel_fla = os.environ.get("RAON_KDA_INKERNEL_FLA", "0") == "1"
        # RAON_KDA_CONV_TRACE: diagnostic-only shape census, see _conv_trace.
        self._conv_trace_budget = int(os.environ.get("RAON_KDA_CONV_TRACE", "0") or 0)
        self._conv_trace_seen: set = set()
        if self.kda_inkernel_fla:
            if self.kda_chunk_replay:
                raise ValueError(
                    "RAON_KDA_INKERNEL_FLA is incompatible with "
                    "RAON_KDA_CHUNK_REPLAY (they route KDA differently); "
                    "unset one."
                )
            if self.kda_prefill_backend == "flashkda":
                raise ValueError(
                    "RAON_KDA_INKERNEL_FLA needs the plain prefill path; set "
                    "RAON_KDA_PREFILL=triton (not flashkda)."
                )
            # Force autotuner creation, then pin them (idempotent, config index
            # from RAON_KDA_AUTOTUNE_CONFIG_INDEX to match the trainer/plugin).
            import fla.modules.conv.causal_conv1d  # noqa: F401
            import fla.modules.conv.triton  # noqa: F401
            import fla.modules.fused_norm_gate  # noqa: F401
            import fla.ops.kda  # noqa: F401

            pin_all_fla_autotune(
                int(os.environ.get("RAON_KDA_AUTOTUNE_CONFIG_INDEX", "0"))
            )
            # kda_query_scale: trainer passes self.config.kda_query_scale (None
            # for RAON -> fla defaults to key_dim**-0.5).  head_dim**-0.5 is the
            # identical float, so it is bitwise-equal to passing None.
            self._inkernel_scale = self.head_dim**-0.5
            # Route the o_norm SITE through the fla-mainline gated RMSNorm.
            self.o_norm = FlaMainlineGatedRMSNorm(
                self.o_norm.weight, self.o_norm.eps, activation="sigmoid"
            )

        # RAON_KDA_FLA_SEAMS=1 (2026-09-22, 발주자 GO -- zcrk d_live 0.0285
        # 의 prompt-구간 이탈 판별/회수): chunk-replay 조성에서 트레이너와
        # 어긋난 prefill 2접합부만 fla-mainline 으로 스왑한다.
        #   (1) prefill conv: vendored causal_conv1d_fn -> fla causal_conv1d
        #       (_fla_prefill_conv 재사용 -- 다중 시퀀스 state 수리
        #       [_fla_conv_initial_state, 127/127·101/101 실증] 포함)
        #   (2) o_norm: vendored gated RMSNorm -> FlaMainlineGatedRMSNorm
        # chunk core 는 스왑하지 않는다: vendored chunk_kda_with_fused_gate
        # 는 deferred 게이트(validate_kda_deferred)가 트레이너 full-seq
        # chunk 와의 bit 동일성을 이미 실증한 커널이다. 스왑 2접합부의
        # 트레이너-산술 일치 근거: RAON_KDA_INKERNEL_FLA=1 격리 bitwise-0
        # (core/conv/o_norm 3축, cross-venv, job 173564/173585,
        # live==isolation; conv 단독은 jobs 179640/179657 -- MORNING-STATUS
        # -2026-09-07.md:79, lever-state-207395-2026-09-10.md). INKERNEL_FLA
        # 와의 차이는 core 스왑과 chunk-replay 배타뿐 -- 이 레버는 replay
        # 와 공존이 존재 이유다. 기본 OFF (미설정 = 구판 byte-identical).
        self.kda_fla_seams = os.environ.get("RAON_KDA_FLA_SEAMS", "0") == "1"
        if self.kda_fla_seams:
            if self.kda_inkernel_fla:
                raise ValueError(
                    "RAON_KDA_FLA_SEAMS=1 is redundant under "
                    "RAON_KDA_INKERNEL_FLA=1 (the full lever already swaps "
                    "conv/core/o_norm); unset one."
                )
            if self.kda_prefill_backend == "flashkda":
                raise ValueError(
                    "RAON_KDA_FLA_SEAMS needs the plain prefill path; set "
                    "RAON_KDA_PREFILL=triton (not flashkda)."
                )
            # fail-closed 설치 단언: 스왑 대상 심볼이 진짜 fla-mainline
            # 네임스페이스에서 오는지 실경로 확인 (bit-c 의 vendored-vs-fla
            # 네임스페이스 사고 동형 방지) + autotune 등록 강제 후 pin.
            from fla.modules.conv.causal_conv1d import (
                causal_conv1d as _seams_conv_check,
            )

            assert _seams_conv_check.__module__.startswith("fla."), (
                f"RAON_KDA_FLA_SEAMS install check failed: causal_conv1d "
                f"resolved to {_seams_conv_check.__module__!r}, expected the "
                f"fla-mainline namespace; refusing to guess."
            )
            import fla.modules.conv.triton  # noqa: F401
            import fla.modules.fused_norm_gate  # noqa: F401
            import fla.ops.kda  # noqa: F401

            pin_all_fla_autotune(
                int(os.environ.get("RAON_KDA_AUTOTUNE_CONFIG_INDEX", "0"))
            )
            self.o_norm = FlaMainlineGatedRMSNorm(
                self.o_norm.weight, self.o_norm.eps, activation="sigmoid"
            )
            if not getattr(RaonRDAAttention, "_fla_seams_logged", False):
                RaonRDAAttention._fla_seams_logged = True
                print(
                    "[parity] RAON_KDA_FLA_SEAMS LIVE: 2 seams swapped to "
                    "fla-mainline -- (1) prefill conv = fla causal_conv1d "
                    "(per-seq state fix incl.), (2) o_norm = "
                    "FlaMainlineGatedRMSNorm; chunk core stays vendored "
                    "(deferred-gate bit-proven vs trainer)",
                    flush=True,
                )

        self.kda_cpr = os.environ.get("RAON_KDA_CPR", "0") == "1"
        if self.kda_cpr:
            if not self.kda_inkernel_fla:
                raise ValueError(
                    "RAON_KDA_CPR=1 requires RAON_KDA_INKERNEL_FLA=1 (the CPR "
                    "state pass and segmented scan are the fla-mainline ops "
                    "the trainer runs; without the reroute the lever would be "
                    "silently inert on the conv and o_norm seams)"
                )
            if self.kda_chunk_replay:
                raise ValueError(
                    "RAON_KDA_CPR is incompatible with RAON_KDA_CHUNK_REPLAY "
                    "(both manage the 64-boundary state; unset one)"
                )
            if self.kda_decode_backend != "fused":
                raise ValueError(
                    "RAON_KDA_CPR requires the fused recurrent decode backend "
                    f"(RAON_KDA_DECODE unset), got {self.kda_decode_backend!r}"
                )
            from vllm.models.raon1_nano.kda_inkernel import CPRSlotTracker

            self._cpr_graph_ledger = None
            self.kda_cpr_graph_ledger = (
                os.environ.get("RAON_KDA_CPR_GRAPH_LEDGER", "0") == "1"
            )
            if self.kda_cpr_graph_ledger:
                if os.environ.get("RAON_KDA_CPR_RING", "0") == "1":
                    raise ValueError(
                        "RAON_KDA_CPR_GRAPH_LEDGER=1 is mutually exclusive "
                        "with RAON_KDA_CPR_RING=1 (the device ledger IS the "
                        "ring; unset one)"
                    )
                from vllm.models.raon1_nano.kda_cpr_graph import attach_graph_ledger

                self._cpr = attach_graph_ledger(
                    self, vllm_config.scheduler_config.max_num_seqs
                )
            else:
                self._cpr = CPRSlotTracker()
            if not getattr(RaonRDAAttention, "_cpr_logged", False):
                RaonRDAAttention._cpr_logged = True
                print(
                    "[raon-kda] RAON_KDA_CPR=1: prefill = chunk state pass + "
                    "segmented fused_recurrent scan (bf16 boundary snapshots "
                    "upcast to fp32); decode = EXACT-STEP fla "
                    "fused_recurrent_kda (one-token segments, jobs "
                    "193224/193329 scan-swap fix) with 64-boundary resync to "
                    "fp32(bf16(entry)); trainer twin megatron "
                    "the training chunk-replay attention core",
                    flush=True,
                )

        self.kda_proj_persistent = (
            os.environ.get("RAON_KDA_PROJ_PERSISTENT", "0") == "1"
        )
        if self.kda_proj_persistent:
            from vllm.distributed import get_tensor_model_parallel_world_size

            if get_tensor_model_parallel_world_size() != 1:
                raise ValueError(
                    "RAON_KDA_PROJ_PERSISTENT requires tensor parallel size 1 "
                    "(it replaces the projection layers' local GEMM without "
                    "their gather/reduce plumbing)"
                )
            # In-process proof, once (labels must not lie): the lever is only
            # real if forward() actually routes through _proj_mm.
            if not getattr(RaonRDAAttention, "_proj_persistent_logged", False):
                RaonRDAAttention._proj_persistent_logged = True
                print(
                    "[raon-kda] RAON_KDA_PROJ_PERSISTENT=1: in_proj_qkvgfab/"
                    "f_b_proj/rda_w_proj/o_proj GEMMs routed through "
                    "matmul_persistent (Track F batch_invariant.py; trainer "
                    "twin megatron/core/ssm/batch_invariant.py)",
                    flush=True,
                )

        # RAON_KDA_SGL_FUSED=1 (va lane, yes-mismatch): one-kernel KDA decode
        # step vendored from SGLang @ f65c70bb (K3 kda_fused_decode, PR #30113)
        # with a 40-head instantiation and an in-kernel RDA v-scale epilogue.
        # Replaces the plain-decode 4-launch chain (causal_conv1d_update ->
        # _scale_packed_v -> fused_recurrent_rda_packed_decode -> o_norm) for
        # decode-only batches; everything else (prefill, spec, other decode
        # backends) is untouched.  NOT bitwise vs the unfused chain (fp32
        # seams kept in registers + fast-math intrinsics) -- never combine
        # with a parity=on census.  Default OFF = byte-identical stock path.
        # Kernel + integration: kda_sgl_fused.py / kda_sgl_fused_kernel.cu.
        self.kda_sgl_fused = os.environ.get("RAON_KDA_SGL_FUSED", "0") == "1"
        if self.kda_sgl_fused:
            from . import kda_sgl_fused as _sgl_fused

            _sgl_fused.validate_module(self, vllm_config)

    def _proj_mm(self, x: torch.Tensor, layer: torch.nn.Module) -> torch.Tensor:
        """RAON_KDA_PROJ_PERSISTENT realization of one projection GEMM.

        PROVENANCE (IsoExec, zero-mismatch-refs/skyrl-isoexec .../ops/mm): the
        seam both runtimes must share is the fixed-schedule Triton
        ``matmul_persistent`` — IsoExec installs it on aten::mm/addmm in BOTH
        runtimes and treats its pinned-cuBLASLt provider (mm_cublaslt.py) as a
        per-shape perf displacement that must first prove bit-equality AGAINST
        this Triton kernel. Here the same seam is applied call-site-scoped (the
        containment style of Track F's mHC usage and of mm_fwd_scope's scoping
        idea) to the KDA projections only. The kernel is the Track F vendored
        copy (batch_invariant.py, from vexact/TML batch_invariant_ops); the
        trainer twin executes the byte-identical copy at
        ``megatron/core/ssm/batch_invariant.py`` (kda_parity.apply_module), and
        the parity manifest's ``kda_proj_gemm_sha`` gates drift between them.
        """
        assert getattr(layer, "bias", None) is None, (
            "RAON_KDA_PROJ_PERSISTENT expects bias-free KDA projections"
        )
        return matmul_persistent(x, layer.weight.t())

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.size(0)
        if self.kda_proj_persistent:
            projected_qkvgfab = self._proj_mm(hidden_states, self.in_proj_qkvgfab)
        else:
            projected_qkvgfab = self.in_proj_qkvgfab(hidden_states)[0]
        split_sizes = [
            3 * self.local_projection_size,
            self.local_projection_size,
            self.head_dim,
            self.local_num_heads,
        ]
        if self.in_proj_padding:
            split_sizes.append(self.in_proj_padding)
        projected = projected_qkvgfab.split(split_sizes, dim=-1)
        mixed_qkv, g_proj_states, f_a, beta = projected[:4]

        if self.kda_proj_persistent:
            g1 = self._proj_mm(f_a, self.f_b_proj)
        else:
            g1 = self.f_b_proj(f_a)[0]
        beta = beta.unsqueeze(0)
        if self.kda_proj_persistent:
            delta = self._proj_mm(hidden_states, self.rda_w_proj).unsqueeze(0)
        else:
            delta = self.rda_w_proj(hidden_states)[0].unsqueeze(0)
        # Under RAON_RDA_FUSED, `ratio` carries the raw delta logits [1, T, H]
        # (same shape/slicing as the ratio) and the seams compute the ratio
        # inside the fused Triton kernel; otherwise it is the fp32 ratio.
        ratio = delta if self.rda_fused else compute_rda_value_ratio(beta, delta)
        g1 = rearrange(g1, "n (h d) -> 1 n h d", d=self.head_dim)
        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)
        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._forward(
            mixed_qkv=mixed_qkv,
            g1=g1,
            g2=g2,
            beta=beta,
            ratio=ratio,
            core_attn_out=core_attn_out,
        )
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        if self.kda_proj_persistent:
            return self._proj_mm(core_attn_out, self.o_proj)
        return self.o_proj(core_attn_out)[0]

    def _scale_packed_v(
        self, mixed_qkv: torch.Tensor, ratio: torch.Tensor, beta: torch.Tensor
    ) -> None:
        """Apply the RDA ratio to the value block of a packed conv output.

        ``mixed_qkv`` is ``[T, 3 * local_projection_size]`` — q | k | v, each
        ``local_num_heads * head_dim`` wide — so only the trailing block is
        scaled.  The slice is row-strided (stride ``3P``) but its last axis is
        contiguous, which is exactly the case ``view`` can split, giving a
        writable ``[T, H, D]`` view into the packed buffer.  ``ratio`` is
        ``[1, T, H]``; the fp32 product is rounded once, matching Megatron's
        ``(value.float() * ratio.unsqueeze(-1)).to(value.dtype)``.

        In-place is safe: both call sites hand us a freshly allocated conv
        output buffer, never an aliased projection.
        """
        v_blk = mixed_qkv[:, 2 * self.local_projection_size :].view(
            -1, self.local_num_heads, self.head_dim
        )
        if self.rda_fused:
            rda_scale_v_(v_blk, beta[0], ratio[0])  # ratio holds delta logits
        else:
            v_blk.copy_((v_blk.float() * ratio[0].unsqueeze(-1)).to(v_blk.dtype))

    # ----- RAON_KDA_INKERNEL_FLA: fla-mainline prefill conv -------------------
    def _conv_trace(
        self,
        cu_seqlens,
        single_seq,
        needs_state,
        has_initial_state,
        cache_indices,
        state,
        weight,
    ) -> None:
        """RAON_KDA_CONV_TRACE=<n>: say what shapes this call actually took.

        Job 204514 cost a full RL arm and two diagnostic jobs to establish
        facts the layer could simply have printed: how many sequences were
        packed, whether their lengths were heterogeneous at all, whether a
        prior conv state was present, and which cache slots were touched.
        Inferring those from a traceback is what made the fault expensive.

        ``n`` is the number of DISTINCT shape signatures to print per layer
        (dedup, so a long rollout emits a bounded census rather than a flood).
        OFF by default and it does device syncs when armed, so it is a
        diagnostic lever, never a parity one.
        """
        if self._conv_trace_budget <= 0:
            return
        if cu_seqlens is None:
            lens, nseq = [], 0
        else:
            cu = [int(v) for v in cu_seqlens.tolist()]
            lens = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
            nseq = len(lens)
        if torch.is_tensor(has_initial_state):
            hs = [bool(v) for v in has_initial_state.tolist()]
        else:
            hs = [bool(has_initial_state)] * max(nseq, 1)
        idx = [int(v) for v in cache_indices.tolist()]
        sig = (
            nseq,
            tuple(sorted(set(lens))),
            single_seq,
            needs_state,
            tuple(sorted(set(hs))),
            len(set(idx)) != len(idx),
        )
        if sig in self._conv_trace_seen:
            return
        self._conv_trace_seen.add(sig)
        self._conv_trace_budget -= 1
        print(
            f"[kda-conv-trace] {self.prefix} nseq={nseq} "
            f"len_min={min(lens) if lens else 0} "
            f"len_max={max(lens) if lens else 0} "
            f"len_sum={sum(lens)} len_distinct={len(set(lens))} "
            f"single_seq={single_seq} needs_state={needs_state} "
            f"has_init_true={sum(hs)}/{len(hs)} "
            f"slots=[{min(idx) if idx else -1},{max(idx) if idx else -1}] "
            f"slots_unique={len(set(idx))}/{len(idx)} "
            f"cache_state={tuple(state.shape)} conv_W={weight.shape[1]}",
            flush=True,
        )

    @staticmethod
    def _fla_conv_initial_state(
        state, cache_indices, has_initial_state, conv_width, dtype
    ):
        """vLLM conv cache slice -> fla's ``initial_state`` [N, D, W] contract.

        THE TWO LAYOUTS ARE NOT THE SAME TENSOR (job 204514, the first GPU
        fault of the integration):

        * vLLM stores ``conv_kernel_size - 1`` history columns per channel
          (``mamba_utils.MambaStateShapeCalculator.kda_state_shape``), so the
          gathered slice is ``[N, D, W-1]`` with a per-channel stride of
          ``W-1`` and a per-sequence stride of ``D*(W-1)``.
        * fla documents ``initial_state`` as ``[N, D, W]`` (``W`` = the conv
          KERNEL width) and its Triton kernel addresses it as
          ``initial_state + i_n*D*W + o_d*W + (o_x + W)`` for ``o_x`` in
          ``[-W+1, -1]`` (``fla/modules/conv/triton/kernels.py``
          ``causal_conv1d_fwd_kernel``; ``causal_conv1d_states_fwd_kernel``
          uses the same ``i_n*D*W + o_d*W`` form).  Column 0 is never read.

        Handing the ``[N, D, W-1]`` slice straight to fla therefore (a) reads
        every channel ``d > 0`` from the wrong columns and (b) addresses up to
        ``(N-1)*D*W + (D-1)*W + W-1`` elements of a buffer that only holds
        ``N*D*(W-1)`` -- an overrun of ``N*D`` elements that GROWS WITH THE
        NUMBER OF SEQUENCES.  At one short sequence the overrun stays inside
        the caching allocator's block and the fault is silent (which is why
        the single-sequence smoke passes); at a batched server rollout it
        leaves the allocation and the launch takes an illegal memory access.

        fla also has no per-sequence ``has_initial_state`` (vLLM's
        ``causal_conv1d_fn`` does), so a FRESH sequence landing on a REUSED
        slot would inherit that slot's conv tail.  Those rows are zeroed here.

        Returns a fresh contiguous ``[N, D, conv_width]`` tensor.
        """
        hist = state.index_select(0, cache_indices).to(dtype)  # [N, D, S]
        n, d, s = hist.shape
        history = conv_width - 1
        assert s >= history, (s, conv_width)
        init = hist.new_zeros(n, d, conv_width)
        # History lives in the LEADING W-1 columns, oldest first: stock
        # causal_conv1d_fn and the update kernel read/write columns
        # [0, W-1), and the write-back below stores fla's trailing W-1
        # columns there.  With speculative decoding vLLM widens the cache to
        # S = W-1+num_spec; the extra columns are draft-token slots that may
        # still hold a previous request's drafts, so reading the LAST
        # columns (the old right-align) fed a chunked-prefill continuation
        # stale history.  S == W-1 (no spec) is unchanged.
        init[..., conv_width - history :] = hist[..., :history]
        if torch.is_tensor(has_initial_state):
            keep_row = has_initial_state.to(torch.bool).reshape(-1, 1, 1)
            # `where`, not a multiply: a never-written cache slot can hold NaN
            # bit patterns and NaN * 0 is NaN.
            init = torch.where(keep_row, init, torch.zeros_like(init))
        return init

    def _fla_prefill_conv(
        self,
        q_ns,
        k_ns,
        v_ns,
        q_conv_weight,
        k_conv_weight,
        v_conv_weight,
        q_conv_state,
        k_conv_state,
        v_conv_state,
        has_initial_state,
        cache_indices,
        cu_seqlens,
    ):
        """fla-mainline ``causal_conv1d(backend='triton')``, one call per q/k/v
        section, preserving the training convolution layout.

        Inputs ``q_ns/k_ns/v_ns`` are ``[T, P]`` packed varlen; fla wants
        ``[B, T, D]`` with ``cu_seqlens``, so each is unsqueezed to ``[1, T, P]``.
        Returns ``[T, P]`` per section (2D, like the vLLM ``_prefill_conv``).

        The conv cache is refreshed from fla's final_state so a following decode
        continues correctly.  The parity dump is a single fresh prefill
        (``has_initial_state=False``) — the branch that matters — and this uses
        no prior state there, exactly like the trainer's fast path.
        """
        from fla.modules.conv.causal_conv1d import causal_conv1d as _fla_conv

        bias = self.conv1d.bias
        if bias is not None:
            q_bias, k_bias, v_bias = bias.split(self.local_projection_size, dim=0)
        else:
            q_bias = k_bias = v_bias = None

        # index_select / index_copy_ require a long index; the metadata carries
        # int32 state indices.
        cache_indices = cache_indices.long()
        needs_state = bool(
            has_initial_state.any()
            if torch.is_tensor(has_initial_state)
            else has_initial_state
        )
        single_seq = cu_seqlens is None or int(cu_seqlens.numel()) <= 2
        conv_cu = None if single_seq else cu_seqlens
        self._conv_trace(
            cu_seqlens,
            single_seq,
            needs_state,
            has_initial_state,
            cache_indices,
            q_conv_state,
            q_conv_weight,
        )
        outs = []
        for x, weight, cbias, state in (
            (q_ns, q_conv_weight, q_bias, q_conv_state),
            (k_ns, k_conv_weight, k_bias, k_conv_state),
            (v_ns, v_conv_weight, v_bias, v_conv_state),
        ):
            init = None
            if needs_state:
                init = self._fla_conv_initial_state(
                    state,
                    cache_indices,
                    has_initial_state,
                    weight.shape[1],
                    x.dtype,
                )
            y, fstate = _fla_conv(
                x=x.unsqueeze(0),
                weight=weight,
                bias=cbias,
                initial_state=init,
                output_final_state=True,
                activation="silu",
                backend="triton",
                cu_seqlens=conv_cu,
            )
            if fstate is not None:
                # vLLM extends the allocated conv-cache width from W-1 to
                # W-1+num_spec when speculative decoding is enabled.  The
                # extra columns are *future draft-token slots*, not prefill
                # history.  Stock causal_conv1d_fn therefore initializes only
                # the leading W-1 columns during prefill; the speculative
                # update kernel fills the extension later.  FLA returns W
                # history values, so mirror the stock layout exactly instead
                # of trying to copy them into the full extended allocation.
                history = weight.shape[1] - 1
                state[..., :history].index_copy_(
                    0,
                    cache_indices,
                    fstate[..., -history:].to(state.dtype),
                )
            outs.append(y.squeeze(0))
        return outs[0], outs[1], outs[2]

    # ----- RAON_KDA_CHUNK_REPLAY: zero-mismatch chunk-replay decode -----------
    def _cr_replay_chunk(self, q, k, v, g, beta, init):
        """One replay call: chunk_kda over REAL tokens from the committed state.

        q/k/v/g are [1, L, H, D], beta [1, L, H], init [1, H, D, D].  Returns
        (out [1, L, H, D], final_state [1, H, D, D]) with BOTH CLONED out of the
        kernel's reused triton workspace -- carrying either as-returned silently
        (and nondeterministically) drifts ~1.8e-3 (job 166242).  No padding: the
        gate is derived from raw_g and the decay factored to the chunk end, so a
        zero pad is not neutral (~9e-2, job 166105) -- replay real tokens only.
        """
        from .ops.kda import (
            chunk_kda_with_fused_gate,
        )

        assert self.gate_lower_bound is not None
        L = q.shape[1]
        # [P1] cu_seqlens site 4 of 4 -- the chunk-replay seam.  This one calls
        # the vLLM-VENDORED kda, whose prepare_chunk_indices lives in
        # vllm.third_party.flash_linear_attention.ops.index and is memoized by
        # the same identity-based tensor_cache, so the object reuse pays here
        # too; that copy takes no cu_seqlens_cpu, so only the reuse applies.
        # (Inert under v1: RAON_KDA_CHUNK_REPLAY=0.)
        cu, _cu_cpu_unused = cu_seqlens_cached(
            lambda: torch.tensor([0, L], device=q.device, dtype=torch.int32),
            (0, L),
            q.device,
        )
        o, fs = chunk_kda_with_fused_gate(
            q=q,
            k=k,
            v=v,
            raw_g=g,
            raw_beta=beta,
            A_log=self.A_log,
            g_bias=self.dt_bias,
            lower_bound=self.gate_lower_bound,
            initial_state=init,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu,
        )
        return o.clone(), fs.clone()

    def _cr_prefill_commit(
        self,
        q_ns,
        k_ns,
        v_ns,
        g1_ns,
        beta_ns,
        cu,
        state_indices,
        initial_state,
        recurrent_state,
        g2_ns=None,
    ):
        """chunk-replay prefill seam: commit each prefill sequence's state at its
        LAST 64-token boundary (not at prompt_len) and buffer the open-chunk tail.

        This keeps chunk boundaries anchored at absolute position 0, the way the
        trainer chunks prompt+generation, so decode continues the same chunking
        and the trailing partial chunk lands at the same length the trainer's
        full-seq decomposition uses.  The prefill OUTPUT (core_attn_out) is left
        untouched -- it already equals the full-prompt chunk_kda.  q_ns/k_ns/v_ns
        are the post-conv, RDA-scaled [1, n, h, d] prefill activations; g1_ns
        [1, n, h, d]; beta_ns [1, n, h].
        """
        C = self.kda_replay_chunk
        starts = cu.tolist()
        for si in range(len(starts) - 1):
            s, e = int(starts[si]), int(starts[si + 1])
            k64 = ((e - s) // C) * C
            slot = int(state_indices[si])
            # Slot reuse across requests: this sequence starts a fresh deferred
            # finalized stream.
            self._cr_finalized.pop(slot, None)
            self._cr_finalized_g2.pop(slot, None)
            if k64 == 0:
                recurrent_state[slot] = initial_state[si].clone()
            else:
                _, committed = self._cr_replay_chunk(
                    q_ns[:, s : s + k64],
                    k_ns[:, s : s + k64],
                    v_ns[:, s : s + k64],
                    g1_ns[:, s : s + k64],
                    beta_ns[:, s : s + k64],
                    initial_state[si : si + 1],
                )
                recurrent_state[slot] = committed[0]
            buf = {"q": [], "k": [], "v": [], "g": [], "beta": [], "g2": []}
            for t in range(s + k64, e):
                buf["q"].append(q_ns[0, t])
                buf["k"].append(k_ns[0, t])
                buf["v"].append(v_ns[0, t])
                buf["g"].append(g1_ns[0, t])
                buf["beta"].append(beta_ns[0, t])
                if g2_ns is not None:
                    buf["g2"].append(g2_ns[t])  # g2_ns is [n, H, D] (3D)
            if buf["q"]:
                self._cr_open[slot] = buf
            else:
                self._cr_open.pop(slot, None)

    def _cr_ensure_slot_buffer(self, slot, q_d, g1_ns, beta_ns, g2_ns):
        """Preallocated PER-SLOT open-chunk buffer (2026-09-22 v2 -- host-RSS
        부검(job vs-b1-official-2177691, host SIGKILL rc=137) 수리의 재수리.

        v1(전 슬롯 [num_slots, chunk_len, ...] 1개씩, 하이브리드 검증 패턴
        0a8aaf34d 전사)은 job vs-b2-zero-3arm-finish-2877820 에서 CUDA
        OOM 으로 즉사했다: num_slots(recurrent_state.shape[0])는 실제
        동시 디코드 시퀀스 수(max_num_seqs=32)가 아니라 엔진의 이론상
        최대 동시성(이 잡의 KV 캐시 실측: "Maximum concurrency ... 1022.71x"
        -- 약 1024 슬롯)이라, k 버퍼 torch.empty(1024, 64, H, D) 단 1개가
        4.37 GiB 를 요구했고 그 시점 GPU 여유는 4.27 GiB 뿐이었다(이미
        263.40/267.69 GiB 사용 중인 조밀한 배경). 슬롯 6종 x 레이어 18개
        규모로는 실제 쓰이는 슬롯 수와 무관하게 처음부터 전체 예산을
        초과했다.

        수리: self._cr_open 자신의 setdefault-per-slot 패턴과 동형으로,
        슬롯 하나당 [chunk_len, ...] 버퍼 하나만 그 슬롯이 처음 열릴 때
        지연 할당한다. 발자국이 "이 레이어가 실제로 만난 서로 다른 슬롯
        수"에 비례하지, 엔진 전체 이론상 슬롯 수에 비례하지 않는다 --
        정상 가동에서는 max_num_seqs(32) 근방에서 안정된다(구 .clone()
        경로의 실측 사용 패턴과 같은 규모). 값 안전성(재생이 덮어쓰기
        전에 값이 복사돼 있다는 성질)은 v1 과 동일, copy_ 이후로는 그
        슬롯에 대해 스텝당 신규 CUDA 할당이 0.
        """
        C = self.kda_replay_chunk
        H, D = q_d.shape[-2], q_d.shape[-1]
        Hb = beta_ns.shape[-1]
        g2_dtype = g2_ns.dtype if g2_ns is not None else None
        key = (H, D, Hb, q_d.dtype, g1_ns.dtype, beta_ns.dtype, g2_dtype, q_d.device)
        bufs = self._cr_slot_bufs.get(slot)
        if bufs is not None and bufs["key"] == key:
            return bufs
        dev = q_d.device
        bufs = {
            "key": key,
            "q": torch.empty(C, H, D, dtype=q_d.dtype, device=dev),
            "k": torch.empty(C, H, D, dtype=q_d.dtype, device=dev),
            "v": torch.empty(C, H, D, dtype=q_d.dtype, device=dev),
            "g": torch.empty(C, H, D, dtype=g1_ns.dtype, device=dev),
            "beta": torch.empty(C, Hb, dtype=beta_ns.dtype, device=dev),
            "g2": (
                torch.empty(C, H, D, dtype=g2_dtype, device=dev)
                if g2_ns is not None
                else None
            ),
        }
        self._cr_slot_bufs[slot] = bufs
        return bufs

    def _cr_decode(
        self,
        mixed_qkv_ns,
        g1_ns,
        beta_ns,
        decode_conv_indices,
        recurrent_state,
        g2_ns=None,
    ):
        """chunk-replay decode: per slot, append the new token to its open-chunk
        buffer and replay the whole open chunk (REAL tokens, no padding) through
        chunk_kda from the committed 64-boundary state.

        Returns per step the GROWING open chunk's newest row -- PROVISIONAL, for
        sampling only, and NOT bitwise (each open length is a distinct kernel
        shape).  The DEFERRED, bitwise output is recorded into self._cr_finalized:
        when a 64-token chunk closes, that one completed-64-chunk replay yields
        all 64 rows bitwise-equal to the trainer's full-seq chunk_kda (under the
        pinned autotune), and those rows -- not this per-step output -- are what
        the rollout worker scores logprobs from.  The trailing open chunk is
        finalized once, at its final length, by _cr_flush.  State AND output are
        cloned out of the reused workspace inside _cr_replay_chunk.
        """
        P, H, D = self.local_projection_size, self.local_num_heads, self.head_dim
        t_dec = mixed_qkv_ns.size(0)
        q_d, k_d, v_d = (
            rearrange(x, "n (h d) -> n h d", d=D) for x in mixed_qkv_ns.split(P, dim=-1)
        )
        out = torch.empty(
            1, t_dec, H, D, dtype=mixed_qkv_ns.dtype, device=mixed_qkv_ns.device
        )
        slots = _decode_conv_slots_host(self, decode_conv_indices)
        for i in range(t_dec):
            slot = int(slots[i])
            buf = self._cr_open.setdefault(
                slot, {"q": [], "k": [], "v": [], "g": [], "beta": [], "g2": []}
            )
            sbuf = self._cr_ensure_slot_buffer(slot, q_d, g1_ns, beta_ns, g2_ns)
            L_prev = len(buf["q"])
            sbuf["q"][L_prev].copy_(q_d[i])
            sbuf["k"][L_prev].copy_(k_d[i])
            sbuf["v"][L_prev].copy_(v_d[i])
            sbuf["g"][L_prev].copy_(g1_ns[0, i])
            sbuf["beta"][L_prev].copy_(beta_ns[0, i])
            buf["q"].append(sbuf["q"][L_prev])
            buf["k"].append(sbuf["k"][L_prev])
            buf["v"].append(sbuf["v"][L_prev])
            buf["g"].append(sbuf["g"][L_prev])
            buf["beta"].append(sbuf["beta"][L_prev])
            if g2_ns is not None:
                sbuf["g2"][L_prev].copy_(g2_ns[i])
                buf["g2"].append(sbuf["g2"][L_prev])  # g2_ns is [n, H, D] (3D)
            L = len(buf["q"])
            o, fs = self._cr_replay_chunk(
                torch.stack(buf["q"]).unsqueeze(0),
                torch.stack(buf["k"]).unsqueeze(0),
                torch.stack(buf["v"]).unsqueeze(0),
                torch.stack(buf["g"]).unsqueeze(0),
                torch.stack(buf["beta"]).unsqueeze(0),
                recurrent_state[slot : slot + 1],
            )
            out[0, i] = o[0, L - 1]
            if self.kda_replay_chunk == L:
                # 64-boundary: commit the completed-chunk state (bitwise) and
                # record its 64 finalized rows for deferred logprob replay.
                recurrent_state[slot] = fs[0]
                self._cr_finalized.setdefault(slot, []).append(o[0])
                if buf["g2"]:
                    self._cr_finalized_g2.setdefault(slot, []).append(
                        torch.stack(buf["g2"])
                    )
                self._cr_open.pop(slot, None)
                self._cr_slot_bufs.pop(slot, None)
        return out

    def _cr_flush(self, slot, recurrent_state):
        """Finalize a slot's trailing OPEN chunk at its actual length (deferred).

        Replays the buffered tail once from the committed 64-boundary state -- a
        single fixed shape, hence bitwise vs the trainer's final partial chunk
        under the pinned autotune -- appends its rows to the finalized queue and
        clears the open buffer.  Returns the slot's full finalized deferred KDA
        output [n, H, D] (every completed chunk + this tail, from the last
        committed prefill boundary onward), or None if the slot has nothing.
        """
        buf = self._cr_open.get(slot)
        if buf and buf["q"]:
            o, _ = self._cr_replay_chunk(
                torch.stack(buf["q"]).unsqueeze(0),
                torch.stack(buf["k"]).unsqueeze(0),
                torch.stack(buf["v"]).unsqueeze(0),
                torch.stack(buf["g"]).unsqueeze(0),
                torch.stack(buf["beta"]).unsqueeze(0),
                recurrent_state[slot : slot + 1],
            )
            self._cr_finalized.setdefault(slot, []).append(o[0])
            if buf["g2"]:
                self._cr_finalized_g2.setdefault(slot, []).append(
                    torch.stack(buf["g2"])
                )
            self._cr_open.pop(slot, None)
            self._cr_slot_bufs.pop(slot, None)
        fin = self._cr_finalized.get(slot)
        return torch.cat(fin, dim=0) if fin else None

    def cr_finalized_snapshot(self, slot, recurrent_state):
        """Public, NON-destructive read of a slot's DEFERRED finalized KDA core
        output [n, H, D] from its last committed prefill 64-boundary onward.

        Every already-closed 64-chunk block (bitwise vs the trainer's full-seq
        chunk_kda under the pinned autotune) PLUS a preview replay of the
        still-open trailing chunk at its current length.  Unlike ``_cr_flush`` it
        does NOT pop the open buffer, so it is safe to call from a per-step
        forward hook.  ``o`` is cloned out of the reused kernel workspace inside
        ``_cr_replay_chunk``.

        Returns None if the slot has recorded nothing yet.  This is the RAW KDA
        core (pre-``o_norm``/``o_proj``) -- the quantity the isolated op gate
        (``run/validate_kda_deferred.py``) compares against the trainer's
        chunk_kda.  For the full KDA *site* (what the layer probe / trainer dump
        compare, kda.py forward: ``o_proj(o_norm(core, g2))``) use
        ``cr_finalized_site_snapshot`` instead.
        """
        core, _ = self._cr_finalized_core_g2(slot, recurrent_state)
        return core

    def _cr_finalized_core_g2(self, slot, recurrent_state):
        """(core [n,H,D], g2 [n,H,D] or None) for the finalized + open-preview
        window, NON-destructively.  g2 is the o_norm gate captured in lockstep
        with each finalized block; it is None when g2 was not threaded in (the
        isolated op harness)."""
        blocks = list(self._cr_finalized.get(slot, []))
        g2_blocks = list(self._cr_finalized_g2.get(slot, []))
        buf = self._cr_open.get(slot)
        if buf and buf["q"]:
            o, _ = self._cr_replay_chunk(
                torch.stack(buf["q"]).unsqueeze(0),
                torch.stack(buf["k"]).unsqueeze(0),
                torch.stack(buf["v"]).unsqueeze(0),
                torch.stack(buf["g"]).unsqueeze(0),
                torch.stack(buf["beta"]).unsqueeze(0),
                recurrent_state[slot : slot + 1],
            )
            blocks.append(o[0])
            if buf["g2"]:
                g2_blocks.append(torch.stack(buf["g2"]))
        if not blocks:
            return None, None
        core = torch.cat(blocks, dim=0)
        g2 = torch.cat(g2_blocks, dim=0) if g2_blocks else None
        return core, g2

    def cr_finalized_site_snapshot(self, slot, recurrent_state):
        """Public, NON-destructive read of a slot's DEFERRED finalized KDA SITE
        output [n, hidden] -- the exact quantity the layer probe / trainer dump
        compare: ``o_proj(o_norm(core, g2))`` (kda.py forward, lines 292-307 and
        the ``o_norm`` seam at the tail of ``_forward``).

        This is what the rollout worker / layer probe reports for the KDA site
        instead of the per-step provisional growing row.  Returns None if the
        slot has nothing yet, or if g2 was not threaded in (raw-core-only path)."""
        core, g2 = self._cr_finalized_core_g2(slot, recurrent_state)
        if core is None or g2 is None:
            return None
        # Match _forward's o_norm call exactly: core [1, n, H, D] (4D), g2
        # [n, H, D] (3D). o_norm normalizes in place, so clone the stored core.
        gated = self.o_norm(core.unsqueeze(0).clone(), g2)
        gated.shape[1]
        flat = rearrange(gated, "1 n h d -> n (h d)")
        out = self.o_proj(flat)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    # ----- RAON_KDA_CPR: chunk-pinned recurrence seams ------------------------
    def _cpr_prefill(
        self, q_ns, k_ns, v_ns, g1_ns, beta_ns, cu, state_indices, has_initial_state
    ):
        """CPR prefill seam (thin wiring; the math is kda_inkernel's
        fla_cpr_prefill_core / CPRSlotTracker).

        Fresh sequences run the CPR forward from zero state.  Resumed
        sequences (chunked prefill) use the donor OPEN-CHUNK REPLAY: the
        slot's buffered open-chunk tokens are prepended and re-run together
        with the new tokens from the committed fp32 entry, keeping the chunk
        grid anchored at absolute position 0 (P3/P4 bitwise, misaligned
        splits included); the replayed rows are stripped from the output.

        Returns ``(core [1, n, H, D], cache_states [nseq, H, V, K] fp32)`` --
        the per-sequence recurrent-cache states (running state mid-chunk, or
        the fp32(bf16(entry)) resync when the sequence ends ON a boundary),
        written back by the stock ``recurrent_state[...] = ...`` below.
        """
        from vllm.models.raon1_nano.kda_inkernel import fla_cpr_prefill_core

        starts = [int(x) for x in cu.tolist()]
        nseq = len(starts) - 1
        prepend, entries0 = [], []
        for si in range(nseq):
            slot = int(state_indices[si])
            hs = bool(has_initial_state[si]) if has_initial_state is not None else False
            if not hs:
                # Fresh sequence on a (possibly reused) slot.
                self._cpr.drop(slot)
                prepend.append(None)
                entries0.append(None)
                continue
            if slot not in self._cpr.entry and self._cpr.open_len(slot) == 0:
                raise RuntimeError(
                    f"RAON_KDA_CPR: sequence resumes slot {slot} but no CPR "
                    "entry-chain bookkeeping exists for it (engine restart / "
                    "state migration is not supported under this lever)"
                )
            prepend.append(self._cpr.open_stacked(slot))
            entries0.append(self._cpr.entry.get(slot))

        if any(p is not None for p in prepend):
            parts = {key: [] for key in ("q", "k", "v", "g", "b")}
            cu_aug, n_pre = [0], []
            for si in range(nseq):
                s, e = starts[si], starts[si + 1]
                npre = 0
                if prepend[si] is not None:
                    pq, pk, pv, pg, pb = prepend[si]
                    npre = pq.shape[1]
                    parts["q"].append(pq[0])
                    parts["k"].append(pk[0])
                    parts["v"].append(pv[0])
                    parts["g"].append(pg[0])
                    parts["b"].append(pb[0])
                n_pre.append(npre)
                parts["q"].append(q_ns[0, s:e])
                parts["k"].append(k_ns[0, s:e])
                parts["v"].append(v_ns[0, s:e])
                parts["g"].append(g1_ns[0, s:e])
                parts["b"].append(beta_ns[0, s:e])
                cu_aug.append(cu_aug[-1] + npre + (e - s))
            q_aug = torch.cat(parts["q"], dim=0).unsqueeze(0)
            k_aug = torch.cat(parts["k"], dim=0).unsqueeze(0)
            v_aug = torch.cat(parts["v"], dim=0).unsqueeze(0)
            g_aug = torch.cat(parts["g"], dim=0).unsqueeze(0)
            b_aug = torch.cat(parts["b"], dim=0).unsqueeze(0)
            core_aug, seq_info = fla_cpr_prefill_core(
                q_aug,
                k_aug,
                v_aug,
                g_aug,
                b_aug,
                self.A_log,
                self.dt_bias,
                self._inkernel_scale,
                self.gate_lower_bound,
                entries0=entries0,
                cu_seqlens=torch.tensor(cu_aug, device=q_ns.device, dtype=torch.int32),
            )
            rows = [
                core_aug[0, cu_aug[si] + n_pre[si] : cu_aug[si + 1]]
                for si in range(nseq)
            ]
            core = torch.cat(rows, dim=0).unsqueeze(0)
        else:
            q_aug, k_aug, v_aug, g_aug, b_aug = q_ns, k_ns, v_ns, g1_ns, beta_ns
            core, seq_info = fla_cpr_prefill_core(
                q_aug,
                k_aug,
                v_aug,
                g_aug,
                b_aug,
                self.A_log,
                self.dt_bias,
                self._inkernel_scale,
                self.gate_lower_bound,
                entries0=entries0,
                cu_seqlens=cu,
            )

        cache_states = []
        for si in range(nseq):
            info = seq_info[si]
            slot = int(state_indices[si])
            ts, tl = info["tail_start"], info["tail_len"]
            if tl > 0:
                self._cpr.reset(
                    slot,
                    info["entry"],
                    q_tail=q_aug[0, ts : ts + tl],
                    k_tail=k_aug[0, ts : ts + tl],
                    v_tail=v_aug[0, ts : ts + tl],
                    g_tail=g_aug[0, ts : ts + tl],
                    beta_tail=b_aug[0, ts : ts + tl],
                )
            else:
                self._cpr.reset(slot, info["entry"])
            cache_states.append(info["running"][0])
        return core, torch.stack(cache_states, dim=0)

    def _cpr_decode_track(
        self, mixed_qkv_ns, g1_ns, beta_ns, decode_conv_indices, recurrent_state
    ):
        """CPR decode seam: buffer each slot's decoded token (post-conv,
        RDA-scaled, PRE-l2norm) and, when its open chunk reaches the absolute
        64-boundary, advance the fp32 entry chain and RESYNC the slot's
        recurrent cache to the bf16-snapshot-upcast (kda_inkernel's
        CPRSlotTracker.append).  Runs AFTER the packed decode kernel produced
        the step's live output from the pre-resync running state."""
        # Capture-safe route (RAON_KDA_CPR_GRAPH_LEDGER=1): scatter the rows
        # into the device-resident ledger — pure device ops, no host readback,
        # so this is legal inside a FULL decode-only capture.  The 64-boundary
        # advance/resync this method's host loop used to do runs in the
        # kda_cpr_graph epilogue before the NEXT step's forward (the next live
        # decode step reads the identical resynced bytes).
        if getattr(self, "_cpr_graph_ledger", None) is not None:
            P, D = self.local_projection_size, self.head_dim
            q_d, k_d, v_d = (
                rearrange(x, "n (h d) -> n h d", d=D)
                for x in mixed_qkv_ns.split(P, dim=-1)
            )
            self._cpr_graph_ledger.append_graph(
                decode_conv_indices, q_d, k_d, v_d, g1_ns[0], beta_ns[0]
            )
            return
        from vllm.models.raon1_nano.kda_inkernel import _FLA_AUTOTUNE_PINNED

        if (
            mixed_qkv_ns.size(0) > 1
            and _FLA_AUTOTUNE_PINNED
            and os.environ.get("RAON_KDA_AUTOTUNE_CONFIG_INDEX", "0") == "0"
            and os.environ.get("RAON_KDA_CPR_BOUNDARY_BATCH", "1") == "1"
        ):
            from vllm.models.raon1_nano.kda_boundary_batch import track

            return track(
                self, mixed_qkv_ns, g1_ns, beta_ns, decode_conv_indices, recurrent_state
            )
        # Each row view owns a private immutable batch snapshot. This replaces
        # five clones per slot with three per batch without retaining live buffers.
        owned = os.environ.get("RAON_KDA_CPR_BATCH_SNAPSHOT", "1") == "1"
        if owned:
            mixed_qkv_ns = mixed_qkv_ns.clone()
            g1_ns = g1_ns.clone()
            beta_ns = beta_ns.clone()
        P, D = self.local_projection_size, self.head_dim
        q_d, k_d, v_d = (
            rearrange(x, "n (h d) -> n h d", d=D) for x in mixed_qkv_ns.split(P, dim=-1)
        )
        # [18] gated: reuse the step-shared host copy (RAON_KDA_CONV_IDX_CACHE;
        # OFF = exactly the old per-layer tolist).
        slots = _decode_conv_slots_host(self, decode_conv_indices)
        for i in range(mixed_qkv_ns.size(0)):
            slot = int(slots[i])
            resync = self._cpr.append(
                slot,
                q_d[i],
                k_d[i],
                v_d[i],
                g1_ns[0, i],
                beta_ns[0, i],
                self.A_log,
                self.dt_bias,
                self._inkernel_scale,
                self.gate_lower_bound,
                _owned=owned,
            )
            if resync is not None:
                recurrent_state[slot] = resync[0]

    # Preserve the Nano RDA value scale at the post-convolution seams.
    # This forward path uses the Mini-derived base and local recurrence kernels.
    @eager_break_during_capture
    def _forward(
        self,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        ratio: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if attn_metadata_raw is None:
            return

        from .ops.kda import chunk_kda_with_fused_gate
        from .rda_recurrent import (
            fused_recurrent_rda,
            fused_recurrent_rda_packed_decode,
        )

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata_narrowed, RaonRDAMetadata)
        m = attn_metadata_narrowed
        if os.environ.get("RAON_KDA_DECODE_GRAPH", "1") == "1":
            from .kda_decode_graph import try_forward

            if try_forward(self, mixed_qkv, g1, g2, beta, ratio, core_attn_out, m):
                return
        has_initial_state = m.has_initial_state
        non_spec_query_start_loc = m.non_spec_query_start_loc
        non_spec_state_indices_tensor = m.non_spec_state_indices_tensor
        spec_token_indx = m.spec_token_indx
        non_spec_token_indx = m.non_spec_token_indx
        spec_state_indices_tensor = m.spec_state_indices_tensor
        spec_query_start_loc = m.spec_query_start_loc
        num_accepted_tokens = m.num_accepted_tokens
        num_actual_tokens = m.num_actual_tokens
        has_spec_decode = m.num_spec_decodes > 0
        if self.kda_cpr and has_spec_decode:
            raise RuntimeError(
                "RAON_KDA_CPR does not support speculative decode (the "
                "spec path bypasses the 64-boundary entry-chain bookkeeping)"
            )
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]
        ratio = ratio[:, :num_actual_tokens]

        conv_state, recurrent_state = self.kv_cache
        # The convolution kernels consume (..., dim, width - 1).
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        q_conv_weight, k_conv_weight, v_conv_weight = conv_weights.split(
            self.local_projection_size, dim=0
        )
        q_conv_state, k_conv_state, v_conv_state = conv_state.split(
            self.local_projection_size, dim=-2
        )

        # Separate multi-query speculative tokens from prefill/plain decode.
        if has_spec_decode:
            if m.num_prefills == 0 and m.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                g1_spec, beta_spec, ratio_spec = g1, beta, ratio
                mixed_qkv_ns = g1_ns = beta_ns = ratio_ns = g2_ns = None
            else:
                assert spec_token_indx is not None
                assert non_spec_token_indx is not None
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                g1_spec = g1.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                ratio_spec = ratio.index_select(1, spec_token_indx)
                mixed_qkv_ns = mixed_qkv.index_select(0, non_spec_token_indx)
                g1_ns = g1.index_select(1, non_spec_token_indx)
                beta_ns = beta.index_select(1, non_spec_token_indx)
                ratio_ns = ratio.index_select(1, non_spec_token_indx)
                # g2 (the o_norm gate) rides the same non-spec split as g1 so the
                # chunk-replay finalize can reproduce the o_norm'd KDA site. NOTE
                # g2 is [num_tokens, H, D] (3D, forward() line 292) -- unlike g1
                # which is [1, num_tokens, H, D] -- so the token axis is dim 0.
                g2_ns = g2.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = g1_spec = beta_spec = ratio_spec = None
            mixed_qkv_ns, g1_ns, beta_ns, ratio_ns = mixed_qkv, g1, beta, ratio
            g2_ns = g2

        # Spec-decode multi-query path.
        core_attn_out_spec = None
        if has_spec_decode:
            assert spec_state_indices_tensor is not None
            assert spec_query_start_loc is not None
            spec_conv_indices = spec_state_indices_tensor[:, 0][: m.num_spec_decodes]
            spec_max_query_len = spec_state_indices_tensor.size(-1)
            spec_conv_out = torch.empty_like(mixed_qkv_spec)
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                activation="silu",
                conv_state_indices=spec_conv_indices,
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_max_query_len,
                validate_data=False,
                out=spec_conv_out,
            )
            self._scale_packed_v(mixed_qkv_spec, ratio_spec, beta_spec)  # RDA
            q_spec, k_spec, v_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                for x in mixed_qkv_spec.split(self.local_projection_size, dim=-1)
            )
            spec_cu_seqlens = spec_query_start_loc[: m.num_spec_decodes + 1]
            spec_out = (
                core_attn_out[:, : q_spec.shape[1]]
                if m.num_prefills == 0 and m.num_decodes == 0
                else None
            )
            core_attn_out_spec, _ = fused_recurrent_rda(
                q=q_spec,
                k=k_spec,
                v=v_spec,
                raw_g=g1_spec,
                raw_beta=beta_spec,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                lower_bound=self.gate_lower_bound,
                initial_state=recurrent_state,
                cu_seqlens=spec_cu_seqlens,
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                out=spec_out,
            )

        # Prefill or plain-decode path.
        core_attn_out_non_spec = None
        if mixed_qkv_ns is not None:
            assert g1_ns is not None and beta_ns is not None
            if m.num_prefills > 0:
                q_ns, k_ns, v_ns = mixed_qkv_ns.split(
                    self.local_projection_size, dim=-1
                )

                # Separate convolution calls accept row-strided packed inputs
                # and produce dense Q/K/V without an additional V copy.
                def _prefill_conv(
                    x: torch.Tensor,
                    state: torch.Tensor,
                    weight: torch.Tensor,
                ) -> torch.Tensor:
                    return causal_conv1d_fn(
                        x.transpose(0, 1),
                        weight,
                        None,
                        activation="silu",
                        conv_states=state,
                        has_initial_state=has_initial_state,
                        cache_indices=non_spec_state_indices_tensor,
                        query_start_loc=non_spec_query_start_loc,
                        metadata=m,
                    ).transpose(0, 1)

                if self.kda_inkernel_fla or self.kda_fla_seams:
                    # fla-mainline causal_conv1d(backend='triton'), one call per
                    # q/k/v section, matching _hf_fla_training_forward.  x is
                    # [1, T, P] packed varlen (cu_seqlens); weight [P, W]; bias
                    # the per-section conv bias (None for RAON).  Update the conv
                    # cache from fla's final_state so a following decode is not
                    # silently wrong (irrelevant for the prefill-only dump).
                    # RAON_KDA_FLA_SEAMS(2026-09-22)도 같은 경로: conv 접합부
                    # 만 fla 로, core 는 아래 분기 그대로 (vendored/plain).
                    q_ns, k_ns, v_ns = self._fla_prefill_conv(
                        q_ns,
                        k_ns,
                        v_ns,
                        q_conv_weight,
                        k_conv_weight,
                        v_conv_weight,
                        q_conv_state,
                        k_conv_state,
                        v_conv_state,
                        has_initial_state,
                        non_spec_state_indices_tensor,
                        non_spec_query_start_loc,
                    )
                else:
                    q_ns = _prefill_conv(q_ns, q_conv_state, q_conv_weight)
                    k_ns = _prefill_conv(k_ns, k_conv_state, k_conv_weight)
                    v_ns = _prefill_conv(v_ns, v_conv_state, v_conv_weight)
                q_ns, k_ns, v_ns = (
                    rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                    for x in (q_ns, k_ns, v_ns)
                )
                # RDA, post-conv: v_ns is [1, n, h, d], ratio_ns is [1, n, h].
                if self.rda_fused:
                    rda_scale_v_(v_ns[0], beta_ns[0], ratio_ns[0])
                else:
                    v_ns = (v_ns.float() * ratio_ns.unsqueeze(-1)).to(v_ns.dtype)

                if self.kda_chunk_replay:
                    _cr_q, _cr_k, _cr_v = (q_ns.clone(), k_ns.clone(), v_ns.clone())
                    _cr_g1, _cr_beta = g1_ns.clone(), beta_ns.clone()

                assert non_spec_state_indices_tensor is not None
                assert has_initial_state is not None
                initial_state = gather_initial_states(
                    recurrent_state,
                    non_spec_state_indices_tensor,
                    has_initial_state,
                )
                if self.kda_cpr:
                    # RAON_KDA_CPR prefill: chunk state pass + segmented
                    # recurrent scan (see the lever block in __init__).  Slot
                    # bookkeeping (fp32 entry chain + open-chunk tail buffers)
                    # is updated inside; the returned per-sequence cache
                    # states ride the stock recurrent_state write below.
                    (
                        core_attn_out_non_spec,
                        last_recurrent_state,
                    ) = self._cpr_prefill(
                        q_ns,
                        k_ns,
                        v_ns,
                        g1_ns,
                        beta_ns,
                        non_spec_query_start_loc,
                        non_spec_state_indices_tensor,
                        has_initial_state,
                    )
                elif self.kda_inkernel_fla:
                    # fla-mainline chunk_kda, IN-KERNEL trainer contract.  g1_ns
                    # is the RAW decay (== trainer decay_raw); beta_ns the RAW
                    # beta logits; v_ns already RDA-scaled above.  key_heads ==
                    # value_heads for RAON so no repeat_interleave is needed.
                    (
                        core_attn_out_non_spec,
                        last_recurrent_state,
                    ) = fla_inkernel_core(
                        q_ns,
                        k_ns,
                        v_ns,
                        g1_ns,
                        beta_ns,
                        self.A_log,
                        self.dt_bias,
                        self._inkernel_scale,
                        self.gate_lower_bound,
                        initial_state=initial_state,
                        cu_seqlens=(
                            None
                            if int(non_spec_query_start_loc.numel()) <= 2
                            else non_spec_query_start_loc
                        ),
                        output_final_state=True,
                    )
                elif self.kda_prefill_backend == "flashkda":
                    assert self.gate_lower_bound is not None
                    # 0.28.0 API change: fetch out/final_state/workspace from
                    # the workspace arena exactly as stock kda.py:723-731 does.
                    assert self._flashkda_buffer_specs is not None
                    workspace_out, final_state, workspace = _flashkda_dedicated_buffers(
                        self._flashkda_buffer_specs, mixed_qkv.device
                    )
                    # Stock writes straight into core_attn_out unless the spec
                    # path also needs it, in which case the output must be
                    # index_copy_'d back by token order (stock kda.py:729-731).
                    flashkda_out = (
                        workspace_out if has_spec_decode else core_attn_out
                    )[:, : q_ns.shape[1]]
                    (
                        core_attn_out_non_spec,
                        last_recurrent_state,
                    ) = _flashkda_prefill(
                        q=q_ns,
                        k=k_ns,
                        v=v_ns,
                        g=g1_ns,
                        beta=beta_ns,
                        A_log=self.A_log,
                        dt_bias=self.dt_bias,
                        lower_bound=self.gate_lower_bound,
                        initial_state=initial_state,
                        cu_seqlens=non_spec_query_start_loc,
                        out=flashkda_out,  # 0.28.0: required out-param
                        # 0.28.0: required out-param, one [H, D, D] slot per
                        # prefill sequence (stock kda.py:747).
                        final_state=final_state[: initial_state.shape[0]],
                        workspace=workspace,  # 0.28.0: required scratch buffer
                    )
                else:
                    (
                        core_attn_out_non_spec,
                        last_recurrent_state,
                    ) = chunk_kda_with_fused_gate(
                        q=q_ns,
                        k=k_ns,
                        v=v_ns,
                        raw_g=g1_ns,
                        raw_beta=beta_ns,
                        A_log=self.A_log,
                        g_bias=self.dt_bias,
                        lower_bound=self.gate_lower_bound,
                        initial_state=initial_state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                        cu_seqlens=non_spec_query_start_loc,
                    )
                if self.kda_chunk_replay:
                    core_attn_out_non_spec = core_attn_out_non_spec.clone()
                    # Commit at the last 64-boundary from the PRISTINE inputs +
                    # buffer the open-chunk tail so decode continues
                    # absolute-position-0 chunking.
                    self._cr_prefill_commit(
                        _cr_q,
                        _cr_k,
                        _cr_v,
                        _cr_g1,
                        _cr_beta,
                        non_spec_query_start_loc,
                        non_spec_state_indices_tensor,
                        initial_state,
                        recurrent_state,
                        g2_ns=g2_ns,
                    )
                else:
                    recurrent_state[non_spec_state_indices_tensor] = (
                        last_recurrent_state
                    )
            else:
                # Pure non-speculative decode.
                assert non_spec_state_indices_tensor is not None
                decode_conv_indices = non_spec_state_indices_tensor[
                    : mixed_qkv_ns.size(0)
                ]
                # RAON_KDA_SGL_FUSED: one-kernel decode step (conv + RDA +
                # recurrence + o_norm).  Handles the WHOLE step including the
                # tail o_norm, so a handled batch returns here -- reaching the
                # tail again would double-normalize.  A False return (pool
                # off-contract, capture-stale caches) falls through to the
                # unchanged chain below, announced once.  Spec-mixed batches
                # are excluded: the early return would skip the spec rows'
                # index_copy merge and their o_norm.
                if self.kda_sgl_fused and not has_spec_decode:
                    from . import kda_sgl_fused as _sgl_fused

                    if _sgl_fused.try_decode(
                        self,
                        mixed_qkv_ns,
                        g1_ns,
                        g2_ns,
                        beta_ns,
                        ratio_ns,
                        decode_conv_indices,
                        core_attn_out,
                        num_actual_tokens,
                    ):
                        return
                packed_conv_out = torch.empty_like(mixed_qkv_ns)
                mixed_qkv_ns = causal_conv1d_update(
                    mixed_qkv_ns,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    activation="silu",
                    conv_state_indices=decode_conv_indices,
                    validate_data=True,
                    out=packed_conv_out,
                )
                self._scale_packed_v(mixed_qkv_ns, ratio_ns, beta_ns)  # RDA
                if self.kda_chunk_replay:
                    if not _CR_DECODE_LIVE_ANNOUNCED[0]:
                        _CR_DECODE_LIVE_ANNOUNCED[0] = True
                        print(
                            "[parity] RAON_KDA_CHUNK_REPLAY LIVE: "
                            "_cr_decode engaged for the KDA decode seam "
                            "(deferred completed-chunk replay path)",
                            flush=True,
                        )
                    # Deferred completed-chunk replay: per-step provisional rows
                    # for sampling; bitwise finalized rows recorded in
                    # self._cr_finalized at each 64-boundary (+ _cr_flush tail).
                    core_attn_out_non_spec = self._cr_decode(
                        mixed_qkv_ns,
                        g1_ns,
                        beta_ns,
                        decode_conv_indices,
                        recurrent_state,
                        g2_ns=g2_ns,
                    )
                elif self.kda_decode_backend == "flashkda":
                    # Each decode token is its own one-token varlen sequence;
                    # gather the per-request recurrent states, run the chunk
                    # kernel, scatter the final states back.  v is already
                    # RDA-scaled in the packed tail above.
                    t_dec = mixed_qkv_ns.size(0)
                    q_d, k_d, v_d = (
                        rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                        for x in mixed_qkv_ns.split(self.local_projection_size, dim=-1)
                    )
                    cu = torch.arange(
                        0,
                        t_dec + 1,
                        device=mixed_qkv_ns.device,
                        dtype=torch.int32,
                    )
                    init = gather_initial_states(
                        recurrent_state,
                        decode_conv_indices,
                        torch.ones(
                            t_dec,
                            dtype=torch.bool,
                            device=mixed_qkv_ns.device,
                        ),
                    )
                    assert self.gate_lower_bound is not None
                    # 0.28.0 API change: same three out-params as the prefill
                    # site.  Unlike stock's prefill this always takes the
                    # workspace output buffer rather than aliasing
                    # core_attn_out: the decode branch already copies its
                    # result out below, and keeping it unaliased means the
                    # mixed spec+decode case needs no special casing.
                    assert self._flashkda_buffer_specs is not None
                    workspace_out, final_state, workspace = _flashkda_dedicated_buffers(
                        self._flashkda_buffer_specs, mixed_qkv.device
                    )
                    (
                        core_attn_out_non_spec,
                        last_state,
                    ) = _flashkda_prefill(
                        q=q_d,
                        k=k_d,
                        v=v_d,
                        g=g1_ns,
                        beta=beta_ns,
                        A_log=self.A_log,
                        dt_bias=self.dt_bias,
                        lower_bound=self.gate_lower_bound,
                        initial_state=init,
                        cu_seqlens=cu,
                        out=workspace_out[:, :t_dec],  # 0.28.0: required
                        final_state=final_state[:t_dec],  # 0.28.0: required
                        workspace=workspace,  # 0.28.0: required
                    )
                    recurrent_state[decode_conv_indices] = last_state
                elif self.kda_cpr:
                    # RAON_KDA_CPR exact-step decode (third-source fix, jobs
                    # 193224/193329): the vendored packed decode kernel's fp32
                    # running-state arithmetic drifts 1 ulp/step (~300 elems)
                    # against the fla scan that produces the CPR prefill bits,
                    # flipping sparse bf16 core outputs within each open chunk
                    # (the 39/119 headline's residual — first logprob
                    # divergence 47 was L02's flip).  The chained one-token
                    # fla fused_recurrent_kda is PROVEN bitwise the scan
                    # (193329: 64/64 outputs, final state 0/655,360), so the
                    # live step runs on that kernel: gather the fp32 slot
                    # states, one one-token segment per request, scatter the
                    # final states back.  CPR semantics now INCLUDE the
                    # exact-step kernel; the 64-boundary entry-chain resync
                    # (_cpr_decode_track below) is unchanged and now discards
                    # a zero drift.
                    from vllm.models.raon1_nano.kda_inkernel import fla_cpr_decode_step

                    q_d, k_d, v_d = (
                        rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                        for x in mixed_qkv_ns.split(self.local_projection_size, dim=-1)
                    )
                    (
                        core_attn_out_non_spec,
                        _cpr_finals,
                    ) = fla_cpr_decode_step(
                        q_d,
                        k_d,
                        v_d,
                        g1_ns,
                        beta_ns,
                        self.A_log,
                        self.dt_bias,
                        self._inkernel_scale,
                        self.gate_lower_bound,
                        recurrent_state[decode_conv_indices],
                    )
                    recurrent_state[decode_conv_indices] = _cpr_finals
                    # Buffer the token and, at each absolute 64-boundary,
                    # advance the fp32 entry chain (chunk state pass over the
                    # REAL buffered tokens) and RESYNC the cache slot to
                    # fp32(bf16(entry)).
                    self._cpr_decode_track(
                        mixed_qkv_ns,
                        g1_ns,
                        beta_ns,
                        decode_conv_indices,
                        recurrent_state,
                    )
                else:
                    (
                        core_attn_out_non_spec,
                        _,
                    ) = fused_recurrent_rda_packed_decode(
                        mixed_qkv=mixed_qkv_ns,
                        raw_g=g1_ns,
                        raw_beta=beta_ns,
                        A_log=self.A_log,
                        dt_bias=self.dt_bias,
                        lower_bound=self.gate_lower_bound,
                        initial_state=recurrent_state,
                        state_indices=decode_conv_indices,
                    )

        # Restore the scheduler's original token order for mixed batches.
        if core_attn_out_spec is not None and core_attn_out_non_spec is not None:
            core_attn_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            core_attn_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
        elif core_attn_out_non_spec is not None:
            # 0.28.0 API change: with the FlashKDA prefill out-param above,
            # core_attn_out_non_spec IS core_attn_out, so this copy is a
            # self-copy.  Stock skips it under the same guard (kda.py:805).
            if self.kda_prefill_backend != "flashkda" or m.num_prefills == 0:
                # TODO: prefill and decode kernels write directly to
                # core_attn_out
                core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
                    0, :num_actual_tokens
                ]
        else:
            assert core_attn_out_spec is not None
        # Triton normalizes in place, so this is a self-copy with no device
        # work. Keep it for the out-of-place native implementation.
        core_attn_out.copy_(self.o_norm(core_attn_out, g2))


if __name__ == "__main__":
    # The two things that corrupt silently: the ratio identity and the
    # packed-tail view. Run on the rootfs interpreter (this module imports vLLM).
    beta = torch.randn(1, 7, 5, dtype=torch.bfloat16)
    assert torch.equal(
        compute_rda_value_ratio(beta, torch.zeros_like(beta)), torch.ones(1, 7, 5)
    ), "zero delta must be exactly identity"
    delta = torch.randn(1, 7, 5, dtype=torch.bfloat16)
    write = torch.sigmoid(beta.float()) * compute_rda_value_ratio(beta, delta)
    assert torch.allclose(write, torch.sigmoid(beta.float() + delta.float()), atol=1e-6)

    h, d = 5, 4
    packed = torch.zeros(2, 3 * h * d)
    packed[:, 2 * h * d :] = 1.0
    v_blk = packed[:, 2 * h * d :].view(-1, h, d)
    v_blk.copy_(v_blk * torch.arange(1, h + 1, dtype=torch.float32)[:, None])
    assert packed[:, : 2 * h * d].eq(0).all(), "q/k blocks must be untouched"
    assert torch.equal(
        packed[0, 2 * h * d :].view(h, d),
        torch.arange(1, h + 1, dtype=torch.float32)[:, None].expand(h, d),
    ), "per-head ratio must broadcast across head_dim, not across heads"
    if torch.cuda.is_available():
        T, H, D = 3, 5, 128
        beta_l = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
        delta_l = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
        packed = torch.randn(T, 3 * H * D, dtype=torch.bfloat16, device="cuda")
        ref = packed.clone()
        r = compute_rda_value_ratio(beta_l.unsqueeze(0), delta_l.unsqueeze(0))
        rv = ref[:, 2 * H * D :].view(-1, H, D)
        rv.copy_((rv.float() * r[0].unsqueeze(-1)).to(rv.dtype))
        rda_scale_v_(packed[:, 2 * H * D :].view(-1, H, D), beta_l, delta_l)
        dmax = (packed.float() - ref.float()).abs().max().item()
        assert dmax == 0.0, f"fused vs torch max abs diff {dmax}"
        print("cuda fused ok")
    print("ok")

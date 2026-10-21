# Copyright (c) 2026, KRAFTON. Zero-mismatch W8A8 backend.
"""RAON_MOE_MXFP8_W8A8_GEMM_MODE=2: trtllm-gen MxFp8 + SiTuGlu fused MoE backend.

  * The TRTLLM_GEN_BMM artifact of the pinned flashinfer (0.6.16.post3)
    ships w8a8 (activations and weights both MxE4m3) SiTuGlu kernels for
    sm100f, including low-latency decode tiles (t128x16x256, tokens=N=16
    swap-AB).
  * SiTuGlu contract (artifact header GemmGatedActOptions.h:79-85):
      left  = beta  * tanh(x0 / beta)      # x0 = first half = linear(value)
      right = alpha * tanh(x1 / alpha) * sigmoid(x1)   # x1 = gate
    Our activation 100*tanh(g/4)*sig(g)*tanh(v/25) is algebraically equal
    with alpha=4.0, beta=25.0. However, two validators in flashinfer
    (python core._validate_fp8_block_scale_gemm1_activation_params and C++
    validateFp8BlockScaleGemm1ActivationParams, launcher.cu:104-117,
    called at :2361) reject alpha/beta together with Situ, even though the
    kernel plumbing passes them through for any activation (runner.cu:846
    -> ptrGatedActAlpha/Beta, consumed natively by the cubin).
  * This module therefore (i) reads the launcher .cu at runtime, replaces
    only that one ICHECK, and JIT-builds the copy under a separate module
    name (the installed package is left untouched; an anchor mismatch
    fails closed), and (ii) monkeypatches the python validator to accept
    Situ.

Precision: the format matches pretraining (E4M3 + UE8M0 1x32, W8A8). The
arithmetic differs from the training reference (quantizer rounding, a2
requantization, prob fold applied by us outside finalize), so bitwise
equality is not a goal of this mode (mode 0 provides that).

Fallback: no failure in this module silently falls back to Triton or any
other path; exceptions propagate (fail-closed). _FALLBACK_COUNT observes
that contract and must always be 0.
"""

import dataclasses
import functools
import os
from pathlib import Path

import torch

__all__ = [
    "expert_rows_trtllm",
    "prepare_expert_weights",
    "quantize_hidden_mxfp8",
    "invalidate_weight_cache",
    "get_situ_moe_module_info",
]

# SiTuGlu per-expert parameters (alpha = gate beta, beta = linear beta).
# Must match the fail-closed check (4.0, 25.0) at expert_chain entry.
_SITU_ALPHA_GATE = 4.0
_SITU_BETA_LINEAR = 25.0

_MODULE_NAME = "fused_moe_trtllm_sm100_situ"

# launcher.cu replacement anchor. It includes the preceding MxFp8 ICHECK
# message to distinguish it from the identical text in the BF16 launcher
# (around launcher.cu:835). If the source changes, count != 1 raises.
_ANCHOR = (
    '         "Fp8QuantizationType::MxFp8 in FP8 block scale MoE.";\n'
    '  TVM_FFI_ICHECK(activation_type == ActivationType::Swiglu)\n'
    '      << "gemm1_alpha, gemm1_beta, and gemm1_clamp_limit are only'
    ' supported for "\n'
    '         "ActivationType::Swiglu.";\n'
    '}\n'
)
_RELAXED = (
    '         "Fp8QuantizationType::MxFp8 in FP8 block scale MoE.";\n'
    '  TVM_FFI_ICHECK(activation_type == ActivationType::Swiglu ||\n'
    '                 activation_type == ActivationType::Situ)\n'
    '      << "gemm1_alpha, gemm1_beta, and gemm1_clamp_limit are only'
    ' supported for "\n'
    '         "ActivationType::Swiglu or ActivationType::Situ.";\n'
    '}\n'
)

_FALLBACK_COUNT = 0        # always 0 (fail-closed, see module docstring)
_ANNOUNCED = False
_WQ_CACHE: dict = {}       # id(w13) -> (prep dict, w13, w2 original refs)

# flashinfer autotune switch (default 0 = default tactic). When enabled, the
# first call for each shape runs inside the flashinfer autotune context so
# the chosen tactic lands in the process cache and later CUDA graph capture
# picks up the tuned tactic. Autotune only runs outside capture: this relies
# on vLLM eager warmup visiting every shape first, and a first-seen shape
# during capture raises instead of silently using the default tactic.
# RAON_MXFP8_BOOT_AUTOTUNE enables boot-time autotune for both the trtllm
# fused MoE (here) and the dense mm_mxfp8 (mxfp8_align) with a single key.
# Tuning happens on the first call per shape, so it always uses the real
# shapes, and it runs during vLLM eager warmup. Per-kernel durations are not
# a valid cost measure under PDL overlap; compare end-to-end wall time.
_AUTOTUNE = (os.environ.get("RAON_MXW8A8_TRTLLM_AUTOTUNE", "0") == "1"
             or os.environ.get("RAON_MXFP8_BOOT_AUTOTUNE", "0") == "1")
_TUNED_SHAPES: set = set()

# Explicit tactic pin. The default tactic selection can pick a different fc1
# tactic on each boot with large per-call latency differences. Value is
# "i1,i2", a [gemm1Config, gemm2Config] pair from trtllm_get_valid_moe_configs
# for our Situ module build (indices are stable because the module .so is
# pinned). The pin is injected via an op-name filter in
# AutoTuner.choose_one, so other ops are unaffected. A pair outside the
# valid set falls back to the stock selector and increments a counter. Boot
# readback runs the first chain call once under the profiler and checks
# that the executed fc1 kernel name contains _TACTIC_EXPECT; a mismatch
# raises (fail-closed).
_TACTIC_PIN = None
_pin_env = os.environ.get("RAON_MXW8A8_TRTLLM_TACTIC_PIN", "")
if _pin_env:
    _TACTIC_PIN = [int(v) for v in _pin_env.split(",")]
    assert len(_TACTIC_PIN) == 2, _pin_env
_TACTIC_EXPECT = os.environ.get("RAON_MXW8A8_TRTLLM_TACTIC_EXPECT", "")
# Separate fc2 readback (fc1 and fc2 are specified and checked separately).
_TACTIC_EXPECT2 = os.environ.get("RAON_MXW8A8_TRTLLM_TACTIC_EXPECT2", "")
# Token count targeted by the pin (decode capture shape). Calls with other
# num_tokens (e.g. prefill) use the stock selector, and readback only fires
# at this shape (the first chain call may be a prefill).
_TACTIC_PIN_M = int(os.environ.get("RAON_MXW8A8_TRTLLM_TACTIC_M", "32"))
_TACTIC_FALLBACKS = 0     # times a pin outside the valid set fell back
_PIN_INSTALLED = False
_READBACK_DONE = False


_CAPTURE_DUMPED = False


def _dump_tuner_at_capture() -> None:
    """On the first detected graph capture, print a one-line summary of the
    AutoTuner profiling_cache keys (key count + sha of sorted keys). Always
    on, one line per process; used to compare tuner state at capture time
    across runs."""
    global _CAPTURE_DUMPED
    if _CAPTURE_DUMPED or not torch.cuda.is_current_stream_capturing():
        return
    _CAPTURE_DUMPED = True
    try:
        import hashlib
        from flashinfer.autotuner import AutoTuner
        pc = AutoTuner.get().profiling_cache
        keys = sorted(str(k) for k in pc)
        h = hashlib.sha256("\n".join(keys).encode()).hexdigest()[:12]
        print(f"[parity] tuner-state@capture: {len(keys)} tuned keys, "
              f"sha {h}, sample={keys[:1]}", flush=True)
    except Exception as exc:
        print(f"[parity] tuner-state@capture dump failed: "
              f"{type(exc).__name__}", flush=True)


def _install_tactic_pin() -> None:
    """Wrap AutoTuner.choose_one with an op-filtered pin (once per process)."""
    global _PIN_INSTALLED
    if _PIN_INSTALLED or _TACTIC_PIN is None:
        return
    _PIN_INSTALLED = True
    from flashinfer.autotuner import AutoTuner

    _orig = AutoTuner.choose_one

    def _pinned(self, custom_op, runners, tuning_config, inputs, **kwargs):
        global _TACTIC_FALLBACKS
        if (custom_op == "flashinfer::trtllm_fp8_block_scale_moe"
                and _TACTIC_PIN is not None):
            try:
                from flashinfer.fused_moe.core import MoeRunnerInputs
                nt = inputs[MoeRunnerInputs.idx("hidden_states")].shape[0]
            except Exception:
                nt = None
            if nt is not None and nt != _TACTIC_PIN_M:
                # Non-target shape (e.g. prefill): stock selector. This is
                # the expected path, so the fallback counter is not bumped.
                return _orig(self, custom_op, runners, tuning_config,
                             inputs, **kwargs)
            try:
                valid = runners[0].get_valid_tactics(inputs, None)
            except TypeError:
                valid = None    # different signature: skip check, keep pin
            except Exception:
                valid = None
            if valid is not None and list(_TACTIC_PIN) not in [
                    list(v) for v in valid]:
                _TACTIC_FALLBACKS += 1
                print(f"[parity] RAON_MXW8A8_TRTLLM_TACTIC_PIN "
                      f"{_TACTIC_PIN} not in valid set "
                      f"(n={len(valid)}) — fallback #{_TACTIC_FALLBACKS} "
                      f"to stock selector", flush=True)
                return _orig(self, custom_op, runners, tuning_config,
                             inputs, **kwargs)
            return runners[0], list(_TACTIC_PIN)
        return _orig(self, custom_op, runners, tuning_config, inputs,
                     **kwargs)

    AutoTuner.choose_one = _pinned
    print(f"[parity] RAON_MXW8A8_TRTLLM_TACTIC_PIN={_TACTIC_PIN} installed "
          f"(expect fc1 kernel ~ {_TACTIC_EXPECT!r})", flush=True)


def _tactic_readback(run_once) -> None:
    """Boot readback: run once under the profiler and check that the fc1
    (gated Bmm) kernel name contains _TACTIC_EXPECT. Eager only (outside
    graph capture)."""
    global _READBACK_DONE
    if _READBACK_DONE or not _TACTIC_EXPECT:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "RAON_MXW8A8_TRTLLM_TACTIC_EXPECT set but first chain call is "
            "inside graph capture — readback must run pre-capture")
    _READBACK_DONE = True
    from torch.profiler import ProfilerActivity, profile
    run_once()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        run_once()
        torch.cuda.synchronize()
    fc1 = [ev.key for ev in prof.key_averages()
           if "bmm_" in ev.key.lower() and "glu" in ev.key.lower()]
    fc2 = [ev.key for ev in prof.key_averages()
           if ev.key.lower().startswith("bmm_bfloat16")]
    hit1 = [n for n in fc1 if _TACTIC_EXPECT in n]
    if not hit1:
        raise RuntimeError(
            f"tactic readback FAIL (fc1): expected substring "
            f"{_TACTIC_EXPECT!r} not in executed set {fc1[:3]} — "
            f"tactic lottery relapsed (fail-closed)")
    if _TACTIC_EXPECT2:
        hit2 = [n for n in fc2 if _TACTIC_EXPECT2 in n]
        if not hit2:
            raise RuntimeError(
                f"tactic readback FAIL (fc2): expected substring "
                f"{_TACTIC_EXPECT2!r} not in executed set {fc2[:3]} — "
                f"tactic lottery relapsed (fail-closed)")
        print(f"[parity] tactic readback PASS fc2: {hit2[0][:110]}",
              flush=True)
    print(f"[parity] tactic readback PASS: {hit1[0][:110]}", flush=True)


def _maybe_autotune(x, topk_weights, topk_ids, prep, do_finalize):
    if not _AUTOTUNE:
        return
    key = (int(x.shape[0]), bool(do_finalize))
    if key in _TUNED_SHAPES:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "RAON_MXW8A8_TRTLLM_AUTOTUNE=1: first call for shape "
            f"{key} arrived inside CUDA graph capture — autotune must run "
            "pre-capture (eager warmup did not cover this shape)")
    from flashinfer.autotuner import autotune
    with autotune(True):
        for _ in range(3):
            moe_rows_trtllm(x, topk_weights, topk_ids, prep,
                            do_finalize=do_finalize)
    torch.cuda.synchronize()
    _TUNED_SHAPES.add(key)
    print(f"[parity] RAON_MXW8A8_TRTLLM_AUTOTUNE=1: trtllm moe tactics "
          f"tuned for shape {key} (pre-capture)", flush=True)

# Weight physical layout is fixed: shuffled (MajorK row reorder) + 128x4
# interleaved scales. The launcher forces use_shuffled_weight=True for MxFp8
# (launcher.cu:2417); unshuffled weights fail. Interleave produces a
# canonical padded buffer (fp4_quantization block_scale_interleave:
# _compute_swizzled_layout_sf_size(rows, cols, 128)), so gemm2's K/32=34
# (not a multiple of 4) is accommodated by padding. The intermediate%128
# requirement in flashinfer's prepare.py only comes from its reshape_as.
# fc1 half order: trtllm x0 = first half = linear(value), x1 = second half =
# gate. Engine w13 is [gate; value], so the default "vg" reorders it to
# [value; gate].
_DEFAULT_HALF_ORDER = os.environ.get("RAON_MXW8A8_TRTLLM_HALF_ORDER", "vg")
_PERMUTE_CACHE: dict = {}   # equivalent of prepare.py _TRTLLM_FP8_PERMUTE_CACHE


def _patched_launcher_path() -> Path:
    """Read the installed launcher source and write a Situ-enabled copy next
    to the JIT directory."""
    from flashinfer.jit import env as jit_env

    src = Path(jit_env.FLASHINFER_CSRC_DIR) / "trtllm_fused_moe_kernel_launcher.cu"
    text = src.read_text()
    n = text.count(_ANCHOR)
    if n != 1:
        raise RuntimeError(
            f"mxw8a8-trtllm: launcher anchor count={n} (expect 1) in {src} — "
            "pinned flashinfer source drifted; refusing to patch blindly")
    patched = text.replace(_ANCHOR, _RELAXED, 1)
    out_dir = Path(jit_env.FLASHINFER_JIT_DIR) / f"{_MODULE_NAME}-src"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "trtllm_fused_moe_kernel_launcher.cu"
    # Do not rewrite identical content: an mtime change triggers a ninja
    # rebuild.
    if not (out.exists() and out.read_text() == patched):
        out.write_text(patched)
    return out


@functools.cache
def _install_situ_patches() -> None:
    """Patch flashinfer to allow Situ (once per process, fail-closed).

    (1) core.gen_trtllm_gen_fused_moe_sm100_module -> returns the same
        JitSpec with only the launcher source swapped for the copy and a
        separate module name.
    (2) core._validate_fp8_block_scale_gemm1_activation_params -> allows
        Situ (all other combinations are rejected as in stock).
    Refuses if the stock module is already loaded in this process: the
    functools.cache'd _get_trtllm_moe_sm100_module_impl would return the
    stock module.
    """
    from flashinfer.fused_moe import core as ficore
    from flashinfer.jit.fused_moe import (
        gen_trtllm_gen_fused_moe_sm100_module as _orig_gen)
    from flashinfer.tllm_enums import ActivationType, Fp8QuantizationType

    ci = ficore._get_trtllm_moe_sm100_module_impl.cache_info()
    if ci.currsize > 0:
        raise RuntimeError(
            "mxw8a8-trtllm: stock trtllm moe module already loaded in this "
            "process — Situ patch must install before first use (fail-closed)")

    def _gen_situ(enable_rubin: bool = False):
        if enable_rubin:
            raise RuntimeError(
                "mxw8a8-trtllm: rubin(sm107) path not supported by the Situ "
                "patch (Rubin BMM pin predates SiTuGlu — KernelRunner.h)")
        spec = _orig_gen(enable_rubin=False)
        patched = _patched_launcher_path()
        sources = [
            patched if Path(s).name == "trtllm_fused_moe_kernel_launcher.cu"
            else Path(s) for s in spec.sources]
        if not any(Path(s) == patched for s in sources):
            raise RuntimeError(
                "mxw8a8-trtllm: launcher source not found in stock JitSpec — "
                "pinned flashinfer layout drifted")
        return dataclasses.replace(spec, name=_MODULE_NAME, sources=sources)

    _orig_validate = ficore._validate_fp8_block_scale_gemm1_activation_params

    def _validate_situ(fp8_quantization_type, activation_type,
                       gemm1_alpha, gemm1_beta, gemm1_clamp_limit):
        if (Fp8QuantizationType(fp8_quantization_type)
                == Fp8QuantizationType.MxFp8
                and int(activation_type) == int(ActivationType.Situ)):
            return  # the kernel consumes ptrGatedActAlpha/Beta natively
        return _orig_validate(fp8_quantization_type, activation_type,
                              gemm1_alpha, gemm1_beta, gemm1_clamp_limit)

    ficore.gen_trtllm_gen_fused_moe_sm100_module = _gen_situ
    ficore._validate_fp8_block_scale_gemm1_activation_params = _validate_situ


def get_situ_moe_module_info() -> dict:
    """Module name, pinned artifact path and patched launcher sha256."""
    import hashlib
    from flashinfer.artifacts import ArtifactPath, CheckSumHash

    info = {"module_name": _MODULE_NAME,
            "bmm_artifact_path": ArtifactPath.TRTLLM_GEN_BMM,
            "bmm_checksum_pin": CheckSumHash.TRTLLM_GEN_BMM}
    try:
        p = _patched_launcher_path()
        info["patched_launcher_sha256"] = hashlib.sha256(
            p.read_bytes()).hexdigest()
    except Exception as exc:  # info only, not on the execution path
        info["patched_launcher_error"] = f"{type(exc).__name__}: {exc}"
    return info


def quantize_hidden_mxfp8(x_bf16: torch.Tensor):
    """(M, H) bf16 -> (q fp8 (M,H), sf u8 (M, H/32) linear).

    Mirrors flashinfer prepare.py:374-377 (MxFp8 activation prep):
    is_sf_swizzled_layout=False + (M, H/32) reshape. Activation scales use
    the [num_tokens, ...] layout (core.py:1515)."""
    from flashinfer import mxfp8_quantize

    q, sf = mxfp8_quantize(x_bf16.contiguous(), is_sf_swizzled_layout=False)
    return q, sf.view(torch.uint8).reshape(x_bf16.shape[0], -1)


def _quantize_weight_matrix_shuffled(w2d: torch.Tensor, *, gated: bool):
    """(N, K) bf16 -> (q_shuffled fp8 (N,K), sf_swizzled u8 flat,
    q_logical fp8 (N,K), sf_logical u8 (N, K/32)).

    Mirrors flashinfer prepare.py:296-345 (MxFp8 shuffled path) with two
    changes:
      * No intermediate%128 precondition. That requirement comes from
        prepare.py's reshape_as(sf) (which assumes no padding); interleave
        itself produces a canonical padded buffer, which is passed through
        as a flat buffer (the launcher only checks the MxFp8 scale dtype,
        launcher.cu:1345).
      * Also returns the logical (pre-shuffle) q/sf, needed by reference
        comparisons that use pre-reorder semantics."""
    from flashinfer import mxfp8_quantize
    from flashinfer.quantization.fp4_quantization import block_scale_interleave
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache)

    n, k = w2d.shape
    q, sf = mxfp8_quantize(w2d.contiguous(), is_sf_swizzled_layout=False)
    sf = sf.view(torch.uint8).reshape(n, k // 32)
    dev = w2d.device
    if gated:
        permute = _maybe_get_cached_w3_w1_permute_indices(
            _PERMUTE_CACHE, q.view(torch.uint8), 128, is_gated_act_gemm=True)
        permute_sf = _maybe_get_cached_w3_w1_permute_indices(
            _PERMUTE_CACHE, sf, 128, num_elts_per_sf=32,
            is_gated_act_gemm=True)
    else:
        permute = get_w2_permute_indices_with_cache(
            _PERMUTE_CACHE, q.view(torch.uint8), 128)
        permute_sf = get_w2_permute_indices_with_cache(
            _PERMUTE_CACHE, sf, 128, num_elts_per_sf=32)
    q_sh = q.view(torch.uint8)[permute.to(dev)].view(q.dtype).contiguous()
    sf_sh = block_scale_interleave(
        sf[permute_sf.to(dev)].contiguous())      # flat, canonical padded
    return q_sh, sf_sh, q, sf


def prepare_expert_weights(w13: torch.Tensor, w2: torch.Tensor, *,
                           half_order: str = None,
                           keep_logical: bool = False,
                           pad_ffn_to: int = 128):
    """Engine bf16 expert weights -> trtllm shuffled MxFp8 bundle (once at
    load time).

    w13 (E, 2F, K) [gate; value] is reordered by half_order="vg" to
    [x0=value; x1=gate], then each expert is quantized and shuffled. With
    keep_logical=True the pre-shuffle logical q/sf are included as well
    (for reference comparisons only; off by default to save memory).

    pad_ffn_to: zero-pad intermediate (F) to this multiple (default 128,
    the 64/128 alignment rule of upstream swizzle_blockscale; same fix as
    vLLM PR #57351 "[Bugfix][NVFP4 MoE] Pad gated intermediate for
    FlashInfer/CUTLASS"; without it getValidConfigIndices rejects every
    config). Zero padding is an arithmetic identity: padded fc1 rows give
    SiTU(0,0) = 0 and the padded K rows of w2 are also 0, so output bits
    are unchanged. Cost: weight reads grow by F_pad/F (1088->1152 = +5.9%).
    0 or 1 disables padding (diagnostics only)."""
    half_order = half_order or _DEFAULT_HALF_ORDER
    E, n13, K = w13.shape
    ffn = n13 // 2
    ffn_orig = ffn
    if pad_ffn_to and pad_ffn_to > 1 and ffn % pad_ffn_to != 0:
        ffn_pad = (ffn + pad_ffn_to - 1) // pad_ffn_to * pad_ffn_to
        zc = torch.zeros((E, ffn_pad - ffn, K), dtype=w13.dtype,
                         device=w13.device)
        gate = torch.cat([w13[:, :ffn], zc], dim=1)
        value = torch.cat([w13[:, ffn:], zc], dim=1)
        w13 = torch.cat([gate, value], dim=1)     # (E, 2*ffn_pad, K) [g; v]
        w2 = torch.cat([w2, torch.zeros((E, K, ffn_pad - ffn),
                                        dtype=w2.dtype, device=w2.device)],
                       dim=2)
        ffn = ffn_pad
    if half_order == "vg":
        w13_r = torch.cat([w13[:, ffn:], w13[:, :ffn]], dim=1)
    elif half_order == "gv":
        w13_r = w13
    else:
        raise ValueError(f"unknown half_order {half_order!r}")
    q1, s1, q1l, s1l = [], [], [], []
    q2, s2, q2l, s2l = [], [], [], []
    for e in range(E):
        qs, ss, ql, sl = _quantize_weight_matrix_shuffled(
            w13_r[e], gated=True)
        q1.append(qs); s1.append(ss)
        if keep_logical:
            q1l.append(ql); s1l.append(sl)
        qs, ss, ql, sl = _quantize_weight_matrix_shuffled(w2[e], gated=False)
        q2.append(qs); s2.append(ss)
        if keep_logical:
            q2l.append(ql); s2l.append(sl)
    dev = w13.device
    prep = {
        "gemm1_weights": torch.stack(q1).contiguous(),
        "gemm1_weights_scale": torch.stack(s1).contiguous(),
        "gemm2_weights": torch.stack(q2).contiguous(),
        "gemm2_weights_scale": torch.stack(s2).contiguous(),
        "gemm1_alpha": torch.full((E,), _SITU_ALPHA_GATE,
                                  dtype=torch.float32, device=dev),
        "gemm1_beta": torch.full((E,), _SITU_BETA_LINEAR,
                                 dtype=torch.float32, device=dev),
        "half_order": half_order, "sf_layout": "shuffled128x4",
        "intermediate_size": ffn, "ffn_orig": ffn_orig,
        "num_experts": E, "hidden_size": K,
    }
    if keep_logical:
        prep["gemm1_q_logical"] = torch.stack(q1l)
        prep["gemm1_sf_logical"] = torch.stack(s1l)
        prep["gemm2_q_logical"] = torch.stack(q2l)
        prep["gemm2_sf_logical"] = torch.stack(s2l)
    return prep


_PREP_TENSOR_KEYS = ("gemm1_weights", "gemm1_weights_scale",
                     "gemm2_weights", "gemm2_weights_scale",
                     "gemm1_alpha", "gemm1_beta")
_PREP_META_KEYS = ("half_order", "sf_layout", "intermediate_size",
                   "ffn_orig", "num_experts", "hidden_size")


def refit_requantize_in_place() -> int:
    """Requantize cached weights in place instead of dropping the cache.
    Same approach as mxfp8_align._requant_inplace.

    Clearing _WQ_CACHE and rebuilding lazily would allocate new storage,
    while FULL_DECODE_ONLY CUDA graph replay does not go through Python and
    keeps reading the old (freed) prep addresses, producing NaNs. The 6 prep
    tensors x 24 MoE layers (144 tensors) are kernel arguments baked into
    the captured decode graph (moe_rows_trtllm(x, ..., prep)).

    The MoE outer graph cannot recover on its own: the stamp in
    moe_outer_graph.py only tracks (id, _version, data_ptr, ...) of
    `tuple(module.parameters()) + tuple(module.buffers())`. prep is a
    module-global cache, neither parameter nor buffer, so the stamp never
    sees it, and refit uses param.data.copy_, so the stamp is unchanged and
    cache.clear() never fires. Extending the stamp is not a fix either,
    since FULL_DECODE_ONLY graphs have no Python recapture path.

    Same prepare_expert_weights and same (not invalidated) _PERMUTE_CACHE
    -> bitwise identical to an independent rebuild. Each layer is rebuilt,
    copied in and freed immediately, bounding transient memory to about
    +1.5GB per layer (regenerating all 144 tensors at once would need
    ~28GB).
    """
    n = 0
    for _key, (prep, w13, w2) in list(_WQ_CACHE.items()):
        fresh = prepare_expert_weights(w13.detach(), w2.detach())   # per layer (~+1.5GB transient)
        for k in _PREP_META_KEYS:
            if prep[k] != fresh[k]:
                raise RuntimeError(
                    f"trtllm refit contract violated: {k} changed {prep[k]}->{fresh[k]}")
        for k in _PREP_TENSOR_KEYS:
            old, new = prep[k], fresh[k]
            if old.shape != new.shape or old.dtype != new.dtype:
                raise RuntimeError(
                    f"trtllm refit contract violated: {k} structure changed (re-registration required)")
            p = old.data_ptr()
            with torch.inference_mode():      # refit arrives from a worker extension outside InferenceMode
                old.copy_(new)
            assert old.data_ptr() == p, f"{k}: prequant buffer pointer moved"
        del fresh
        n += 1
    return n


def invalidate_weight_cache() -> int:
    """Requantize in place (returns the number of refreshed entries).

    Ordering matters: this is called before each weight bucket is written,
    where requantizing bakes stale values (harmless, redundant). The call
    that counts is the final one after all buckets are written (at the end
    of _reprocess_weights_after_refit, collective transport only). Refit
    transports without a post-write call leave the last requantization
    baking pre-refit weights: not NaNs, but silently wrong numerics.
    Disabling refit reprocessing has the same effect.
    """
    return refit_requantize_in_place()


def _cached_weights(w13, w2):
    key = id(w13)
    ent = _WQ_CACHE.get(key)
    if ent is None:
        prep = prepare_expert_weights(w13.detach(), w2.detach())
        ent = (prep, w13, w2)      # hold originals so id() stays valid
        _WQ_CACHE[key] = ent
    return ent[0]


def pack_topk(topk_weights: torch.Tensor, topk_ids: torch.Tensor):
    """(T,K) fp32 prob + (T,K) int32 id -> packed int32
    (expert_id << 16) | bf16(prob).view(int16), the routed API contract
    (core.py trtllm_fp8_block_scale_routed_moe docstring). Rounding prob to
    bf16 matches the training reference."""
    w16 = (topk_weights.to(torch.bfloat16).view(torch.int16)
           .to(torch.int32) & 0xFFFF)
    return ((topk_ids.to(torch.int32) << 16) | w16).contiguous()


def moe_rows_trtllm(x, topk_weights, topk_ids, prep, *,
                    do_finalize: bool = False, enable_pdl=None):
    """Quantize + one trtllm routed MoE call. do_finalize=False returns
    (T, K, H) prob-folded rows (same return contract as the mode 0/1
    chain); True returns the final (T, H) output."""
    from flashinfer.fused_moe import core as ficore
    from flashinfer.tllm_enums import ActivationType, Fp8QuantizationType

    _install_situ_patches()
    T, K = topk_ids.shape
    H = prep["hidden_size"]
    a_q, a_sf = quantize_hidden_mxfp8(x)
    packed = pack_topk(topk_weights, topk_ids)
    result = ficore.trtllm_fp8_block_scale_routed_moe(
        packed, None, a_q, a_sf,
        prep["gemm1_weights"], prep["gemm1_weights_scale"],
        prep["gemm2_weights"], prep["gemm2_weights_scale"],
        num_experts=prep["num_experts"], top_k=K,
        n_group=None, topk_group=None,
        intermediate_size=prep["intermediate_size"],
        local_expert_offset=0, local_num_experts=prep["num_experts"],
        routed_scaling_factor=None,
        # launcher.cu:2417-2418: MxFp8 requires use_shuffled_weight=True
        # and weight_layout=0 (MajorK).
        use_shuffled_weight=True, weight_layout=0,
        do_finalize=do_finalize, enable_pdl=enable_pdl,
        fp8_quantization_type=Fp8QuantizationType.MxFp8,
        activation_type=ActivationType.Situ.value,
        gemm1_alpha=prep["gemm1_alpha"], gemm1_beta=prep["gemm1_beta"])
    if do_finalize:
        return result if isinstance(result, torch.Tensor) else result[0]
    gemm2_out, _expert_w, expanded_to_permuted = result
    return _rows_tail_reference(
        gemm2_out, expanded_to_permuted, topk_weights, T, K, H)


def _rows_tail_reference(gemm2_out, expanded_to_permuted, topk_weights,
                         T, K, H):
    """do_finalize=False row tail (unpermute + prob fold), factored out of
    moe_rows_trtllm unchanged so the RAON_MOE_COMBINE_EP_FINALIZE
    cross-check oracle can consume the same trtllm output tensors."""
    idx = expanded_to_permuted.reshape(-1).to(torch.long)
    valid = (idx >= 0).unsqueeze(1)
    rows = gemm2_out.index_select(0, idx.clamp_min(0))
    # Prob fold: the trainer folds the bf16 prob in the fc1 epilogue, but
    # trtllm folds it in finalize, so for do_finalize=False rows we fold the
    # same bf16 prob with an fp32 multiply. The different fold position is
    # an accepted arithmetic difference of this mode.
    probs = (topk_weights.reshape(-1, 1).to(torch.bfloat16)
             .to(torch.float32))
    rows = torch.where(valid, (rows.to(torch.float32) * probs)
                       .to(rows.dtype), torch.zeros_like(rows))
    return rows.view(T, K, H)


_EP_FINALIZE_XCHECKED = False


def moe_ep2_finalized_trtllm(x, topk_weights, topk_ids, prep, *,
                             num_local_experts, enable_pdl=None):
    """RAON_MOE_COMBINE_EP_FINALIZE=1 (EP=2 only): after one trtllm call with
    do_finalize=False, the tail [_rows_tail_reference 5 passes + (T,K,H)
    materialization + moe_align.combine_ep_partial_fp32 re-gather] is
    replaced by a single moe_ep_kernel.finalize_ep2_permuted kernel that
    returns (T, H) bf16.

    Arithmetic is unchanged: the kernel follows the same per-element
    operation order, and the first call cross-checks both implementations
    on the same trtllm outputs (gemm2_out/expanded_to_permuted) with
    torch.equal, failing closed (this excludes run-to-run nondeterminism)."""
    global _EP_FINALIZE_XCHECKED
    from flashinfer.fused_moe import core as ficore
    from flashinfer.tllm_enums import ActivationType, Fp8QuantizationType

    from . import moe_ep_kernel as MEK

    _install_situ_patches()
    T, K = topk_ids.shape
    H = prep["hidden_size"]
    a_q, a_sf = quantize_hidden_mxfp8(x)
    packed = pack_topk(topk_weights, topk_ids)
    result = ficore.trtllm_fp8_block_scale_routed_moe(
        packed, None, a_q, a_sf,
        prep["gemm1_weights"], prep["gemm1_weights_scale"],
        prep["gemm2_weights"], prep["gemm2_weights_scale"],
        num_experts=prep["num_experts"], top_k=K,
        n_group=None, topk_group=None,
        intermediate_size=prep["intermediate_size"],
        local_expert_offset=0, local_num_experts=prep["num_experts"],
        routed_scaling_factor=None,
        # launcher.cu:2417-2418: MxFp8 requires use_shuffled_weight=True
        # and weight_layout=0 (MajorK).
        use_shuffled_weight=True, weight_layout=0,
        do_finalize=False, enable_pdl=enable_pdl,
        fp8_quantization_type=Fp8QuantizationType.MxFp8,
        activation_type=ActivationType.Situ.value,
        gemm1_alpha=prep["gemm1_alpha"], gemm1_beta=prep["gemm1_beta"])
    gemm2_out, _expert_w, expanded_to_permuted = result
    ids_c = topk_ids.contiguous()
    order = torch.argsort(ids_c, dim=-1, stable=True)
    fused = MEK.finalize_ep2_permuted(
        gemm2_out.contiguous(), expanded_to_permuted.reshape(-1),
        topk_weights.to(torch.float32), ids_c, num_local_experts,
        order=order)
    if not _EP_FINALIZE_XCHECKED:
        _EP_FINALIZE_XCHECKED = True
        from . import moe_align as MA
        rows_ref = _rows_tail_reference(
            gemm2_out, expanded_to_permuted, topk_weights, T, K, H)
        ref = MA.combine_ep_partial_fp32(
            rows_ref.contiguous(), ids_c, ep_size=2,
            num_local_experts=num_local_experts,
            out_dtype=torch.bfloat16)
        if not torch.equal(fused, ref):
            d = (fused.float() - ref.float()).abs()
            raise RuntimeError(
                "RAON_MOE_COMBINE_EP_FINALIZE=1 live cross-check FAILED: "
                "fused ep2-finalize kernel != reference tail on a "
                f"{tuple(x.shape)} input -- "
                f"{int((fused != ref).sum())}/{ref.numel()} elements, "
                f"max|d|={d.max().item():.6e}. The tail's output bits are "
                "the contract; refusing to run the fused plumbing. Set "
                "RAON_MOE_COMBINE_EP_FINALIZE=0 to fall back.")
        _single = MA.combine_unweighted_fp32(
            rows_ref, ids_c, out_dtype=torch.bfloat16)
        _d = (fused.float() - _single.float()).abs()
        print(
            "[parity] RAON_MOE_COMBINE_EP_FINALIZE=1 LIVE: trtllm FIN0 "
            "tail (unpermute+prob fold+EP2 combine) absorbed into ONE "
            f"kernel on a {tuple(x.shape)} input -- cross-check PASS "
            f"(torch.equal vs reference tail, {fused.numel()} elements); "
            f"EP boundary live: vs single-stage contract "
            f"max|d|={_d.max().item():.6e} over "
            f"{int((_d > 0).sum())}/{_d.numel()} differing elements",
            flush=True)
    return fused


def expert_rows_trtllm(x, topk_weights, topk_ids, w13, w2, layer_idx=None,
                       do_finalize=False, ep_finalize_local=None):
    """Mode 2 entry point, called from the branch in
    mxfp8_w8a8_triton.expert_chain_mxw8a8. do_finalize=False (default):
    (T, K, H) prob-folded bf16 rows (combine tail contract unchanged).
    do_finalize=True (RAON_MOE_MXFP8_W8A8_FINALIZE=1): (T, H) bf16 with
    unpermute + prob fold + sum done in the kernel's finalize; the combine
    tail is skipped. This is faster; the different fold position and
    summation order are ULP-level differences of the same class as the
    rest of mode 2."""
    global _ANNOUNCED
    _first_install = id(w13) not in _WQ_CACHE
    prep = _cached_weights(w13, w2)
    if _first_install:
        # Record te_precision_scope with the same API as the Triton branch,
        # at prequant install time (mxfp8_align._record_scope: env gated,
        # skips non-integer layer_idx; record_te_precision is idempotent).
        from . import mxfp8_align as MX
        MX._record_scope(layer_idx, ("routed_fc1", "routed_fc2"))
    if not _ANNOUNCED:
        _ANNOUNCED = True
        print(f"[parity] RAON_MOE_MXFP8_W8A8_TRITON=1 LIVE (gemm_mode=2): "
              f"trtllm-gen MxFp8+SiTuGlu fused MoE on layer_idx={layer_idx}, "
              f"module={_MODULE_NAME}, alpha={_SITU_ALPHA_GATE}, "
              f"beta={_SITU_BETA_LINEAR}, half_order={prep['half_order']}, "
              f"sf_layout={prep['sf_layout']}, "
              f"ffn={prep['ffn_orig']}->pad{prep['intermediate_size']}, "
              f"finalize={'1' if do_finalize else '0'}, "
              f"autotune={'1' if _AUTOTUNE else '0'}, "
              f"tactic_pin={_TACTIC_PIN}, "
              f"tactic_fallbacks={_TACTIC_FALLBACKS}, "
              f"fallback=forbidden"
              f"(fail-closed, count={_FALLBACK_COUNT})", flush=True)
    _install_tactic_pin()
    _dump_tuner_at_capture()
    _maybe_autotune(x, topk_weights, topk_ids, prep, do_finalize)
    if x.shape[0] == _TACTIC_PIN_M:
        # Readback only at the pin's target shape (if the first call is a
        # prefill, it fires on the first decode-shaped call instead).
        _tactic_readback(lambda: moe_rows_trtllm(
            x, topk_weights, topk_ids, prep, do_finalize=do_finalize))
    # RAON_MOE_COMBINE_EP_FINALIZE branch: shares all preparation above
    # (cache, scope record, announce, pin, autotune) and only swaps the
    # final execution for the fused tail. ep_finalize_local=None (default)
    # keeps the existing path.
    if ep_finalize_local is not None:
        if do_finalize:
            raise RuntimeError(
                "expert_rows_trtllm: do_finalize=True and ep_finalize_local are "
                "mutually exclusive (finalize cannot reproduce the two-stage EP combine)")
        return moe_ep2_finalized_trtllm(
            x, topk_weights, topk_ids, prep,
            num_local_experts=ep_finalize_local)
    return moe_rows_trtllm(x, topk_weights, topk_ids, prep,
                           do_finalize=do_finalize)

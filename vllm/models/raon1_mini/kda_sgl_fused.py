"""RAON_KDA_SGL_FUSED: one-kernel KDA decode step, vendored from SGLang.

Replaces the plain-decode four-launch chain
``causal_conv1d_update -> _scale_packed_v (RDA) -> fused_recurrent_kda_packed_
decode -> o_norm`` with a single kernel (conv1d update + RDA v-scale +
delta-rule recurrence + sigmoid-gated RMSNorm), vendored from
sgl-project/sglang @ f65c70bb (kernels/jit/csrc/attention/kda_fused_decode.cuh,
PR #30113, NVIDIA x Moonshot Kimi K3 package) with a 40-head instantiation and
an in-kernel RDA epilogue -- see kda_sgl_fused_kernel.cu's header for the
anchored patch list.

Contract notes (not bitwise-exact against the unfused chain):

* The kernel keeps conv output / RDA-scaled v / recurrence output in fp32
  registers where the unfused chain rounds to bf16 at each seam, and uses
  fast-math intrinsics (__expf, fast sigmoid/softplus/silu).  Output is
  allclose to, not bitwise-equal with, the unfused chain.  Do not enable it
  together with the bitwise-exact decode paths.
* RDA rides in-kernel: the fp32 ratio sigmoid(beta+delta)/sigmoid(beta)
  (computed on the host exactly as today, aten logsigmoid chain) multiplies
  the SiLU'd conv output of v -- the same seam _scale_packed_v uses.
  Under RAON_RDA_FUSED=1 (`ratio` carries raw delta logits) try_decode
  computes the fp32 ratio itself with the same logsigmoid math; the unfused
  fallback still receives the raw delta it expects.
* Decode-only batches (num_prefills == 0, no spec decode).  Prefill and every
  other decode backend fall through to the existing chain untouched.
* CUDA-graph-safe: fixed launch shape per batch size, no host syncs, padded
  slots (index < 0) zero their output row in-kernel.  Weight-derived caches
  are prepared OUTSIDE capture only (vLLM's eager warmup passes run first);
  a stale cache during capture falls back to the unfused chain for that
  capture rather than allocating inside it.

Build: torch cpp_extension JIT into a node-local directory (directory renames
on shared filesystems are unsafe).  CUDA_HOME must point at a CUDA 13.0
toolkit.
"""

from __future__ import annotations

import os

import torch

_LEVER = "RAON_KDA_SGL_FUSED"
_BUILD_DIR_ENV = "RAON_KDA_SGL_FUSED_BUILD_DIR"
_DEFAULT_BUILD_DIR = "/var/tmp/{user}/sgl-kda-fused-build"

_module = None
_announced: set = set()


def lever_on() -> bool:
    return os.environ.get(_LEVER, "0") == "1"


def _announce(key: str, msg: str) -> None:
    if key not in _announced:
        _announced.add(key)
        print(msg, flush=True)


def _load_module():
    """Compile (or load the cached build of) the vendored kernel."""
    global _module
    if _module is not None:
        return _module
    from torch.utils.cpp_extension import load

    src = os.path.join(os.path.dirname(__file__), "kda_sgl_fused_kernel.cu")
    build_dir = os.environ.get(
        _BUILD_DIR_ENV,
        _DEFAULT_BUILD_DIR.format(user=os.environ.get("USER", "unknown")),
    )
    os.makedirs(build_dir, exist_ok=True)
    _module = load(
        name="kda_sgl_fused",
        sources=[src],
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=os.environ.get("RAON_KDA_SGL_FUSED_VERBOSE", "0") == "1",
    )
    return _module


def validate_module(self, vllm_config=None) -> None:
    """Called from __init__ when enabled; checks the kernel contract."""
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is not None and (getattr(spec, "num_speculative_tokens", 0) or 0) > 0:
        # vLLM widens the conv pool to W-1+num_spec columns, which try_decode's
        # pool contract rejects, so the kernel could never run: every decode
        # would run the unfused chain while a spec-off reference with the
        # same env runs this kernel, which is not bitwise with that chain.
        raise ValueError(
            f"{_LEVER}=1 cannot take effect with speculative decoding (conv "
            "pool widened to W-1+num_speculative_tokens).  Set it to 0 for "
            "speculative engines and for their spec-off parity reference."
        )
    if getattr(self, "kda_chunk_replay", False) or getattr(self, "kda_cpr", False):
        raise ValueError(
            f"{_LEVER}=1 is not bitwise-exact and cannot be combined with "
            "the bitwise-exact decode paths (RAON_KDA_CHUNK_REPLAY / "
            "RAON_KDA_CPR own the decode seam there).  Unset one."
        )
    if getattr(self, "kda_decode_backend", "fused") == "flashkda":
        raise ValueError(
            f"{_LEVER}=1 conflicts with RAON_KDA_DECODE=flashkda (both claim "
            "the plain-decode branch).  Unset one."
        )
    if self.head_dim != 128 or self.conv_size != 4:
        raise ValueError(
            f"{_LEVER}=1 requires head_dim=128 / conv width 4 (kernel "
            f"constants); got head_dim={self.head_dim}, "
            f"conv_size={self.conv_size}"
        )
    if self.local_num_heads not in (3, 6, 12, 40):
        raise ValueError(
            f"{_LEVER}=1: kernel instantiated for local head counts "
            f"3/6/12/40, got {self.local_num_heads}"
        )


def _prep(self) -> dict | None:
    """Weight-derived kernel constants, cached on the module and re-derived
    when any source parameter is rebound (refit) or mutated in place.

    Not prepared during graph capture: allocation inside a capture would pin
    the buffers to the graph pool and a stale stamp could not be repaired.
    """
    w = self.conv1d.weight  # [3P, 1, W] fp32
    bias = self.conv1d.bias
    onw = self.o_norm.weight
    stamp = (
        w.data_ptr(), w._version,
        None if bias is None else (bias.data_ptr(), bias._version),
        onw.data_ptr(), onw._version,
        self.A_log.data_ptr(), self.A_log._version,
        self.dt_bias.data_ptr(), self.dt_bias._version,
    )
    cache = getattr(self, "_sgl_fused_prep", None)
    if cache is not None and cache["stamp"] == stamp:
        return cache
    if torch.cuda.is_current_stream_capturing():
        return None  # repaired on the next eager step

    P = self.local_projection_size
    W = self.conv_size
    taps = w.view(3 * P, W).float()
    q_t, k_t, v_t = (
        t.t().contiguous() for t in taps.split(P, dim=0)
    )  # each [W, P], width-major as the kernel reads them
    if bias is None:
        bias_flat = torch.zeros(3 * P, dtype=torch.float32, device=w.device)
    else:
        bias_flat = bias.reshape(-1).float().contiguous()
    cache = {
        "stamp": stamp,
        "w_q_t": q_t,
        "w_k_t": k_t,
        "w_v_t": v_t,
        "conv_bias": bias_flat,
        "A_log": self.A_log.detach().float().contiguous(),
        "dt_bias": self.dt_bias.detach().reshape(-1).float().contiguous(),
        "onorm_weight": onw.detach().float().contiguous(),
        "scale": float(self.head_dim) ** -0.5,
    }
    self._sgl_fused_prep = cache
    return cache


def try_decode(
    self,
    mixed_qkv_ns: torch.Tensor,   # [n, 3P] bf16, PRE-conv, row-strided ok
    g1_ns: torch.Tensor,          # [1, n, H, D] bf16 raw forget gate
    g2_ns: torch.Tensor,          # [n, H, D] bf16 o_norm gate
    beta_ns: torch.Tensor,        # [1, n, H] bf16 raw beta logits
    ratio_ns: torch.Tensor,       # [1, n, H] fp32 RDA ratio
    decode_conv_indices: torch.Tensor,  # [n] int32 (< 0 = padded slot)
    core_attn_out: torch.Tensor,  # [1, T, H, D] bf16
    num_actual_tokens: int,
) -> bool:
    """Run the fused decode step; True when handled (output INCLUDES o_norm,
    caller must return without running the tail o_norm), False to fall back.
    """
    n = mixed_qkv_ns.size(0)
    H, D = self.local_num_heads, self.head_dim
    seg = H * D

    conv_pool, state_pool = self.kv_cache
    # Time-major [slots, width-1, conv_dim] pool (layout "SD", the default);
    # a DS pool would need a transposed kernel and does not occur here.
    if not (
        conv_pool.dim() == 3
        and conv_pool.size(1) == 3
        and conv_pool.size(2) == 3 * seg
        and conv_pool.dtype == torch.bfloat16
        and conv_pool.stride(2) == 1
        and state_pool.dim() == 4
        and state_pool.shape[1:] == (H, D, D)
        and state_pool.dtype == torch.float32
        # both state staging paths 16B-load from state + slot*stride
        and state_pool.stride(0) % 4 == 0
    ):
        _announce(
            "pool",
            f"[perf] {_LEVER} SKIPPED: kv_cache pool layout off-contract "
            f"(conv {tuple(conv_pool.shape)} {conv_pool.dtype}, state "
            f"{tuple(state_pool.shape)} {state_pool.dtype}); expected "
            f"time-major [slots, 3, {3 * seg}] bf16 + [slots, {H}, {D}, {D}] "
            "fp32 -- decode falls back to the unfused chain",
        )
        return False

    prep = _prep(self)
    if prep is None:
        _announce(
            "capture-prep",
            f"[perf] {_LEVER} SKIPPED for this capture: weight caches were "
            "stale inside a CUDA-graph capture; the unfused chain was "
            "captured instead (an eager step repairs the cache)",
        )
        return False

    a = g1_ns[0].reshape(n, seg)
    b = beta_ns[0]
    onorm_g = g2_ns.reshape(n, seg)
    if getattr(self, "rda_fused", False):
        # Under RAON_RDA_FUSED, `ratio` carries the RAW delta logits (the
        # unfused seam computes the ratio inside its Triton kernel).  Compute
        # the fp32 ratio here with the exact compute_rda_value_ratio math
        # (aten logsigmoid chain, [1, n, H] -- a few microseconds); the fused
        # kernel consumes the ratio directly.  The unfused fallback below
        # still receives the raw delta it expects.
        import torch.nn.functional as F

        beta_fp32 = beta_ns.float()
        ratio = torch.exp(
            F.logsigmoid(beta_fp32 + ratio_ns.float()) - F.logsigmoid(beta_fp32)
        )[0]
    else:
        ratio = ratio_ns[0]
    if ratio.dtype != torch.float32:
        _announce(
            "ratio-dtype",
            f"[perf] {_LEVER} SKIPPED: ratio dtype {ratio.dtype} != fp32 "
            "-- unfused chain",
        )
        return False
    out = core_attn_out[0, :n].view(n, seg)

    mod = _load_module()
    mod.run(
        mixed_qkv_ns,
        a,
        b,
        conv_pool,
        prep["w_q_t"],
        prep["w_k_t"],
        prep["w_v_t"],
        prep["conv_bias"],
        prep["A_log"],
        prep["dt_bias"],
        onorm_g,
        prep["onorm_weight"],
        state_pool,
        decode_conv_indices,
        out,
        ratio,
        prep["scale"],
        float(self.o_norm.eps),
        0.0 if self.gate_lower_bound is None else float(self.gate_lower_bound),
        self.gate_lower_bound is not None,
    )
    _announce(
        "live",
        f"[perf] {_LEVER} LIVE: fused KDA decode step (conv+RDA+recurrence+"
        f"o_norm, 1 launch) handled a decode batch of n={n} "
        f"(H={H}, lower_bound={self.gate_lower_bound}, "
        f"tma={'on' if state_pool.stride(0) % 4 == 0 else 'off'})",
    )
    return True

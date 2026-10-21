"""RAON_KDA_SGL_FUSED=1 under speculative decoding: one-kernel KDA step for speculative verification.

kda.py enables this module from RAON_KDA_SGL_FUSED whenever
num_speculative_tokens > 0.

The speculative-decode counterpart of RAON_KDA_SGL_FUSED (kda_sgl_fused.py,
which refuses spec engines because vLLM widens the conv pool).  Replaces the
spec branch of ``_forward``

    causal_conv1d_update(num_accepted_tokens) -> _scale_packed_v (RDA)
    -> fused_recurrent_kda(ssm_state_indices [N, 1+k]) -> o_norm

with one launch (kda_sgl_fused_spec_kernel.cu) for batches made only of spec
rows -- in a spec engine that is every decode step, because kda_spec_state.py
relabels zero-draft decode rows as spec rows of query length 1, so this kernel
gives spec engines what RAON_KDA_SGL_FUSED gives plain ones.  Batches with
prefill or plain-decode rows fall through to the unchanged chain (announced
once).

Cache contract: identical to vLLM's spec kernels (read state at
``idx[n, num_accepted-1]``, write token t's state to ``idx[n, t]``, conv
rolling window of length T+2 in ``idx[n, 0]``), which is also FlashInfer
v0.7.1 ``packed_fused_kda_decode``'s contract.  Design notes and upstream
anchors are in the kernel header.

Contract notes (not bitwise-exact against the unfused chain):
bf16 rounding is kept at the unfused chain's seams but reduction order / FMA
contraction differ from the Triton kernels, so outputs and checkpoints are
allclose, not bitwise.  The conv cache update is a pure copy and IS bitwise.

Build: torch cpp_extension JIT into a node-local directory, like
kda_sgl_fused.py.
"""

from __future__ import annotations

import os

import torch

_LEVER = "RAON_KDA_SGL_FUSED"  # spec-verify kernel; message label
_BUILD_DIR_ENV = "RAON_KDA_SGL_FUSED_SPEC_BUILD_DIR"
_DEFAULT_BUILD_DIR = "/var/tmp/{user}/sgl-kda-fused-spec-build"
_MAX_VERIFY = 8  # kernel instantiations: 2 / 4 / 8 positions
# Value-row split of one head over a thread-block cluster (1, 2 or 4 blocks);
# 0 = auto: split 2 while the grid is below one wave of SMs (N*H < 148 -- only
# N=1 at H=40), else 1.  On B300, split 1 is fastest for N >= 16 and split 2
# wins only at N = 1 (15.9 vs 22.7 us).
_SPLIT = int(os.environ.get("RAON_KDA_SGL_FUSED_SPEC_SPLIT", "0"))
_ONE_WAVE_BLOCKS = 148

_module = None
_announced: set = set()
_calls = [0]


def lever_on() -> bool:
    return os.environ.get(_LEVER, "0") == "1"


def _announce(key: str, msg: str) -> None:
    if key not in _announced:
        _announced.add(key)
        print(msg, flush=True)


def _load_module():
    global _module
    if _module is not None:
        return _module
    from torch.utils.cpp_extension import load

    src = os.path.join(os.path.dirname(__file__), "kda_sgl_fused_spec_kernel.cu")
    build_dir = os.environ.get(
        _BUILD_DIR_ENV,
        _DEFAULT_BUILD_DIR.format(user=os.environ.get("USER", "unknown")),
    )
    os.makedirs(build_dir, exist_ok=True)
    _module = load(
        name="kda_sgl_fused_spec",
        sources=[src],
        build_directory=build_dir,
        extra_cuda_cflags=["-O3"],
        verbose=os.environ.get("RAON_KDA_SGL_FUSED_SPEC_VERBOSE", "0") == "1",
    )
    return _module


def validate_module(self, vllm_config=None) -> None:
    """Called from __init__ when enabled; checks the kernel contract."""
    if self.head_dim != 128 or self.conv_size != 4:
        raise ValueError(
            f"{_LEVER}=1 requires head_dim=128 / conv width 4 (kernel "
            f"constants); got head_dim={self.head_dim}, "
            f"conv_size={self.conv_size}"
        )
    if getattr(self, "kda_chunk_replay", False) or getattr(self, "kda_cpr", False):
        raise ValueError(
            f"{_LEVER}=1 is not bitwise-exact and cannot be combined with "
            "the bitwise-exact decode paths (RAON_KDA_CHUNK_REPLAY / "
            "RAON_KDA_CPR).  Unset one."
        )
    spec = getattr(vllm_config, "speculative_config", None)
    k = (getattr(spec, "num_speculative_tokens", 0) or 0) if spec is not None else 0
    if k <= 0:
        _announce(
            "inert",
            f"[perf] {_LEVER}=1 is inert: no speculative decoding configured "
            "(this kernel only replaces the spec-verify branch)",
        )
    elif k + 1 > _MAX_VERIFY:
        raise ValueError(
            f"{_LEVER}=1 supports up to {_MAX_VERIFY - 1} speculative tokens "
            f"(kernel smem tiles), got num_speculative_tokens={k}"
        )
    else:
        # JIT-build at model init, not inside the first forward.
        _load_module()
        _announce(
            "armed",
            f"[perf] {_LEVER}=1 armed: fused spec-verify KDA kernel built "
            f"(num_speculative_tokens={k}, local heads={self.local_num_heads})",
        )


def _prep(self) -> dict | None:
    """Weight-derived kernel inputs, re-derived on rebind/in-place refit."""
    w = self.conv1d.weight  # [3P, 1, W]
    bias = self.conv1d.bias
    onw = self.o_norm.weight
    stamp = (
        w.data_ptr(), w._version,
        None if bias is None else (bias.data_ptr(), bias._version),
        onw.data_ptr(), onw._version,
        self.A_log.data_ptr(), self.A_log._version,
        self.dt_bias.data_ptr(), self.dt_bias._version,
    )
    cache = getattr(self, "_sgl_fused_spec_prep", None)
    if cache is not None and cache["stamp"] == stamp:
        return cache
    if torch.cuda.is_current_stream_capturing():
        return None
    W = self.conv_size
    cache = {
        "stamp": stamp,
        "conv_w": w.detach().reshape(-1, W).float().contiguous(),
        "conv_b": (
            torch.empty(0, dtype=torch.float32, device=w.device)
            if bias is None
            else bias.detach().reshape(-1).float().contiguous()
        ),
        "A_log": self.A_log.detach().reshape(-1).float().contiguous(),
        "dt_bias": self.dt_bias.detach().reshape(-1).float().contiguous(),
        "onorm_weight": onw.detach().float().contiguous(),
        "scale": float(self.head_dim) ** -0.5,
    }
    self._sgl_fused_spec_prep = cache
    return cache


def _as_i32(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.int32 and t.is_contiguous() else t.to(torch.int32).contiguous()


def try_spec(
    self,
    mixed_qkv: torch.Tensor,      # [T, 3P] bf16, PRE-conv, row-strided ok
    g1: torch.Tensor,             # [1, T, H, D] bf16 raw forget gate
    g2: torch.Tensor,             # [>=T, H, D] bf16 o_norm gate
    beta: torch.Tensor,           # [1, T, H] bf16 raw beta logits
    ratio: torch.Tensor,          # [1, T, H] fp32 ratio, or raw delta (RDA_FUSED)
    core_attn_out: torch.Tensor,  # [1, >=T, H, D] bf16
    m,                            # KimiK3KDAMetadata
) -> bool:
    """Run the fused spec-verify step; True when handled (output INCLUDES
    o_norm, the caller must return without the tail o_norm)."""
    if m.num_prefills != 0 or m.num_decodes != 0:
        _announce(
            "mixed",
            f"[perf] {_LEVER} SKIPPED for a mixed batch (spec rows with "
            f"prefill/plain-decode rows: prefills={m.num_prefills}, "
            f"decodes={m.num_decodes}); such batches run the unfused chain",
        )
        return False
    n = int(m.num_spec_decodes)
    H, D = self.local_num_heads, self.head_dim
    seg = H * D
    idx = m.spec_state_indices_tensor
    tcols = idx.size(-1)
    conv_pool, state_pool = self.kv_cache
    if not (
        conv_pool.dim() == 3
        and conv_pool.size(1) >= tcols + 2
        and conv_pool.size(2) == 3 * seg
        and conv_pool.dtype == torch.bfloat16
        and conv_pool.stride(2) == 1
        and state_pool.dim() == 4
        and state_pool.shape[1:] == (H, D, D)
        and state_pool.dtype == torch.float32
        and state_pool.stride(0) % 4 == 0
        and state_pool.stride()[1:] == (D * D, D, 1)
        and 1 <= tcols <= _MAX_VERIFY
        and mixed_qkv.stride(-1) == 1
        and g1.stride()[2:] == (D, 1)
        and g2.stride()[1:] == (D, 1)
        and beta.stride(-1) == 1
        and ratio.stride(-1) == 1
        and core_attn_out.stride()[2:] == (D, 1)
    ):
        _announce(
            "pool",
            f"[perf] {_LEVER} SKIPPED: pool layout off-contract (conv "
            f"{tuple(conv_pool.shape)} {conv_pool.dtype}, state "
            f"{tuple(state_pool.shape)} {state_pool.dtype}, verify width "
            f"{tcols}) or an input view is not inner-contiguous; expected "
            f"time-major [slots, >= T+2, {3 * seg}] bf16 + [slots, {H}, {D}, {D}] "
            "fp32 -- spec verify runs the unfused chain",
        )
        return False
    prep = _prep(self)
    if prep is None:
        _announce(
            "capture-prep",
            f"[perf] {_LEVER} SKIPPED for this capture: weight caches stale "
            "inside a CUDA-graph capture; unfused chain captured instead",
        )
        return False
    if getattr(self, "rda_fused", False):
        ratio_mode = 1 if ratio.dtype == torch.bfloat16 else 2
    else:
        ratio_mode = 0
        if ratio.dtype != torch.float32:
            _announce(
                "ratio-dtype",
                f"[perf] {_LEVER} SKIPPED: ratio dtype {ratio.dtype} != fp32",
            )
            return False
    if self.gate_lower_bound is None:
        # The kernel implements only the lower-bound gate (RAON: -5).
        _announce(
            "no-lb",
            f"[perf] {_LEVER} SKIPPED: gate_lower_bound is None (softplus "
            "gate not ported) -- spec verify runs the unfused chain",
        )
        return False
    num_tokens = mixed_qkv.size(0)
    qsl = _as_i32(m.spec_query_start_loc[: n + 1])
    accepted = _as_i32(m.num_accepted_tokens[:n])
    out = core_attn_out[0, :num_tokens]

    split = _SPLIT or (2 if n * H < _ONE_WAVE_BLOCKS else 1)
    mod = _load_module()
    mod.run(
        mixed_qkv,
        prep["conv_w"],
        prep["conv_b"],
        conv_pool,
        g1[0],
        prep["A_log"],
        prep["dt_bias"],
        beta[0],
        ratio[0],
        ratio_mode,
        g2[:num_tokens],
        prep["onorm_weight"],
        state_pool,
        _as_i32(idx) if idx.dtype != torch.int32 or idx.stride(-1) != 1 else idx,
        qsl,
        accepted,
        out,
        n,
        prep["scale"],
        float(self.gate_lower_bound),
        float(self.o_norm.eps),
        split,
    )
    _calls[0] += 1
    c = _calls[0]
    if c & (c - 1) == 0 and (c == 1 or c >= 1024):
        print(
            f"[perf] {_LEVER} LIVE: fused spec-verify KDA step (conv+RDA+"
            f"recurrence+o_norm, 1 launch) call #{c}: n={n} tokens={num_tokens} "
            f"verify_width={tcols} H={H} lower_bound={self.gate_lower_bound} "
            f"ratio_mode={ratio_mode} split={split}",
            flush=True,
        )
    return True

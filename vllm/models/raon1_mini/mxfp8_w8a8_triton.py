# Copyright (c) 2026, KRAFTON. Zero-mismatch W8A8 backend.
"""RAON_MOE_MXFP8_W8A8_TRITON: routed-expert chain that reproduces the
training-time mxfp8 W8A8 + SiTU arithmetic on top of the Triton fused MoE
structure. Off by default.

All numeric constants and rounding rules are transcribed from
TransformerEngine release_v2.18 (ptx.cuh / quantize_mxfp8.cuh) and
situ_cudnn_kernels.py. The reference implementation is
mxfp8_align.fused_expert_rows.

Derived from:
  * GEMM kernel: vLLM fused_moe/fused_moe.py fused_moe_kernel and
    invoke_fused_moe_triton_kernel.
  * Chain structure: the 6-launch chain in experts/triton_moe.py.
  * Install convention: actquant.py (no patching when the env var is off,
    re-entry is a no-op).

Changes relative to the upstream kernel (marked [C#] at each site):
  C1 per-tensor/block-wise scale branch -> mxfp8 1x32 block scales
     (a: (M,K/32) e8m0 u8, b: (E,N,K/32) e8m0 u8).
  C2 tl.dot -> tl.dot_scaled(e4m3+e8m0) (default) / SLICED32 mode
     (BLOCK_K=32 split + explicit 2^(e8m0-127) scaling) to isolate
     accumulation order.
  C3 GEMM13 writes fp32 (the training epilogue consumes the fp32
     accumulator; no bf16 round trip) and disallows MUL_ROUTED_WEIGHT;
     GEMM2 writes bf16 and disallows MUL_ROUTED_WEIGHT (the router prob is
     folded in the SiTU epilogue).
  C4 Removed: int8/int4/w8a16/bias/LoRA/SWAP_AB/USE_TD/
     naive_block_assignment/SPLIT_K. SPLIT_K is forbidden because it changes
     the fp32 accumulation order and breaks batch invariance.

Batch invariance: all scales are row-local 1x32 blocks (no global amax).
"""

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

__all__ = [
    "enabled",
    "quantize_mxfp8_rows",
    "quantize_expert_weights",
    "situ_probfold_quant_mxfp8",
    "invoke_moe_mxfp8_gemm",
    "fused_experts_mxw8a8",
    "expert_chain_mxw8a8",
    "expert_chain_mxw8a8_finalized",
    "invalidate_weight_cache",
]

# TE Quantized_Limits<fp8e4m3>::max_norm_rcp = fp32(1.0/448.0). TE multiplies
# by this reciprocal rather than dividing (quantize_mxfp8.cuh:
# thread_amax * max_norm_rcp). The Python double 1/448 is cast to fp32 in the
# kernel, giving the same value as C++ Float32(1.0/448.0).
_MAX_NORM_RCP = 1.0 / 448.0
_FP8_DTYPE = torch.float8_e4m3fn

# SiTU constants, as in situ_cudnn_kernels.py (including the folded beta product).
_SITU_BETA_PRODUCT = 4.0 * 25.0
_SITU_RCP_BETA_GATE = 1.0 / 4.0
_SITU_RCP_BETA_VALUE = 1.0 / 25.0


def enabled() -> bool:
    return os.environ.get("RAON_MOE_MXFP8_W8A8_TRITON", "0") not in (
        "0", "", "false")


# ---------------------------------------------------------------------------
# TE transcription: e8m0 scale and reciprocal (ptx.cuh float_to_e8m0 / exp2f_rcp)
# ---------------------------------------------------------------------------

@triton.jit
def _float_to_e8m0(x):
    """Transcription of ptx.cuh float_to_e8m0. x = amax * max_norm_rcp
    (non-negative fp32).

    nan->0xFF, inf->0xFE, 0->0x00; otherwise round up to the next power of
    two: with exp=bits>>23, man=bits&0x7FFFFF, return exp+1 if
    (man>0 && exp!=0xFE) && !(exp==0 && man<=0x400000).
    """
    bits = x.to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    is_nan = (exp == 255) & (man != 0)
    is_inf = (exp == 255) & (man == 0)
    round_up = (man > 0) & (exp != 0xFE) & (
        ~((exp == 0) & (man <= 0x400000)))
    e = exp + tl.where(round_up, 1, 0)
    e = tl.where(is_nan, 255, tl.where(is_inf, 254, e))
    return e  # int32, 0..255


@triton.jit
def _exp2f_rcp(e):
    """Transcription of ptx.cuh exp2f_rcp: 2^(127-e). e==254 -> 0x00400000
    (=2^-127 denormal), e==255 -> 0x7fffffff. Multiplier for the data cast."""
    rbits = (254 - e) << 23
    rbits = tl.where(e == 254, 0x00400000, rbits)
    rbits = tl.where(e == 255, 0x7FFFFFFF, rbits)
    return rbits.to(tl.float32, bitcast=True)


@triton.jit
def _float_to_e8m0_rz(x):
    """Transcription of cvt.rz.satfinite.ue8m0x2.f32 (mantissa truncation,
    i.e. floor). The reference fc1 epilogue (quant_sfd_row) computes scales
    with a CuTe DSL dtype conversion rather than the TE cast kernel
    (cudnn/gemm/cutedsl/grouped/glu/moe_blockscaled_grouped_
    gemm_glu_bias.py tmp_f32.to(sf_dtype)). The PTX ue8m0 cvt only supports
    rz/rp, so its rounding can differ from TE's round-up. A2_SCALE_MODE=1."""
    bits = x.to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    is_nan = (exp == 255) & (man != 0)
    is_inf = (exp == 255) & (man == 0)
    e = tl.where(is_nan, 255, tl.where(is_inf, 254, exp))
    return e


@triton.jit
def _e8m0_to_pow2(e):
    """Transcription of TE ptx.cuh exp2f: __int_as_float(e<<23) = 2^(e-127)
    (e=0 -> +0). Used for explicit scaling in SLICED32 mode; e=0 only when
    the whole block is zero, so returning +0 is harmless."""
    return (e << 23).to(tl.float32, bitcast=True)


# ---------------------------------------------------------------------------
# Activation/weight 1x32 block quantization (TE quantize_mxfp8_kernel semantics)
# ---------------------------------------------------------------------------

@triton.jit
def _mxfp8_quant_rows_kernel(
    x_ptr, q_ptr, s_ptr,
    K, stride_xm, stride_xk, stride_qm, stride_qk, stride_sm, stride_sk,
    BLOCK_K: tl.constexpr,
):
    """Row-wise 1x32 block quantization: x (rows, K) bf16/fp32 -> q e4m3 +
    s e8m0.

    One program per (row, k-chunk). Rounding:
    scale = float_to_e8m0(block_amax * max_norm_rcp) [round up],
    q = cvt.rn.satfinite.e4m3(x * exp2f_rcp(scale)).
    """
    row = tl.program_id(0)
    kb = tl.program_id(1)
    offs = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_xm + offs * stride_xk,
                mask=mask, other=0.0).to(tl.float32)
    xr = tl.reshape(x, (BLOCK_K // 32, 32))
    amax = tl.max(tl.abs(xr), axis=1)                    # (BLOCK_K//32,)
    e = _float_to_e8m0(amax * _MAX_NORM_RCP_C)
    rcp = _exp2f_rcp(e)
    q = (xr * rcp[:, None]).to(tl.float8e4nv)            # cvt.rn.satfinite
    tl.store(q_ptr + row * stride_qm + offs * stride_qk,
             tl.reshape(q, (BLOCK_K,)), mask=mask)
    soffs = kb * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32)
    smask = soffs < tl.cdiv(K, 32)
    tl.store(s_ptr + row * stride_sm + soffs * stride_sk,
             e.to(tl.uint8), mask=smask)


# Wrapped in tl.constexpr so the jit body can see the module constant.
_MAX_NORM_RCP_C = tl.constexpr(_MAX_NORM_RCP)


def quantize_mxfp8_rows(x: torch.Tensor, block_k: int = 512,
                        num_warps: int | None = None):
    """x (rows, K) -> (q (rows,K) float8_e4m3fn, s (rows, K//32) uint8).

    K must be a multiple of 32 (h=2560 and ffn=1088 both are)."""
    rows, K = x.shape
    assert K % 32 == 0, K
    block_k = min(block_k, triton.next_power_of_2(K))
    assert block_k % 32 == 0
    q = torch.empty((rows, K), dtype=_FP8_DTYPE, device=x.device)
    s = torch.empty((rows, K // 32), dtype=torch.uint8, device=x.device)
    if rows == 0:
        return q, s
    grid = (rows, triton.cdiv(K, block_k))
    kw = {} if num_warps is None else {"num_warps": num_warps}
    _mxfp8_quant_rows_kernel[grid](
        x, q, s, K,
        x.stride(0), x.stride(1), q.stride(0), q.stride(1),
        s.stride(0), s.stride(1),
        BLOCK_K=block_k, **kw,
    )
    return q, s


def quantize_expert_weights(w: torch.Tensor):
    """One-time weight quantization at load (same as PREQUANT in
    mxfp8_align.py).

    w (E, N, K) bf16 -> (q (E,N,K) e4m3, s (E,N,K//32) u8). Rowwise only;
    inference does not need columnwise. Uses the same kernel as activations,
    so rounding semantics are identical by construction."""
    E, N, K = w.shape
    q2, s2 = quantize_mxfp8_rows(w.reshape(E * N, K).contiguous())
    return (q2.reshape(E, N, K).contiguous(),
            s2.reshape(E, N, K // 32).contiguous())


# ---------------------------------------------------------------------------
# Fused SiTU + prob fold + a2 quantization kernel
# (transcription of situ_cudnn_kernels.py _situ_glu_act + 1x32 quantization)
# ---------------------------------------------------------------------------

@triton.jit
def _rcp_approx(x):
    """Transcription of rcp.approx.f32 used by CuTe quant_sfd_row
    (glu_bias.py). Assumed exact for powers of two."""
    return tl.inline_asm_elementwise(
        "rcp.approx.f32 $0, $1;", "=r,r", [x],
        dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _tanh_approx(x):
    """CuTe fastmath tanh, i.e. hardware tanh.approx.f32
    (situ_cudnn_kernels.py)."""
    return tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", "=r,r", [x],
        dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sf_mma_addr(sfr, ks, RMX128, RK, SLAB):
    """Closed-form scale-factor address in the MMA layout:
    sfr = e*RMX128 + rank.

    Shared by the SiTU kernel's direct SF write path and the swap-AB glue
    kernels in mxfp8_w8a8_cutlass.py."""
    e = sfr // RMX128
    r = sfr % RMX128
    return (e * SLAB + (r // 128) * (512 * RK) + (ks // 4) * 512
            + (r % 32) * 16 + ((r // 32) % 4) * 4 + (ks % 4))


@triton.jit
def _situ_probfold_quant_kernel(
    x_ptr,            # (rows, 2*FFN) fp32 GEMM13 output [gate; value]
    w_ptr,            # (rows,) fp32 per-row router prob (flattened topk_weights)
    q_ptr, s_ptr,     # (rows, FFN) e4m3, (rows, FFN//32) u8
    out_f32_ptr,      # (rows, FFN) fp32 for comparison (only if WRITE_F32)
    FFN, stride_xm, stride_qm, stride_sm, stride_om,
    sf_row_ptr, RMX128, RK, SLAB,   # used only when SF_MMA=1
    BLOCK_I: tl.constexpr,
    TANH_APPROX: tl.constexpr,
    SIG_VIA_TANH: tl.constexpr,
    PROB_FOLD: tl.constexpr,
    WRITE_F32: tl.constexpr,
    A2_SCALE_MODE: tl.constexpr,   # 0=TE round-up (float_to_e8m0) 1=RZ (cvt.rz)
    A2_RCP_APPROX: tl.constexpr,   # 1 = data multiplier is
    # rcp.approx.f32(decode(e8m0)) as in CuTe (default 0 = TE exp2f_rcp; the
    # two are assumed equal for powers of two).
    PROB_BF16: tl.constexpr,       # 1 = round prob to bf16 before folding.
    # Matches training: TE fused grouped MLP casts fc1_prob_tensor to the
    # module dtype (bf16) before the kernel (transformer_engine/
    # pytorch/ops/fused/grouped_mlp.py dtype=fc1_weight_param.dtype,
    # scales.detach().to(dtype=dtype)).
    SF_MMA: tl.constexpr = 0,      # 1 = write SF to the closed-form MMA
    # arena address instead of s_ptr rows (mode4 only; saves a scatter_sf
    # launch). Values are unchanged, only the store address differs.
):
    """Replaces the upstream situ_and_mul + quant(a2) launches (not fused
    into the GEMM epilogue). Formula and association order follow the CuTe
    epilogue: out = (100*tanh(g*0.25)) * (((1+tanh(g*0.5))*0.5) * tanh(v*0.04)),
    then the prob fold, then quantization directly from fp32 with no bf16
    round trip.

    TANH_APPROX / SIG_VIA_TANH: both True matches training; the libdevice
    variants are kept for comparison.
    """
    row = tl.program_id(0)
    ib = tl.program_id(1)
    offs = ib * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = offs < FFN
    g = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0)
    v = tl.load(x_ptr + row * stride_xm + FFN + offs, mask=mask, other=0.0)
    g = g.to(tl.float32)
    v = v.to(tl.float32)
    if TANH_APPROX:
        tanh_gate = _tanh_approx(g * _SITU_RCP_BETA_GATE_C)
        tanh_value = _tanh_approx(v * _SITU_RCP_BETA_VALUE_C)
    else:
        tanh_gate = libdevice.tanh(g * _SITU_RCP_BETA_GATE_C)
        tanh_value = libdevice.tanh(v * _SITU_RCP_BETA_VALUE_C)
    if SIG_VIA_TANH:
        # sig = (1 + tanh(gate * 0.5)) * 0.5 (one MUFU op)
        if TANH_APPROX:
            sig_gate = (1.0 + _tanh_approx(g * 0.5)) * 0.5
        else:
            sig_gate = (1.0 + libdevice.tanh(g * 0.5)) * 0.5
    else:
        sig_gate = 1.0 / (1.0 + libdevice.exp(-g))
    out = (_SITU_BETA_PRODUCT_C * tanh_gate) * (sig_gate * tanh_value)
    if PROB_FOLD:
        w = tl.load(w_ptr + row).to(tl.float32)
        if PROB_BF16:
            w = w.to(tl.bfloat16).to(tl.float32)
        out = out * w
    if WRITE_F32:
        tl.store(out_f32_ptr + row * stride_om + offs, out, mask=mask)
    # ---- 1x32 quantization (no bf16 round trip) ----
    outr = tl.reshape(out, (BLOCK_I // 32, 32))
    amax = tl.max(tl.abs(outr), axis=1)
    if A2_SCALE_MODE == 1:
        e = _float_to_e8m0_rz(amax * _MAX_NORM_RCP_C)
    else:
        e = _float_to_e8m0(amax * _MAX_NORM_RCP_C)
    if A2_RCP_APPROX:
        rcp = _rcp_approx(_e8m0_to_pow2(e))
        rcp = tl.minimum(rcp, 3.4028234663852886e38)   # CuTe fmin(nan=True)
    else:
        rcp = _exp2f_rcp(e)
    q = (outr * rcp[:, None]).to(tl.float8e4nv)
    tl.store(q_ptr + row * stride_qm + offs,
             tl.reshape(q, (BLOCK_I,)), mask=mask)
    soffs = ib * (BLOCK_I // 32) + tl.arange(0, BLOCK_I // 32)
    smask = soffs < tl.cdiv(FFN, 32)
    if SF_MMA:
        sfr = tl.load(sf_row_ptr + row)
        tl.store(s_ptr + _sf_mma_addr(sfr, soffs, RMX128, RK, SLAB),
                 e.to(tl.uint8), mask=smask)
    else:
        tl.store(s_ptr + row * stride_sm + soffs, e.to(tl.uint8),
                 mask=smask)


_SITU_RCP_BETA_GATE_C = tl.constexpr(_SITU_RCP_BETA_GATE)
_SITU_RCP_BETA_VALUE_C = tl.constexpr(_SITU_RCP_BETA_VALUE)
_SITU_BETA_PRODUCT_C = tl.constexpr(_SITU_BETA_PRODUCT)


def situ_probfold_quant_mxfp8(
    gate_up: torch.Tensor,          # (rows, 2*FFN) fp32
    probs_rows: torch.Tensor | None,  # (rows,) fp32 or None (no fold)
    *, tanh_approx: bool = True, sig_via_tanh: bool = True,
    a2_scale_mode: int = 0, prob_bf16: bool = True,
    a2_rcp_approx: bool = False,
    return_f32: bool = False, block_i: int = 2048,
    num_warps: int | None = None,
    q_out: torch.Tensor | None = None, s_out: torch.Tensor | None = None,
    sf_mma_out: torch.Tensor | None = None,
    sf_row: torch.Tensor | None = None,
    sf_mma_geom: tuple | None = None,   # (RMX128, RK, SLAB)
):
    # q_out/s_out: optional caller-owned output buffers so mode4 can write a2
    # directly into the fc2 input arena without a copy. The kernel only uses
    # the strides, so the arithmetic is unchanged.
    rows, n2 = gate_up.shape
    ffn = n2 // 2
    assert ffn % 32 == 0, ffn
    block_i = min(block_i, triton.next_power_of_2(ffn))
    assert block_i % 32 == 0
    if q_out is not None:
        assert (q_out.shape == (rows, ffn) and q_out.dtype == _FP8_DTYPE
                and q_out.stride(1) == 1), (q_out.shape, q_out.dtype)
        q = q_out
    else:
        q = torch.empty((rows, ffn), dtype=_FP8_DTYPE, device=gate_up.device)
    sf_mma = sf_mma_out is not None
    if sf_mma:
        # Write SF directly into the MMA arena (saves a scatter launch).
        # The returned s is the arena itself, not a row array (mode4 only).
        assert s_out is None and sf_row is not None and sf_mma_geom
        assert sf_mma_out.dtype == torch.uint8
        s = sf_mma_out
    elif s_out is not None:
        assert (s_out.shape == (rows, ffn // 32)
                and s_out.dtype == torch.uint8
                and s_out.stride(1) == 1), (s_out.shape, s_out.dtype)
        s = s_out
    else:
        s = torch.empty((rows, ffn // 32), dtype=torch.uint8,
                        device=gate_up.device)
    out_f32 = (torch.empty((rows, ffn), dtype=torch.float32,
                           device=gate_up.device) if return_f32 else q)
    prob_fold = probs_rows is not None
    wp = probs_rows if prob_fold else gate_up  # dummy ptr (branch off)
    if rows:
        grid = (rows, triton.cdiv(ffn, block_i))
        g0, g1, g2 = (sf_mma_geom if sf_mma else (0, 0, 0))
        _situ_probfold_quant_kernel[grid](
            gate_up, wp, q, s, out_f32,
            ffn, gate_up.stride(0), q.stride(0),
            0 if sf_mma else s.stride(0),
            out_f32.stride(0) if return_f32 else 0,
            sf_row if sf_mma else gate_up,   # dummy ptr (branch off)
            g0, g1, g2,
            BLOCK_I=block_i,
            TANH_APPROX=tanh_approx, SIG_VIA_TANH=sig_via_tanh,
            PROB_FOLD=prob_fold, WRITE_F32=return_f32,
            A2_SCALE_MODE=a2_scale_mode, PROB_BF16=prob_bf16,
            A2_RCP_APPROX=a2_rcp_approx, SF_MMA=1 if sf_mma else 0,
            **({} if num_warps is None else {"num_warps": num_warps}),
        )
    return (q, s, out_f32) if return_f32 else (q, s)


# ---------------------------------------------------------------------------
# mxfp8 W8A8 fused MoE GEMM: copy of upstream fused_moe_kernel + C1-C4
# ---------------------------------------------------------------------------

@triton.jit
def _fused_moe_mxfp8_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bse, stride_bsk, stride_bsn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr, compute_type: tl.constexpr,
    GEMM_MODE: tl.constexpr,   # 0=dot_scaled 1=sliced32 2=dqbf16 3=free-BK sliced
):
    # [C4] Removed from the upstream signature: b_bias/topk_weights
    # (MUL_ROUTED_WEIGHT)/int8 and int4 flags/per_channel_quant/
    # naive_block_assignment/HAS_BIAS/SWAP_AB/USE_TD/SPLIT_K. The routed
    # weight is folded in the SiTU epilogue; split-K would change the
    # accumulation order.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + offs
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # Inlined upstream write_zeros_to_output (expert not on this EP rank).
        offs_cn0 = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs0 = (c_ptr + stride_cm * offs_token[:, None]
                   + stride_cn * offs_cn0[None, :])
        z = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
        tl.store(c_ptrs0, z,
                 mask=token_mask[:, None] & (offs_cn0[None, :] < N))
        return

    offs_bn = (pid_n * BLOCK_SIZE_N
               + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am
                      + offs_k[None, :] * stride_ak)
    b_ptrs = (b_ptr + off_experts * stride_be
              + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn))

    # [C1] mxfp8 1x32 block scale pointers: the upstream block-wise branch
    # (fused_moe.py) extended to 1x32 e8m0 scales with several per K tile.
    offs_ks32 = tl.arange(0, BLOCK_SIZE_K // 32)
    a_scale_ptrs = (a_scale_ptr
                    + (offs_token[:, None] // top_k) * stride_asm
                    + offs_ks32[None, :] * stride_ask)
    b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse
                    + offs_bn[:, None] * stride_bsn
                    + offs_ks32[None, :] * stride_bsk)
    n_k32 = tl.cdiv(K, 32)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if GEMM_MODE == 3:
        # [C2-d] free-BK sliced (mode3): same arithmetic as mode1
        # (sliced32), an e4m3 tl.dot per 32-wide slice inside the K tile
        # followed by an explicit 2^(sa+sb) multiply. Slices accumulate
        # serially, so the fp32 addition order matches mode1 for any BLOCK_K
        # (BK only changes the K-loop trip count and pipeline depth). This
        # avoids mode1's forced BK=32, which quadruples the K loop.
        offs_k32 = tl.arange(0, 32)
        a_ptrs3 = a_ptr + (offs_token[:, None] // top_k * stride_am
                           + offs_k32[None, :] * stride_ak)
        b_ptrs3 = (b_ptr + off_experts * stride_be
                   + (offs_k32[:, None] * stride_bk
                      + offs_bn[None, :] * stride_bn))
        as_ptrs3 = (a_scale_ptr
                    + (offs_token[:, None] // top_k) * stride_asm)  # (M,1)
        bs_ptrs3 = (b_scale_ptr + off_experts * stride_bse
                    + offs_bn[:, None] * stride_bsn)                # (N,1)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            for s in tl.static_range(BLOCK_SIZE_K // 32):
                k_mask = (k * BLOCK_SIZE_K + s * 32 + offs_k32) < K
                a32 = tl.load(
                    a_ptrs3 + s * 32 * stride_ak,
                    mask=token_mask[:, None] & k_mask[None, :], other=0.0)
                b32 = tl.load(b_ptrs3 + s * 32 * stride_bk,
                              mask=k_mask[:, None], other=0.0)
                ks_ok = (k * (BLOCK_SIZE_K // 32) + s) < n_k32
                sa = tl.load(as_ptrs3 + (k * (BLOCK_SIZE_K // 32) + s)
                             * stride_ask,
                             mask=token_mask[:, None] & ks_ok, other=0)
                sb = tl.load(bs_ptrs3 + (k * (BLOCK_SIZE_K // 32) + s)
                             * stride_bsk,
                             mask=ks_ok, other=0)
                acc32 = tl.dot(a32, b32)
                fa = _e8m0_to_pow2(sa.to(tl.int32))          # (M,1)
                fb = _e8m0_to_pow2(sb.to(tl.int32))          # (N,1)
                accumulator += acc32 * fa * tl.trans(fb)
            a_ptrs3 += BLOCK_SIZE_K * stride_ak
            b_ptrs3 += BLOCK_SIZE_K * stride_bk
    if GEMM_MODE != 3:
        # (constexpr branch; mode3 is handled above. The dead side is
        # removed at compile time, so SMEM/registers are not allocated twice.)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None]
                & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                        other=0.0)
            ks_mask = (offs_ks32 + k * (BLOCK_SIZE_K // 32)) < n_k32
            a_s = tl.load(a_scale_ptrs,
                          mask=token_mask[:, None] & ks_mask[None, :], other=0)
            b_s = tl.load(b_scale_ptrs, mask=ks_mask[None, :], other=0)
            if GEMM_MODE == 1:
                # [C2-b] Diagnostic mode: BLOCK_SIZE_K==32 (forced by the
                # wrapper), e4m3 dot then explicit 2^(e8m0-127). Power-of-two
                # multiplies are exact, so only the in-dot addition order differs.
                tl.static_assert(BLOCK_SIZE_K == 32)
                acc32 = tl.dot(a, b)
                fa = _e8m0_to_pow2(a_s.to(tl.int32))     # (M,1)
                fb = _e8m0_to_pow2(b_s.to(tl.int32))     # (N,1)
                accumulator += acc32 * fa * tl.trans(fb)
            elif GEMM_MODE == 2:
                # [C2-c] dqbf16 diagnostic mode: dequantize e4m3*2^(e8m0-127)
                # exactly to bf16 (the 4-bit mantissa fits in 8 bits), then
                # bf16 MMA with fp32 accumulation. Reference for ruling out
                # fp8 MMA accumulation precision.
                fa = _e8m0_to_pow2(a_s.to(tl.int32))     # (M, BK/32)
                af = tl.reshape(a.to(tl.float32),
                                (BLOCK_SIZE_M, BLOCK_SIZE_K // 32, 32))
                af = af * tl.reshape(fa,
                                     (BLOCK_SIZE_M, BLOCK_SIZE_K // 32, 1))
                ab16 = tl.reshape(
                    af, (BLOCK_SIZE_M, BLOCK_SIZE_K)).to(tl.bfloat16)
                fb = _e8m0_to_pow2(b_s.to(tl.int32))     # (N, BK/32)
                bf = tl.reshape(b.to(tl.float32),
                                (BLOCK_SIZE_K // 32, 32, BLOCK_SIZE_N))
                bf = bf * tl.reshape(tl.trans(fb),
                                     (BLOCK_SIZE_K // 32, 1, BLOCK_SIZE_N))
                bb16 = tl.reshape(
                    bf, (BLOCK_SIZE_K, BLOCK_SIZE_N)).to(tl.bfloat16)
                accumulator = tl.dot(ab16, bb16, acc=accumulator)
            else:
                # [C2-a] Native block-scaled MMA (tcgen05), the same
                # instruction family as the training cuDNN kernel.
                accumulator = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3",
                                            acc=accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
            a_scale_ptrs += (BLOCK_SIZE_K // 32) * stride_ask
            b_scale_ptrs += (BLOCK_SIZE_K // 32) * stride_bsk

    # [C3] Epilogue: no scale reapplication (block scales are applied in the
    # K loop), no bias/routed weight. GEMM13 uses compute_type=fp32 (the
    # training epilogue consumes the fp32 accumulator), GEMM2 uses bf16.
    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (c_ptr + stride_cm * offs_token[:, None]
              + stride_cn * offs_cn[None, :])
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_moe_mxfp8_gemm(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
    A_scale: torch.Tensor, B_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int, config: dict, compute_type: tl.dtype,
    gemm_mode: int = 0,
):
    """Trimmed copy of upstream invoke_fused_moe_triton_kernel. Removed:
    SWAP_AB/TD/naive/wna16/bias branches (C4) and SPLIT_K. The small-batch EM
    optimization is kept as upstream."""
    assert sorted_token_ids.stride(0) == 1
    assert A.dtype == _FP8_DTYPE and B.dtype == _FP8_DTYPE
    assert A_scale.dtype == torch.uint8 and B_scale.dtype == torch.uint8
    assert B.size(2) == A.size(1) and B_scale.size(2) == A.size(1) // 32

    M = A.size(0)
    num_tokens = M * top_k
    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(EM, A.size(0) * top_k * config["BLOCK_SIZE_M"])
    config = dict(config)
    assert gemm_mode in (0, 1, 2, 3), gemm_mode
    if gemm_mode == 1:
        config["BLOCK_SIZE_K"] = 32   # [C2-b] forced in diagnostic mode
    assert config["BLOCK_SIZE_K"] % 32 == 0
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),)
    _fused_moe_mxfp8_kernel[grid](
        A, B, C, A_scale, B_scale,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        B.size(1), B.size(2), EM, num_tokens,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(0), C.stride(1),
        A_scale.stride(0), A_scale.stride(1),
        B_scale.stride(0), B_scale.stride(2), B_scale.stride(1),
        top_k=top_k, compute_type=compute_type,
        GEMM_MODE=gemm_mode,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        num_warps=config.get("num_warps", 4),
        num_stages=config.get("num_stages", 3),
    )


# Default config. dot_scaled lowers to native tcgen05.mma.kind::mxf8f6f4
# only with BLOCK_M=128 and warps>=4 (Triton AccelerateMatmul.cpp); otherwise
# it silently falls back to bf16 mma.sync emulation. With this config the
# output is bitwise identical to the reference implementation; smaller
# BLOCK_M values are not. Check that the TTGIR for this config contains
# tc_gen5_mma_scaled before relying on it.
_DEFAULT_CONFIG = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256,
                   "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1,
                   "num_warps": 8, "num_stages": 4}

# Default config for mode3 (free-BK sliced). BLOCK_M=128 wastes most of each
# tile at decode batch sizes, so BLOCK_M is fixed at 16.
_MODE3_CONFIG = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128,
                 "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1,
                 "num_warps": 4, "num_stages": 3}


def fused_experts_mxw8a8(
    hidden_states: torch.Tensor,      # (M, K) bf16
    w13_q: torch.Tensor, w13_s: torch.Tensor,   # (E, 2*FFN, K) e4m3 / (E, 2*FFN, K/32) u8
    w2_q: torch.Tensor, w2_s: torch.Tensor,     # (E, K, FFN) e4m3 / (E, K, FFN/32) u8
    topk_weights: torch.Tensor,       # (M, top_k) fp32
    topk_ids: torch.Tensor,           # (M, top_k) int32
    *, config: dict | None = None, gemm_mode: int = 0,
    tanh_approx: bool = True, sig_via_tanh: bool = True,
    a2_scale_mode: int = 0, prob_bf16: bool = True,
    a2_rcp_approx: bool = False,
    return_intermediates: bool = False,
):
    """mxfp8 version of the upstream 6-launch chain. Does not include
    moe_sum: returns (M, top_k, K) fc2 rows with the prob already folded, so
    they can be compared directly with mxfp8_align.fused_expert_rows."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size)

    M, K = hidden_states.shape
    E, N13, Kb = w13_q.shape
    ffn = N13 // 2
    top_k = topk_ids.size(1)
    assert Kb == K and w2_q.shape == (E, K, ffn), (w13_q.shape, w2_q.shape)
    cfg = dict(_DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], E))

    # quant(a1): one Triton 1x32 block launch instead of per-tensor CUDA
    a1_q, a1_s = quantize_mxfp8_rows(hidden_states)

    # GEMM13: fp32 output (C3)
    cache1 = torch.empty((M * top_k, N13), dtype=torch.float32,
                         device=hidden_states.device)
    invoke_moe_mxfp8_gemm(
        a1_q, w13_q, cache1, a1_s, w13_s,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        top_k=top_k, config=cfg, compute_type=tl.float32, gemm_mode=gemm_mode)

    # situ + prob fold + quant(a2), replacing upstream situ_and_mul + quant
    probs_rows = topk_weights.reshape(-1).to(torch.float32)
    r = situ_probfold_quant_mxfp8(
        cache1, probs_rows, tanh_approx=tanh_approx,
        sig_via_tanh=sig_via_tanh, a2_scale_mode=a2_scale_mode,
        prob_bf16=prob_bf16, a2_rcp_approx=a2_rcp_approx,
        return_f32=return_intermediates)
    a2_q, a2_s = r[0], r[1]

    # GEMM2: bf16 output, no MUL_ROUTED_WEIGHT (C3; prob already folded)
    cache3 = torch.empty((M * top_k, K), dtype=torch.bfloat16,
                         device=hidden_states.device)
    invoke_moe_mxfp8_gemm(
        a2_q, w2_q, cache3, a2_s, w2_s,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        top_k=1, config=cfg, compute_type=tl.bfloat16, gemm_mode=gemm_mode)

    out = cache3.view(M, top_k, K)
    if return_intermediates:
        return out, {"a1_q": a1_q, "a1_s": a1_s, "cache1": cache1,
                     "situ_f32": r[2], "a2_q": a2_q, "a2_s": a2_s}
    return out


# ---------------------------------------------------------------------------
# Entry points used by model.py _moe_expert_chain_mxfp8
# ---------------------------------------------------------------------------

# Per-layer prequantized weight cache. Key = id(w13); each entry holds a
# reference to the source tensors so the id cannot be reused (same convention
# as mxfp8_align._STATES).
_WQ_CACHE: dict = {}
_REFIT_HOOKED = False
_CHAIN_ANNOUNCED = False
_GEMM_MODE = int(os.environ.get("RAON_MOE_MXFP8_W8A8_GEMM_MODE", "0"))

# RAON_MOE_MXFP8_SHADOW=1: for every (T,K,H) row output of mode2/mode4, also
# run the mode0 reference (dot_scaled, bitwise identical to the reference
# implementation) and compare bitwise. MoE counterpart of the dense shadow
# check in mxfp8_align. Diagnostic only (slow), off by default, skipped during
# CUDA graph capture.
_MOE_SHADOW = os.environ.get("RAON_MOE_MXFP8_SHADOW", "0") == "1"
_MOE_SHADOW_STATS = {"match": 0, "mismatch": 0, "nan_bitwise_equal": 0}
if _MOE_SHADOW:
    import atexit

    atexit.register(lambda: print(
        f"[moe-shadow] SUMMARY match={_MOE_SHADOW_STATS['match']} "
        f"mismatch={_MOE_SHADOW_STATS['mismatch']} "
        f"nan_bitwise_equal={_MOE_SHADOW_STATS['nan_bitwise_equal']}",
        flush=True))


def _moe_shadow_compare(out2, ref, layer_idx, t, topk_ids) -> None:
    """Bitwise compare against the mode0 reference via int16 views (so NaNs
    are not misclassified); logs T and the number of active experts."""
    du = (out2.view(torch.int16).to(torch.int32)
          - ref.view(torch.int16).to(torch.int32)).abs()
    if int(du.max()) != 0:
        _MOE_SHADOW_STATS["mismatch"] += 1
        if _MOE_SHADOW_STATS["mismatch"] <= 24:
            dd = (out2.float() - ref.float()).abs()
            ne = int(topk_ids.unique().numel())
            print(f"[moe-shadow] MISMATCH L{layer_idx} T={t} experts={ne} "
                  f"frac={float((du != 0).float().mean()):.4f} "
                  f"max_ulp={int(du.max())} "
                  f"max_abs={float(dd.max()):.3e} "
                  f"mean_abs={float(dd.mean()):.3e}", flush=True)
    else:
        _MOE_SHADOW_STATS["match"] += 1
        if bool(torch.isnan(out2).any()):
            _MOE_SHADOW_STATS["nan_bitwise_equal"] += 1


def invalidate_weight_cache() -> int:
    """Invalidate all weight caches after a refit so the next forward
    requantizes from the live weights.

    Also invalidates the mode2 (trtllm) and mode4 (cutlass) caches if those
    modules are loaded (looked up in sys.modules, so no import cost)."""
    import sys as _sys
    # The local cache is only used by the eager paths (mode0/1/shadow); mode2
    # returns via expert_rows_trtllm and never fills it. Only
    # trtllm._WQ_CACHE is referenced by captured graphs, and its invalidate
    # requantizes in place instead of dropping entries (see
    # mxfp8_w8a8_trtllm.invalidate_weight_cache).
    n = len(_WQ_CACHE)
    _WQ_CACHE.clear()
    _trt = _sys.modules.get((__package__ or "vllm.models.raon1_mini")
                            + ".mxfp8_w8a8_trtllm")
    if _trt is not None:
        n += _trt.invalidate_weight_cache()
    _swab = _sys.modules.get((__package__ or "vllm.models.raon1_mini")
                             + ".mxfp8_w8a8_cutlass")
    if _swab is not None:
        n += _swab.invalidate_weight_cache()
    return n


def _hook_refit_once() -> None:
    """Wrap mxfp8_align.refit_invalidate so it also clears these caches.

    Refit writes through param.data.copy_, which bypasses the tensor version
    counter, so there is nothing to detect; hook the same invalidation point
    as mxfp8_align instead. When the env var is off this is never called.
    """
    global _REFIT_HOOKED
    if _REFIT_HOOKED:
        return
    _REFIT_HOOKED = True
    from . import mxfp8_align as MX
    _orig = MX.refit_invalidate

    def _wrapped(model):
        n = invalidate_weight_cache()
        if n:
            print(f"[parity] RAON_MOE_MXFP8_W8A8_TRITON: refit_invalidate "
                  f"requantized {n} prequantized expert weight sets IN PLACE "
                  f"(trtllm ptr contract kept)", flush=True)
        return _orig(model)

    MX.refit_invalidate = _wrapped


def _quantized_weights(w13: torch.Tensor, w2: torch.Tensor):
    key = id(w13)
    ent = _WQ_CACHE.get(key)
    if ent is None:
        q13, s13 = quantize_expert_weights(w13.detach())
        q2, s2 = quantize_expert_weights(w2.detach())
        ent = (q13, s13, q2, s2, w13, w2)   # keep source refs so the id stays unique
        _WQ_CACHE[key] = ent
    return ent[0], ent[1], ent[2], ent[3]


def expert_chain_mxw8a8_finalized(x, topk_weights, topk_ids, w13, w2,
                                  beta, linear_beta, layer_idx=None):
    """RAON_MOE_MXFP8_W8A8_FINALIZE=1: (T, H) bf16 output with the combine
    tail (unpermute/prob fold/fp32 sum) done in the kernel finalize step.
    Only mode2 and mode4 support this; other modes raise instead of silently
    falling back to the rows path. Beta check and refit hook are the same as
    expert_chain_mxw8a8."""
    if (float(beta), float(linear_beta or -1.0)) != (4.0, 25.0):
        raise RuntimeError(
            "RAON_MOE_MXFP8_W8A8_TRITON=1: kernel transcribes CuTe betas "
            f"(4.0, 25.0) as constexpr; got ({beta}, {linear_beta})")
    _hook_refit_once()
    if _GEMM_MODE == 2:
        from . import mxfp8_w8a8_trtllm as TRT
        return TRT.expert_rows_trtllm(
            x, topk_weights, topk_ids, w13, w2, layer_idx=layer_idx,
            do_finalize=True)
    if _GEMM_MODE == 4:
        # mode4 finalize: fp32 sum over the swap-AB compact rows in ascending
        # expert order (same per-element order as combine_unweighted_fp32).
        from . import mxfp8_w8a8_cutlass as SWAB
        return SWAB.expert_rows_swab(
            x, topk_weights, topk_ids, w13, w2, layer_idx=layer_idx,
            do_finalize=True)
    raise RuntimeError(
        f"RAON_MOE_MXFP8_W8A8_FINALIZE=1 requires GEMM_MODE 2 (trtllm) "
        f"or 4 (swap-AB); got {_GEMM_MODE}")


def expert_chain_mxw8a8_ep_finalized(x, topk_weights, topk_ids, w13, w2,
                                     beta, linear_beta, layer_idx=None,
                                     num_local_experts=None):
    """RAON_MOE_COMBINE_EP_FINALIZE=1 (EP=2): (T, H) bf16 output with the
    combine plumbing around the mode2 rows path (unpermute/prob fold/(T,K,H)
    materialization/EP2 combine re-gather) replaced by one trtllm call plus
    one fused kernel. mode2 only; other modes raise instead of silently
    falling back. Beta check and refit hook are the same as
    expert_chain_mxw8a8."""
    if (float(beta), float(linear_beta or -1.0)) != (4.0, 25.0):
        raise RuntimeError(
            "RAON_MOE_MXFP8_W8A8_TRITON=1: kernel transcribes CuTe betas "
            f"(4.0, 25.0) as constexpr; got ({beta}, {linear_beta})")
    _hook_refit_once()
    if _GEMM_MODE == 2:
        from . import mxfp8_w8a8_trtllm as TRT
        return TRT.expert_rows_trtllm(
            x, topk_weights, topk_ids, w13, w2, layer_idx=layer_idx,
            do_finalize=False, ep_finalize_local=num_local_experts)
    raise RuntimeError(
        f"RAON_MOE_COMBINE_EP_FINALIZE=1 requires GEMM_MODE 2 (trtllm); "
        f"got {_GEMM_MODE}")


def expert_chain_mxw8a8(x, topk_weights, topk_ids, w13, w2,
                        beta, linear_beta, layer_idx=None):
    """(T, K, H) expert rows (prob already folded) from token-major input.
    Bitwise identical to mxfp8_align.fused_expert_rows with the default
    config.

    The kernel hardcodes the CuTe betas (4.0/25.0), so other values raise
    rather than silently computing a different activation."""
    global _CHAIN_ANNOUNCED
    global _GEMM_MODE
    if (float(beta), float(linear_beta or -1.0)) != (4.0, 25.0):
        raise RuntimeError(
            "RAON_MOE_MXFP8_W8A8_TRITON=1: kernel transcribes CuTe betas "
            f"(4.0, 25.0) as constexpr; got ({beta}, {linear_beta})")
    _hook_refit_once()
    if _GEMM_MODE == 2:
        # mode2: trtllm-gen MxFp8+SiTuGlu fused MoE backend
        # (mxfp8_w8a8_trtllm.py). The kernel argument gemm_mode=2 still means
        # dqbf16; the env value 2 selects this backend instead. Exceptions
        # propagate (no silent Triton fallback).
        from . import mxfp8_w8a8_trtllm as TRT
        out2 = TRT.expert_rows_trtllm(
            x, topk_weights, topk_ids, w13, w2, layer_idx=layer_idx)
        # Shadow check against the mode0 reference (diagnostic, skipped during capture)
        if _MOE_SHADOW and not torch.cuda.is_current_stream_capturing():
            w13_q, w13_s, w2_q, w2_s = _quantized_weights(w13, w2)
            ref = fused_experts_mxw8a8(
                x, w13_q, w13_s, w2_q, w2_s,
                topk_weights.to(torch.float32), topk_ids.to(torch.int32),
                gemm_mode=0, config=None)
            _moe_shadow_compare(out2, ref, layer_idx, x.shape[0], topk_ids)
        return out2
    if _GEMM_MODE == 4:
        # mode4: swap-AB CUTLASS grouped backend (mxfp8_w8a8_cutlass.py,
        # vendored Sm100 grouped GEMM). Exceptions propagate (no silent
        # fallback). Supports the RAON_MOE_MXFP8_SHADOW check.
        from . import mxfp8_w8a8_cutlass as SWAB
        out4 = SWAB.expert_rows_swab(
            x, topk_weights, topk_ids, w13, w2, layer_idx=layer_idx)
        if _MOE_SHADOW and not torch.cuda.is_current_stream_capturing():
            w13_q, w13_s, w2_q, w2_s = _quantized_weights(w13, w2)
            ref = fused_experts_mxw8a8(
                x, w13_q, w13_s, w2_q, w2_s,
                topk_weights.to(torch.float32), topk_ids.to(torch.int32),
                gemm_mode=0, config=None)
            _moe_shadow_compare(out4, ref, layer_idx, x.shape[0], topk_ids)
        return out4
    _first_install = id(w13) not in _WQ_CACHE
    w13_q, w13_s, w2_q, w2_s = _quantized_weights(w13, w2)
    if _first_install:
        # Record te_precision_scope at prequant install time, using the same
        # API mxfp8_align calls when it wraps a layer (mxfp8_align.py
        # _record_scope, which honors the SCOPE_RECORD env gate and skips
        # non-integer layer_idx; recording is idempotent, so reinstalling
        # after a refit is harmless). Without it, the scope check treats
        # routed fc1/fc2 as not covered.
        from . import mxfp8_align as MX
        MX._record_scope(layer_idx, ("routed_fc1", "routed_fc2"))
    if not _CHAIN_ANNOUNCED:
        _CHAIN_ANNOUNCED = True
        print(f"[parity] RAON_MOE_MXFP8_W8A8_TRITON=1 LIVE (gemm_mode={_GEMM_MODE}): triton mxfp8 "
              f"W8A8+SiTU chain on layer_idx={layer_idx}, config "
              f"{_DEFAULT_CONFIG} (bitwise identical to reference at this "
              f"config; native tcgen05 requires "
              f"BLOCK_M>=128 — verify the TTGIR contains "
              f"tc_gen5_mma_scaled before relying on other configs)",
              flush=True)
    # RAON_MOE_MXFP8_W8A8_GEMM_MODE:
    # 0=dot_scaled (native tcgen05, bitwise reference)
    # 1=sliced32 (forced BLOCK_K=32; slow at decode, diagnostic only)
    # 2=trtllm-gen MxFp8+SiTuGlu backend (handled above, never reaches here;
    #   the kernel argument gemm_mode=2 still means dqbf16)
    # 3=free-BK sliced (mode1 arithmetic with free BLOCK_K)
    # 4=swap-AB CUTLASS backend (handled above)
    # The announce message includes the mode.
    out = fused_experts_mxw8a8(
        x, w13_q, w13_s, w2_q, w2_s,
        topk_weights.to(torch.float32), topk_ids.to(torch.int32),
        gemm_mode=_GEMM_MODE,
        config=(_MODE3_CONFIG if _GEMM_MODE == 3 else None))
    return out

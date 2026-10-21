# KDA prefill front-end 4-launch fusion (FUSED_FLOW=1).
#
# WHAT: the fla in-kernel prefill path spends 4 separate launches on pure
# element/row-local math before the chunk core: l2norm(q), l2norm(k),
# beta sigmoid, gate+chunk-cumsum.  All four are mutually independent
# (cumsum consumes the RAW decay g1, not the normalized q/k), so they can
# ride ONE kernel launch with program-id range dispatch.
#
# BIT CONTRACT: each segment below is a VERBATIM copy of the wheel kernel
# body it replaces (the fla arithmetic is concatenated, never reimplemented):
#   seg q/k : fla/modules/l2norm.py::l2norm_fwd_kernel   (pinned cfg BT=8)
#   seg b   : fla/ops/common/gate.py::fused_beta_sigmoid_fwd_kernel (BLOCK 2048)
#   seg g   : fla/ops/kda/gate.py::kda_gate_chunk_cumsum_vector_kernel
#             (pinned cfg BS=32; v2 uses HAS_BIAS + USE_LOWER_BOUND + HAS_SCALE)
# Per-element arithmetic order is preserved segment-wise; the only shared
# compile knob is num_warps (single kernel = single value).  We pin
# num_warps=2 == the cumsum pinned config; every l2norm/cumsum config is
# bit-equal on this path, and (core, final_state) is torch.equal for every
# shape-branch coordinate.
#
# LEVER: RAON_KDA_PREFILL_FUSED_FLOW=1 (default OFF => this module is never
# imported by the hot path; unset means byte-identical behavior).

import os

import torch
import triton
import triton.language as tl

from fla.ops.kda.chunk_fwd import chunk_kda_fwd  # noqa: F401 (kept for parity reference)
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.op import exp
from fla.ops.utils.softplus import softplus
from fla.modules.l2norm import l2norm_fwd

_BT_L2 = 8        # l2norm pinned configs[0]: BT=8 (warps list outer, BT inner)
_BB = 2048        # beta sigmoid fixed BLOCK_SIZE (fla/ops/common/gate.py)
_BS = 32          # cumsum pinned configs[0]: BS=32
_NUM_WARPS = 2    # == cumsum pinned num_warps


@triton.jit(do_not_specialize=['T', 'TR', 'n_beta'])
def kda_frontend_fused_kernel(
    xq, yq, rstd_q,
    xk, yk, rstd_k,
    xb, yb, beta_scale, n_beta,
    s, A_log, dt_bias, o_g, g_scale, cu_seqlens, chunk_indices, lower_bound,
    T, TR,
    NPQ, NPK, NPB, NS, NT,
    eps,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT_L2: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BB: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    if pid < NPQ:
        # ---- seg q: l2norm_fwd_kernel verbatim --------------------------
        i_t = pid
        o_t = i_t * BT_L2 + tl.arange(0, BT_L2)
        o_d = tl.arange(0, BD)
        m_t = o_t < TR
        m_x = m_t[:, None] & (o_d[None, :] < D)
        p_x = xq + o_t[:, None] * D + o_d[None, :]
        p_y = yq + o_t[:, None] * D + o_d[None, :]
        p_rstd = rstd_q + o_t
        b_x = tl.load(p_x, mask=m_x, other=0.0).to(tl.float32)
        b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
        b_y = b_x * b_rstd[:, None]
        tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_x)
        tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), mask=m_t)
    elif pid < NPQ + NPK:
        # ---- seg k: l2norm_fwd_kernel verbatim --------------------------
        i_t = pid - NPQ
        o_t = i_t * BT_L2 + tl.arange(0, BT_L2)
        o_d = tl.arange(0, BD)
        m_t = o_t < TR
        m_x = m_t[:, None] & (o_d[None, :] < D)
        p_x = xk + o_t[:, None] * D + o_d[None, :]
        p_y = yk + o_t[:, None] * D + o_d[None, :]
        p_rstd = rstd_k + o_t
        b_x = tl.load(p_x, mask=m_x, other=0.0).to(tl.float32)
        b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
        b_y = b_x * b_rstd[:, None]
        tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_x)
        tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), mask=m_t)
    elif pid < NPQ + NPK + NPB:
        # ---- seg b: fused_beta_sigmoid_fwd_kernel verbatim -------------
        pid_b = pid - (NPQ + NPK)
        offs = pid_b * BB + tl.arange(0, BB).to(tl.int64)
        mask = offs < n_beta
        b_x = tl.load(xb + offs, mask=mask, other=0).to(tl.float32)
        b_y = beta_scale * tl.sigmoid(b_x)
        tl.store(yb + offs, b_y.to(yb.dtype.element_ty), mask=mask)
    else:
        # ---- seg g: kda_gate_chunk_cumsum_vector_kernel verbatim -------
        r = pid - (NPQ + NPK + NPB)
        i_s = r % NS
        r2 = r // NS
        i_t = (r2 % NT).to(tl.int64)
        i_bh = r2 // NT
        i_b, i_h = i_bh // H, i_bh % H
        if IS_VARLEN:
            i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
            i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T

        o_t = i_t * BT + tl.arange(0, BT)
        o_s = i_s * BS + tl.arange(0, BS)
        m_s = (o_t[:, None] < T) & (o_s[None, :] < S)
        p_s = s + (bos * H + i_h) * S + o_t[:, None] * (H * S) + o_s[None, :]
        p_o = o_g + (bos * H + i_h) * S + o_t[:, None] * (H * S) + o_s[None, :]
        b_s = tl.load(p_s, mask=m_s, other=0.0).to(tl.float32)

        if HAS_BIAS:
            b_bias = tl.load(dt_bias + i_h * S + o_s, mask=o_s < S, other=0.0).to(tl.float32)
            b_s = b_s + b_bias[None, :]

        b_A = tl.load(A_log + i_h).to(tl.float32)
        if not USE_LOWER_BOUND:
            b_gate = -exp(b_A) * softplus(b_s)
        else:
            b_gate = lower_bound * tl.sigmoid(exp(b_A) * b_s)

        b_o = tl.cumsum(b_gate, axis=0)

        if HAS_SCALE:
            b_o *= g_scale
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_s)


def frontend_fused(q, k, beta_f32, g1, A_log, dt_bias_flat, lower_bound,
                   cu_seqlens=None, chunk_indices=None, chunk_size=64):
    """One launch replacing [l2norm(q), l2norm(k), sigmoid(beta), gate-cumsum].

    Inputs exactly as ChunkKDAFunction.forward would consume them (q/k raw
    [B,T,H,D] bf16, beta fp32 raw logits, g1 raw decay [B,T,H,D], dt_bias
    flattened [H*D]).  Returns (q_n, k_n, beta, g_cum) with the same
    allocation dtypes as the wheel path (bf16/bf16/fp32/fp32).
    """
    B, T, H, D = q.shape
    assert B == 1 or cu_seqlens is None, "varlen requires B == 1 (fla contract)"
    xq = q.view(-1, D)
    xk = k.view(-1, D)
    TR = xq.shape[0]
    # retreat mode (RAON_KDA_FE_FUSE_L2NORM=0): l2norm stays on the ORIGINAL
    # wheel launches (2), only beta+cumsum ride the fused kernel (3-segment
    # fallback authorized for one automatic retreat on ladder FAIL --
    # num_warps risk isolation: l2norm keeps its own pinned config).
    fuse_l2 = os.environ.get("RAON_KDA_FE_FUSE_L2NORM", "1") == "1"
    if not fuse_l2:
        yq_t, _ = l2norm_fwd(q)
        yk_t, _ = l2norm_fwd(k)
    # allocations: byte-for-byte the wheel hosts' choices
    yq = torch.empty_like(xq) if fuse_l2 else None
    yk = torch.empty_like(xk) if fuse_l2 else None
    rstd_q = torch.empty((TR,), dtype=torch.float32, device=q.device)
    rstd_k = torch.empty((TR,), dtype=torch.float32, device=q.device)
    yb = torch.empty_like(beta_f32, dtype=torch.float32)
    g_cum = torch.empty_like(g1, dtype=torch.float32)

    BD = triton.next_power_of_2(D)
    n_beta = beta_f32.numel()
    NPQ = triton.cdiv(TR, _BT_L2) if fuse_l2 else 0
    NPK = NPQ
    NPB = triton.cdiv(n_beta, _BB)
    NS = triton.cdiv(D, _BS)
    NT = triton.cdiv(T, chunk_size) if cu_seqlens is None else len(chunk_indices)
    NPG = NS * NT * B * H
    grid = (NPQ + NPK + NPB + NPG,)

    kda_frontend_fused_kernel[grid](
        xq, yq if fuse_l2 else xq, rstd_q,
        xk, yk if fuse_l2 else xk, rstd_k,
        beta_f32, yb, 1.0, n_beta,
        g1, A_log, dt_bias_flat, g_cum, RCP_LN2, cu_seqlens, chunk_indices,
        lower_bound if lower_bound is not None else 0.0,
        T, TR,
        NPQ, NPK, NPB, NS, NT,
        1e-6,
        H=H, S=D, D=D, BD=BD,
        BT_L2=_BT_L2, BT=chunk_size, BS=_BS, BB=_BB,
        HAS_BIAS=dt_bias_flat is not None,
        USE_LOWER_BOUND=lower_bound is not None,
        HAS_SCALE=True,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=_NUM_WARPS,
    )
    if not fuse_l2:
        return yq_t, yk_t, yb, g_cum
    return yq.view_as(q), yk.view_as(k), yb, g_cum


def chunk_kda_frontend_fused(
    q, k, v, g, beta, A_log, dt_bias, scale,
    initial_state=None, output_final_state=True,
    lower_bound=None, cu_seqlens=None, state_v_first=False,
    chunk_size=64,
):
    """Inference-only replica of ChunkKDAFunction.forward with the four
    front-end launches fused into one.  Everything from intra onward calls
    the SAME wheel functions with the SAME arguments (copy of
    fla/ops/kda/chunk_fwd.py::chunk_kda_fwd with the cumsum block replaced
    by the fused precompute; cp_context SUBTRACTED — engine never passes it).
    """
    assert torch.is_inference_mode_enabled() or not torch.is_grad_enabled(), \
        "fused front-end path is inference-only (no autograd ctx)"

    chunk_indices = None
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    q_n, k_n, beta_s, g_cum = frontend_fused(
        q, k, beta, g, A_log, dt_bias, lower_bound,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )

    # ---- below: chunk_kda_fwd verbatim, cumsum block replaced by g_cum ----
    w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
        q=q_n, k=k_n, v=v, gk=g_cum, beta=beta_s, scale=scale,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size,
        chunk_indices=chunk_indices, safe_gate=(lower_bound is not None),
        disable_recompute=False,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=g_cum, initial_state=initial_state,
        output_final_state=output_final_state, cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=None, chunk_indices=chunk_indices,
        chunk_size=chunk_size, state_v_first=state_v_first,
    )
    o = chunk_gla_fwd_o_gk(
        q=q_n, v=v_new, g=g_cum, A=Aqk, h=h, scale=scale,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size,
        chunk_indices=chunk_indices, state_v_first=state_v_first,
    )
    return o.type_as(q), final_state


def fused_flow_enabled() -> bool:
    return os.environ.get("RAON_KDA_PREFILL_FUSED_FLOW", "0") == "1"


_FE_PREWARM_DONE = False


def prewarm_frontend_fused(device=0):
    """With FUSED_FLOW=1, run the fused front-end kernel once on a tiny dummy
    input at model build (before worker threads multiply) so the Triton JIT
    finishes in a safe window. The dummy output is discarded and does not
    affect any results. No-op when the lever is unset.
    """
    global _FE_PREWARM_DONE
    if _FE_PREWARM_DONE or not fused_flow_enabled():
        return False
    _FE_PREWARM_DONE = True
    dev = torch.device("cuda", device)
    T, H, D = 8, 40, 128
    g = lambda *s, **k: torch.randn(*s, device=dev, **k)
    frontend_fused(
        g(1, T, H, D, dtype=torch.bfloat16),
        g(1, T, H, D, dtype=torch.bfloat16),
        g(1, T, H, dtype=torch.float32),
        g(1, T, H, D, dtype=torch.bfloat16),
        g(H, dtype=torch.float32) * 0.1,
        (g(H, D, dtype=torch.float32) * 0.1).reshape(-1),
        -5.0,
    )
    torch.cuda.synchronize(dev)
    print("[raon-kda] FUSED_FLOW prewarm: front-end fused kernel JIT done "
          "at model build (freeze fix)", flush=True)
    return True

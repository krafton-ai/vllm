# SPDX-License-Identifier: Apache-2.0
"""Zero-mismatch MoE combine: trainer-aligned slot-order FP32 reduction.

This is the *combine* half of the MoE branch -- the weighted reduction of a
token's top-k expert (FC2) outputs back into one hidden vector -- implemented
to match the Megatron trainer **bitwise** when routing is held fixed
(identical top-k id sets).

Contract (distilled from three references):

  Megatron-LM exact_reference (megatron/core/transformer/moe):
    activations.py  apply_expert_weights_after_projection:
        out = down_output.float() * permuted_probs.float()      # FP32 after FC2
    moe_utils.py    unpermute: FP32 index_add over expert-major rows
    moe_layer.py:545-546: single cast FP32 -> params_dtype (bf16) AFTER combine

  Vime  (docs/en/advanced/reproducibility.md:62): "the token owner performs the
    weighted reduction in slot order and FP32."

  SkyRL IsoExec (ops/moe leaftree): FP32 balanced-tree fold, lower index left.

Distilled rules implemented below:
  (1) multiply each (token, slot) FC2 output by its router weight in FP32;
  (2) reduce a token's k contributions entirely in FP32;
  (3) reduce in a FIXED order: ascending expert id (== Megatron expert-major
      index_add == SkyRL stable-argsort for exact top-k);
  (4) apply routed_scaling_factor in FP32;
  (5) cast to the output dtype exactly ONCE, at the end.

vLLM's stock fused-MoE combine folds the router weight into a bf16 GEMM epilogue
and sums the k outputs with a bf16 rounding boundary in selection order, which
does not match the trainer bitwise. ``combine()`` here is the reference numeric
contract a fused replacement must reproduce.

Gated by env ``RAON_MOE_FP32_COMBINE=1`` (see model.py). Pure torch; the CPU
``_self_check`` runs at import when the flag is on and fails closed.
"""

from __future__ import annotations

import os

import torch

from . import raon_phase


def ascending_expert_order(topk_ids: torch.Tensor) -> torch.Tensor:
    """Per-token slot order visiting experts in ascending id (Megatron order)."""
    return torch.argsort(topk_ids, dim=1, stable=True)


def combine(
    expert_out: torch.Tensor,      # [T, K, H] FC2 outputs per selected slot
    topk_ids: torch.Tensor,        # [T, K] long, selected expert ids (pinned)
    topk_weights: torch.Tensor,    # [T, K] router weights
    *,
    routed_scaling_factor: float = 1.0,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Trainer-aligned combine. Returns [T, H] in ``out_dtype`` (default: the
    ``expert_out`` dtype). Bitwise-equal to Megatron exact_reference."""
    if out_dtype is None:
        out_dtype = expert_out.dtype
    T, K, H = expert_out.shape
    rows = torch.arange(T, device=expert_out.device)
    order = ascending_expert_order(topk_ids)
    ef = expert_out.float()
    wf = topk_weights.float()
    acc = torch.zeros(T, H, dtype=torch.float32, device=expert_out.device)
    for j in range(K):
        slot = order[:, j]
        acc = acc + ef[rows, slot] * wf[rows, slot].unsqueeze(1)  # FP32, no cast
    acc = acc * float(routed_scaling_factor)
    return acc.to(out_dtype)


def _eager_reference(expert_out, topk_ids, topk_weights, routed_scaling_factor):
    """Independent oracle: fp32 weighted sum via scatter over expert order."""
    order = ascending_expert_order(topk_ids)
    T, K, H = expert_out.shape
    rows = torch.arange(T)
    acc = torch.zeros(T, H, dtype=torch.float32)
    for j in range(K):
        s = order[:, j]
        acc.index_add_(
            0, rows,
            expert_out.float()[rows, s] * topk_weights.float()[rows, s].unsqueeze(1),
        )
    return (acc * float(routed_scaling_factor)).to(expert_out.dtype)


def _self_check() -> None:
    """Fail-closed bitwise self-check (SkyRL pattern). Raises on mismatch."""
    g = torch.Generator().manual_seed(0)
    eo = torch.randn(16, 6, 256, generator=g, dtype=torch.float32).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(128, generator=g)[:6] for _ in range(16)])
    w = torch.rand(16, 6, generator=g, dtype=torch.float32)
    got = combine(eo, ids, w, routed_scaling_factor=2.5)
    ref = _eager_reference(eo, ids, w, 2.5)
    if not torch.equal(got, ref):
        raise RuntimeError(
            "moe_align.combine failed its bitwise self-check — refusing to "
            "enable RAON_MOE_FP32_COMBINE with a non-conforming combine")


def enabled() -> bool:
    return os.environ.get("RAON_MOE_FP32_COMBINE", "0") not in ("0", "", "false")


# ---------------------------------------------------------------------------
# RAON_MOE_TRAINER_REALIZED — trainer-STRUCTURED expert chain (default OFF)
#
# On identical inputs and weights, the trainer's bf16 expert realization
# differs from the ``combine`` contract above structurally, not just in
# kernels:
#
#   * the trainer folds the router weight into the ACTIVATION output before
#     FC2 (TE ``_ScaledGLU.fuser_forward``: scales dequantized to the compute
#     dtype -> ``situ_bf16 * probs_bf16``), never after FC2 in FP32
#     (``apply_expert_weights_after_projection`` is exact_reference-only dead
#     code -- no recipe sets model_math_mode);
#   * FC1/FC2 GEMMs need no change: TE's cublasLt grouped GEMM and this
#     module's per-expert ``torch.mm`` are bitwise equal on the same rows;
#   * the SiTU op must be the fp32 torch formula (megatron
#     ``ScaledSituGLU._glu_forward``): the compiled ``_C.situ_and_mul`` CUDA
#     kernel rounds a small fraction of elements to a neighboring bf16;
#   * the trainer combine (probs pre-applied) is an UNWEIGHTED per-token sum
#     (hybridep); the FP32 ascending-expert accumulation below realizes it
#     (DeepEP combine is fp32 slot-order).
#
# These helpers are the engine twin of that chain. Gated by
# ``RAON_MOE_TRAINER_REALIZED=1`` (see model.py); implies the eager expert
# path and supersedes the weighted ``combine`` when set.
# ---------------------------------------------------------------------------


def situ_reference(
    gate_up: torch.Tensor, gate_beta: float, value_beta: float
) -> torch.Tensor:
    """FP32 SiTU-GLU on contiguous [gate; value] halves, single cast back.

    Byte-level transcription of megatron ScaledSituGLU._glu_forward
    (situ_te_op_fuser.py:120-127) after TE's fuser has de-interleaved — the
    engine's [gate; up] rows are already contiguous, so no interleave step.
    """
    d = gate_up.shape[-1] // 2
    gate = gate_up[..., :d].float()
    value = gate_up[..., d:].float()
    gate_act = gate_beta * torch.tanh(gate / gate_beta) * torch.sigmoid(gate)
    value_act = value_beta * torch.tanh(value / value_beta)
    return (gate_act * value_act).to(gate_up.dtype)


def fold_probs(situ_out: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """Router-weight fold, trainer realization (_ScaledGLU.fuser_forward):
    scales are dequantized to the COMPUTE dtype first, then multiplied —
    ``situ_bf16 * probs_bf16``, one rounding at the multiply."""
    return situ_out * probs.to(situ_out.dtype).unsqueeze(-1)


def combine_unweighted_fp32(
    expert_out: torch.Tensor,      # [T, K, H] FC2 rows, probs ALREADY folded
    topk_ids: torch.Tensor,        # [T, K] long (pinned routing)
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Unweighted per-token sum in FP32, ascending expert id, single cast.

    Candidate realization of the trainer's hybridep combine for the
    pre-weighted chain. Same reduction order as ``combine``; no weights."""
    if out_dtype is None:
        out_dtype = expert_out.dtype
    # RAON_MOE_COMBINE_UNW_FUSED (default off): bitwise-identical Triton
    # kernel implementation of the same contract.
    if (os.environ.get("RAON_MOE_COMBINE_UNW_FUSED", "0") == "1"
            and expert_out.is_cuda and expert_out.dtype == torch.bfloat16
            and out_dtype == torch.bfloat16
            and expert_out.is_contiguous() and topk_ids.is_contiguous()):
        from .moe_ep_kernel import combine_unweighted as _unw_kernel
        return _unw_kernel(expert_out, topk_ids)
    T, K, H = expert_out.shape
    rows = torch.arange(T, device=expert_out.device)
    order = ascending_expert_order(topk_ids)
    acc = torch.zeros(T, H, dtype=torch.float32, device=expert_out.device)
    for j in range(K):
        slot = order[:, j]
        acc = acc + expert_out[rows, slot].float()
    return acc.to(out_dtype)


def trainer_realized() -> bool:
    return os.environ.get("RAON_MOE_TRAINER_REALIZED", "0") not in (
        "0", "", "false")


# ---------------------------------------------------------------------------
# RAON_MOE_COMBINE_EP_REALIZED — hybridep's TWO-STAGE combine (default OFF)
#
# ``combine_unweighted_fp32`` above assumes a SINGLE fp32 accumulation over all
# k slots with one cast at the end (Vime's DeepEP slot-order contract). That is
# what a one-rank combine does. The trainer does not run a one-rank combine: at
# EP=N a token's k experts are split across N expert-parallel ranks and
# HybridEP reduces them in two stages, with a bf16 rounding boundary in
# between. From the kernel source (deep_ep/backend/hybrid_ep_backend.cuh):
#
#   stage 1 -- "unpermute reduction warp group", runs on the
#     EXPERT-side rank. Per destination token it zero-inits
#     ``float2 acc_token_fp32[]`` ("Token must be accumulated in FP32"), loads
#     each of THAT RANK's contributions as ``__nv_bfloat162``, widens with
#     ``__bfloat1622float2`` and ``+=`` in fp32 -- then stores the partial with
#     ``__float22bfloat162_rn`` into a ``uint16_t`` smem/network buffer
#     (``unpermute_token_S2G_buffer``). The per-rank partial is ROUNDED TO BF16
#     before it leaves the rank.
#   stage 2 -- "intra-node reduction warp group", runs on the
#     token-owning rank. Same pattern: fp32 accumulator, bf16 loads of the N
#     per-rank partials, one final ``__float22bfloat162_rn``.
#
# So the trainer computes, for a token whose experts split {A} over rank 0 and
# {B} over rank 1,
#
#     bf16( fp32( bf16(fp32 sum A) + bf16(fp32 sum B) ) )
#
# and ``combine_unweighted_fp32`` computes ``bf16(fp32 sum (A u B))``. One
# extra bf16 rounding per EP rank. At EP=1 the two collapse (rounding is
# idempotent).
#
# Reduction ORDER is not the difference: megatron assigns experts to ranks in
# contiguous blocks (``local_expert_indices = ep_rank * num_local + i``), the
# permuted buffer is expert-major, and the intra-node stage sees at most
# NUM_OF_RANKS_PER_NODE partials in rank order -- both stages are therefore
# ascending expert id, which is what ``ascending_expert_order`` already gives.
#
# Value-typed env: ``RAON_MOE_COMBINE_EP_REALIZED=<ep_size>`` (``0`` = OFF).
# The trainer side reads the SAME env name and asserts its live HybridEP
# topology equals it (megatron token_dispatcher.py), so a wrong EP size on
# either side fails instead of silently emulating the wrong reduction.
# ---------------------------------------------------------------------------


def combine_ep_size() -> int:
    """0 when the lever is OFF, else the expert-parallel size to emulate.

    Under RAON_PREFILL_ALIGNED (value-typed counterpart of ``aligned()``):
    this is evaluated per call rather than latched at import, so the phase
    branch lives entirely inside it -- prefill returns the aligned table value
    (2), decode/gate-off returns the env-resolved value unchanged
    (``raon_phase.combine_ep_value`` announces through the shared
    ``aligned()`` site)."""
    raw = os.environ.get("RAON_MOE_COMBINE_EP_REALIZED", "0")
    if raw in ("0", "", "false"):
        env_ep = 0
    else:
        try:
            env_ep = int(raw)
        except ValueError as exc:  # fail closed: never guess an EP size
            raise RuntimeError(
                "RAON_MOE_COMBINE_EP_REALIZED must be 0 (off) or the integer "
                f"expert-parallel size the trainer ran; got {raw!r}") from exc
        if env_ep < 1:
            raise RuntimeError(
                f"RAON_MOE_COMBINE_EP_REALIZED must be >= 1 when on; got {env_ep}")
    return raon_phase.combine_ep_value(env_ep)


def combine_ep_realized() -> bool:
    return combine_ep_size() != 0


def _combine_ep_partial_fp32_loop(
    expert_out: torch.Tensor,      # [T, K, H] FC2 rows, probs ALREADY folded
    topk_ids: torch.Tensor,        # [T, K] long (pinned routing)
    *,
    ep_size: int,
    num_local_experts: int,
    out_dtype: torch.dtype | None = None,
    transport_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Per-rank/per-slot loop realization.

    Reference that the vectorized ``combine_ep_partial_fp32`` below is
    bit-checked against (import-time self-check + the fused-chain cross-check
    in model.py). The reference MoE chain (``RAON_MOE_CHAIN_FUSED=0``) calls
    this function.
    """
    if out_dtype is None:
        out_dtype = expert_out.dtype
    if transport_dtype is None:
        transport_dtype = expert_out.dtype
    if ep_size < 1 or num_local_experts < 1:
        raise ValueError(
            f"bad topology: ep_size={ep_size} num_local_experts="
            f"{num_local_experts}")
    T, K, H = expert_out.shape
    rows = torch.arange(T, device=expert_out.device)
    order = ascending_expert_order(topk_ids)
    ids_ordered = torch.gather(topk_ids, 1, order)
    rank_ordered = torch.div(ids_ordered, num_local_experts, rounding_mode="floor")
    if int(rank_ordered.max()) >= ep_size:
        raise RuntimeError(
            f"expert id {int(ids_ordered.max())} maps to EP rank "
            f"{int(rank_ordered.max())} outside ep_size={ep_size} "
            f"(num_local_experts={num_local_experts})")
    acc = torch.zeros(T, H, dtype=torch.float32, device=expert_out.device)
    for r in range(ep_size):
        # Stage 1: this rank's own contributions, fp32, ascending expert id.
        part = torch.zeros(T, H, dtype=torch.float32, device=expert_out.device)
        for j in range(K):
            on_rank = (rank_ordered[:, j] == r).unsqueeze(1)
            part = torch.where(
                on_rank, part + expert_out[rows, order[:, j]].float(), part)
        # Stage 2: the partial crosses the network as transport_dtype, and the
        # owner widens it back to fp32. Tokens with no expert on this rank
        # contribute an exact 0.
        acc = acc + part.to(transport_dtype).float()
    return acc.to(out_dtype)


_EP_RANK_MAP_VALIDATED: set = set()


def combine_ep_partial_fp32(
    expert_out: torch.Tensor,      # [T, K, H] FC2 rows, probs ALREADY folded
    topk_ids: torch.Tensor,        # [T, K] long (pinned routing)
    *,
    ep_size: int,
    num_local_experts: int,
    out_dtype: torch.dtype | None = None,
    transport_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """HybridEP two-stage unweighted combine (see the block comment above).

    ``transport_dtype`` is the dtype of HybridEP's per-rank partial buffer,
    i.e. the width of the intermediate rounding (``uint16_t`` -> bf16 in the
    shipped kernel). Defaults to ``expert_out.dtype``, NOT to ``out_dtype``:
    the transport width is a property of the network buffer, not of what the
    caller wants back.

    Vectorized realization: the per-rank
    ``part`` accumulators are stacked into one ``[ep, T, H]`` tensor and each
    slot ``j`` is applied to every rank in ONE broadcast ``torch.where`` —
    per element this evaluates the IDENTICAL fp32 add sequence the original
    loop did (``_combine_ep_partial_fp32_loop`` above; the selected rank sees
    ``part + x_j``, the others keep ``part`` — same values, same order, one
    kernel instead of ep*K*3). Bit-checked against the loop at import
    (``_self_check_combine_ep``) and in the model-level fused-chain
    cross-check. The out-of-range rank sanity check is hoisted to the first
    call per (device, shape) key: it is derived-impossible for a caller that
    already checked ``num_experts % ep == 0`` (topk ids < num_experts =
    ep * num_local), and running ``int(tensor.max())`` per call is a
    per-layer host sync.
    """
    if out_dtype is None:
        out_dtype = expert_out.dtype
    if transport_dtype is None:
        transport_dtype = expert_out.dtype
    if ep_size < 1 or num_local_experts < 1:
        raise ValueError(
            f"bad topology: ep_size={ep_size} num_local_experts="
            f"{num_local_experts}")
    T, K, H = expert_out.shape
    order = ascending_expert_order(topk_ids)
    ids_ordered = torch.gather(topk_ids, 1, order)
    rank_ordered = torch.div(ids_ordered, num_local_experts, rounding_mode="floor")
    vkey = (str(expert_out.device), ep_size, num_local_experts)
    if vkey not in _EP_RANK_MAP_VALIDATED or not expert_out.is_cuda:
        if int(rank_ordered.max()) >= ep_size:
            raise RuntimeError(
                f"expert id {int(ids_ordered.max())} maps to EP rank "
                f"{int(rank_ordered.max())} outside ep_size={ep_size} "
                f"(num_local_experts={num_local_experts})")
        _EP_RANK_MAP_VALIDATED.add(vkey)
    # Preserve the existing topology validation and stable expert order. The
    # specialized kernel implements the same per-slot FP32 additions and each
    # rank's BF16 transport rounding; other dtypes keep the reference below.
    if (os.environ.get("RAON_MOE_EP_COMBINE_FUSED", "1") == "1"
            and expert_out.is_cuda and expert_out.dtype == torch.bfloat16
            and out_dtype == torch.bfloat16 and transport_dtype == torch.bfloat16
            and expert_out.is_contiguous() and topk_ids.is_contiguous()):
        from .moe_ep_kernel import combine as fused_ep_combine
        return fused_ep_combine(expert_out, topk_ids, ep_size,
                                num_local_experts, order=order)
    ranks = torch.arange(ep_size, device=expert_out.device).view(ep_size, 1, 1)
    parts = torch.zeros(ep_size, T, H, dtype=torch.float32,
                        device=expert_out.device)
    for j in range(K):
        # x_j = expert_out[rows, order[:, j]].float() of the loop realization,
        # gathered without materializing ``rows`` (same elements bit-for-bit).
        x_j = torch.gather(
            expert_out, 1,
            order[:, j].view(T, 1, 1).expand(T, 1, H)).squeeze(1).float()
        on = rank_ordered[:, j].view(1, T, 1) == ranks       # [ep, T, 1]
        parts = torch.where(on, parts + x_j.unsqueeze(0), parts)
    acc = torch.zeros(T, H, dtype=torch.float32, device=expert_out.device)
    for r in range(ep_size):
        acc = acc + parts[r].to(transport_dtype).float()
    return acc.to(out_dtype)


def _self_check_combine_ep() -> None:
    """Fail-closed import-time checks for the EP-partial combine.

    Three properties:

      1. EP=1 is BITWISE the single-stage contract -- otherwise the emulation
         would add a rounding the trainer does not have.
      2. EP=2 on a hand-computed tie case DIFFERS from the single-stage
         contract -- a lever that cannot change a bit is a silent no-op.
      3. EP=2 equals an independent per-token python oracle on random data.
    """
    g = torch.Generator().manual_seed(7)
    T, K, H, E, EP = 12, 6, 32, 16, 2
    num_local = E // EP
    eo = torch.randn(T, K, H, generator=g, dtype=torch.float32).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)])

    one = combine_ep_partial_fp32(
        eo, ids, ep_size=1, num_local_experts=E)
    if not torch.equal(one, combine_unweighted_fp32(eo, ids)):
        raise RuntimeError(
            "moe_align.combine_ep_partial_fp32 at ep_size=1 is not bitwise "
            "equal to combine_unweighted_fp32 -- refusing to enable "
            "RAON_MOE_COMBINE_EP_REALIZED")

    # (2) tie case, hand-computed. bf16 has 7 mantissa bits, so ulp(1.0) is
    # 2^-7. Contributions {1.0, 2^-8, 2^-8}, all exact in bf16: one fp32
    # accumulation gives 1 + 2^-7, exact in bf16. Split {1.0, 2^-8} | {2^-8}
    # over two ranks and stage 1 must round 1 + 2^-8 -- exactly halfway
    # between 1.0 and 1 + 2^-7, so ties-to-even drops it to 1.0 -- and the
    # 2^-7 is gone for good (stage 2 then rounds 1 + 2^-8 back to 1.0 again).
    # Keeping all three on ONE rank must reproduce the single-stage answer,
    # because the second rounding is then idempotent.
    ulp = 2.0 ** -7
    tie = torch.zeros(1, 3, 1, dtype=torch.bfloat16)
    tie[0, 0, 0] = 1.0
    tie[0, 1, 0] = ulp / 2
    tie[0, 2, 0] = ulp / 2
    split = torch.tensor([[0, 1, num_local]])       # rank 0, rank 0 | rank 1
    same = torch.tensor([[0, 1, 2]])                # all three on rank 0
    got_split = combine_ep_partial_fp32(
        tie, split, ep_size=EP, num_local_experts=num_local)
    got_same = combine_ep_partial_fp32(
        tie, same, ep_size=EP, num_local_experts=num_local)
    want_single = float(combine_unweighted_fp32(tie, same)[0, 0])
    if (want_single != 1.0 + ulp
            or float(got_same[0, 0]) != want_single
            or float(got_split[0, 0]) != 1.0):
        raise RuntimeError(
            "moe_align.combine_ep_partial_fp32 tie case: expected "
            f"single={1.0 + ulp!r} same-rank={1.0 + ulp!r} split=1.0, got "
            f"single={want_single!r} same-rank={float(got_same[0, 0])!r} "
            f"split={float(got_split[0, 0])!r} -- the EP rounding boundary is "
            "not being realized")

    # (3) independent oracle.
    ref = torch.zeros(T, H, dtype=torch.float32)
    for t in range(T):
        acc = torch.zeros(H, dtype=torch.float32)
        for r in range(EP):
            part = torch.zeros(H, dtype=torch.float32)
            for j in torch.argsort(ids[t], stable=True).tolist():
                if int(ids[t, j]) // num_local == r:
                    part = part + eo[t, j].float()
            acc = acc + part.to(torch.bfloat16).float()
        ref[t] = acc
    two = combine_ep_partial_fp32(eo, ids, ep_size=EP, num_local_experts=num_local)
    if not torch.equal(two, ref.to(torch.bfloat16)):
        raise RuntimeError(
            "moe_align.combine_ep_partial_fp32 disagrees with its per-token "
            "oracle -- refusing to enable RAON_MOE_COMBINE_EP_REALIZED")

    # (4) the vectorized realization is BITWISE the loop realization.
    # Checked on the random data AND on the tie case, both ep sizes -- a fused
    # rewrite that moves one bit is a different function and must fail closed
    # here.
    for eo_c, ids_c, ep_c in ((eo, ids, EP), (eo, ids, 1),
                              (tie, split, EP), (tie, same, EP)):
        nl = E // ep_c if eo_c is eo else num_local
        a = combine_ep_partial_fp32(
            eo_c, ids_c, ep_size=ep_c, num_local_experts=nl)
        b = _combine_ep_partial_fp32_loop(
            eo_c, ids_c, ep_size=ep_c, num_local_experts=nl)
        if not torch.equal(a, b):
            raise RuntimeError(
                "moe_align.combine_ep_partial_fp32 (vectorized) is not "
                "bitwise the original loop realization -- refusing to enable "
                "RAON_MOE_COMBINE_EP_REALIZED")


# ---------------------------------------------------------------------------
# RAON_MOE_SHARED_SITU_REALIZED — the SHARED expert's SiTU (default OFF)
#
# RAON_MOE_TRAINER_REALIZED aligns the ROUTED experts' activation onto the fp32
# SiTU formula (``situ_reference``, megatron ``ScaledSituGLU._glu_forward``) in
# place of vLLM's compiled ``torch.ops._C.situ_and_mul``. It does not touch the
# SHARED expert: the eager MoE forward still ends in
# ``combined + self.shared_experts(x)``, and ``self.shared_experts`` is a stock
# vLLM ``KimiMLP`` whose ``act_fn`` is ``SituAndMul`` -> the compiled kernel.
# Without this lever the two halves of one MoE block use two different SiTU
# realizations. The compiled kernel rounds a small fraction of intermediate
# elements to a neighbouring bf16, which is enough to flip shared-expert
# outputs that sit close to a bf16 boundary.
#
# The lever swaps ONLY the activation and leaves the shared expert's own
# gate_up/down GEMMs (vLLM's F.linear) as they are.
# ---------------------------------------------------------------------------

_SHARED_SITU_ANNOUNCED = False


def shared_situ_realized() -> bool:
    return os.environ.get("RAON_MOE_SHARED_SITU_REALIZED", "0") not in (
        "0", "", "false")


class SharedSituReference(torch.nn.Module):
    """Drop-in replacement for vLLM ``SituAndMul`` on the SHARED expert.

    Evaluates megatron's fp32 SiTU-GLU formula (``situ_reference``) on the
    contiguous ``[gate; value]`` halves the shared expert's
    ``MergedColumnParallelLinear`` produces. Constructed eagerly at model build
    so an OFF arm never reaches this module at all.
    """

    def __init__(self, beta: float, linear_beta: float | None):
        super().__init__()
        if linear_beta is None or float(linear_beta) <= 0:
            raise RuntimeError(
                "RAON_MOE_SHARED_SITU_REALIZED requires a positive "
                "activation_situ_linear_beta (megatron moe_situ_value_beta is "
                f"always > 0); got {linear_beta!r}")
        self.beta = float(beta)
        self.linear_beta = float(linear_beta)

    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        global _SHARED_SITU_ANNOUNCED
        out = situ_reference(gate_up, self.beta, self.linear_beta)
        if not _SHARED_SITU_ANNOUNCED:
            _SHARED_SITU_ANNOUNCED = True
            print(
                "[parity] RAON_MOE_SHARED_SITU_REALIZED=1: shared-expert SiTU "
                f"is moe_align.situ_reference (beta={self.beta}, "
                f"linear_beta={self.linear_beta}) on a {tuple(gate_up.shape)} "
                "gate_up -- the compiled torch.ops._C.situ_and_mul is OFF the "
                "shared path",
                flush=True)
        return out


def install_shared_expert_situ(moe_module, beta: float,
                               linear_beta: float | None) -> str:
    """Swap a ``KimiMoE``'s shared-expert ``act_fn`` for ``SharedSituReference``.

    Fails closed on every way the lever could be armed against nothing: a block
    with no shared expert, a shared expert with no ``act_fn``, an ``act_fn``
    that has already been swapped, or a non-positive ``linear_beta``.
    Returns a one-line description of what was replaced.
    """
    shared = getattr(moe_module, "shared_experts", None)
    if shared is None:
        raise RuntimeError(
            "RAON_MOE_SHARED_SITU_REALIZED is set but this MoE block has no "
            "shared expert; the lever cannot be armed against nothing")
    act = getattr(shared, "act_fn", None)
    if act is None:
        raise RuntimeError(
            "RAON_MOE_SHARED_SITU_REALIZED expects the shared expert's act_fn "
            f"to exist; {type(shared).__name__} has none")
    if isinstance(act, SharedSituReference):
        raise RuntimeError(
            "RAON_MOE_SHARED_SITU_REALIZED: the shared expert's act_fn is "
            "already SharedSituReference; installing twice would hide a "
            "double-wiring bug")
    shared.act_fn = SharedSituReference(beta, linear_beta)
    return (f"{type(act).__name__} -> moe_align.SharedSituReference "
            f"(beta={float(beta)}, linear_beta={float(linear_beta)})")


def _self_check_shared_situ() -> None:
    """Fail-closed import-time checks for the shared-expert SiTU lever.

    Three properties:

      1. ``SharedSituReference`` evaluates ``situ_reference`` bitwise and keeps
         the caller's dtype/shape -- it is a drop-in for ``SituAndMul``.
      2. It refuses a non-positive ``linear_beta``, and ``install_...`` refuses
         a module with no shared expert / no ``act_fn`` / an already-installed
         one, so the lever cannot be armed against nothing.
      3. On CUDA, the replacement is NOT bitwise equal to the compiled
         ``torch.ops._C.situ_and_mul`` over a large sample. A lever that cannot
         change a bit is worse than no lever. The kernel disagrees at a low
         rate (about 2e-5 of elements), so the sample must be big enough that
         zero disagreements is evidence rather than small numbers.
    """
    global _SHARED_SITU_ANNOUNCED
    _announced = _SHARED_SITU_ANNOUNCED
    _SHARED_SITU_ANNOUNCED = True      # the self-check must not spend the latch
    g = torch.Generator().manual_seed(11)
    gu = (torch.randn(37, 2 * 128, generator=g, dtype=torch.float32) * 3.0).to(torch.bfloat16)
    act = SharedSituReference(4.0, 25.0)
    got, want = act(gu), situ_reference(gu, 4.0, 25.0)
    _SHARED_SITU_ANNOUNCED = _announced
    if got.dtype != gu.dtype or got.shape != (37, 128) or not torch.equal(got, want):
        raise RuntimeError(
            "moe_align.SharedSituReference does not evaluate situ_reference — "
            "refusing to enable RAON_MOE_SHARED_SITU_REALIZED")
    for bad in (None, 0.0, -1.0):
        try:
            SharedSituReference(4.0, bad)
        except RuntimeError:
            continue
        raise RuntimeError(
            "SharedSituReference accepted a non-positive linear_beta; the "
            "lever would evaluate a different activation than it replaces")

    class _NoShared:
        shared_experts = None

    class _NoAct:
        shared_experts = type("_S", (), {})()

    class _Already:
        shared_experts = type("_S", (), {"act_fn": SharedSituReference(4.0, 25.0)})()

    for mod, why in ((_NoShared(), "a block with no shared expert"),
                     (_NoAct(), "a shared expert with no act_fn"),
                     (_Already(), "an already-installed act_fn")):
        try:
            install_shared_expert_situ(mod, 4.0, 25.0)
        except RuntimeError:
            continue
        raise RuntimeError(
            f"moe_align.install_shared_expert_situ accepted {why} — the lever "
            "must fail closed")

    if not (torch.cuda.is_available()
            and hasattr(torch.ops, "_C")
            and hasattr(torch.ops._C, "situ_and_mul")):
        return
    big = (torch.randn(4096, 2 * 1088, generator=g, dtype=torch.float32) * 3.0).to(torch.bfloat16).cuda()
    out = torch.empty(big.shape[:-1] + (big.shape[-1] // 2,),
                      dtype=big.dtype, device=big.device)
    torch.ops._C.situ_and_mul(out, big, 4.0, 25.0)
    ref = situ_reference(big, 4.0, 25.0)
    ndiff = int((out != ref).sum())
    if ndiff == 0:
        raise RuntimeError(
            "moe_align: torch.ops._C.situ_and_mul and situ_reference agree "
            f"bitwise over {ref.numel()} elements, so "
            "RAON_MOE_SHARED_SITU_REALIZED cannot change a bit — refusing to "
            "arm a lever that is a silent no-op")
    print(
        "[parity] RAON_MOE_SHARED_SITU_REALIZED: shared-expert SiTU swap "
        "VERIFIED -- SharedSituReference is situ_reference bitwise, it fails "
        "closed on an unarmed module, and it differs from the compiled "
        f"torch.ops._C.situ_and_mul at {ndiff}/{ref.numel()} elements "
        f"({100.0 * ndiff / ref.numel():.5f}%)",
        flush=True)


def _self_check_trainer_realized() -> None:
    """Fail-closed: the structured chain equals an independent fp64-shaped
    eager oracle on CPU (structure check; dtypes exercised as on GPU)."""
    g = torch.Generator().manual_seed(1)
    T, K, H, I = 5, 3, 16, 4
    eo = torch.randn(T, K, H, generator=g, dtype=torch.float32).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(8, generator=g)[:K] for _ in range(T)])
    got = combine_unweighted_fp32(eo, ids)
    # The oracle must accumulate in FP32 even inside vLLM's BF16 context.
    ref = torch.zeros(T, H, dtype=torch.float32)
    for t in range(T):
        for j in torch.argsort(ids[t]):
            ref[t] += eo[t, j].float()
    if not torch.equal(got, ref.to(torch.bfloat16)):
        raise RuntimeError(
            "moe_align.combine_unweighted_fp32 failed its self-check — "
            "refusing to enable RAON_MOE_TRAINER_REALIZED")
    gu = torch.randn(7, 2 * I, generator=g, dtype=torch.float32).to(torch.bfloat16)
    s = situ_reference(gu, 4.0, 25.0)
    gate = gu[..., :I].float()
    val = gu[..., I:].float()
    ref_s = ((4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate))
             * (25.0 * torch.tanh(val / 25.0))).to(torch.bfloat16)
    if not torch.equal(s, ref_s):
        raise RuntimeError("moe_align.situ_reference failed its self-check")


# Import may run inside vLLM's CUDA default-device context. The seeded
# reference checks use CPU generators; restore the caller's device afterward.
with torch.device("cpu"):
    if enabled():
        _self_check()
    if trainer_realized():
        _self_check_trainer_realized()
    if shared_situ_realized():
        if not trainer_realized():
            raise RuntimeError(
                "RAON_MOE_SHARED_SITU_REALIZED aligns the SHARED expert of the "
                "trainer-structured MoE chain and is meaningless without it; set "
                "RAON_MOE_TRAINER_REALIZED=1")
        _self_check_shared_situ()
    if combine_ep_realized():
        if not trainer_realized():
            raise RuntimeError(
                "RAON_MOE_COMBINE_EP_REALIZED replaces the combine of the "
                "trainer-structured chain and is meaningless without it; set "
                "RAON_MOE_TRAINER_REALIZED=1")
        _self_check_combine_ep()

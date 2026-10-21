# SPDX-License-Identifier: Apache-2.0
"""RAON_MOE_RENORM_REALIZED: shared router-gate renormalization.

When enabled, the eager MoE forward computes its top-k gate weights with a
transcription of Megatron's unfused renormalization (same ops, same order,
fp32) instead of the engine's fused ``torch.ops._moe_C.topk_sigmoid``. The
training side has a matching check in Megatron
``moe_utils.topk_routing_with_score_function``.

## Why

On identical fp32 logits, the fused ``topk_sigmoid`` kernel and Megatron's
unfused formula produce gate weights that differ by a few fp32 ULP (the fused
kernel is systematically low) on roughly half of the slots. A 1-ULP
difference can flip a bf16 round-to-nearest tie, so engine and trainer
diverge. The top-k gather order also feeds the renormalization denominator,
so the shared realization fixes the order too: ``torch.topk``'s
descending-score slot order is the order both sides gather and sum in.

This does not address router-GEMM logit differences between trainer and
engine; those are handled separately by RAON_MOE_ROUTER_GEMM_PIN
(moe_router_gemm.py).

## The realization

An exact transcription of ``megatron.core.transformer.moe.moe_utils.
topk_routing_with_score_function`` for the branch RAON uses --
score_function="sigmoid", expert_bias set, topk>1, no groups, no scaling
factor, dense_output=True:

    scores = torch.sigmoid(logits.float()).type_as(logits)   # fp32 in, fp32 out
    scores_for_routing = scores + expert_bias                # fp32
    _, top_indices = torch.topk(scores_for_routing, k=topk, dim=1)
    scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
    probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)

RAON's routed_scaling_factor is 1.0 (the combine applies it in fp32 anyway)
and Megatron applies no scaling here either.

Default OFF and byte-identical when off: ``model.py`` resolves the flag once
at import and the OFF path never reaches this module's tensor ops.

Fail-closed, mirroring RAON_MOE_SHARED_SITU_REALIZED (moe_align.py):
  * import-time self-check (enabled only): the vectorized transcription equals
    an independent per-token loop oracle bitwise on random fp32 logits, and
    the constructor refuses non-fp32 logits / a missing bias.
  * first-live-forward self-check: the reference and the fused kernel are
    both run on the live logits; their per-token top-k sets must agree and,
    on a forward wide enough to carry evidence, the weights must differ
    somewhere (a lever that cannot change a bit is a silent no-op). If
    megatron is importable, the reference is additionally compared against
    the real ``topk_routing_with_score_function`` (advisory).
"""

from __future__ import annotations

import os

import torch

ENV = "RAON_MOE_RENORM_REALIZED"


def renorm_realized() -> bool:
    return os.environ.get(ENV, "0") not in ("0", "", "false")


def topk_renorm_reference(
    logits: torch.Tensor, expert_bias: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Megatron's unfused sigmoid+bias top-k renormalization, transcribed.

    Returns ``(probs [T, topk] fp32, top_indices [T, topk] int64)`` in
    ``torch.topk``'s descending-score slot order -- the trainer's own gather
    and summation order.
    """
    if logits.dtype != torch.float32:
        raise RuntimeError(
            f"{ENV}: the shared renormalization is an fp32 contract and the "
            f"router logits arrived as {logits.dtype}. The trainer computes "
            "fp32 logits (moe_router_dtype fp32); refusing to realize a "
            "different function."
        )
    if expert_bias is None:
        raise RuntimeError(
            f"{ENV}: no e_score_correction_bias -- this transcription is the "
            "sigmoid + expert-bias branch (the branch RAON's trainer runs); "
            "arming it against a bias-less router would realize a different "
            "selection rule."
        )
    if topk <= 1:
        raise RuntimeError(
            f"{ENV}: topk={topk} -- megatron skips the renormalization when "
            "topk <= 1 (probs = scores), so this lever has nothing to share."
        )
    # Exact transcription of megatron-lm moe_utils.topk_routing_with_score_
    # function, score_function == "sigmoid", expert_bias not None.
    # ``.type_as(logits)`` is the no-op fp32 cast megatron also executes.
    scores = torch.sigmoid(logits.float()).type_as(logits)
    scores_for_routing = scores + expert_bias.to(torch.float32)
    _, top_indices = torch.topk(scores_for_routing, k=topk, dim=1)
    scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
    probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    return probs, top_indices


_ANNOUNCED = False
_LIVE_CHECKED = False
_CAPTURE_ANNOUNCED = False


def _validate_router(router) -> torch.Tensor:
    """Fail closed on every router configuration this transcription does not
    realize. Returns the fp32 expert bias."""
    scoring = getattr(router, "scoring_func", None)
    if scoring != "sigmoid":
        raise RuntimeError(
            f"{ENV}: router.scoring_func={scoring!r}, transcription is the "
            "sigmoid branch only")
    if not bool(getattr(router, "renormalize", False)):
        raise RuntimeError(
            f"{ENV}: router.renormalize is off -- there is no renormalization "
            "to share")
    rsf = float(getattr(router, "routed_scaling_factor", 1.0))
    if rsf != 1.0:
        raise RuntimeError(
            f"{ENV}: routed_scaling_factor={rsf} != 1.0 -- megatron applies "
            "no scaling inside the renorm for RAON and neither does this "
            "transcription")
    if int(getattr(router, "num_fused_shared_experts", 0)) != 0:
        raise RuntimeError(
            f"{ENV}: num_fused_shared_experts != 0 -- the fused-shared-slot "
            "append is not part of the shared realization")
    if getattr(router, "_hash_indices_table", None) is not None:
        raise RuntimeError(
            f"{ENV}: router carries a hash_indices_table -- hash routing is "
            "not top-k and cannot be realized by this transcription")
    bias = getattr(router, "e_score_correction_bias", None)
    if bias is None:
        raise RuntimeError(
            f"{ENV}: router has no e_score_correction_bias -- see "
            "topk_renorm_reference")
    bias = bias.data if hasattr(bias, "data") else bias
    return bias.to(torch.float32)


def select_experts_realized(
    router, hidden_states: torch.Tensor, router_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The armed replacement for ``router.select_experts`` on the eager path.

    First live forward: runs BOTH realizations on the live logits and fails
    closed unless (a) every token's top-k SET agrees and (b) the lever is
    demonstrably capable of changing a bit (weights differ somewhere on a
    forward wide enough that zero differences would be evidence of a no-op,
    not of luck: kernel and formula typically differ on ~55% of slots).
    """
    global _ANNOUNCED, _LIVE_CHECKED
    bias = _validate_router(router)
    topk = int(router.top_k)
    probs, top_indices = topk_renorm_reference(router_logits, bias, topk)

    if not _LIVE_CHECKED:
        _LIVE_CHECKED = True
        # RAON_MOE_RENORM_LIVECHECK=0 (default 1): skip the first-forward
        # cross-check. It can misfire when the lever is fine: a degenerate
        # first forward (e.g. profiling input) where fused == reference on
        # every slot trips the no-op guard, and the megatron clause may
        # compare against a different co-resident megatron version. The skip
        # is logged.
        if os.environ.get("RAON_MOE_RENORM_LIVECHECK", "1") in ("0", "false"):
            print(
                f"[parity] {ENV}=1: first-forward live cross-check SKIPPED by "
                "RAON_MOE_RENORM_LIVECHECK=0",
                flush=True,
            )
        else:
            _live_cross_check(
                router, hidden_states, router_logits, probs, top_indices,
                bias, topk)
    if not _ANNOUNCED:
        _ANNOUNCED = True
        print(
            f"[parity] {ENV}=1: MoE router-gate renormalization is the "
            "Megatron unfused transcription (moe_renorm.topk_renorm_"
            "reference, fp32, torch.topk gather/sum order) -- the fused "
            f"torch.ops._moe_C.topk_sigmoid is OFF the eager path "
            f"(first forward: {tuple(router_logits.shape)} logits, "
            f"topk={topk})",
            flush=True,
        )
    # vLLM's enable_return_routed_experts records routing inside
    # BaseRouter.select_experts (`if self.capture_fn is not None:
    # self.capture_fn(topk_ids)`). This path does not go through that method
    # (_live_cross_check calls it once, and its kern_ids are discarded), so
    # call capture_fn here with the ids actually dispatched, as the fused
    # path does. Otherwise the capture buffer stays all zeros.
    # BaseRouter.__init__ always defines capture_fn (None by default).
    capture_fn = getattr(router, "capture_fn", None)
    if capture_fn is not None:
        global _CAPTURE_ANNOUNCED
        if not _CAPTURE_ANNOUNCED:
            # One-time consistency check before the first capture: compare
            # top_indices with a top-k independently recomputed from
            # router_logits + bias on a few tokens, so capture_fn is known to
            # receive the selection that was actually dispatched.
            _N = min(8, top_indices.shape[0])
            _ref_topk = torch.topk(
                (torch.sigmoid(router_logits[:_N].float()) + bias).to(torch.float32),
                k=topk, dim=1,
            ).indices
            _cap_sets = torch.zeros(_N, router_logits.shape[1], dtype=torch.bool,
                                     device=router_logits.device)
            _cap_sets.scatter_(1, top_indices[:_N].long(), True)
            _ref_sets = torch.zeros_like(_cap_sets)
            _ref_sets.scatter_(1, _ref_topk.long(), True)
            assert bool((_cap_sets == _ref_sets).all()), (
                f"{ENV} route-capture consistency check failed: "
                f"top_indices[:{_N}] passed to capture_fn differ from the "
                "top-k SET recomputed from router_logits -- the capture is "
                "not seeing the dispatched selection"
            )
            _CAPTURE_ANNOUNCED = True
            print(
                f"[parity] {ENV}=1 route-capture LIVE: router.capture_fn "
                "invoked from the eager renorm-realized path with the "
                "ACTUAL dispatched top_indices (not the cross-check-only "
                f"kern_ids) -- internal-consistency check passed on {_N} "
                "sampled tokens (independent recompute from router_logits "
                "matches the captured selection SET) -- "
                "enable_return_routed_experts now sees the real routing",
                flush=True,
            )
        capture_fn(top_indices)
    return probs, top_indices


def _live_cross_check(
    router, hidden_states, router_logits, probs, top_indices, bias, topk
) -> None:
    with torch.no_grad():
        kern_w, kern_ids = router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_indices_dtype=torch.long,
        )
        ref_sets = torch.zeros(
            router_logits.shape, dtype=torch.bool, device=router_logits.device)
        ref_sets.scatter_(1, top_indices, True)
        kern_sets = torch.zeros_like(ref_sets)
        kern_sets.scatter_(1, kern_ids.long(), True)
        set_mism = int((ref_sets != kern_sets).any(dim=1).sum())
        if set_mism:
            raise RuntimeError(
                f"{ENV} live cross-check FAILED: the reference renormalization "
                f"and the fused kernel disagree about the top-{topk} SET on "
                f"{set_mism}/{router_logits.shape[0]} tokens of the first live "
                "forward. A set flip means the selection landed on a tie or "
                "this environment differs from the validated one. Refusing to "
                "dispatch a selection the fused path would not have made."
            )
        # Weights compared at the reference slot order (kernel order may
        # legitimately differ; the SET is the dispatch contract).
        kern_dense = torch.zeros_like(router_logits)
        kern_dense.scatter_(1, kern_ids.long(), kern_w.to(torch.float32))
        kern_at_ref = torch.gather(kern_dense, 1, top_indices)
        ndiff = int(
            (kern_at_ref.view(torch.int32)
             != probs.view(torch.int32)).sum())
        nslots = probs.numel()
        # Evidence floor: at the typical ~55% differing rate, zero
        # differences on a 512-token forward are essentially impossible if the
        # lever is live; below the floor a zero is only reported.
        if ndiff == 0 and nslots >= 6 * 512:
            raise RuntimeError(
                f"{ENV} live cross-check FAILED: reference and fused kernel "
                f"agree bitwise on all {nslots} slots of a "
                f"{router_logits.shape[0]}-token forward. Typically ~55% of "
                "slots differ; a lever that cannot change a bit is a silent "
                "no-op. Refusing to arm."
            )
        # Clause 3 (advisory, not fail-closed): compare against whatever
        # megatron is importable in this process. In a colocated RL run that
        # can be a different megatron version than the one this transcribes,
        # so a mismatch indicates comparand skew rather than transcription
        # drift (the unfused branch is pure torch, so only a different source
        # can disagree). This clause reports instead of raising; clauses 1 and
        # 2 (vs the engine's own fused kernel, which this lever replaces) are
        # the fail-closed contract. Set RAON_MOE_RENORM_SELFCHECK_MEGATRON=strict
        # to raise on mismatch.
        mega_line = "megatron not importable in this venv (skipped)"
        try:
            import megatron.core.transformer.moe.moe_utils as _mm
            _mega_topk = _mm.topk_routing_with_score_function
            _mega_file = getattr(_mm, "__file__", "<unknown>")
        except Exception:
            _mega_topk = None
            _mega_file = None
        if _mega_topk is not None:
            m_probs, m_idx = _mega_topk(
                router_logits, topk, score_function="sigmoid",
                expert_bias=bias, dense_output=True)
            eq = (m_probs.shape == probs.shape
                  and torch.equal(m_probs, probs)
                  and torch.equal(m_idx.long(), top_indices.long()))
            if eq:
                mega_line = (f"megatron ({_mega_file}) "
                             "topk_routing_with_score_function == reference "
                             "BITWISE on the live forward")
            else:
                nd = (int((m_probs.reshape(-1).view(torch.int32)
                           != probs.reshape(-1).view(torch.int32)).sum())
                      if m_probs.shape == probs.shape else -1)
                mx = (int((m_probs.view(torch.int32)
                           - probs.view(torch.int32)).abs().max())
                      if m_probs.shape == probs.shape else -1)
                mega_line = (
                    f"megatron ({_mega_file}) DIFFERS from reference on "
                    f"{nd}/{probs.numel()} slots (max {mx} fp32 ULP) -- "
                    "COMPARAND SKEW (imported megatron is not this "
                    "transcription's lineage; the transcription's match to the "
                    "RL trainer is verified offline). Advisory only; clauses "
                    "1-2 (vs the engine's own fused kernel) are the hard check.")
                if os.environ.get(
                        "RAON_MOE_RENORM_SELFCHECK_MEGATRON") == "strict":
                    raise RuntimeError(
                        f"{ENV} live cross-check FAILED (strict): {mega_line}")
        print(
            f"[parity] {ENV} live cross-check PASS: top-{topk} sets equal on "
            f"all {router_logits.shape[0]} tokens; reference vs fused "
            f"topk_sigmoid weights differ on {ndiff}/{nslots} slots "
            f"({100.0 * ndiff / max(nslots, 1):.2f}%; typical rate "
            f"~55%); {mega_line}",
            flush=True,
        )


def _self_check() -> None:
    """Import-time fail-closed checks (armed only). CPU, no CUDA required."""
    g = torch.Generator().manual_seed(207295)
    T, E, K = 64, 128, 6
    logits = torch.randn(T, E, generator=g, dtype=torch.float32) * 2.0
    bias = torch.randn(E, generator=g, dtype=torch.float32) * 0.1
    probs, idx = topk_renorm_reference(logits, bias, K)
    # Independent per-token oracle: same formula, scalar order.
    for t in range(T):
        s = torch.sigmoid(logits[t].float())
        sfr = s + bias
        _, want_idx = torch.topk(sfr, k=K, dim=0)
        if not torch.equal(want_idx, idx[t]):
            raise RuntimeError(
                "moe_renorm.topk_renorm_reference selection differs from the "
                f"per-token oracle at row {t} -- refusing to enable {ENV}")
        gs = torch.gather(s, 0, want_idx)
        want = gs / (gs.sum(dim=-1, keepdim=True) + 1e-20)
        if not torch.equal(want, probs[t]):
            raise RuntimeError(
                "moe_renorm.topk_renorm_reference weights differ from the "
                f"per-token oracle at row {t} -- refusing to enable {ENV}")
    if probs.dtype != torch.float32 or idx.shape != (T, K):
        raise RuntimeError(
            f"moe_renorm.topk_renorm_reference output contract violated -- "
            f"refusing to enable {ENV}")
    for bad_logits, bad_bias, bad_k, why in (
        (logits.to(torch.bfloat16), bias, K, "bf16 logits"),
        (logits, None, K, "missing bias"),
        (logits, bias, 1, "topk == 1"),
    ):
        try:
            topk_renorm_reference(bad_logits, bad_bias, bad_k)
        except RuntimeError:
            continue
        raise RuntimeError(
            f"moe_renorm.topk_renorm_reference accepted {why} -- the lever "
            "must fail closed")


if renorm_realized():
    # The reference check is CPU-only, including when vLLM imports on CUDA.
    with torch.device("cpu"):
        _self_check()

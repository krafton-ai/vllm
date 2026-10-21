# SPDX-License-Identifier: Apache-2.0
"""raon-1-mini: the layer loop that carries the four mHC streams.

RAON threads one ``[T, streams, hidden]`` activation between layers, and the
*entire* residual lives inside each site's doubly stochastic ``H_res`` mix —
the branch adds nothing. Adding vLLM's usual ``hidden = residual +
branch(norm(hidden))`` residual anywhere in this model produces a different
network.

Layers: ``MultiHeadLatentAttention`` (mla_attention.py; NoPE, output-gated,
scale 192**-0.5), ``RaonMoE`` (mlp.py; sigmoid router + expert bias +
SiTU-GLU), ``RaonMLP`` and the RDA layer from kda.py.
The MLA wrapper only preserves inverse-scale buffer addresses on refit.

Nothing about mHC persists across decode steps and nothing goes in the KV
cache: every site reads only that token's own ``(streams, hidden)`` slice. The
stream tensor is an ordinary activation.

Out of scope for v1 and asserted, not silently degraded: pipeline parallelism
(the carrier would have to move through ``IntermediateTensors``), sequence
parallelism (nobody has sharded the 3-D carrier), MTP (weights are converted, the layer is not built).
"""

import os
import re
import types
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    SupportsEagle3,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    make_layers,
    maybe_prefix,
)

from vllm.transformers_utils.configs.raon1_mini import Raon1MiniConfig
from .kda import RaonKDAAttention
from .mla_attention import MultiHeadLatentAttention
from .mlp import RaonMLP, RaonMoE
from .mla_refit import StableMLAScaleBuffers
from .mla_align import install_mla_output_gate_two_round
from .mla_proj_align import apply_module as apply_mla_proj_persistent
from .mhc import _AGGNORM_FUSED_ENV as _MHC_AGGNORM_FUSED_ENV
from .mhc import RaonMHC
from . import raon_phase
from .mlp_align import apply_module as apply_dense_mlp_persistent
from .mlp_align import apply_situ_native as apply_dense_mlp_situ_native
from .moe_align import _combine_ep_partial_fp32_loop as moe_combine_ep_partial_loop
from .moe_align import combine as aligned_moe_combine
from .moe_align import combine_ep_partial_fp32 as moe_combine_ep_partial
from .moe_align import combine_ep_size as moe_combine_ep_size
from .moe_align import combine_unweighted_fp32 as moe_combine_unweighted
from .moe_align import enabled as fp32_moe_combine_enabled
from .moe_align import fold_probs as moe_fold_probs
from .moe_align import install_shared_expert_situ as moe_install_shared_situ
from .moe_align import shared_situ_realized as moe_shared_situ_realized
from .moe_align import situ_reference as moe_situ_reference
from .moe_align import trainer_realized as moe_trainer_realized
from .mxfp8_align import apply_mla_mxfp8 as mxfp8_apply_mla
from .mxfp8_align import apply_shared_mxfp8 as mxfp8_apply_shared
from .mxfp8_align import expert_mxfp8_realized as moe_expert_mxfp8_realized
from .mxfp8_align import MIN_M as _MXFP8_MIN_M
from .mxfp8_align import rows_reach_min_m as mxfp8_rows_reach_min_m
from .mxfp8_align import fused_expert_rows as mxfp8_fused_expert_rows
from .mxfp8_w8a8_triton import enabled as mxw8a8_triton_enabled
from .mxfp8_w8a8_triton import expert_chain_mxw8a8
from .mxfp8_w8a8_triton import expert_chain_mxw8a8_ep_finalized
from .mxfp8_w8a8_triton import expert_chain_mxw8a8_finalized
from .mxfp8_align import (
    fused_expert_rows_static as mxfp8_fused_expert_rows_static)
from .moe_renorm import renorm_realized as moe_renorm_realized
from .moe_renorm import select_experts_realized as moe_select_experts_realized
from .moe_route import enabled as moe_route_capture_enabled
from .moe_route import record as moe_route_record


class RaonRefitMLA(StableMLAScaleBuffers, MultiHeadLatentAttention):
    """Upstream MLA math with graph-safe reciprocal-scale registration."""


# Resolved once at import so an OFF arm is byte-identical: the eager MoE
# forward below then executes one already-evaluated boolean and no tensor op.
_MOE_ROUTE_CAPTURE = moe_route_capture_enabled()
# RAON_MOE_RENORM_REALIZED (default OFF): shared router-gate renormalization
# (moe_renorm.py). Same one-shot resolution rule as the other levers.
#
# Named _ENV because it only reflects the env, which cannot change at
# runtime; the per-forward RAON_PREFILL_ALIGNED phase gate is ORed in at
# the branch condition, not in this import-time latch.
_MOE_RENORM_REALIZED_ENV = moe_renorm_realized()
# RAON_MOE_COMBINE_EP_REALIZED one-time announce latch (see the combine below).
_MOE_COMBINE_EP_ANNOUNCED = False
# RAON_MOE_SHARED_SITU_REALIZED one-time announce latch (24 MoE layers install
# the same swap; the log wants it once).
_MOE_SHARED_SITU_ANNOUNCED = False
# RAON_MOE_CHAIN_FUSED (default ON): launch-fused realization of the MoE
# realized chain. NOT a parity lever: it is proven bit-identical to the
# reference chain (live cross-check below, fail closed; moe_align import
# self-checks; run/moe_fuse_bitcheck.py sweep). =0 selects the
# byte-identical pre-optimization chain.
_MOE_CHAIN_FUSED = os.environ.get(
    "RAON_MOE_CHAIN_FUSED", "1") not in ("0", "", "false")
# One-time live cross-check latch (fused vs reference on the first forward).
_MOE_FUSED_CROSSCHECKED = False

# One line per distinct verdict per process for the two levers below, so
# logs show whether a lever actually executed.
# Keyed in a set, like mxfp8_align._ANNOUNCED, so a lever that engages on 24
# MoE layers x every decode step still costs one line.
_LEVER_ANNOUNCED: set = set()

# Execution witness for the EP-combine call. _moe_combine_realized_tail is
# shared by the reference, mxfp8 and fused chains, so an
# announce placed inside it cannot name the lever that chose the function.
# It bumps this counter instead and the mxfp8 chain reads the counter ACROSS
# its own tail call: a bump is proof its _ep_fn ran, no bump is proof the tail
# never called it. Bookkeeping only -- one dict store on a path that already
# launches a combine.
_EP_PARTIAL_CALLS = {"n": 0, "fn": None}


def _situ_and_mul(gate_up: torch.Tensor, beta: float, linear_beta: float | None):
    """SiTU-GLU gate: the exact op the fused MoE expert kernel applies.

    Mirrors ``vllm.model_executor.layers.fused_moe.activation.apply_moe_activation``
    for ``MoEActivation.SITU`` (and ``SituAndMul.forward_cuda``): the compiled
    ``torch.ops._C.situ_and_mul`` on ``[gate; up]`` with ``linear_beta<=0`` == unset.
    Using the same activation op keeps the SiTU nonlinearity faithful to the
    fused path. NOTE: the surrounding FC1/FC2 are plain per-expert ``torch.mm``
    here (expert-major dispatch), not the fused triton grouped GEMM, so the
    eager per-expert output is *not* bit-identical to the fused kernel's
    pre-combine output, so RAON_MOE_FP32_COMBINE=0/1 differences at L*.moe
    include that GEMM-kernel difference, not only the combine change.
    """
    d = gate_up.shape[-1] // 2
    out = torch.empty(
        gate_up.shape[:-1] + (d,), dtype=gate_up.dtype, device=gate_up.device
    )
    torch.ops._C.situ_and_mul(
        out, gate_up, float(beta), -1.0 if linear_beta is None else float(linear_beta)
    )
    return out


def _raon_fp32_combine_moe_impl(self, hidden_states):
    if os.environ.get("RAON_MOE_OUTER_GRAPH", "1") == "1":
        from .moe_outer_graph import try_forward
        result = try_forward(self, hidden_states, _raon_fp32_combine_moe_impl_eager)
        if result is not None:
            return result
    return _raon_fp32_combine_moe_impl_eager(self, hidden_states)


# ---------------------------------------------------------------------------
# shared experts on a side stream
# ---------------------------------------------------------------------------
# combined = combined + self.shared_experts(x) reads only x, never combined --
# a self-contained 2-GEMM + activation subgraph sitting in series behind the
# routed chain for no data-dependency reason. Launching it on a second stream
# lets it overlap the routed chain's kernels. The addition and BOTH operands'
# bits are untouched; only the relative timing moves.
#
# DEFAULT OFF, deliberately: (a) the engine is host-paced and a side stream
# does not reduce host launch cost, so the expected decode gain is small
# (1-2%); (b) under breakable CUDA graphs the shared experts sit in the
# same eager-break region and the interaction needs re-verification.
# The plugin default stays OFF; the MXFP8 kit profile (kit/mxfp8.env) arms
# it together with RAON_EXPERIMENT_SHARED_GRAPH. The BF16 profile does not.
#
# A SECOND stream, not the model's existing aux_stream: that one is already
# spoken for by the MLA g_proj overlap (RaonModel.streams), and sharing it
# would serialize the two overlaps against each other.
_SHARED_SIDE_STREAMS = {}


def _shared_side_stream(device):
    st = _SHARED_SIDE_STREAMS.get(device)
    if st is None:
        st = torch.cuda.Stream(device=device)
        _SHARED_SIDE_STREAMS[device] = st
    return st


def _raon_fp32_combine_moe_impl_eager(
    self: RaonMoE, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Eager reference-reduction MoE forward, bound onto ``RaonMoE`` (via
    ``_raon_fp32_combine_moe_forward`` below) when ``RAON_MOE_FP32_COMBINE=1``.

    vLLM's fused KimiMoE folds the router weight into a bf16 GEMM epilogue and
    sums a token's k expert outputs at a bf16 rounding boundary in *selection*
    order — missing the Megatron ``exact_reference`` trainer by ~5e-2 max on the
    ``L*.moe`` parity site (route-pinned). This path instead:

      1. selects the top-k with the *same* router object (identical, FP32,
         route-pinned; the router already folds routed_scaling_factor into the
         weights, so ``combine`` re-applies scaling of 1.0);
      2. runs the experts eagerly in bf16 (the weights and situ op the fused
         kernel uses) via EXPERT-MAJOR DISPATCH, giving [T, K, H] FC2 rows;
      3. reduces them with ``moe_align.combine`` — FP32, ascending-expert-id
         (Megatron expert-major) order, single cast to bf16 — the primitive
         bit-checked against the trainer oracle;
      4. adds the shared expert, matching the trainer ``L*.moe`` site (routing +
         shared already combined).

    Step 2 uses expert-major dispatch instead of gathering a per-token copy
    of the expert weights (``w13.index_select(0, eid)`` -> [T, 2I, H] + bmm),
    which at vLLM's 8192-token memory-profiling batch would be a single
    85.00 GiB allocation (T=8192 x 2I=2176 x H=2560 x bf16):

      * stable-argsort the T*K (token, slot) routed rows by expert id, run
        each expert's rows through FC1 -> act -> FC2 against that expert's
        RESIDENT weight, and scatter rows back by the inverse permutation.
        The grouped GEMM is realized as a per-expert ``torch.mm`` loop over
        the cu_seqlens segments, the same structure as Megatron's
        sequential/grouped expert MLP.
      * the stable argsort gives the canonical expert-major (ascending expert
        id) row order, which is exactly the order ``moe_align.combine``
        reduces in.
      * every route-sized temporary is bounded by rows actually routed,
        never by tokens x weights.

    Activation footprint at the 8192-token profiling batch: O(T*K*(H+2I)) ~
    0.8 GiB, zero weight copies. Numerics note: per-expert ``mm`` vs the old
    per-token ``bmm`` is a different bf16 GEMM kernel choice, so pre-combine
    FC2 rows are not bit-identical to the OLD eager path — no oracle pins them
    (see ``_situ_and_mul``); the bit-checked contract is ``moe_align.combine``
    on whatever FC2 rows are produced, and that primitive is untouched.

    Fail-closed: only the RAON config path is supported (non-latent, non-MegaMoE,
    unquantized bf16 experts, TP1/EP-off so every expert is local). Anything else
    raises rather than silently mis-measure. One host sync per MoE layer (the
    per-expert row counts) is accepted: under enforce_eager there is no CUDA-graph
    capture to break, and under VLLM_USE_BREAKABLE_CUDAGRAPH the whole chain runs
    as an eager break (``_raon_fp32_combine_moe_forward`` below), outside every
    captured segment, re-executed per replay with the live routing.
    """
    num_tokens, hidden_size = hidden_states.shape
    x = hidden_states.view(-1, hidden_size)

    # Launch the shared experts now, before the router and the routed
    # chain, so their kernels overlap. See _shared_side_stream.
    # RAON_EXPERIMENT_SHARED_GRAPH=1 also forks/joins the side stream inside
    # CUDA-graph capture, so captured decode shapes get the overlap too;
    # without it they run the shared experts in series (INERT line below).
    _shared_side = None
    if (getattr(self, "shared_experts", None) is not None
            and os.environ.get("RAON_MOE_SHARED_SIDE_STREAM", "0") == "1"
            and x.is_cuda
            and (not torch.cuda.is_current_stream_capturing()
                 or os.environ.get("RAON_EXPERIMENT_SHARED_GRAPH", "0") == "1")):
        _main = torch.cuda.current_stream()
        _side = _shared_side_stream(x.device)
        _side.wait_stream(_main)
        # x was allocated on the main stream; the caching allocator must not
        # hand its memory to anything else while the side stream still reads
        # it. Same reason the shared output needs record_stream at the join.
        x.record_stream(_side)
        with torch.cuda.stream(_side):
            _shared_out = self.shared_experts(x)
        _shared_evt = torch.cuda.Event()
        _shared_evt.record(_side)
        _shared_side = (_shared_out, _shared_evt)
    elif os.environ.get("RAON_MOE_SHARED_SIDE_STREAM", "0") == "1":
        # Print only -- nothing here touches _shared_side, so the computation
        # is the same whether or not this arm is compiled in. The lever is
        # armed and the launch site was reached, so the run is entitled to know
        # WHICH of the three remaining conditions turned it away rather than
        # reading the arm as "a side stream that did not help".
        _why = ("no shared_experts on this MoE layer"
                if getattr(self, "shared_experts", None) is None
                else "hidden states are not on CUDA" if not x.is_cuda
                else "the current stream is capturing a CUDA graph")
        if ("shared_side_inert", _why) not in _LEVER_ANNOUNCED:
            _LEVER_ANNOUNCED.add(("shared_side_inert", _why))
            print(f"[parity] RAON_MOE_SHARED_SIDE_STREAM INERT: the lever is "
                  f"on and the launch site was reached, but the side stream "
                  f"was declined because {_why}; the shared experts run in "
                  f"series on the main stream, exactly as with the lever off",
                  flush=True)

    if getattr(self, "use_mega_moe", False) or getattr(self, "use_latent_moe", False):
        raise RuntimeError(
            "RAON_MOE_FP32_COMBINE eager combine supports neither MegaMoE nor "
            "latent MoE; the routed_output_transform tail is out of scope"
        )
    runner = self.experts
    routed = getattr(runner, "routed_experts", None)
    router = getattr(runner, "router", None)
    if routed is None or router is None:
        raise RuntimeError("RAON_MOE_FP32_COMBINE: experts is not a MoERunner")
    w13 = getattr(routed, "w13_weight", None)  # [E, 2I, H] in [gate; up] order
    w2 = getattr(routed, "w2_weight", None)  # [E, H, I]
    if w13 is None or w2 is None or w13.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            "RAON_MOE_FP32_COMBINE eager combine requires unquantized bf16/fp16 "
            f"expert weights; got w13={None if w13 is None else w13.dtype}"
        )

    # Router: identical selection to the fused path (same router object).
    router_logits, _ = self.gate(x)
    # OR with the cached per-forward RAON_PREFILL_ALIGNED phase verdict so a
    # decode forward keeps the raw env value even while the gate is armed.
    if _MOE_RENORM_REALIZED_ENV or raon_phase.aligned("moe.renorm"):
        # RAON_MOE_RENORM_REALIZED (default OFF): the SHARED router-gate
        # renormalization -- Megatron's unfused fp32 formula in torch.topk's
        # gather/sum order, in place of the fused torch.ops._moe_C.
        # topk_sigmoid, which sits 1-2 fp32 ULP low and can flip bf16 ties in
        # the fold. First-forward cross-check + announce live in
        # moe_renorm.py, fail closed.
        topk_weights, topk_ids = moe_select_experts_realized(
            router, x, router_logits)
    else:
        topk_weights, topk_ids = router.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            topk_indices_dtype=torch.long,
        )
    topk_ids = topk_ids.long()

    # RAON_MOE_ROUTE_CAPTURE (default OFF): record the selection this forward
    # is about to dispatch on, for the trainer's MCore RouterReplay arm. The
    # capture sits HERE, on the tensor the expert dispatch actually consumes,
    # rather than on vLLM's enable_return_routed_experts capturer -- see
    # moe_route.py for why those are not the same routing.
    # RAON_MOE_ROUTE_CAPTURE_WEIGHTS additionally records the fp32 top-k
    # WEIGHTS attached to that selection, on the same tensor and at the same
    # site, so the trainer can pin them too (moe_route.py explains why: the
    # router gate GEMM still contributes a residual logit term once the
    # renormalization is aligned).
    if _MOE_ROUTE_CAPTURE:
        moe_route_record(self.layer_idx, topk_ids, topk_weights)

    mc = routed.moe_config
    beta = mc.activation_situ_beta
    linear_beta = mc.activation_situ_linear_beta

    layer_idx = getattr(self, "layer_idx", "?")
    # RAON_MOE_CHAIN_FUSED (default ON): the launch-fused realization of the
    # SAME chain (see _moe_expert_chain_fused). Proven bit-identical to the
    # reference chain on the FIRST live forward (cross-check below, fail
    # closed), by the moe_align import-time self-checks and by the shape-sweep
    # harness (run/moe_fuse_bitcheck.py).
    # =0 falls back to the byte-identical pre-optimization chain.
    global _MOE_FUSED_CROSSCHECKED
    # RAON_MOE_EXPERT_MXFP8_REALIZED (mxfp8 prototype): the
    # engine runs PRODUCTION's fused CuTe MXFP8 grouped MLP as the expert
    # GEMM realization (mxfp8_align.py).
    # PRECEDENCE: this lever REPLACES the expert
    # GEMM+SiTU+fold chain wholesale, so RAON_MOE_CHAIN_FUSED — a launch
    # fusion of the bf16 chain proven bit-identical TO THE BF16 CHAIN — is
    # inapplicable and is bypassed (announced once). The dispatch, combine
    # tail and shared expert stay the certified realizations.
    if moe_expert_mxfp8_realized():
        combined = _moe_expert_chain_mxfp8(
            x, topk_weights, topk_ids, w13, w2, beta, linear_beta, layer_idx)
    elif _MOE_CHAIN_FUSED:
        if not _MOE_FUSED_CROSSCHECKED:
            _MOE_FUSED_CROSSCHECKED = True
            ref = _moe_expert_chain_reference(
                x, topk_weights, topk_ids, w13, w2, beta, linear_beta,
                layer_idx)
            combined = _moe_expert_chain_fused(
                x, topk_weights, topk_ids, w13, w2, beta, linear_beta,
                layer_idx)
            if not torch.equal(ref, combined):
                d = (ref.float() - combined.float()).abs()
                raise RuntimeError(
                    "RAON_MOE_CHAIN_FUSED live cross-check FAILED: fused MoE "
                    f"chain differs from the reference chain at layer_idx="
                    f"{layer_idx} on a {tuple(x.shape)} input -- "
                    f"{int((ref != combined).sum())}/{ref.numel()} elements, "
                    f"max|d|={d.max().item():.6e}. The chain's output bits "
                    "are the contract; refusing to run the fused path. Set "
                    "RAON_MOE_CHAIN_FUSED=0 to fall back.")
            print(
                "[parity] RAON_MOE_CHAIN_FUSED=1 live cross-check PASS: "
                "fused chain == reference chain bitwise on layer_idx="
                f"{layer_idx}, {tuple(x.shape)} input "
                f"({ref.numel()} elements)", flush=True)
        else:
            combined = _moe_expert_chain_fused(
                x, topk_weights, topk_ids, w13, w2, beta, linear_beta,
                layer_idx)
    else:
        combined = _moe_expert_chain_reference(
            x, topk_weights, topk_ids, w13, w2, beta, linear_beta, layer_idx)
    if self.shared_experts is not None:
        if _shared_side is not None:
            _shared_out, _shared_evt = _shared_side
            _main = torch.cuda.current_stream()
            _main.wait_event(_shared_evt)
            _shared_out.record_stream(_main)
            combined = combined + _shared_out
            # Emitted from inside the JOIN, not from the launch: reaching here
            # means the side-stream shared-expert work was launched at the top
            # of this forward, ran concurrently with the whole routed chain,
            # and its result is the operand being added. A launch-site line
            # could not tell that apart from a stream nobody ever joined.
            if ("shared_side_live",) not in _LEVER_ANNOUNCED:
                _LEVER_ANNOUNCED.add(("shared_side_live",))
                print(f"[parity] RAON_MOE_SHARED_SIDE_STREAM LIVE: shared "
                      f"experts ran on the side stream and joined the main "
                      f"stream on layer_idx={layer_idx} -- {tuple(x.shape)} "
                      f"input, overlapped against the routed chain; the "
                      f"addition and both operands are bit-unchanged, only "
                      f"the timing moved", flush=True)
        else:
            combined = combined + self.shared_experts(x)
    return combined.view(num_tokens, hidden_size)


def _moe_combine_realized_tail(
    expert_out: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
    out_dtype: torch.dtype, num_experts: int, layer_idx, realized: bool,
    ep_partial_fn,
) -> torch.Tensor:
    """The combine stage shared by the reference and fused chains.

    ``ep_partial_fn`` selects the EP-combine realization: the ORIGINAL loop
    (reference chain) or the vectorized twin (fused chain) — bit-checked
    equal at import (moe_align._self_check_combine_ep check 4) and in the
    live cross-check. Everything else is byte-moved from the pre-optimization
    ``_raon_fp32_combine_moe_impl``.
    """
    if realized:
        # RAON_MOE_COMBINE_EP_REALIZED=<ep> (default 0/OFF): realize hybridep's
        # TWO-STAGE combine instead of the single-rank fp32 contract -- per-EP-
        # rank fp32 partial, rounded to bf16 on the wire, then fp32 across
        # ranks. See the moe_align.py block comment for the kernel-source
        # provenance. OFF is byte-identical to the single-stage contract.
        _ep = moe_combine_ep_size()
        if _ep:
            if num_experts % _ep:
                raise RuntimeError(
                    f"RAON_MOE_COMBINE_EP_REALIZED={_ep} does not divide "
                    f"num_experts={num_experts}; the trainer's expert->rank "
                    "block mapping cannot be reproduced")
            combined = ep_partial_fn(
                expert_out, topk_ids, ep_size=_ep,
                num_local_experts=num_experts // _ep, out_dtype=out_dtype)
            # Execution witness, read back by _announce_mxfp8_combine_vector.
            # Bookkeeping only: nothing downstream branches on it.
            _EP_PARTIAL_CALLS["n"] += 1
            _EP_PARTIAL_CALLS["fn"] = getattr(ep_partial_fn, "__name__", "?")
            if not _MOE_COMBINE_EP_ANNOUNCED:
                # One-time, in-process proof the lever is LIVE and not inert:
                # the delta against the single-stage contract on this layer's
                # real expert rows. A zero there would mean the EP boundary
                # never fires (a silent no-op).
                _single = moe_combine_unweighted(expert_out, topk_ids,
                                                 out_dtype=out_dtype)
                _d = (combined.float() - _single.float()).abs()
                globals()["_MOE_COMBINE_EP_ANNOUNCED"] = True
                print(
                    f"[parity] RAON_MOE_COMBINE_EP_REALIZED={_ep}: hybridep "
                    f"two-stage combine live on layer_idx="
                    f"{layer_idx} -- vs the single-stage "
                    f"fp32 contract max|d|={_d.max().item():.6e} "
                    f"mean|d|={_d.mean().item():.6e} over "
                    f"{int((_d > 0).sum())}/{_d.numel()} differing elements",
                    flush=True)
        else:
            combined = moe_combine_unweighted(expert_out, topk_ids,
                                              out_dtype=out_dtype)
    else:
        combined = aligned_moe_combine(
            expert_out,
            topk_ids,
            topk_weights,
            routed_scaling_factor=1.0,  # router already folded it into the weights
            out_dtype=out_dtype,
        )
    return combined


def _moe_expert_chain_reference(
    x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
    w13: torch.Tensor, w2: torch.Tensor, beta, linear_beta, layer_idx,
) -> torch.Tensor:
    """The certified pre-optimization expert chain, byte-moved out of
    ``_raon_fp32_combine_moe_impl`` (RAON_MOE_CHAIN_FUSED=0 fallback and the
    live cross-check reference). Same ops, same order, same intermediates."""
    hidden_size = x.shape[1]
    T, K = topk_ids.shape
    num_experts = w13.shape[0]

    # Expert-major dispatch (stable argsort by expert id; see the docstring for
    # why a per-token weight gather is not viable). One [T*K] stable sort, one
    # row gather, then per expert one FC1/situ/FC2 chain against the resident
    # weights.
    flat_ids = topk_ids.reshape(-1)  # [T*K]
    sort_idx = torch.argsort(flat_ids, stable=True)  # expert-major row order
    token_of_row = torch.div(sort_idx, K, rounding_mode="floor")
    xs = x.index_select(0, token_of_row)  # [T*K, H] expert-major
    # The only host sync on this path: 128 per-expert row counts. Legal here
    # because this function NEVER runs inside stream capture -- eager engines
    # by definition, graphed engines via the eager break in
    # ``_raon_fp32_combine_moe_forward``. The cu_seqlens segments are the
    # grouped GEMM's cu_seqlens_m.
    counts = torch.bincount(flat_ids, minlength=num_experts).tolist()
    # RAON_MOE_TRAINER_REALIZED: trainer-STRUCTURED chain (see moe_align.py's
    # lever block):
    # fp32-formula SiTU (not the _C kernel), router weight folded into the
    # activation output in bf16 BEFORE FC2 (TE _ScaledGLU realization), then an
    # UNWEIGHTED fp32 ascending-expert combine. Default OFF keeps the original
    # RAON_MOE_FP32_COMBINE structure (weights after FC2, moe_align.combine).
    realized = moe_trainer_realized()
    probs_rows = None
    if realized:
        if linear_beta is None or float(linear_beta) <= 0:
            raise RuntimeError(
                "RAON_MOE_TRAINER_REALIZED requires a positive "
                "activation_situ_linear_beta (megatron moe_situ_value_beta is "
                f"always > 0); got {linear_beta!r}")
        probs_rows = topk_weights.reshape(-1).float().index_select(0, sort_idx)
    out_flat = x.new_empty(T * K, hidden_size)
    start = 0
    for eid, cnt in enumerate(counts):
        if cnt == 0:
            continue
        seg = xs[start : start + cnt]  # [cnt, H]
        gate_up = seg @ w13[eid].t()  # [cnt, 2I]
        if realized:
            inter = moe_situ_reference(gate_up, beta, linear_beta)  # [cnt, I]
            inter = moe_fold_probs(inter, probs_rows[start : start + cnt])
        else:
            inter = _situ_and_mul(gate_up, beta, linear_beta)  # [cnt, I]
        # Inverse permutation back to (token, slot) rows.
        out_flat.index_copy_(0, sort_idx[start : start + cnt], inter @ w2[eid].t())
        start += cnt
    expert_out = out_flat.view(T, K, hidden_size)

    return _moe_combine_realized_tail(
        expert_out, topk_ids, topk_weights, x.dtype, num_experts, layer_idx,
        realized, moe_combine_ep_partial_loop)


# Pinned host staging for the fused chain's per-expert row counts: one buffer
# + event per (device, num_experts), allocated once. The fused chain launches
# the bincount D2H FIRST and lets the host wait on the COPY (event), not on
# the whole stream (`.tolist()` on a device tensor) -- the argsort/gather
# launches queued after it keep the GPU busy while the host reads the counts.
_MOE_COUNTS_PIN: dict = {}


_MOE_WEIGHT_VIEWS: dict = {}


def _moe_weight_views(w: torch.Tensor) -> list:
    """Per-expert ``w[e].t()`` views, cached by the weight tensor's identity.

    The views alias the SAME storage with the SAME strides ``torch.mm`` saw
    before (``w[eid].t()``), so the GEMM operands are unchanged — this only
    removes per-call Python view construction from the hot loop. Keyed by
    ``id(w)`` with the tensor kept referenced by the cache entry itself, and
    re-derived if the data_ptr moved (weight reload)."""
    key = id(w)
    ent = _MOE_WEIGHT_VIEWS.get(key)
    if ent is None or ent[0] is not w or ent[1] != w.data_ptr():
        _MOE_WEIGHT_VIEWS[key] = ent = (
            w, w.data_ptr(), [w[e].t() for e in range(w.shape[0])])
    return ent[2]


def _moe_counts_pin(num_experts: int, device: torch.device):
    key = (device.index, num_experts)
    ent = _MOE_COUNTS_PIN.get(key)
    if ent is None:
        ent = (
            torch.empty(num_experts, dtype=torch.long, pin_memory=True),
            torch.cuda.Event(),
        )
        _MOE_COUNTS_PIN[key] = ent
    return ent


# Row chunk for the batched elementwise stage of the fused chain: bounds the
# fp32 SiTU intermediates to ~64k x 2I (~0.57 GiB transient at 2I=2176)
# regardless of prefill size. Elementwise ops are per-element pure functions;
# the chunk boundary cannot move a bit (cross-checked live + in the sweep).
_MOE_FUSED_ROW_CHUNK = 65536


_MXFP8_CHAIN_ANNOUNCED = False

# RAON_MOE_MXFP8_W8A8_FINALIZE: absorb the mode-2 finalize (read once at
# import, like _GEMM_MODE; default 0 = byte-identical rows + combine tail).
_MXW8A8_FINALIZE = os.environ.get(
    "RAON_MOE_MXFP8_W8A8_FINALIZE", "0") == "1"

# RAON_MOE_COMBINE_EP_FINALIZE: fused EP2 combine (read once at import,
# like _MXW8A8_FINALIZE; default 0 = byte-identical rows + tail path).
# Like EP_COMBINE_FUSED/EP_FASTPATH it only selects between bit-identical
# implementations, so it is not listed in the parity registry. Bit
# equality rests on the per-position sequence match documented in
# moe_ep_kernel.kernel_ep2_finalize_permuted plus a fail-closed
# first-call cross-check.
_MOE_EP_FINALIZE = os.environ.get(
    "RAON_MOE_COMBINE_EP_FINALIZE", "0") == "1"


def _moe_expert_chain_mxfp8(
    x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
    w13: torch.Tensor, w2: torch.Tensor, beta, linear_beta, layer_idx,
) -> torch.Tensor:
    """RAON_MOE_EXPERT_MXFP8_REALIZED=1: the reference chain's expert-major
    dispatch and combine tail, with the per-expert bf16 GEMM loop replaced by
    production's fused CuTe MXFP8 grouped MLP (mxfp8_align.fused_expert_rows,
    fail-closed on fusion evidence and on the quantization band). Requires the
    trainer-realized contract: probs are folded INSIDE the fused epilogue, so
    the unweighted fp32 combine must follow."""
    global _MXFP8_CHAIN_ANNOUNCED
    hidden_size = x.shape[1]
    T, K = topk_ids.shape
    num_experts = w13.shape[0]

    if not moe_trainer_realized():
        raise RuntimeError(
            "RAON_MOE_EXPERT_MXFP8_REALIZED=1 requires "
            "RAON_MOE_TRAINER_REALIZED=1 (probs-fold + unweighted-combine "
            "contract around the expert GEMMs)")
    if linear_beta is None or float(linear_beta) <= 0:
        raise RuntimeError(
            "RAON_MOE_EXPERT_MXFP8_REALIZED requires a positive "
            f"activation_situ_linear_beta; got {linear_beta!r}")
    if not _MXFP8_CHAIN_ANNOUNCED:
        _MXFP8_CHAIN_ANNOUNCED = True
        print("[parity] RAON_MOE_EXPERT_MXFP8_REALIZED=1: mxfp8 expert chain "
              f"selected on layer_idx={layer_idx}; RAON_MOE_CHAIN_FUSED="
              f"{'1' if _MOE_CHAIN_FUSED else '0'} is BYPASSED for the "
              "routed-expert GEMMs (the bf16 chain it fuses does not run)",
              flush=True)

    # RAON_MOE_MXFP8_W8A8_TRITON=1: the same mxfp8 arithmetic as a native
    # Triton chain (bitwise-equal to the twin; the default config targets
    # BM128 native tcgen05 only, so run the run/dot_scaled_lowering_probe.py
    # compile check before enabling it). It skips the expert-major
    # sort/Fp8Padding/TE and produces (T,K,H) directly; the rows are
    # bit-identical, so the combine tail contract is unchanged (probs are
    # already folded in the epilogue).
    # The five tail lines duplicate the original below to avoid re-indenting
    # that block; keep both copies in sync when editing.
    #
    # RAON_PREFILL_ALIGNED: the aligned realization does not use this
    # chain, so the condition is AND NOT aligned(). During aligned prefill
    # this falls through to the alignment reference chain below
    # (byte-identical to _moe_expert_chain_reference); decode is unaffected.
    if mxw8a8_triton_enabled() and not raon_phase.aligned("moe.expert_chain"):
        # RAON_MOE_MXFP8_W8A8_FINALIZE=1 (default 0 = byte-identical rows path
        # below): absorb mode 2's (T,K,H) row materialization and combine tail
        # (unpermute/prob fold/fp32 sum) into the trtllm finalize kernel and
        # return (T, H) directly. Requires EP combine OFF (the two-stage combine
        # cannot be reproduced by finalize; fail closed). The accumulation order
        # differs, in the same ULP class as mode 2 itself.
        #
        # Same AND-NOT RAON_PREFILL_ALIGNED pattern and the SAME site key as
        # the enclosing branch: both are one phase decision (enter the trtllm
        # mxw8a8 chain or not), so the two branches never split within one
        # phase. Aligned prefill skips finalize entirely, so the EP-combine
        # mutual-exclusion raise below is never reached there.
        if _MXW8A8_FINALIZE and not raon_phase.aligned("moe.expert_chain"):
            if moe_combine_ep_size():
                raise RuntimeError(
                    "RAON_MOE_MXFP8_W8A8_FINALIZE=1 with "
                    "RAON_MOE_COMBINE_EP_REALIZED!=0: the fused finalize "
                    "cannot reproduce the two-stage EP combine; set "
                    "exactly one")
            return expert_chain_mxw8a8_finalized(
                x, topk_weights, topk_ids, w13, w2, beta, linear_beta,
                layer_idx=layer_idx)
        # RAON_MOE_COMBINE_EP_FINALIZE=1 (default 0 = byte-identical rows+tail
        # path below): with EP combine active (=2), bypass row materialization
        # and _moe_combine_realized_tail entirely and return (T, H) in one kernel
        # from the trtllm FIN0 output (permuted rows + inverse permutation map).
        # The arithmetic sequence matches the existing path position by
        # position (see the kernel comments; first-call cross-check, fail
        # closed). EP!=2 has no fused kernel and fails closed rather than
        # silently falling back.
        _ep_now = moe_combine_ep_size()
        if _MOE_EP_FINALIZE and _ep_now:
            if _ep_now != 2:
                raise RuntimeError(
                    "RAON_MOE_COMBINE_EP_FINALIZE=1 requires "
                    f"RAON_MOE_COMBINE_EP_REALIZED=2; got {_ep_now} "
                    "(the fused ep2-finalize kernel is EP=2-specialized)")
            if num_experts % _ep_now:
                raise RuntimeError(
                    f"RAON_MOE_COMBINE_EP_REALIZED={_ep_now} does not divide "
                    f"num_experts={num_experts}; the trainer's expert->rank "
                    "block mapping cannot be reproduced")
            return expert_chain_mxw8a8_ep_finalized(
                x, topk_weights, topk_ids, w13, w2, beta, linear_beta,
                layer_idx=layer_idx,
                num_local_experts=num_experts // _ep_now)
        expert_out = expert_chain_mxw8a8(
            x, topk_weights, topk_ids, w13, w2, beta, linear_beta,
            layer_idx=layer_idx).view(T, K, hidden_size)
        _vector = os.environ.get("RAON_MOE_MXFP8_COMBINE_VECTOR", "1") == "1"
        _ep_fn = (moe_combine_ep_partial
                  if _vector
                  else moe_combine_ep_partial_loop)
        _calls_before = _EP_PARTIAL_CALLS["n"]
        _tail = _moe_combine_realized_tail(
            expert_out, topk_ids, topk_weights, x.dtype, num_experts,
            layer_idx, True, _ep_fn)
        _announce_mxfp8_combine_vector(_vector, _calls_before, layer_idx)
        return _tail

    # RAON_PREFILL_ALIGNED_W8A8_SWAB=1 (default 0 = byte-identical path
    # below): realize aligned prefill's routed experts with swab (mode 4,
    # swap-AB CUTLASS, bitwise-equal to the CuTe path) instead of falling
    # back to CuTe. GEMM_MODE env is unchanged (decode stays on trtllm); the
    # mode-4 function is called directly. The first call is cross-checked
    # against the CuTe path with torch.equal, fail closed. The combine tail
    # reuses the certified _moe_combine_realized_tail; swab rows are in
    # token-slot order, so no index_copy inverse permutation is needed.
    if (_ALIGNED_W8A8_SWAB and mxw8a8_triton_enabled()
            and raon_phase.aligned("moe.expert_chain")):
        return _moe_aligned_swab_chain(
            x, topk_weights, topk_ids, w13, w2, beta, linear_beta,
            layer_idx, num_experts, T, K, hidden_size)

    # expert-major dispatch — byte-identical to _moe_expert_chain_reference
    flat_ids = topk_ids.reshape(-1)
    sort_idx = torch.argsort(flat_ids, stable=True)
    token_of_row = torch.div(sort_idx, K, rounding_mode="floor")
    xs = x.index_select(0, token_of_row)
    probs_rows = topk_weights.reshape(-1).float().index_select(0, sort_idx)

    # RAON_MOE_MXFP8_STATIC_PAD: the capture-safe twin. The ONLY reason
    # moe_outer_graph refuses this chain is that the three host reads below
    # (bincount().tolist(), the data-dependent xs slice, Fp8Padding's python
    # m_splits) cannot live inside a CUDA graph. The static-pad path derives
    # every index on the device instead; see
    # mxfp8_align.fused_expert_rows_static for the bit-neutrality evidence
    # and for the cost, which is shape-dependent and must be measured.
    # torch.bincount is NOT graph-safe. It sizes its output from the maximum
    # element, which is a device->host read, and minlength does not remove it.
    # Inside capture that surfaces as
    #   Cannot copy between CPU and CUDA tensors during CUDA graph capture
    #   unless the CPU tensor is pinned
    #
    # scatter_add_ into a fixed-size buffer is the same tally with no host
    # interaction. Exact, not approximately: these are integer counts, so the
    # result cannot depend on accumulation order. Verified on B300 across 26
    # shapes (E 8..256, n 1..100000, plus the all-one-expert and empty-expert
    # edges): 0 mismatches, and the scatter_add_ form captures where bincount
    # raises the error above.
    counts_t = torch.zeros(num_experts, dtype=torch.long, device=flat_ids.device)
    counts_t.scatter_add_(0, flat_ids.long(),
                          torch.ones_like(flat_ids, dtype=torch.long))
    # The static path gives every expert exactly ALIGN rows and derives
    #     dest = expert_id * ALIGN + (row's position within its expert)
    # so it is only correct while no expert receives more than ALIGN rows.
    # n_rows <= ALIGN guarantees that for ANY routing, however skewed, because
    # one expert cannot receive more rows than exist. Without this guard the
    # env var alone selected the static path for every shape, including vLLM's
    # memory-profiling forward (max_num_batched_tokens x top_k rows, far above
    # ALIGN): `within` then exceeds ALIGN, `dest` runs past the padded buffer,
    # and the out-of-bounds index_copy_ surfaces as
    #   CUDA error: device-side assert triggered
    # at whatever device->host sync happens next, often several modules away.
    #
    # Shapes above the bound take the dynamic path and are simply not captured,
    # which is correct: moe_outer_graph only captures x.shape[0] <= 64 anyway.
    _align = 256
    _static_ok = (os.environ.get("RAON_MOE_MXFP8_STATIC_PAD", "0") == "1"
                  and xs.shape[0] <= _align)
    if _static_ok:
        rows = mxfp8_fused_expert_rows_static(
            None, xs, counts_t, flat_ids.index_select(0, sort_idx), probs_rows,
            w13, w2, beta, linear_beta, cache_on=w13, layer_idx=layer_idx)
    else:
        rows = mxfp8_fused_expert_rows(
            None, xs, counts_t.tolist(), probs_rows, w13, w2, beta, linear_beta,
            cache_on=w13, layer_idx=layer_idx)
    # RAON_MOE_COMBINE_EP_FINALIZE=1 (default 0 = byte-identical
    # index_copy+tail path below): the EP=2 tail of the aligned/mxfp8 chain
    # (out_flat.index_copy_ inverse permutation + (T,K,H) materialization +
    # combine re-gather) is absorbed into a no-fold kernel variant; aligned
    # prefill reaches this chain, not the trtllm branch. The rows already
    # carry the prob fold, so the only arithmetic is the combine; the rest is
    # pure data movement. The static-pad path has a different row layout and
    # is excluded (prefill always takes the dynamic path, xs.shape[0] > 256).
    # EP!=2 fails closed.
    _ep_now = moe_combine_ep_size()
    if _MOE_EP_FINALIZE and _ep_now and not _static_ok:
        if _ep_now != 2:
            raise RuntimeError(
                "RAON_MOE_COMBINE_EP_FINALIZE=1 requires "
                f"RAON_MOE_COMBINE_EP_REALIZED=2; got {_ep_now} "
                "(the fused ep2-finalize kernel is EP=2-specialized)")
        if num_experts % _ep_now:
            raise RuntimeError(
                f"RAON_MOE_COMBINE_EP_REALIZED={_ep_now} does not divide "
                f"num_experts={num_experts}; the trainer's expert->rank "
                "block mapping cannot be reproduced")
        return _moe_aligned_ep2_finalized(
            x, rows, sort_idx, topk_ids, num_experts // _ep_now,
            layer_idx, T, K, hidden_size)
    out_flat = x.new_empty(T * K, hidden_size)
    out_flat.index_copy_(0, sort_idx, rows)
    expert_out = out_flat.view(T, K, hidden_size)

    # The MXFP8 chain was the only realized chain still handing the tail
    # the ORIGINAL loop. Its sibling _moe_expert_chain_fused passes the
    # vectorized twin, and the two are bit-checked equal at import
    # (moe_align._self_check_combine_ep check 4) plus in the live cross-check
    # (see _moe_combine_realized_tail). The loop costs, per MoE layer per
    # step: an UNCONDITIONAL int(rank_ordered.max()) in
    # moe_align._combine_ep_partial_fp32_loop (a _local_scalar_dense that
    # drains the stream, which combine_ep_partial_fp32 hoists behind a
    # (device, ep, local) key), plus ep_size x K = 2 x 6 iterations of
    # roughly five kernels each, which the twin does as one broadcast
    # torch.where over an [ep, T, H] stack.
    # Reaching the twin also makes the fused EP-combine kernel reachable
    # from this chain: it gates on expert_out.dtype == bfloat16, and
    # out_flat here is x.new_empty, i.e. bf16 -- so the tail must be
    # re-verified, which is why this is an env lever.
    _vector = os.environ.get("RAON_MOE_MXFP8_COMBINE_VECTOR", "1") == "1"
    _ep_fn = (moe_combine_ep_partial
              if _vector
              else moe_combine_ep_partial_loop)
    # The selection above is an assignment, not an execution: the tail only
    # calls _ep_fn when RAON_MOE_COMBINE_EP_REALIZED names an EP size, and
    # otherwise reduces with moe_combine_unweighted and never looks at it.
    # Read the witness across the call so the announce reports what RAN.
    _calls_before = _EP_PARTIAL_CALLS["n"]
    _tail = _moe_combine_realized_tail(
        expert_out, topk_ids, topk_weights, x.dtype, num_experts, layer_idx,
        True, _ep_fn)
    _announce_mxfp8_combine_vector(_vector, _calls_before, layer_idx)
    return _tail


# RAON_PREFILL_ALIGNED_W8A8_SWAB: swap aligned prefill's routed expert
# realization (read once at import, like _MOE_EP_FINALIZE; default 0 =
# byte-identical CuTe fallback). Guarded by the first-call cross-check
# below.
_ALIGNED_W8A8_SWAB = os.environ.get(
    "RAON_PREFILL_ALIGNED_W8A8_SWAB", "0") == "1"

_ALIGNED_SWAB_XCHECKED = False


def _moe_aligned_swab_chain(x, topk_weights, topk_ids, w13, w2,
                            beta, linear_beta, layer_idx, num_experts,
                            T, K, hidden_size):
    """Aligned-prefill swab branch: swab rows + certified combine tail.

    On the first call, recompute expert_out through the CuTe fallback path
    (a verbatim copy of the dispatch block in the main chain) on the same
    input and compare with torch.equal, fail closed.
    """
    global _ALIGNED_SWAB_XCHECKED
    from .mxfp8_w8a8_cutlass import expert_rows_swab
    if (float(beta), float(linear_beta or -1.0)) != (4.0, 25.0):
        raise RuntimeError(
            "RAON_PREFILL_ALIGNED_W8A8_SWAB=1: swab kernels transcribe CuTe "
            f"betas (4.0, 25.0); got ({beta}, {linear_beta})")
    # EP2 fused tail (mode-4 counterpart, RAON_MOE_COMBINE_EP_FINALIZE=1 +
    # EP=2): return (T, H) directly from the compact rows, skipping the
    # inverse-gather materialization and the model-side tail. Guarded by the
    # cutlass module's fail-closed first-call cross-check and the swab vs
    # CuTe rows check below (the fused path also runs the classic path on
    # its first call to pass that latch). OFF = byte-identical existing path.
    _ep_now = moe_combine_ep_size()
    _ep_fin = bool(_MOE_EP_FINALIZE and _ep_now)
    if _ep_fin:
        if _ep_now != 2:
            raise RuntimeError(
                "RAON_MOE_COMBINE_EP_FINALIZE=1 requires "
                f"RAON_MOE_COMBINE_EP_REALIZED=2; got {_ep_now} "
                "(the fused ep2-finalize kernel is EP=2-specialized)")
        if num_experts % _ep_now:
            raise RuntimeError(
                f"RAON_MOE_COMBINE_EP_REALIZED={_ep_now} does not divide "
                f"num_experts={num_experts}")
        if _ALIGNED_SWAB_XCHECKED:
            return expert_rows_swab(
                x, topk_weights, topk_ids, w13, w2, layer_idx=layer_idx,
                ep_finalize_local=num_experts // _ep_now)
    expert_out = expert_rows_swab(
        x, topk_weights, topk_ids, w13, w2,
        layer_idx=layer_idx).view(T, K, hidden_size)
    if not _ALIGNED_SWAB_XCHECKED:
        _ALIGNED_SWAB_XCHECKED = True
        # expert_out via the CuTe fallback: verbatim copy of the main chain's
        # dispatch block (dynamic path only; this branch is prefill-only).
        flat_ids = topk_ids.reshape(-1)
        sort_idx = torch.argsort(flat_ids, stable=True)
        token_of_row = torch.div(sort_idx, K, rounding_mode="floor")
        xs = x.index_select(0, token_of_row)
        probs_rows = topk_weights.reshape(-1).float().index_select(0, sort_idx)
        counts_t = torch.zeros(num_experts, dtype=torch.long,
                               device=flat_ids.device)
        counts_t.scatter_add_(0, flat_ids.long(),
                              torch.ones_like(flat_ids, dtype=torch.long))
        rows = mxfp8_fused_expert_rows(
            None, xs, counts_t.tolist(), probs_rows, w13, w2, beta,
            linear_beta, cache_on=w13, layer_idx=layer_idx)
        out_flat = x.new_empty(T * K, hidden_size)
        out_flat.index_copy_(0, sort_idx, rows)
        ref = out_flat.view(T, K, hidden_size)
        if not torch.equal(expert_out, ref):
            d = (expert_out.float() - ref.float()).abs()
            raise RuntimeError(
                "RAON_PREFILL_ALIGNED_W8A8_SWAB=1 live cross-check FAILED: "
                "swab expert rows != CuTe aligned rows on layer_idx="
                f"{layer_idx}, {tuple(x.shape)} input -- "
                f"{int((expert_out != ref).sum())}/{ref.numel()} elements, "
                f"max|d|={d.max().item():.6e}. Refusing to substitute the "
                "aligned realization. Set RAON_PREFILL_ALIGNED_W8A8_SWAB=0 "
                "to fall back.")
        print(
            "[parity] RAON_PREFILL_ALIGNED_W8A8_SWAB=1 LIVE: ALIGNED-W8A8-"
            f"SWAB rows on layer_idx={layer_idx}, {tuple(x.shape)} input -- "
            f"cross-check PASS (torch.equal vs CuTe aligned rows, "
            f"{ref.numel()} elements); combine tail unchanged (certified "
            "_moe_combine_realized_tail)", flush=True)
    if _ep_fin:
        # First call: return the fused result after the swab vs CuTe check
        # above ran on the classic rows (fused == classic tail is proven, fail
        # closed, by the cutlass module's own cross-check). Later calls take
        # the early return above.
        return expert_rows_swab(
            x, topk_weights, topk_ids, w13, w2, layer_idx=layer_idx,
            ep_finalize_local=num_experts // _ep_now)
    _vector = os.environ.get("RAON_MOE_MXFP8_COMBINE_VECTOR", "1") == "1"
    _ep_fn = (moe_combine_ep_partial
              if _vector
              else moe_combine_ep_partial_loop)
    _calls_before = _EP_PARTIAL_CALLS["n"]
    _tail = _moe_combine_realized_tail(
        expert_out, topk_ids, topk_weights, x.dtype, num_experts, layer_idx,
        True, _ep_fn)
    _announce_mxfp8_combine_vector(_vector, _calls_before, layer_idx)
    return _tail


_ALIGNED_EP_FINALIZE_XCHECKED = False


def _moe_aligned_ep2_finalized(x, rows, sort_idx, topk_ids, num_local,
                               layer_idx, T, K, hidden_size):
    """EP2 fused tail for the aligned/mxfp8 chain.

    Feeds the expert-major rows and the sort_idx inverse-permutation map
    straight into the no-fold kernel and returns (T, H) bf16. The first call
    is cross-checked, fail closed: the existing path (index_copy +
    combine_ep_partial_fp32) consumes the same rows/sort_idx and the results
    must match with torch.equal."""
    global _ALIGNED_EP_FINALIZE_XCHECKED
    from .moe_ep_kernel import finalize_ep2_expertmajor
    if x.dtype != torch.bfloat16:
        raise RuntimeError(
            "RAON_MOE_COMBINE_EP_FINALIZE=1: the fused ep2-finalize kernel "
            f"emits bf16; chain dtype is {x.dtype} -- refusing a silent "
            "dtype change")
    inv = torch.empty_like(sort_idx)
    inv.index_copy_(0, sort_idx,
                    torch.arange(T * K, device=sort_idx.device))
    ids_c = topk_ids.contiguous()
    order = torch.argsort(ids_c, dim=-1, stable=True)
    fused = finalize_ep2_expertmajor(rows.contiguous(), inv, ids_c,
                                     num_local, order=order)
    if not _ALIGNED_EP_FINALIZE_XCHECKED:
        _ALIGNED_EP_FINALIZE_XCHECKED = True
        out_flat = x.new_empty(T * K, hidden_size)
        out_flat.index_copy_(0, sort_idx, rows)
        expert_out = out_flat.view(T, K, hidden_size)
        ref = moe_combine_ep_partial(
            expert_out, ids_c, ep_size=2, num_local_experts=num_local,
            out_dtype=x.dtype)
        if not torch.equal(fused, ref):
            d = (fused.float() - ref.float()).abs()
            raise RuntimeError(
                "RAON_MOE_COMBINE_EP_FINALIZE=1 (aligned chain) live "
                "cross-check FAILED: fused ep2-finalize kernel != reference "
                f"tail on a {tuple(x.shape)} input -- "
                f"{int((fused != ref).sum())}/{ref.numel()} elements, "
                f"max|d|={d.max().item():.6e}. Refusing to run the fused "
                "plumbing. Set RAON_MOE_COMBINE_EP_FINALIZE=0 to fall back.")
        _single = moe_combine_unweighted(expert_out, ids_c,
                                         out_dtype=x.dtype)
        _d = (fused.float() - _single.float()).abs()
        print(
            "[parity] RAON_MOE_COMBINE_EP_FINALIZE=1 LIVE: ALIGNED-chain "
            "tail (index_copy unpermute + EP2 combine) absorbed into ONE "
            f"kernel on layer_idx={layer_idx}, {tuple(x.shape)} input -- "
            f"cross-check PASS (torch.equal vs reference tail, "
            f"{fused.numel()} elements); EP boundary live: vs single-stage "
            f"contract max|d|={_d.max().item():.6e} over "
            f"{int((_d > 0).sum())}/{_d.numel()} differing elements",
            flush=True)
    return fused


def _announce_mxfp8_combine_vector(vector, calls_before, layer_idx) -> None:
    """Say ONCE whether the MXFP8 chain's chosen EP-combine actually ran.

    ``_EP_PARTIAL_CALLS`` is bumped inside ``_moe_combine_realized_tail`` at
    the point the function object is invoked, so a bump across the tail call is
    proof of execution on this layer's real expert rows. No bump means the tail
    was reached and took ``moe_combine_unweighted`` instead -- the lever then
    picks between two functions that nobody calls.
    """
    if not vector:
        return                  # the loop twin is the documented =0 fallback
    ran = _EP_PARTIAL_CALLS["n"] > calls_before
    key = ("mxfp8_combine_vector", bool(ran))
    if key in _LEVER_ANNOUNCED:
        return
    _LEVER_ANNOUNCED.add(key)
    if ran:
        print(f"[parity] RAON_MOE_MXFP8_COMBINE_VECTOR LIVE: the vectorized "
              f"EP-combine twin ({_EP_PARTIAL_CALLS['fn']}) executed on the "
              f"mxfp8 chain's tail on layer_idx={layer_idx} -- the original "
              f"loop's unconditional int(rank_ordered.max()) host sync and its "
              f"ep_size x K iterations are gone from this chain", flush=True)
    else:
        print(f"[parity] RAON_MOE_MXFP8_COMBINE_VECTOR INERT: the lever is on "
              f"and selected the vectorized EP-combine twin, but the mxfp8 "
              f"chain's tail on layer_idx={layer_idx} never called it -- "
              f"RAON_MOE_COMBINE_EP_REALIZED is unset/0, so the tail reduces "
              f"with the single-stage moe_combine_unweighted and neither the "
              f"twin nor the loop it replaces runs. This arm measures nothing "
              f"about the lever", flush=True)


def _moe_expert_chain_fused(x, topk_weights, topk_ids, w13, w2,
                            beta, linear_beta, layer_idx):
    if moe_trainer_realized() and os.environ.get("RAON_MOE_GROUPED_GRAPH", "1") == "1":
        from .moe_grouped import try_forward
        result = try_forward(x, topk_weights, topk_ids, w13, w2,
                             beta, linear_beta, layer_idx)
        if result is not None:
            return result
    return _moe_expert_chain_fused_host(x, topk_weights, topk_ids, w13, w2,
                                      beta, linear_beta, layer_idx)


def _moe_expert_chain_fused_host(
    x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
    w13: torch.Tensor, w2: torch.Tensor, beta, linear_beta, layer_idx,
) -> torch.Tensor:
    """Launch-fused realization of ``_moe_expert_chain_reference`` — the SAME
    arithmetic per element, restructured for the GPU (RAON_MOE_CHAIN_FUSED=1):

      * the per-expert FC1/FC2 GEMMs stay per-expert ``torch.mm`` calls with
        the IDENTICAL [cnt,H]x[H,2I] shapes and transposed-view operands (the
        GEMM kernel choice is part of the realized bit contract; grouped/bmm
        batching was rejected for exactly that reason) — but they write into
        row slices of two whole-layer buffers (same row stride, ldc unchanged)
        so everything BETWEEN them batches;
      * the SiTU + probs-fold elementwise stage runs ONCE over the whole
        expert-major row buffer (row-chunked), not once per expert:
        elementwise ops are per-element pure functions, so batching changes
        no bit — this deletes ~10 kernel launches x num_experts per layer;
      * the inverse permutation back to (token, slot) rows is ONE
        ``index_copy_`` on the full buffer (a permutation is a permutation,
        chunked or not) instead of one per expert;
      * the per-expert row counts still cross to the host (the mm loop needs
        python ints — that IS the sync floor of the per-expert-GEMM
        contract), but as an early-launched async copy into pinned memory
        waited on by EVENT, so the host wait no longer drains the whole
        stream and the queued argsort/gather kernels keep the GPU busy;
      * the EP combine is the vectorized twin (moe_align, bit-checked at
        import against the loop).

    Bit-safety checks: the one-time live cross-check in
    ``_raon_fp32_combine_moe_impl`` (fail closed), moe_align's import-time
    self-checks, and run/moe_fuse_bitcheck.py's GPU shape sweep.
    """
    hidden_size = x.shape[1]
    T, K = topk_ids.shape
    num_experts = w13.shape[0]
    n_rows = T * K

    flat_ids = topk_ids.reshape(-1)  # [T*K]
    # Counts first: same bincount (integer atomics — exact, order-free), host
    # copy overlapped with the launches below.
    counts_gpu = torch.bincount(flat_ids, minlength=num_experts)
    on_gpu = flat_ids.is_cuda
    if on_gpu:
        pin, ev = _moe_counts_pin(num_experts, flat_ids.device)
        pin.copy_(counts_gpu, non_blocking=True)
        ev.record()
    sort_idx = torch.argsort(flat_ids, stable=True)  # expert-major row order
    token_of_row = torch.div(sort_idx, K, rounding_mode="floor")
    xs = x.index_select(0, token_of_row)  # [T*K, H] expert-major
    realized = moe_trainer_realized()
    probs_rows = None
    if realized:
        if linear_beta is None or float(linear_beta) <= 0:
            raise RuntimeError(
                "RAON_MOE_TRAINER_REALIZED requires a positive "
                "activation_situ_linear_beta (megatron moe_situ_value_beta is "
                f"always > 0); got {linear_beta!r}")
        probs_rows = topk_weights.reshape(-1).float().index_select(0, sort_idx)
    gate_up_all = x.new_empty(n_rows, w13.shape[1])   # [T*K, 2I] expert-major
    inter_all = x.new_empty(n_rows, w2.shape[2])      # [T*K, I]  expert-major
    fc2_all = x.new_empty(n_rows, hidden_size)        # [T*K, H]  expert-major

    if on_gpu:
        ev.synchronize()      # waits for the COPY, not the stream
        counts = pin.tolist()
    else:
        counts = counts_gpu.tolist()

    # FC1: the bit contract — per-expert torch.mm, same shapes/operands. The
    # transposed per-expert views are built once per weight tensor and cached
    # (same storage, same strides — a view cache cannot move a bit; it only
    # deletes 2 view constructions x experts x layers x steps of host time).
    w13_t = _moe_weight_views(w13)
    w2_t = _moe_weight_views(w2)
    segs = []
    start = 0
    for eid, cnt in enumerate(counts):
        if cnt == 0:
            continue
        torch.mm(xs[start : start + cnt], w13_t[eid],
                 out=gate_up_all[start : start + cnt])
        segs.append((eid, start, cnt))
        start += cnt

    # SiTU (+ probs fold), batched over ALL rows (chunked): per-element pure.
    for c0 in range(0, n_rows, _MOE_FUSED_ROW_CHUNK):
        c1 = min(c0 + _MOE_FUSED_ROW_CHUNK, n_rows)
        gu = gate_up_all[c0:c1]
        if realized:
            s = moe_situ_reference(gu, beta, linear_beta)
            s = moe_fold_probs(s, probs_rows[c0:c1])
        else:
            s = _situ_and_mul(gu, beta, linear_beta)
        inter_all[c0:c1].copy_(s)

    # FC2: per-expert torch.mm again (the contract), into the staging buffer.
    for eid, start, cnt in segs:
        torch.mm(inter_all[start : start + cnt], w2_t[eid],
                 out=fc2_all[start : start + cnt])

    # ONE inverse permutation back to (token, slot) rows.
    out_flat = x.new_empty(n_rows, hidden_size)
    out_flat.index_copy_(0, sort_idx, fc2_all)
    expert_out = out_flat.view(T, K, hidden_size)

    return _moe_combine_realized_tail(
        expert_out, topk_ids, topk_weights, x.dtype, num_experts, layer_idx,
        realized, moe_combine_ep_partial)


# --- CUDA-graph capture safety for the eager MoE chain ---------------------
#
# The reference-reduction chain above is host-driven: one ``bincount().tolist()``
# D2H sync per MoE layer feeds the per-expert segment loop (plus the pad8
# self-check and the EP-combine announce ``.item()`` reads). Inside CUDA-graph
# stream capture a host copy is illegal ("Cannot copy between CPU and CUDA
# tensors during CUDA graph capture unless the CPU tensor is pinned") -- and
# merely PINNING the
# copy would be worse than the crash: the capture batch's routing counts and
# the per-expert loop structure would be baked into the graph and silently
# replayed against every later batch's different routing.
#
# The fix is the seam this vLLM fork provides for exactly this (and that
# kda.py's ``_forward`` already uses): ``eager_break_during_capture``. During
# capture the current graph segment ends, the chain runs eagerly on the
# capture stream, and the recorded callable re-runs the WHOLE chain per replay
# with the live routing -- bit-for-bit the enforce_eager realization. The
# eager-break contract requires a stable-address output buffer, so the graphs
# path writes through an ``empty_like`` allocated inside the captured region
# (graph-pool memory; address baked at capture, reused by every replay).
#
# When breakable graphs are OFF (every enforce_eager parity arm; the decorator
# is an identity at import time), ``_raon_fp32_combine_moe_forward`` takes the
# plain branch below -- byte-identical to the unwrapped binding, no
# extra buffer, no copy.

_MOE_EAGER_BREAK = False
try:
    from vllm.compilation.breakable_cudagraph import (
        eager_break_during_capture as _eager_break_during_capture,
    )
    from vllm.compilation.breakable_cudagraph import (
        is_breakable_cudagraph_enabled as _is_breakable_enabled,
    )

    _MOE_EAGER_BREAK = bool(_is_breakable_enabled())
except ImportError:  # eager-only vLLM build: nothing to break out of

    def _eager_break_during_capture(fn):
        return fn


@_eager_break_during_capture
def _raon_fp32_combine_moe_eager_break(
    self: RaonMoE, hidden_states: torch.Tensor, out: torch.Tensor
) -> None:
    """In-place eager-break wrapper: recomputes the routed chain per replay."""
    out.copy_(_raon_fp32_combine_moe_impl(self, hidden_states))


def _raon_fp32_combine_moe_forward(
    self: RaonMoE, hidden_states: torch.Tensor
) -> torch.Tensor:
    # RAON_MXFP8_MIN_M phase policy (default 0 = this branch is never
    # taken and the function is unchanged): when the routed rows
    # (tokens * top_k) are below the threshold, run the STOCK fused MoE
    # forward that was rebound away at install time -- the stock bf16 decode
    # realization, captured
    # in the same piecewise graph. Above the threshold (prefill), the eager
    # MXFP8 chain below runs as before. The branch is on a SHAPE, so under
    # breakable capture each captured size records one consistent path.
    if _MXFP8_MIN_M > 0:
        stock = getattr(self, "_raon_stock_forward", None)
        if stock is not None:
            router = getattr(self.experts, "router", None)
            top_k = getattr(router, "top_k", None)
            rows = hidden_states.shape[0] * (
                top_k if isinstance(top_k, int) and top_k > 0 else 1)
            if not mxfp8_rows_reach_min_m(rows, "routed_experts"):
                return stock(hidden_states)
    if _MOE_EAGER_BREAK:
        out = torch.empty_like(hidden_states)
        _raon_fp32_combine_moe_eager_break(self, hidden_states, out)
        return out
    return _raon_fp32_combine_moe_impl(self, hidden_states)


# Checkpoint shard name -> (packed param, shard id). Matched as a SUBSTRING of
# the checkpoint name, which is load-bearing in two places:
#
#  * ``.g_proj`` names both KDA's gate (shard 3 of the packed in_proj) and
#    MLA's standalone output gate. The loader resolves it by rewriting the name
#    and skipping the mapping when the rewritten name is absent from the
#    model's params — the KDA layer has ``in_proj_qkvgfab``, the MLA layer does
#    not, so each gets the right home. Drop that skip and one of them loads as
#    garbage or not at all.
#  * ``.b_proj`` does NOT match ``.kv_b_proj`` / ``.q_b_proj`` only because the
#    preceding character there is ``_``, not ``.``. The __main__ check below
#    pins that.
#
# The in_proj shard ids come from RaonRDABase's
# ``in_proj_output_sizes = [P, P, P, P, head_dim, num_heads]`` — q, k, v, gate,
# f_a (the replicated shard), beta. RAON always has ``use_full_rank_gate`` and
# always has ``q_lora_rank``, so unlike K3 there is nothing to switch on here.
_STACKED_PARAMS_MAPPING = [
    # (param_name, shard_name, shard_id)
    (".in_proj_qkvgfab", ".q_proj", 0),
    (".in_proj_qkvgfab", ".k_proj", 1),
    (".in_proj_qkvgfab", ".v_proj", 2),
    (".in_proj_qkvgfab", ".g_proj", 3),
    (".in_proj_qkvgfab", ".f_a_proj", 4),
    (".in_proj_qkvgfab", ".b_proj", 5),
    (".conv1d", ".q_conv1d", 0),
    (".conv1d", ".k_conv1d", 1),
    (".conv1d", ".v_conv1d", 2),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
    (".fused_qkv_a_proj", ".q_a_proj", 0),
    (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
]

# RAON_MLA_OUTPUT_GATE_TWO_ROUND (default OFF, no-op).  Installed here rather
# than only from plugin.register because it rebinds a module global of
# `vllm.models.raon1_mini.mla_attention`, and the import above is the point at which
# that module is provably loaded -- a lever that depends on plugin ordering is
# a lever that can be silently absent.  Re-entrant: the installer no-ops once
# it has run.
install_mla_output_gate_two_round()


def get_spec_layer_idx_from_weight_name(config, weight_name: str) -> int | None:
    """Layer index of an MTP weight (``layers.{num_hidden_layers + i}.``), else None."""
    num_nextn = getattr(config, "num_nextn_predict_layers", 0) or 0
    for i in range(num_nextn):
        if f"layers.{config.num_hidden_layers + i}." in weight_name:
            return config.num_hidden_layers + i
    return None


# The three checkpoint tensors that make up one fused `phi` in mhc.py.
_PHI_BLOCKS = frozenset({"phi_pre", "phi_post", "phi_z"})


# Megatron's _build_mla_qk_norm uses 1e-6 for the q/kv latent norms whenever
# rope_type == "none", which RAON is. Not layernorm_epsilon.
MLA_QK_LAYERNORM_EPS = 1e-6


# RMSNorm unification with the trainer.
#
# The trainer (megatron KimiRMSNorm, --kimi-rmsnorm-backend quack) normalizes
# every standalone RMSNorm through stock QuACK: reduction, normalization and
# affine multiply all happen in FP32 and only the final result is rounded to
# BF16 (torch_norm.py: KimiRMSNorm.forward, backend=='quack'). vLLM's own
# RMSNorm rounds the affine multiply to the input dtype, so its output diverges
# from the trainer by ~1 ulp per element even given the same input hidden — a
# nonzero norm_f site in the parity dump. Calling the exact same QuACK entry
# point here makes norm_f bitwise-identical whenever the input is identical.
#
# QuACK's eligibility contract (bf16 CUDA input+weight, last-dim stride 1) is
# the same one KimiRMSNorm asserts. When it is not met (CPU init probes, an
# fp32 path, a non-contiguous view) we fall back to vLLM's native RMSNorm
# rather than crash — the parity forward always hits the eligible path.
_QUACK_RMSNORM_ENV = "RAON_VLLM_QUACK_RMSNORM"  # default "1" (parity)


class QuackRMSNorm(RMSNorm):
    """vLLM RMSNorm whose eligible forward routes through the trainer's QuACK
    kernel so norm_f matches megatron KimiRMSNorm(backend='quack') bitwise."""

    def _quack_eligible(self, x: torch.Tensor) -> bool:
        # RAON_PREFILL_ALIGNED: this call already reads the env per call
        # (unlike the boolean levers, which latch at import), so one term
        # covers all 3 consumption sites --
        # input_layernorm, post_attention_layernorm and norm_f all route
        # through this same forward().
        # Tensor eligibility FIRST, phase predicate LAST: aligned() emits the
        # once-per-(site,verdict) LIVE line, and announcing before the
        # trailing conditions are known would stamp "takes the zero(prod)
        # realization" on a site that then falls through to the fastest path
        # Pure predicates -- the reorder cannot change the boolean result.
        return (
            x.is_cuda
            and x.dtype == torch.bfloat16
            and self.weight.dtype == torch.bfloat16
            and x.stride(-1) == 1
            and (os.environ.get(_QUACK_RMSNORM_ENV, "1") == "1"
                 or raon_phase.aligned("norm.quack"))
        )

    def forward(self, x, residual=None):  # type: ignore[override]
        # Fused add-then-norm: RAON's mHC owns all residual, so the final norm
        # and the branch pre-norms are always called standalone (residual=None)
        # in this model. Keep the residual arm faithful anyway.
        h = x if residual is None else x + residual
        if self._quack_eligible(h):
            try:
                from quack import rmsnorm

                out = rmsnorm(h, self.weight, eps=self.variance_epsilon)
                return out if residual is None else (out, h)
            except Exception:
                pass
        return super().forward(x, residual)


class RaonDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Raon1MiniConfig,
        vllm_config: VllmConfig,
        prefix: str,
        aux_stream: torch.cuda.Stream,
    ) -> None:
        super().__init__()
        layer_idx = int(prefix.rsplit(".", 1)[1])
        quant_config = vllm_config.quant_config

        # Both attention classes self-register into
        # ``compilation_config.static_forward_context[prefix]`` from their own
        # __init__, which is how the KV-cache / hybrid-state machinery finds
        # them. It keys off exactly this prefix, so it must stay
        # ``<layer prefix>.self_attn``.
        if config.is_kda_layer(layer_idx):
            self.self_attn = RaonKDAAttention(
                config, vllm_config, prefix=f"{prefix}.self_attn"
            )
        else:
            assert config.mla_use_nope, "RAON MLA is NoPE-only"
            self.self_attn = RaonRefitMLA(
                config=config,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                use_output_gate=bool(config.mla_use_output_gate),
                cache_config=vllm_config.cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                aux_stream=aux_stream,
            )
            # Megatron trains the MLA latent norms at 1e-6, not at
            # layernorm_epsilon: _build_mla_qk_norm() hardcodes
            # `1e-6 if rope_type == "none"` (multi_latent_attention.py:363-368)
            # and the recipe sets no mla_qk_layernorm_epsilon override. vLLM's
            # MultiHeadLatentAttention builds both from config.rms_norm_eps
            # (mla.py:195,220) and its fused path reads self.rms_norm_eps
            # (mla.py:444), so all three have to be corrected. Every other norm
            # in this model stays at rms_norm_eps=1e-5.
            self.self_attn.rms_norm_eps = MLA_QK_LAYERNORM_EPS
            self.self_attn.q_a_layernorm.variance_epsilon = MLA_QK_LAYERNORM_EPS
            self.self_attn.kv_a_layernorm.variance_epsilon = MLA_QK_LAYERNORM_EPS

            # RAON_MLA_PROJ_PERSISTENT (default OFF): run MLA's
            # fused_qkv_a/q_b/g/o projections through the shared Triton
            # matmul_persistent, the same fixed-schedule GEMM
            # RAON_KDA_PROJ_PERSISTENT already gives the KDA projections. These
            # were the last GEMM family in the model still on cuBLAS, whose
            # split-K is chosen from M -- and M in a decode forward is the
            # number of live sequences. See mla_proj_align.py.
            self._mla_proj_parity = apply_mla_proj_persistent(self.self_attn)

            # RAON_MLA_MXFP8_REALIZED (default OFF): under the UNMODIFIED
            # production map (configs/te-precision.yaml) the trainer's MLA
            # projections are MXFP8, not bf16 -- only the KDA projections and
            # layer 0's dense FFN stay bf16 there. So the bf16 persistent-GEMM
            # pin above is the right realization for the te-precision-PARITY
            # map and the WRONG one for production. This lever runs the six
            # projections as production's MXFP8 TE Linears instead, splitting
            # the engine's merged qkv_a into the trainer's two GEMMs. It
            # therefore SUPERSEDES RAON_MLA_PROJ_PERSISTENT on the projections
            # it wraps; arming both is refused in mxfp8_align.
            self._mla_mxfp8_parity = mxfp8_apply_mla(
                self.self_attn,
                layer_idx=layer_idx,
                q_lora_rank=config.q_lora_rank,
            )

        if (
            config.is_moe
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            from .mxfp8_loader import compact_loading_enabled, make_quant_config

            moe_quant_config = quant_config
            if compact_loading_enabled():
                if quant_config is not None:
                    raise ValueError("Compact routed MXFP8 cannot replace another quantization configuration")
                moe_quant_config = make_quant_config(
                    layer_idx, getattr(config, "raon_expert_quantization", None))
            # Registered under ``block_sparse_moe`` first, so that is the name
            # named_parameters() keeps and the name the checkpoint uses; the
            # ``mlp`` alias is only for the forward below.
            self.block_sparse_moe = RaonMoE(
                config=config,
                vllm_config=vllm_config,
                quant_config=moe_quant_config,
                prefix=f"{prefix}.block_sparse_moe",
                layer_idx=layer_idx,
            )
            # When RAON_MOE_FP32_COMBINE=1, swap the MoE
            # module's forward for the eager reference-reduction path
            # (``_raon_fp32_combine_moe_forward``) so the trainer-aligned
            # slot-order FP32 combine (moe_align.combine, bit-checked at import)
            # is ACTUALLY on the live forward — not just published.
            # vLLM's fused RaonMoE never exposes pre-combine FC2 outputs, so the
            # combine cannot be spliced into the fused kernel; the eager path
            # reruns the experts (same weights + situ op) and reduces in FP32,
            # replacing the bf16 fused-epilogue reduction that misses the
            # Megatron trainer by ~5e-2 max on L*.moe (route-pinned). The router
            # already scores in FP32 (RaonMoE.gate out_dtype=torch.float32); only
            # the combine precision/order differs. The layer_probe hook on
            # ``layer.mlp`` captures this forward's output as the L*.moe site.
            # RAON_MOE_TRAINER_REALIZED also rides this eager forward: it
            # reuses the same expert-major dispatch but restructures the
            # SiTU/probs/combine to the trainer's measured bf16 realization
            # (moe_align.py lever block).
            parallel = vllm_config.parallel_config
            distributed_moe = parallel.tensor_parallel_size > 1 or parallel.data_parallel_size > 1
            if distributed_moe and compact_loading_enabled():
                # Native MoERunner owns TP reductions and DP/EP dispatch/combine.
                # Its quant method runs the same preconverted MXFP8 expert chain.
                print(f"[raon-parallel] layer={layer_idx} native MoERunner communication; "
                      "packed MXFP8 experts; native shared-expert scheduling", flush=True)
            elif fp32_moe_combine_enabled() or moe_trainer_realized():
                # Keep the stock fused forward reachable: the
                # RAON_MXFP8_MIN_M phase policy (default off) returns to it
                # for below-threshold (decode-sized) row counts.
                self.block_sparse_moe._raon_stock_forward = (
                    self.block_sparse_moe.forward)
                self.block_sparse_moe.raon_aligned_combine = aligned_moe_combine
                self.block_sparse_moe.forward = types.MethodType(
                    _raon_fp32_combine_moe_forward, self.block_sparse_moe
                )
            # RAON_MOE_SHARED_SITU_REALIZED (default OFF): RAON_MOE_TRAINER_
            # REALIZED restructured the ROUTED experts onto the trainer's fp32
            # SiTU but left the SHARED expert on vLLM's compiled
            # torch.ops._C.situ_and_mul -- two SiTU realizations inside one MoE
            # block, only one of them aligned. Once the router weights are pinned
            # this is the sole remaining L*.moe difference: swapping ONLY this
            # activation makes the offline chain reproduce the TRAINER dump bitwise
            # for every combination of the probs / routed-SiTU / combine / add-dtype
            # axes. See the lever block in moe_align.py.
            if moe_shared_situ_realized():
                _swap = moe_install_shared_situ(
                    self.block_sparse_moe,
                    config.activation_situ_beta,
                    config.activation_situ_linear_beta,
                )
                global _MOE_SHARED_SITU_ANNOUNCED
                if not _MOE_SHARED_SITU_ANNOUNCED:
                    _MOE_SHARED_SITU_ANNOUNCED = True
                    print(
                        "[parity] RAON_MOE_SHARED_SITU_REALIZED=1: shared "
                        f"expert act_fn on layer_idx={layer_idx}: {_swap}",
                        flush=True,
                    )
            # RAON_MOE_SHARED_MXFP8_REALIZED (default OFF): the shared expert
            # is MXFP8 under the UNMODIFIED production map (only the parity map
            # demotes mlp.shared_experts.linear_fc*). Runs fc1/fc2 as TE MXFP8
            # Linears; the SiTU between them is unchanged and still owned by
            # RAON_MOE_SHARED_SITU_REALIZED above.
            self._moe_shared_mxfp8 = mxfp8_apply_shared(
                self.block_sparse_moe, layer_idx=layer_idx
            )
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = RaonMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                activation_situ_beta=config.activation_situ_beta,
                activation_situ_linear_beta=config.activation_situ_linear_beta,
            )
            # RAON_MLP_PROJ_PERSISTENT (default OFF): run the dense FFN's
            # gate_up/down GEMMs through the shared Triton matmul_persistent,
            # the realization the trainer's TE linear_fc1/fc2 reproduce
            # bitwise. Once the mHC glue is aligned this GEMM is the only remaining
            # L00.mhc_post difference; see mlp_align.py.
            self._mlp_proj_parity = apply_dense_mlp_persistent(self.mlp)
            # RAON_MLP_SITU_NATIVE (default OFF): run the dense FFN's SiTU
            # activation as the fp32 reference formula, the realization the
            # trainer's dense fallback situ_glu_reference computes op-for-op
            # (its Triton kernel refuses hidden 12032). Under RAON_KDA_CPR the
            # CUDA-vs-reference SiTU split otherwise leaves 1-2 bf16-ulp elements at
            # L00.mhc_post. See mlp_align.apply_situ_native.
            self._mlp_situ_parity = apply_dense_mlp_situ_native(self.mlp)

        self.input_layernorm = QuackRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = QuackRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # GLOBAL site index: 2*layer for the attention site, 2*layer+1 for the
        # MLP site. It sets b_pre's soft one-hot rotation; a local index would
        # give trained weights the wrong rotation on a partial load.
        n = config.raon_mhc_streams
        self.self_attn_raon_mhc = RaonMHC(n, config.hidden_size, 2 * layer_idx)
        self.mlp_raon_mhc = RaonMHC(n, config.hidden_size, 2 * layer_idx + 1)

    def forward(self, positions: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """s: [T, streams, hidden] in, same out."""
        # Neither branch adds a residual — H_res inside the site is the
        # residual. See the module docstring.
        #
        # The wiring decision uses the RAW env (_MHC_AGGNORM_FUSED_ENV), not
        # mhc.py's OR'd _AGGNORM_FUSED module constant (that OR only exists
        # to import-gate mhc.py's fused kernel definitions), ORed with the
        # cached per-forward RAON_PREFILL_ALIGNED verdict from the SAME
        # predicate RaonMHC.forward reads. Both must agree inside one forward
        # (a load-bearing invariant), which holds because both funnel through
        # raon_phase.prefill_now()'s per-ForwardContext cache.
        if _MHC_AGGNORM_FUSED_ENV or raon_phase.aligned("mhc.aggnorm"):
            # RAON_MHC_AGGNORM_FUSED: the site computes the branch pre-norm
            # inside the trainer's fused aggregate->norm kernel (from the fp32
            # aggregate accumulator), exactly as Megatron's transformer_layer
            # passes branch_norm_weight into raon_mhc_aggregate_norm — so the
            # branch consumes `a` directly and the separate input_layernorm /
            # post_attention_layernorm calls are SKIPPED (their weights and eps
            # are consumed by the kernel instead). This aggregate->norm boundary
            # determines L00.kda parity.
            s = self.self_attn_raon_mhc(
                s,
                lambda a: self.self_attn(hidden_states=a, positions=positions),
                branch_norm_weight=self.input_layernorm.weight,
                branch_norm_eps=self.input_layernorm.variance_epsilon,
            )
            return self.mlp_raon_mhc(
                s,
                lambda a: self.mlp(a),
                branch_norm_weight=self.post_attention_layernorm.weight,
                branch_norm_eps=self.post_attention_layernorm.variance_epsilon,
            )
        s = self.self_attn_raon_mhc(
            s,
            lambda a: self.self_attn(
                hidden_states=self.input_layernorm(a), positions=positions
            ),
        )
        return self.mlp_raon_mhc(s, lambda a: self.mlp(self.post_attention_layernorm(a)))


class RaonModel(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Raon1MiniConfig = vllm_config.model_config.hf_text_config
        self.config = config
        self.streams = config.raon_mhc_streams
        self.vocab_size = config.vocab_size
        # Accumulates each mHC site's seen phi blocks across load_weights calls;
        # see the comment at its use site.
        self._phi_seen: dict[str, set[str]] = {}

        parallel_config = vllm_config.parallel_config
        assert parallel_config.pipeline_parallel_size == 1, (
            "RAON's inter-layer carrier is a [T, streams, hidden] activation; "
            "PP would need it in IntermediateTensors. The model fits on one "
            "B200 at TP1."
        )
        # We never shard the carrier across sequence-parallel ranks, so we
        # refuse the launch that would turn sequence parallelism on rather than
        # run half of it.
        assert not (
            parallel_config.enable_expert_parallel
            and parallel_config.tensor_parallel_size > 1
            and parallel_config.data_parallel_size > 1
        ), "RAON has no sequence-parallel path for the mHC stream carrier"
        assert config.num_attention_heads % get_tensor_model_parallel_world_size() == 0

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )

        # Aux stream for overlapping MLA's g_proj GEMM with the attention
        # front-end (DeepseekV4 convention: one stream, made here, threaded
        # into every MLA layer).
        aux_stream = torch.cuda.Stream()

        def get_layer(prefix: str):
            return RaonDecoderLayer(config, vllm_config, prefix, aux_stream)

        _, _, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        # norm_f parity site — QuACK-backed to match the trainer's final
        # decoder.final_layernorm (megatron KimiRMSNorm backend='quack').
        self.norm = QuackRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        # expand -> layer loop -> mean. Megatron creates the streams from the
        # embedding on every forward and destroys them at the block output;
        # there is nothing to cache.
        s = inputs_embeds.unsqueeze(1).expand(-1, self.streams, -1).contiguous()
        aux_hidden_states: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            s = layer(positions, s)
            # vLLM's Eagle3/DFlash convention numbers layer *boundaries* from
            # one, whereas the training config stores zero-based target layer
            # ids.  eagle3_utils converts [1, 6, ...] to [2, 7, ...], so the
            # comparison here captures the same post-MLP carrier as the native
            # Megatron training hook.  Preserve tap-major, stream-major order:
            # each [T, 4, H] carrier becomes one [T, 4*H] feature, and the
            # runner concatenates the five list entries in requested order.
            if (layer_idx + 1) in self.aux_hidden_state_layers:
                aux_hidden_states.append(s.flatten(1))
        # K3 defers the final norm to compute_logits so its MTP draft sees
        # pre-norm hidden states. v1 has no MTP, so the norm lives here where
        # the reduce is; move it back if MTP is ever built.
        hidden_states = self.norm(s.mean(1))
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(
        self,
        weights: Iterable[
            tuple[str, torch.Tensor] | tuple[str, torch.Tensor, dict[str, Any]]
        ],
    ) -> set[str]:
        """Names arrive with AutoWeightsLoader's ``model.`` prefix stripped.

        Returns MODEL PARAMETER names, not checkpoint names — the default
        loader does ``named_parameters() - returned`` and raises on the
        remainder, so a stacked shard must report the packed param it landed
        in. That also makes the returned set a produced-vs-consumed check.
        """
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_experts,
        )
        params_dict = dict(self.named_parameters())
        mhc_sites = {n: m for n, m in self.named_modules() if isinstance(m, RaonMHC)}
        # Instance state, not a local: AutoWeightsLoader calls this once per
        # contiguous `model.*` run in the weight stream (models/utils.py
        # _groupby_prefix), and a site's three phi blocks can straddle two
        # runs. A local accumulator would forget the earlier blocks and
        # silently skip assembling that site's phi.
        phi_seen = self._phi_seen
        loaded_params: set[str] = set()

        for args in weights:
            name, loaded_weight = args[0], args[1]
            # Accept previous export names while exposing canonical module paths.
            name = name.replace(".self_attn_mhc.", ".self_attn_raon_mhc.")
            name = name.replace(".mlp_mhc.", ".mlp_raon_mhc.")
            kwargs: dict[str, Any] = args[2] if len(args) > 2 else {}

            # convert.py emits the MTP block as model.layers.{num_hidden_layers}.*
            # so the checkpoint stays complete; v1 does not build the layer.
            if get_spec_layer_idx_from_weight_name(self.config, name) is not None:
                continue
            # TransformerEngine's opaque per-module blob. set_extra_state raises
            # on any mismatch; never round-trip it.
            if name.endswith("_extra_state"):
                continue

            if "_mhc." in name:
                # phi_pre/phi_post/res.phi_z are row blocks of one fused `phi`
                # and the row order is owned by mhc.py, so the site loads
                # itself rather than going through params_dict.
                site_name, suffix = name.split("_mhc.", 1)
                site_name += "_mhc"
                mhc_sites[site_name].load_weight(suffix, loaded_weight)
                attr = suffix.removeprefix("res.")
                if attr in _PHI_BLOCKS:
                    seen = phi_seen.setdefault(site_name, set())
                    seen.add(attr)
                    # `phi` counts as loaded only once all three blocks are in;
                    # reporting it early would hide a dropped block.
                    if seen != _PHI_BLOCKS:
                        continue
                    attr = "phi"
                loaded_params.add(f"{site_name}.{attr}")
                continue

            for param_name, weight_name, shard_id in _STACKED_PARAMS_MAPPING:
                if weight_name not in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                # The packed projection exists only on the layers that have it
                # (see _STACKED_PARAMS_MAPPING's note on .g_proj).
                if name_mapped not in params_dict:
                    continue
                param = params_dict[name_mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name_mapped)
                break
            else:
                for (
                    expert_param_name,
                    expert_weight_name,
                    expert_id,
                    expert_shard_id,
                ) in expert_params_mapping:
                    if expert_weight_name not in name:
                        continue
                    name = name.replace(expert_weight_name, expert_param_name)
                    param = params_dict[name]
                    if param.is_meta:
                        # Immutable sidecar weights are installed by the MoE
                        # post-load hook. Never copy original BF16 experts to GPU.
                        from .mxfp8_loader import CHECKPOINT_PARAMS, compact_loading_enabled

                        if not compact_loading_enabled():
                            raise RuntimeError(f"Unexpected meta expert parameter: {name}")
                        loaded_params.add(name)
                        # An MXFP8 checkpoint's FP8 experts go to the staging
                        # stack that the post-load hook packs.
                        prefix, _, anchor = name.rpartition(".")
                        staging = f"{prefix}.{CHECKPOINT_PARAMS[anchor]}"
                        if staging not in params_dict:
                            break
                        name, param = staging, params_dict[staging]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=expert_shard_id,
                    )
                    loaded_params.add(name)
                    break
                else:
                    # No silent skip: everything convert.py emits has a home, and
                    # a KeyError names any key that does not. BF16-only, so there
                    # are no quant scales to ignore.
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, **kwargs)
                    loaded_params.add(name)
        return loaded_params


# ---------------------------------------------------------------------------
# Fail-closed cudagraph readback + replay counter
# ---------------------------------------------------------------------------
# vllm/config/vllm.py silently downgrades cudagraph_mode to NONE when
# piecewise compilation is not available, so a run can report graphs on
# while executing eagerly; the opposite misconfiguration fails with
# cudaErrorStreamCaptureInvalidated. Throughput numbers are only
# attributable if a run can prove which mode it ran in.
#
# So: read the RESOLVED mode back, and refuse to start if it is not the mode
# the caller pinned. The readback runs in the model constructor because that
# is downstream of config resolution and runs once per worker process.
_CUDAGRAPH_READBACK_DONE = False
_GRAPH_REPLAYS = {"n": 0}


def _cudagraph_readback(vllm_config) -> None:
    """Refuse to run if the resolved cudagraph mode is not the pinned one.

    RAON_EXPECT_CUDAGRAPH_MODE pins it by name (NONE / PIECEWISE /
    FULL_DECODE_ONLY / FULL / FULL_AND_PIECEWISE). Unset means "do not judge,
    just report" -- the honest default, because most runs of this engine are
    deliberately eager and must not start failing.

    mode: vllm_compile is refused unconditionally and separately: inductor
    fuses scored ops, which destroys the reason capture under mode: none is
    bit-transparent (capture records the eager two-kernel sequence, not
    inductor's fusion): any scored op under torch.compile breaks parity.
    """
    global _CUDAGRAPH_READBACK_DONE
    if _CUDAGRAPH_READBACK_DONE:
        return
    _CUDAGRAPH_READBACK_DONE = True

    cc = getattr(vllm_config, "compilation_config", None)
    if cc is None:
        if raon_phase.PREFILL_ALIGNED:
            # The phase gate's whole safety argument
            # rests on knowing the RESOLVED capture mode; "cannot judge" is
            # not an option while it is armed.
            raise RuntimeError(
                "RAON_PREFILL_ALIGNED=1 but vllm_config has no "
                "compilation_config to read the resolved cudagraph mode "
                "from; refusing to run the phase gate against an unknown "
                "capture composition."
            )
        print("[parity] cudagraph readback: no compilation_config on "
              "vllm_config; cannot judge", flush=True)
        return

    raw_mode = getattr(cc, "cudagraph_mode", None)
    resolved = str(raw_mode)
    comp_mode = str(getattr(cc, "mode", None))
    print(f"[parity] cudagraph readback: cudagraph_mode={resolved} "
          f"compilation mode={comp_mode}", flush=True)

    if raon_phase.PREFILL_ALIGNED:
        # Judge the AUTHORITATIVE resolved mode, not the
        # RAON_EXPECT_CUDAGRAPH_MODE env label (which is optional and can
        # silently disagree with what vLLM resolved). The gate's phase
        # branch is evaluated eagerly per forward;
        # a capture region that records a PREFILL shape (PIECEWISE / FULL /
        # FULL_AND_PIECEWISE) would freeze one verdict into the graph.
        # FULL_DECODE_ONLY captures only uniform-decode shapes (verdict False
        # by construction) and NONE captures nothing -- both safe.
        if raw_mode is None:
            raise RuntimeError(
                "RAON_PREFILL_ALIGNED=1 but compilation_config has no "
                "cudagraph_mode attribute; cannot verify the capture "
                "composition the gate requires (FULL_DECODE_ONLY or NONE). "
                "Refusing rather than risk freezing the phase verdict into "
                "a captured graph."
            )
        _r = resolved.upper()
        if "FULL_DECODE_ONLY" not in _r and not _r.endswith("NONE"):
            raise RuntimeError(
                f"RAON_PREFILL_ALIGNED=1 with resolved cudagraph_mode="
                f"{resolved}: v1 of the phase gate only has evidence for "
                "FULL_DECODE_ONLY (and trivially NONE) capture; a piecewise/"
                "full capture region can record a prefill shape and freeze "
                "the phase branch's verdict into the graph. Refusing rather "
                "than silently mismeasure."
            )
        # Spec verification uses the scheduler's true request-phase flag.
        # Mixed batches run all aligned operations eagerly; local MoE graphs
        # are bypassed for those forwards, retaining request/cache row order.

    if "vllm_compile" in comp_mode.lower() or "inductor" in comp_mode.lower():
        raise RuntimeError(
            "raon_vllm refuses compilation mode "
            f"{comp_mode!r}: inductor fuses scored ops, which breaks the "
            "bit-transparency argument that lets this engine capture at all. "
            "Use mode: none (survey M49)."
        )

    want = os.environ.get("RAON_EXPECT_CUDAGRAPH_MODE")
    if not want:
        return
    if want.upper() not in resolved.upper():
        raise RuntimeError(
            f"RAON_EXPECT_CUDAGRAPH_MODE={want} but vLLM RESOLVED "
            f"{resolved} (compilation mode={comp_mode}). vllm/config/vllm.py "
            "downgrades to NONE when "
            "piecewise compilation is unavailable, and the job then reports "
            "graphs on while running eager. Refusing to start rather than "
            "produce an unattributable throughput number."
        )
    print(f"[parity] cudagraph readback OK: pinned {want}, resolved "
          f"{resolved}", flush=True)
    _install_graph_replay_counter()


def _install_graph_replay_counter() -> None:
    """Count graph replays so "the graph was live" is a tally, not a guess.

    Wrapping torch.cuda.CUDAGraph.replay catches every replayer (vLLM's
    runner and any library that captured its own), which a vLLM-side hook
    would not.
    """
    if os.environ.get("RAON_CUDAGRAPH_REPLAY_COUNTER", "1") != "1":
        return
    g = torch.cuda.CUDAGraph
    if getattr(g.replay, "_raon_counted", False):
        return
    orig = g.replay

    def replay(self, *a, **k):
        _GRAPH_REPLAYS["n"] += 1
        return orig(self, *a, **k)

    replay._raon_counted = True
    g.replay = replay
    import atexit
    atexit.register(
        lambda: print(f"[parity] cudagraph replays this process: "
                      f"{_GRAPH_REPLAYS['n']}", flush=True))


class Raon1MiniForCausalLM(nn.Module, SupportsEagle3, HasInnerState, IsHybrid):
    # The HF release names the MoE container ``moe`` and the routed expert
    # projections ``gate_proj``/``up_proj``/``down_proj``; the modules keep the
    # bundle names ``block_sparse_moe`` and ``w1``/``w3``/``w2``. Bundle exports
    # pass through unchanged. The model loader also applies this mapper to
    # quantization layer lists (ModelOpt NVFP4 ignore/quantized_layers).
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_regex={
        re.compile(r"(\.moe\.experts\.\d+\.)gate_proj(?=\.|$)"): r"\1w1",
        re.compile(r"(\.moe\.experts\.\d+\.)up_proj(?=\.|$)"): r"\1w3",
        re.compile(r"(\.moe\.experts\.\d+\.)down_proj(?=\.|$)"): r"\1w2",
        re.compile(r"(?<=\.)moe(?=\.|$)"): "block_sparse_moe",
    })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        _cudagraph_readback(vllm_config)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.config = self.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        if self.quant_config is not None and self.quant_config.get_name().startswith("modelopt"):
            # Before any layer is built: unloaded NVFP4 per-partition scales
            # become a load error instead of uninitialized memory.
            from .nvfp4_padding import install_strict_scale_check

            install_strict_scale_check()
        # RAON_FP8_IGNORE (default unset = no change): scope lever for
        # NATIVE load-time quantization. `--quantization fp8` builds Fp8Config
        # with an EMPTY ignore list and no CLI surface to fill it, so an arm
        # that wants fp8 on a subset of the standard Linears (e.g. MoE experts
        # only, attention kept bf16) has nowhere to say so. This env is a
        # comma-separated list of SUBSTRING patterns matched against each
        # layer's prefix (is_layer_skipped match_mode="substring"); matched
        # layers get UnquantizedLinearMethod / UnquantizedFusedMoEMethod, i.e.
        # the stock bf16 realization. Mutating quant_config HERE is
        # sound because every get_quant_method call happens later, inside the
        # Linear/RoutedExperts constructors under RaonModel below. Without
        # --quantization the config is None and the lever is refused rather
        # than silently ignored.
        _fp8_ignore = os.environ.get("RAON_FP8_IGNORE", "")
        if _fp8_ignore:
            if self.quant_config is None:
                raise RuntimeError(
                    "RAON_FP8_IGNORE is set but the engine has no "
                    "quantization config (--quantization was not passed); "
                    "the lever would be a silent no-op"
                )
            _pats = [p for p in
                     (s.strip() for s in _fp8_ignore.split(",")) if p]
            self.quant_config.ignored_layers = list(
                getattr(self.quant_config, "ignored_layers", None) or []
            ) + _pats
            self.quant_config.ignored_layers_match_mode = "substring"
            print(
                f"[q4] RAON_FP8_IGNORE: quantization skips layers whose "
                f"prefix contains any of {_pats} (substring match); skipped "
                "layers run the unquantized bf16 realization",
                flush=True,
            )
        self.model = RaonModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # RAON trains with --untie-embeddings-and-output-weights, so lm_head is
        # always its own tensor.
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        from .head_cache import FP32HeadWeightCache
        self._fp32_head_weight_cache = FP32HeadWeightCache()
        self._deferred_refit_active = False
        self._deferred_refit_batches = 0
        # Run the new kernels' JIT at the very start of boot (before worker
        # threads multiply); each is a no-op when its lever is unset.
        try:
            from .mxfp8_w8a8_cutlass import prewarm_shape_map_kernels
            prewarm_shape_map_kernels(device=torch.cuda.current_device(),
                                      top_k=self.config.num_experts_per_token)
        except Exception as _e:  # noqa: BLE001
            print(f"[va] SHAPE_MAP prewarm skipped: {_e}", flush=True)
        try:
            from .kda_frontend_fused import prewarm_frontend_fused
            prewarm_frontend_fused(device=torch.cuda.current_device())
        except Exception as _e:  # noqa: BLE001
            print(f"[raon-kda] FUSED_FLOW prewarm skipped: {_e}", flush=True)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        # intermediate_tensors is always None: PP is asserted off in RaonModel.
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        # The final norm already ran in RaonModel.forward.
        #
        # LM head precision unification with the trainer. The default
        # path is vLLM's LogitsProcessor: a BF16 lm_head GEMM whose cuBLAS
        # variant is chosen by the batch's M dimension, so its rounding is
        # batch-variant and cannot be pinned to the trainer's projection.
        # VLLM_KPAMO_FP32_LM_HEAD=1 selects a single canonical FP32
        # representation instead — hidden and lm_head weight are widened to
        # FP32 and multiplied once in FP32 (no BF16 rounding, no bf16x9
        # Nemotron-legacy emulation of FP32). The trainer counterpart is
        # output_layer with fp32_output=True; the trainer must use that flag
        # for the two heads to agree.
        if os.environ.get("VLLM_KPAMO_FP32_LM_HEAD", "0") == "1":
            if not getattr(self, "_fp32_head_announced", False):
                self._fp32_head_announced = True
                print(
                    "[parity] VLLM_KPAMO_FP32_LM_HEAD LIVE: fp32 head GEMM "
                    "engaged (hidden+weight widened to fp32, single fp32 "
                    "matmul — not a post-cast)",
                    flush=True,
                )
            weight = self.lm_head.weight
            hidden_fp32 = hidden_states.to(torch.float32)
            weight_fp32 = self._fp32_head_weight_cache.get(weight)
            # RAON_HEAD_GEMM_PERSISTENT=1 pins the FP32 head GEMM to the
            # shared batch-invariant matmul_persistent (fixed tile schedule
            # and K-loop, fp32 in/out — the RAON_MOE_ROUTER_GEMM_PIN=realized
            # mechanism on the head), removing the cuBLAS schedule's M
            # dependence: the logit row a token gets is the same at any
            # logprob batch size. BIT-MOVING vs the torch.matmul product;
            # engine-side unpaired, must go through bitwise recertification
            # of the head/logprob sites before any certified arm turns it on.
            # Default OFF is the certified torch.matmul path, byte-identical.
            if os.environ.get("RAON_HEAD_GEMM_PERSISTENT", "0") == "1":
                from .batch_invariant import matmul_persistent
                logits = matmul_persistent(
                    hidden_fp32, weight_fp32.t(), out_dtype=torch.float32
                )
            else:
                logits = torch.matmul(hidden_fp32, weight_fp32.t())
            # LogitsProcessor truncates the padded vocabulary to the real size.
            return logits[..., : self.config.vocab_size]
        if os.environ.get("RAON_HEAD_GEMM_PERSISTENT", "0") == "1":
            # Fail closed: the lever realizes the FP32 head GEMM. The default
            # BF16 LogitsProcessor path has no persistent twin, so honoring
            # the env there would silently mean something else.
            raise RuntimeError(
                "RAON_HEAD_GEMM_PERSISTENT=1 requires VLLM_KPAMO_FP32_LM_HEAD=1"
            )
        return self.logits_processor(self.lm_head, hidden_states)

    def begin_deferred_refit(self) -> None:
        """Defer derived-weight refresh until a complete streamed transaction."""
        if self._deferred_refit_active:
            raise RuntimeError("a deferred Raon refit is already active")
        self._deferred_refit_active = True
        self._deferred_refit_batches = 0

    def finish_deferred_refit(self) -> int:
        """Refresh graph-stable derived buffers after the last source copy."""
        if not self._deferred_refit_active or not self._deferred_refit_batches:
            raise RuntimeError("no streamed Raon weights were loaded")
        self._invalidate_refit_derived()
        self._finalize_mhc_refit()
        batches = self._deferred_refit_batches
        self._deferred_refit_batches = 0
        self._deferred_refit_active = False
        return batches

    def _invalidate_refit_derived(self) -> None:
        self._fp32_head_weight_cache.clear()
        from .moe_grouped import clear
        clear()
        # Transport-independent derived-cache invalidation. The refit paths
        # OUTSIDE the collective transport (ipc_zmq /
        # striped / vllm_zmq_sparse -- the transport production.yaml declares)
        # never reach vllm_backend's NRL_REFIT_REPROCESS_ATTENTION hook
        # (collective-only), but EVERY refit path passes through this
        # function (the sparse applier calls model.load_weights too,
        # vllm_sparse_delta.py). Invalidation here removes the
        # transport/env condition; the owner invalidates its own caches --
        # the same convention as the clears above. This also refreshes
        # MXFP8 inference buffers, so streamed refits call it only after all
        # source tensors have arrived. MUST stay a module-attribute
        # reference: mxfp8_w8a8_triton._hook_refit_once replaces
        # mxfp8_align.refit_invalidate with a wrapper that also drops the
        # W8A8 PREQUANT cache (_WQ_CACHE), so an early-bound
        # `from .mxfp8_align import refit_invalidate` would silently skip it.
        # Safe under the sparse-delta TorchDispatchMode: delattr/dict.clear
        # issue no aten ops.
        from . import mxfp8_align as _mxfp8_align_mod
        _mxfp8_align_mod.refit_invalidate(self)
        for module in self.modules():
            for name in ("_raon_kda_decode_graph", "_raon_kda_decode_graph_stamp",
                         "_raon_moe_outer_cache", "_raon_moe_outer_stamp"):
                if hasattr(module, name):
                    delattr(module, name)

    def _finalize_mhc_refit(self) -> None:
        for module in self.modules():
            if isinstance(module, RaonMHC):
                module.finalize()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if os.environ.get("RAON_MXFP8_WEIGHTS") and getattr(self, "_offline_weights_loaded", False):
            raise RuntimeError("Preconverted MXFP8 serving does not support weight refit; reconvert and restart")
        if not self._deferred_refit_active:
            self._invalidate_refit_derived()
        # AutoWeightsLoader routes ``model.*`` into RaonModel.load_weights and
        # loads ``lm_head.weight`` itself. MultiHeadLatentAttention's
        # kv_b_proj absorption runs from the loader's generic post-load hook.
        loaded = AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self._offline_weights_loaded = True
        if self._deferred_refit_active:
            self._deferred_refit_batches += 1
        else:
            self._finalize_mhc_refit()
        return loaded

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        hf_config = vllm_config.model_config.hf_config
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            hf_config.linear_attn_config["num_heads"],
            hf_config.linear_attn_config["head_dim"],
            conv_kernel_size=hf_config.linear_attn_config["short_conv_kernel_size"],
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()


if __name__ == "__main__":
    # The one piece of logic here that fails silently rather than loudly is the
    # substring routing of _STACKED_PARAMS_MAPPING, and RAON is the config that
    # exercises its two hazards: one `.g_proj` name for two different homes,
    # and `.b_proj` living inside `.kv_b_proj`/`.q_b_proj`. Pure string work —
    # no tensors, no CUDA.
    kda, mla = "layers.1.self_attn.", "layers.3.self_attn."
    param_names = {
        kda + "in_proj_qkvgfab.weight",
        kda + "conv1d.weight",
        kda + "f_b_proj.weight",
        kda + "rda_w_proj.weight",
        mla + "fused_qkv_a_proj.weight",
        mla + "q_b_proj.weight",
        mla + "kv_b_proj.weight",
        mla + "g_proj.weight",
        "layers.0.mlp.gate_up_proj.weight",
        "layers.1.block_sparse_moe.shared_experts.gate_up_proj.weight",
    }

    def route(name: str) -> tuple[str, int | None]:
        for param_name, weight_name, shard_id in _STACKED_PARAMS_MAPPING:
            if weight_name in name:
                mapped = name.replace(weight_name, param_name)
                if mapped in param_names:
                    return mapped, shard_id
        return name, None

    packed = kda + "in_proj_qkvgfab.weight"
    assert route(kda + "q_proj.weight") == (packed, 0)
    assert route(kda + "k_proj.weight") == (packed, 1)
    assert route(kda + "v_proj.weight") == (packed, 2)
    assert route(kda + "g_proj.weight") == (packed, 3)
    assert route(kda + "f_a_proj.weight") == (packed, 4)
    assert route(kda + "b_proj.weight") == (packed, 5)
    assert route(kda + "v_conv1d.weight") == (kda + "conv1d.weight", 2)
    # Fall-throughs: no packed home, so the default path loads them as-is.
    for standalone in (
        mla + "g_proj.weight",
        mla + "kv_b_proj.weight",
        mla + "q_b_proj.weight",
        kda + "f_b_proj.weight",
        kda + "rda_w_proj.weight",
    ):
        assert route(standalone) == (standalone, None), standalone
    assert route(mla + "q_a_proj.weight") == (mla + "fused_qkv_a_proj.weight", 0)
    assert route(mla + "kv_a_proj_with_mqa.weight") == (
        mla + "fused_qkv_a_proj.weight",
        1,
    )
    assert route("layers.0.mlp.up_proj.weight") == (
        "layers.0.mlp.gate_up_proj.weight",
        1,
    )
    assert route("layers.1.block_sparse_moe.shared_experts.gate_proj.weight") == (
        "layers.1.block_sparse_moe.shared_experts.gate_up_proj.weight",
        0,
    )
    # Release names map back to the bundle names before any routing.
    release = Raon1MiniForCausalLM.hf_to_vllm_mapper._map_name
    for name, bundle in (
        ("model.layers.1.moe.experts.7.gate_proj.weight", "model.layers.1.block_sparse_moe.experts.7.w1.weight"),
        ("model.layers.1.moe.experts.7.up_proj.weight_scale_inv",
         "model.layers.1.block_sparse_moe.experts.7.w3.weight_scale_inv"),
        ("model.layers.1.moe.experts.7.down_proj.weight", "model.layers.1.block_sparse_moe.experts.7.w2.weight"),
        ("model.layers.1.moe.shared_experts.gate_proj.weight",
         "model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight"),
        ("model.layers.1.moe.gate.e_score_correction_bias",
         "model.layers.1.block_sparse_moe.gate.e_score_correction_bias"),
        ("model.layers.1.block_sparse_moe.experts.7.w1.weight", "model.layers.1.block_sparse_moe.experts.7.w1.weight"),
    ):
        assert release(name) == bundle, (name, release(name))
    # Routed experts must not be touched by the stacked mapping at all.
    for expert in ("layers.1.block_sparse_moe.experts.7.w1.weight",
                   "layers.1.block_sparse_moe.experts.7.w3.weight"):
        assert route(expert) == (expert, None), expert
    print("ok")

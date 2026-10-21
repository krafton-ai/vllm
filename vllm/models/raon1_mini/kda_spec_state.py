# SPDX-License-Identifier: Apache-2.0
"""Keep KDA decode rows on the accepted-offset kernels under speculation.

A speculative verification of ``1 + d`` tokens stores the recurrent state
after each verified position in ``block_table[:, t]`` and leaves the conv
history window at offset ``t`` of the widened conv cache
(``W - 1 + num_spec`` columns).  The next step must continue from position
``num_accepted - 1``: the spec kernels read exactly that
(``fused_recurrent.py`` ``initial_token = num_accepted_tokens - 1``;
``causal_conv1d.py`` ``conv_state_token_offset = num_accepted_tokens - 1``).

Both runners label a decode row that has no scheduled draft ``-1`` in
``num_decode_draft_tokens_cpu`` (V2 ``MambaHybridModelState.prepare_attn``
requires ``num_draft_tokens_per_req > 0``; V1 only fills rows present in
``scheduled_spec_decode_tokens``), and when no row has drafts at all the Kimi
builder collapses the whole batch to the non-spec path.  Either way the row
runs the packed decode, which reads ``block_table[:, 0]`` and conv offset 0
and drops ``num_accepted_tokens``.  After a verification that accepted
``a > 1`` tokens that is the state after only the FIRST verified token, so
the request silently loses ``a - 1`` tokens of KDA conv/recurrent history,
and the error is written back and carried by every later token.  The
scheduler produces such rows whenever the token budget or ``max_model_len``
trims a decode request to its single sampled token (``scheduler.py``
``num_scheduled_spec_tokens``), and the mamba "none" cache mode performs no
compensating state copy (``mamba_hybrid.py`` pre/postprocess are align-only).

Relabel true decode rows (not prefilling, one query token, label -1) as spec
rows with zero drafts.  The builder reads the draft label only for the spec
row mask and its all-zero fallback; every kernel input is derived from
``query_start_loc``, the block table and ``num_accepted_tokens``.  Those rows
therefore run the spec conv/recurrent kernels with query length 1: they read
slot/offset ``num_accepted - 1`` and write the new state to slot 0 and the
conv window to offset 0, which is what the following step (acceptance reset
to 1) reads.  With ``num_accepted == 1`` they read and write the same state
slot and conv window as the packed decode and apply the same recurrence
formula (the launch tiling differs, BV/warps, so bitwise equality of the
two kernels is an empirical GPU check, not a property of this relabel).

CUDA graphs: in the supported composition (fixed ``num_speculative_tokens``,
no adaptive verification) FULL decode graphs exist only for uniform
``1 + num_spec`` query lengths, whose rows all carry drafts, and graph capture
labels every row nonnegative, so neither is changed here.  Compositions that
capture shorter decode graphs are refused rather than risk a replay whose
staged buffers disagree with this relabel.
"""

import torch

_ANNOUNCED: set = set()


def relabel_zero_draft_decodes(
    num_decode_draft_tokens_cpu: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    is_prefilling_cpu: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Return (draft labels, relabelled row count); inputs are not mutated."""
    labels = num_decode_draft_tokens_cpu
    n = labels.numel()
    zero_draft = (
        (labels < 0)
        & (query_lens_cpu[:n] == 1)
        & ~is_prefilling_cpu[:n].to(torch.bool)
    )
    count = int(zero_draft.sum().item())
    if count == 0:
        return labels, 0
    labels = labels.clone()
    labels[zero_draft] = 0
    if int(labels[labels >= 0].sum().item()) == 0:
        # The builder would treat an all-zero batch as non-spec and take the
        # packed decode.  The label value is not a kernel input (the rows
        # still have query length 1), so a positive label only keeps them
        # on the spec branch.
        labels[zero_draft] = 1
    return labels, count


def relabel_spec_decode_state(builder, common_attn_metadata, num_accepted_tokens,
                              num_decode_draft_tokens_cpu):
    """Draft labels for the builder, with zero-draft decode rows relabelled."""
    if (not builder.use_spec_decode or num_accepted_tokens is None
            or num_decode_draft_tokens_cpu is None):
        return num_decode_draft_tokens_cpu
    _check_graph_shapes(builder)
    m = common_attn_metadata
    flags = m.is_prefilling
    if flags is None or flags.device.type != "cpu":
        raise RuntimeError(
            "Raon KDA speculative decode needs CPU per-request "
            "is_prefilling metadata to keep zero-draft decode rows "
            "on the accepted-offset kernels")
    num_decode_draft_tokens_cpu, count = relabel_zero_draft_decodes(
        num_decode_draft_tokens_cpu,
        m.query_start_loc_cpu.diff(),
        flags,
    )
    if count and "live" not in _ANNOUNCED:
        _ANNOUNCED.add("live")
        print("[parity] Raon KDA spec state: zero-draft decode rows "
              "run the accepted-offset spec kernels (query length 1) "
              f"instead of the slot-0 packed decode; first batch "
              f"relabelled {count} row(s)", flush=True)
    return num_decode_draft_tokens_cpu


def _check_graph_shapes(builder) -> None:
    if not builder.use_full_cuda_graph or "graph-ok" in _ANNOUNCED:
        return
    spec = builder.speculative_config
    dynamic = getattr(spec, "uses_dynamic_speculative_decoding", None)
    if (callable(dynamic) and dynamic()) or getattr(
            spec, "enable_adaptive_verification", False):
        raise RuntimeError(
            "Raon KDA spec state relabel is qualified only for FULL decode "
            "graphs of uniform 1+num_speculative_tokens queries; dynamic "
            "speculation or adaptive verification captures shorter decode "
            "graphs whose staged state indices would disagree with it")
    _ANNOUNCED.add("graph-ok")

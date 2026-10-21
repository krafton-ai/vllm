"""Keep expired DFlash2 SWA pages out of draft attention's input table.

The worker can retain physical IDs for prefix pages returned by the scheduler.
FA4 may load those pages in a partially masked attention tile. Replace only
wholly expired kernel blocks with the null block before metadata is built.
Persistent tables, cache storage and slot mappings are not changed.
This fix requires context parallel size one for the selected draft SWA groups.
"""

from __future__ import annotations


_GROUPS_ATTRIBUTE = "_raon_dflash_expired_prefix_groups"
_PREFIX_KERNEL = None


def _zero_expired_prefix(
    Table,
    SeqLens,
    TableRowStride: tl.constexpr,
    TableColumnStride: tl.constexpr,
    SeqStride: tl.constexpr,
    NumColumns: tl.constexpr,
    QueryTokens: tl.constexpr,
    Window: tl.constexpr,
    KernelBlockSize: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    seq_len = tl.load(SeqLens + row * SeqStride).to(tl.int64)
    # The first query has the earliest left edge of every query in this row.
    first_key = tl.maximum(seq_len - QueryTokens - (Window - 1), 0)
    expired = tl.minimum(first_key // KernelBlockSize, NumColumns)
    for start in range(0, expired, BLOCK):
        column = start + tl.arange(0, BLOCK)
        tl.store(
            Table + row * TableRowStride + column * TableColumnStride,
            0,
            mask=column < expired,
        )


def _prefix_kernel():
    # Registration and CPU contract tests can import this module without CUDA
    # or Triton. Compile only when a DFlash2 SWA input table needs this kernel.
    global _PREFIX_KERNEL, tl
    if _PREFIX_KERNEL is None:
        import triton
        import triton.language as tl

        _PREFIX_KERNEL = triton.jit(_zero_expired_prefix)
    return _PREFIX_KERNEL


def _draft_swa_groups(speculator, sliding_window_spec_type) -> tuple:
    groups = speculator.kv_cache_config.kv_cache_groups
    draft_layers = speculator.draft_attn_layer_names
    selected = []
    for gid in speculator.draft_kv_cache_group_ids:
        group = groups[gid]
        spec = group.kv_cache_spec
        if (
            isinstance(spec, sliding_window_spec_type)
            and group.layer_names
            and all(name in draft_layers for name in group.layer_names)
        ):
            # Input tables contain kernel blocks, including when a scheduler
            # page is expanded into several smaller virtual kernel blocks.
            selected.append(
                (gid, speculator.block_tables.kernel_block_sizes[gid], spec.sliding_window)
            )
    if selected and speculator.block_tables.cp_size != 1:
        raise RuntimeError("DFlash2 expired prefix handling requires context parallel size 1")
    return tuple(selected)


def _clear_expired_prefix(speculator, num_reqs: int, groups: tuple) -> None:
    if num_reqs == 0 or not groups:
        return
    kernel = _prefix_kernel()
    seq_lens = speculator.input_buffers.seq_lens
    for gid, block_size, window in groups:
        table = speculator.block_tables.input_block_tables[gid]
        kernel[(num_reqs,)](
            table,
            seq_lens,
            table.stride(0),
            table.stride(1),
            seq_lens.stride(0),
            table.shape[1],
            speculator.num_query_per_req,
            window,
            block_size,
            BLOCK=256,
        )


def clear_expired_prefix(speculator, num_reqs: int) -> None:
    """Null wholly expired SWA kernel blocks before the draft metadata is built."""
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    if (not num_reqs or speculator.method != "dflash"
            or "DFlash2DraftModel" not in speculator.draft_model_config.hf_config.architectures):
        return
    groups = getattr(speculator, _GROUPS_ATTRIBUTE, None)
    if groups is None:
        groups = _draft_swa_groups(speculator, SlidingWindowSpec)
        setattr(speculator, _GROUPS_ATTRIBUTE, groups)
    _clear_expired_prefix(speculator, num_reqs, groups)

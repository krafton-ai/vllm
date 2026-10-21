"""Align Raon MLA/KDA pages with the DFlash draft's native KV page unit.

APC OFF (mamba cache mode "none"): the shared page is the smallest multiple of
both the 128-token MLA page and the draft's kernel-block page that holds the
KDA/RDA state, so draft rows fill whole pages.

APC ON (mamba cache mode "align"): vLLM's own hybrid alignment already sets one
grid T for MLA blocks and KDA state checkpoints (``block_size ==
mamba_block_size``) on a page of exactly T MLA rows. That grid is kept, and the
draft sliding-window block D is the largest kernel-aligned divisor of T whose
rows fit the page (the remainder is padding). D dividing T keeps the scheduler
block at lcm = T and the prefix-hash unit at gcd = D; a draft block that does
not divide T would make every hit land on lcm(T, D) tokens instead. Hits land
on the T grid (keep_hits_on_block_grid).
"""

import copy
import dataclasses
import math


def common_page_size(minimum, primary_unit, draft_unit):
    if any(type(value) is not int or value <= 0
           for value in (minimum, primary_unit, draft_unit)):
        raise ValueError("Cache page sizes must be positive integers")
    unit = math.lcm(primary_unit, draft_unit)
    return ((minimum + unit - 1) // unit) * unit


def draft_block_size(target_block, page, unit, bytes_per_token):
    """Largest multiple of ``unit`` that divides ``target_block`` and fits ``page``."""
    if any(type(value) is not int or value <= 0
           for value in (target_block, page, unit, bytes_per_token)):
        raise ValueError("Cache block and page sizes must be positive integers")
    for block in range(min(target_block, page // bytes_per_token) // unit * unit, 0, -unit):
        if target_block % block == 0:
            return block
    raise ValueError("No draft block divides the target block and fits the shared page")


def raon_dflash(config):
    """raon-1-mini served with a DFlash drafter."""
    spec = config.speculative_config
    return (spec is not None and spec.method == "dflash"
            and config.model_config.hf_config.model_type == "raon1_mini")


def align_draft_pages(platform, config, primary_backend):
    from vllm.config.cache import CacheConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
    from vllm.v1.attention.backend import MultipleOf
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MLAAttentionSpec,
        get_kv_quant_mode,
    )

    if not raon_dflash(config):
        return
    spec = config.speculative_config
    cache = config.cache_config
    prefix_caching = bool(cache.enable_prefix_caching)
    if (cache.cache_dtype != "fp8_e4m3" or cache.user_specified_block_size
            or cache.mamba_cache_mode != ("align" if prefix_caching else "none")
            or (prefix_caching and cache.user_specified_mamba_block_size)):
        raise ValueError("Raon draft-page alignment requires FP8 KV, automatic block sizing and "
                         "mamba cache mode none with APC OFF or align with APC ON")
    draft = spec.draft_model_config
    if "DFlash2DraftModel" not in draft.hf_config.architectures:
        raise ValueError("Raon draft-page alignment requires the qualified DFlash2 draft")
    dtype = STR_DTYPE_TO_TORCH_DTYPE[cache.cache_dtype]
    quant_mode = get_kv_quant_mode(cache.cache_dtype)
    with set_current_vllm_config(config):
        sizes = primary_backend.get_supported_kernel_block_sizes()
    # vLLM's MLA hybrid alignment additionally requires 128-token alignment.
    alignment = max(128, min(s.base if isinstance(s, MultipleOf) else s for s in sizes))
    primary = MLAAttentionSpec(
        block_size=alignment,
        num_kv_heads=config.model_config.get_num_kv_heads(config.parallel_config),
        head_size=config.model_config.get_head_size(), dtype=dtype,
        kv_quant_mode=quant_mode,
    )
    draft_config = copy.copy(config)
    draft_config.model_config = draft
    draft_config.speculative_config = None
    draft_config.parallel_config = spec.draft_parallel_config
    draft_config.cache_config = copy.copy(cache)
    draft_config.cache_config.block_size = CacheConfig.DEFAULT_BLOCK_SIZE
    draft_backend = platform._find_non_ssm_backend(draft_config)
    if draft_backend is None:
        raise ValueError("No attention backend found for the DFlash draft")
    with set_current_vllm_config(draft_config):
        draft_block = draft_backend.get_preferred_block_size(CacheConfig.DEFAULT_BLOCK_SIZE)
    draft_page = FullAttentionSpec(
        block_size=draft_block,
        num_kv_heads=draft.get_num_kv_heads(draft_config.parallel_config),
        head_size=draft.get_head_size(), dtype=dtype, kv_quant_mode=quant_mode,
    )
    bytes_per_token = primary.page_size_bytes // alignment
    if prefix_caching:
        # Keep vLLM's align-mode grid: one T-token page per MLA block and KDA state.
        page = cache.block_size * bytes_per_token
        if (cache.block_size % alignment or cache.mamba_block_size != cache.block_size
                or (cache.mamba_page_size_padded or page) != page):
            raise ValueError("Raon APC alignment requires one MLA/KDA block grid on a full MLA page")
        block = draft_block_size(cache.block_size, page, draft_block,
                                 draft_page.page_size_bytes // draft_block)
        cache.mamba_page_size_padded = page
        print(f"[dflash-cache] prefix_caching=align common_page_bytes={page} "
              f"target_block={cache.block_size} draft_block={block}", flush=True)
        return
    minimum = cache.mamba_page_size_padded or cache.block_size * bytes_per_token
    page = common_page_size(minimum, primary.page_size_bytes, draft_page.page_size_bytes)
    cache.block_size = page // bytes_per_token
    cache.mamba_page_size_padded = page
    print(f"[dflash-cache] common_page_bytes={page} target_block={cache.block_size} "
          f"draft_block_after_unification={draft_block * (page // draft_page.page_size_bytes)}", flush=True)


def align_draft_window(spec, config):
    """Put the draft sliding-window block on the APC grid, padded to the shared page."""
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    cache = config.cache_config
    if (not isinstance(spec, SlidingWindowSpec) or not cache.enable_prefix_caching
            or not raon_dflash(config)):
        return spec
    page = cache.mamba_page_size_padded
    if not page or cache.mamba_cache_mode != "align" or cache.mamba_block_size != cache.block_size:
        raise ValueError("Raon APC draft window requires the aligned MLA/KDA page")
    bytes_per_token = spec.real_page_size_bytes // spec.block_size
    block = draft_block_size(cache.block_size, page, spec.block_size, bytes_per_token)
    return dataclasses.replace(spec, block_size=block, page_size_padded=page)


def keep_hits_on_block_grid(scheduler):
    """Put Raon DFlash prefix hits on the T grid without the EAGLE last-unit drop.

    - EAGLE drop: vLLM treats DFlash as EAGLE-family and, with no group flagged,
      drops one unit from MLA/draft hits and backs the Mamba split off one block,
      so a repeated or extended prompt loses a whole T block of its hit. EAGLE needs
      that because its draft KV at a chunk's last position mixes in the next prompt
      token. DFlash projects draft context K/V from the target hidden state at each
      position only (precompute_and_store_context_kv) and writes query rows at
      positions >= the chunk end, outside every cached unit. Draft K/V never changes
      the target distribution, only acceptance.
    - Prompt-tail entries: with a D-token hash unit vLLM also stops every prompt's
      prefill at its last D boundary to cache a partial KDA state there. The extra
      step costs more TTFT than the shorter recomputation saves, so hits use
      only the T-aligned KDA checkpoints.
    """
    coordinator = scheduler.kv_cache_manager.coordinator
    scheduler.use_eagle = False
    scheduler.mamba_partial_cache_hit = False
    coordinator.enable_partial_hash_hits = False
    coordinator.eagle_group_ids = set()
    if hasattr(coordinator, "attention_groups"):
        coordinator.attention_groups = [group._replace(use_eagle=False)
                                        for group in coordinator.attention_groups]
    for manager in coordinator.single_type_managers:
        manager.use_eagle = False

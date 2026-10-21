"""Map EP slots without duplicating a local expert within one token.

The grouped GEMM's scale arena reserves at most T rows per expert. Mapping all
remote slots to expert zero breaks that bound even when their probabilities
are zero. Assign each remote slot a distinct unused local expert instead.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _map_routes(expert_map, ids, weights, local_ids, local_weights,
                TOPK: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    slot = tl.arange(0, BLOCK)
    mask = slot < TOPK
    global_id = tl.load(ids + token * TOPK + slot, mask, other=0)
    local = tl.load(expert_map + global_id, mask, other=-1)
    owned = mask & (local >= 0)
    # At most TOPK-1 other experts are occupied, so [0, TOPK) contains
    # enough free experts for every remote slot when local E >= TOPK.
    occupied = tl.sum(((local[:, None] == slot[None, :])
                       & owned[:, None]).to(tl.int32), axis=0) > 0
    free = mask & ~occupied
    free_rank = tl.cumsum(free.to(tl.int32), axis=0) - 1
    remote_rank = tl.cumsum((mask & ~owned).to(tl.int32), axis=0) - 1
    dummy = tl.sum(tl.where(free[None, :]
                           & (free_rank[None, :] == remote_rank[:, None]),
                           slot[None, :], 0), axis=1)
    value = tl.load(weights + token * TOPK + slot, mask, other=0)
    tl.store(local_ids + token * TOPK + slot, tl.where(owned, local, dummy), mask)
    tl.store(local_weights + token * TOPK + slot, tl.where(owned, value, 0), mask)


def local_routes(expert_map, topk_ids, topk_weights, local_experts):
    """Preserve owned IDs/probabilities and zero remote contributions."""
    if not topk_ids.is_cuda or topk_ids.ndim != 2 or topk_ids.shape != topk_weights.shape:
        raise ValueError("EP routing requires matching CUDA [token, top-k] tensors")
    tokens, topk = topk_ids.shape
    if not 1 <= topk <= local_experts:
        raise ValueError("EP fixed-slot routing requires local expert count >= top-k")
    ids = topk_ids.contiguous()
    weights = topk_weights.contiguous()
    output_ids = torch.empty_like(ids)
    output_weights = torch.empty_like(weights)
    if tokens:
        _map_routes[(tokens,)](expert_map, ids, weights, output_ids, output_weights,
                              TOPK=topk, BLOCK=triton.next_power_of_2(topk), num_warps=1)
    return output_ids, output_weights

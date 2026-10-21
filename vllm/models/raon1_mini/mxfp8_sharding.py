"""Shard preconverted MXFP8 bytes without changing their 32-channel scales."""

import json
import os
from pathlib import Path


def source_shapes(layer_idx):
    root = Path(os.environ["RAON_MXFP8_WEIGHTS"])
    entry = json.loads((root / "manifest.json").read_text())["entries"][f"layer-{layer_idx}-routed"]
    return entry["source_shapes"]


def intermediate_partition(intermediate, size, rank):
    if intermediate % 32 or not 0 <= rank < size:
        raise ValueError("MXFP8 partitions require complete 32-channel blocks")
    blocks = intermediate // 32
    return 32 * (blocks * rank // size), 32 * (blocks * (rank + 1) // size)


def unpack_scales(packed, rows, columns):
    """Invert the CUTLASS MMA layout into [expert, row, K/32]."""
    experts = packed.shape[0]
    row_tiles, group_tiles = (rows + 127) // 128, (columns // 32 + 3) // 4
    return (packed.reshape(experts, row_tiles, group_tiles, 32, 4, 4)
            .permute(0, 1, 4, 3, 2, 5).reshape(experts, row_tiles * 128, group_tiles * 4)
            [:, :rows, :columns // 32])


def pack_scales(scales, fill=127):
    import torch

    experts, rows, groups = scales.shape
    row_tiles, group_tiles = (rows + 127) // 128, (groups + 3) // 4
    padded = torch.full((experts, row_tiles * 128, group_tiles * 4), fill,
                        dtype=torch.uint8, device=scales.device)
    padded[:, :rows, :groups] = scales
    return (padded.reshape(experts, row_tiles, 4, 32, group_tiles, 4)
            .permute(0, 1, 4, 3, 2, 5).contiguous().reshape(experts, -1))


def shard(tensors, experts, tp_size, tp_rank, local_intermediate):
    """Select EP experts and TP channel blocks; padding has exactly zero weights.

    Gate/up rows and FC2 columns share the same partition. In particular, an
    intermediate size of 1088 cannot be divided into four 272-channel shards:
    that would split MXFP8 scale blocks. Use whole blocks and pad local tails.
    """
    import torch

    q13, q2 = tensors["q13"], tensors["q2"]
    _, twice_intermediate, hidden = q13.shape
    intermediate = twice_intermediate // 2
    start, stop = intermediate_partition(intermediate, tp_size, tp_rank)
    width = stop - start
    if local_intermediate < width or local_intermediate % 32:
        raise ValueError("Local MXFP8 intermediate shape cannot hold this partition")
    indices = torch.tensor(experts, dtype=torch.long)
    # CPU index_select does not implement all float8 dtypes; select raw bytes.
    gate_up = q13.view(torch.uint8).index_select(0, indices)
    down = q2.view(torch.uint8).index_select(0, indices)
    out13 = torch.zeros((len(experts), 2 * local_intermediate, hidden), dtype=torch.uint8)
    out2 = torch.zeros((len(experts), hidden, local_intermediate), dtype=torch.uint8)
    out13[:, :width] = gate_up[:, start:stop]
    out13[:, local_intermediate:local_intermediate + width] = gate_up[:, intermediate + start:intermediate + stop]
    out2[:, :, :width] = down[:, :, start:stop]
    scales13 = unpack_scales(tensors["s13_mma"], twice_intermediate, hidden).index_select(0, indices)
    scales2 = unpack_scales(tensors["s2_mma"], hidden, intermediate).index_select(0, indices)
    out_s13 = torch.full((len(experts), 2 * local_intermediate, hidden // 32), 127, dtype=torch.uint8)
    out_s2 = torch.full((len(experts), hidden, local_intermediate // 32), 127, dtype=torch.uint8)
    out_s13[:, :width] = scales13[:, start:stop]
    out_s13[:, local_intermediate:local_intermediate + width] = scales13[:, intermediate + start:intermediate + stop]
    out_s2[:, :, :width // 32] = scales2[:, :, start // 32:stop // 32]
    return {"q13": out13.view(q13.dtype), "q2": out2.view(q2.dtype),
            "s13_mma": pack_scales(out_s13), "s2_mma": pack_scales(out_s2)}

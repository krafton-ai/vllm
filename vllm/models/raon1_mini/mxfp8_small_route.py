"""Small-batch stable expert routing fused with activation preparation."""
import triton
import triton.language as tl
from .mxfp8_w8a8_triton import (
    _float_to_e8m0, _exp2f_rcp, _MAX_NORM_RCP_C, _sf_mma_addr,
)


@triton.jit
def _prepare(x, weights, ids, dim1, dim2, pabc1, pabc2,
             b1, c1, b2, c2, scales, probs, positions, src_tok, src_slot,
             sf_rows, starts_out, source_p1, source_p2, source_s1, source_s2, target_s1, target_s2,
             K: tl.constexpr, N: tl.constexpr, FFN: tl.constexpr,
             E: tl.constexpr, R: tl.constexpr, TOPK: tl.constexpr,
             RM: tl.constexpr, RK: tl.constexpr, SLAB: tl.constexpr,
             SX0: tl.constexpr, SX1: tl.constexpr,
             BR: tl.constexpr, BK: tl.constexpr, COMPACT: tl.constexpr):
    program = tl.program_id(0)
    rr = tl.arange(0, BR)
    experts = tl.load(ids + rr, rr < R, E).to(tl.int32)
    if program == 0:
        if COMPACT:
            # C1 top-k experts are distinct; one active row per grouped problem.
            rows = tl.sum(((experts[:, None] > experts[None, :]) & (rr[None, :] < R)).to(tl.int32), 1).to(tl.int64)
            valid = rr < R
            ee = experts.to(tl.int64)
            tl.store(dim1 + rows * 4 + 1, 1, valid)
            tl.store(dim2 + rows * 4 + 1, 1, valid)
            tl.store(pabc1 + rows * 3, tl.load(source_p1 + ee * 3, valid, 0), valid)
            tl.store(pabc2 + rows * 3, tl.load(source_p2 + ee * 3, valid, 0), valid)
            tl.store(pabc1 + rows * 3 + 1, b1.to(tl.int64) + rows * K, valid)
            tl.store(pabc1 + rows * 3 + 2, c1.to(tl.int64) + rows * N * 4, valid)
            tl.store(pabc2 + rows * 3 + 1, b2.to(tl.int64) + rows * FFN, valid)
            tl.store(pabc2 + rows * 3 + 2, c2.to(tl.int64) + rows * K * 2, valid)
            for col in tl.static_range(2):
                tl.store(target_s1 + rows * 2 + col, tl.load(source_s1 + ee * 2 + col, valid, 0), valid)
                tl.store(target_s2 + rows * 2 + col, tl.load(source_s2 + ee * 2 + col, valid, 0), valid)
        else:
            ee = tl.arange(0, E)
            counts = tl.sum((experts[None, :] == ee[:, None]).to(tl.int32), 1)
            starts = tl.cumsum(counts, 0) - counts
            tl.store(dim1 + ee * 4 + 1, counts)
            tl.store(dim2 + ee * 4 + 1, counts)
            offset = starts.to(tl.int64)
            tl.store(pabc1 + ee * 3 + 1, b1.to(tl.int64) + offset * K)
            tl.store(pabc1 + ee * 3 + 2, c1.to(tl.int64) + offset * N * 4)
            tl.store(pabc2 + ee * 3 + 1, b2.to(tl.int64) + offset * FFN)
            tl.store(pabc2 + ee * 3 + 2, c2.to(tl.int64) + offset * K * 2)
            tl.store(starts_out + ee, starts)
    else:
        slot = (program - 1) // tl.cdiv(K, BK)
        tile = (program - 1) % tl.cdiv(K, BK)
        expert = tl.load(ids + slot).to(tl.int32)
        rank = tl.sum(((experts == expert) & (rr < slot)).to(tl.int32), 0)
        start = tl.sum((experts < expert).to(tl.int32), 0)
        row = (start + rank).to(tl.int64)
        sfr = expert.to(tl.int64) * RM + rank
        token = slot // TOPK
        if tile == 0:
            tl.store(positions + slot, row)
            tl.store(src_tok + row, token)
            tl.store(src_slot + row, slot)
            tl.store(sf_rows + row, sfr)
            tl.store(probs + row, tl.load(weights + slot))
        kk = tile * BK + tl.arange(0, BK)
        values = tl.load(x + token * SX0 + kk * SX1, kk < K, 0).to(tl.float32)
        blocks = tl.reshape(values, (BK // 32, 32))
        exponent = _float_to_e8m0(tl.max(tl.abs(blocks), 1) * _MAX_NORM_RCP_C)
        quantized = (blocks * _exp2f_rcp(exponent)[:, None]).to(tl.float8e4nv)
        tl.store(b1 + row * K + kk, tl.reshape(quantized, (BK,)), kk < K)
        sk = tile * (BK // 32) + tl.arange(0, BK // 32)
        tl.store(scales + _sf_mma_addr(sfr, sk, RM, RK, SLAB), exponent.to(tl.uint8), sk < K // 32)


def prepare(x, weights, ids, ar, arrays, source_arrays=None):
    T, K = x.shape
    topk = ids.shape[1]
    R = T * topk
    N = ar['c1'].shape[1]
    source = arrays if source_arrays is None else source_arrays
    if source_arrays is not None and T != 1:
        raise ValueError('Compact expert preparation requires T=1')
    E = source['fc1']['dim_torch'].shape[0]
    _prepare[(1 + R * triton.cdiv(K, 512),)](
        x, weights, ids, arrays['fc1']['dim_torch'], arrays['fc2']['dim_torch'],
        arrays['fc1']['pabc_torch'], arrays['fc2']['pabc_torch'],
        ar['b1'], ar['c1'], ar['b2'], ar['c2'], ar['sfb1_mma'], ar['probs'],
        ar['pos_for_slot'], ar['src_tok'], ar['src_slot'], ar['sf_row'], ar['route_starts'],
        source['fc1']['pabc_torch'], source['fc2']['pabc_torch'],
        source['fc1']['_keep'][1], source['fc2']['_keep'][1],
        arrays['fc1']['_keep'][1], arrays['fc2']['_keep'][1],
        K, N, N // 2, E, R, topk, ar['rmx128'], ar['rk1'], ar['slab1'],
        x.stride(0), x.stride(1), triton.next_power_of_2(R), 512, source_arrays is not None, num_warps=4)

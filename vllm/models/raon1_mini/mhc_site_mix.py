"""Experimental four-stream inference mix with multiple tokens per program.

The chart normalization and sequential FP32 FMA chain preserve the consumer's
SGL-style site arithmetic.
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as tld


@triton.jit
def _site_mix(s_ptr, y_ptr, head_ptr, chart_ptr, h0_ptr, neg_h0_ptr,
              g_eye_ptr, g_gate_ptr, out_ptr, TOKENS,
              HIDDEN: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
              MIX_ONE: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    streams = tl.arange(0, 4)
    matrix = streams[:, None] * 4 + streams[None, :]
    chart = tl.load(chart_ptr + rows[:, None, None] * 16 + matrix[None, :, :],
                    mask=rows[:, None, None] < TOKENS, other=0.0)
    neg_h0 = tl.load(neg_h0_ptr + matrix)
    rho = tl.maximum(tl.max(tl.max(tld.div_rn(chart, neg_h0[None, :, :]), 2), 1), 0.0)
    h0 = tl.load(h0_ptr + matrix)
    phi = h0[None, :, :] + tld.div_rn(chart, 1.0 + rho[:, None, None])
    g_eye = tl.load(g_eye_ptr + matrix)
    gate = tl.load(g_gate_ptr)
    hres = tld.add_rn(g_eye[None, :, :], tld.mul_rn(gate, phi))
    hpost = tl.load(head_ptr + rows[:, None] * 8 + 4 + streams[None, :],
                   mask=rows[:, None] < TOKENS, other=0.0)
    c0 = tl.sum(tl.where(streams[None, None, :] == 0, hres, 0.0), 2)
    c1 = tl.sum(tl.where(streams[None, None, :] == 1, hres, 0.0), 2)
    c2 = tl.sum(tl.where(streams[None, None, :] == 2, hres, 0.0), 2)
    c3 = tl.sum(tl.where(streams[None, None, :] == 3, hres, 0.0), 2)
    mask = (rows[:, None] < TOKENS) & (cols[None, :] < HIDDEN)
    base = rows[:, None] * 4 * HIDDEN + cols[None, :]
    s0 = tl.load(s_ptr + base, mask, other=0.0).to(tl.float32)
    s1 = tl.load(s_ptr + base + HIDDEN, mask, other=0.0).to(tl.float32)
    s2 = tl.load(s_ptr + base + 2 * HIDDEN, mask, other=0.0).to(tl.float32)
    s3 = tl.load(s_ptr + base + 3 * HIDDEN, mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + rows[:, None] * HIDDEN + cols[None, :], mask, other=0.0).to(tl.float32)
    # Store one stream at a time so four output tiles do not stay live together.
    for out_stream in tl.static_range(4):
        a0 = tl.sum(tl.where(streams[None, :] == out_stream, c0, 0.0), 1)
        a1 = tl.sum(tl.where(streams[None, :] == out_stream, c1, 0.0), 1)
        a2 = tl.sum(tl.where(streams[None, :] == out_stream, c2, 0.0), 1)
        a3 = tl.sum(tl.where(streams[None, :] == out_stream, c3, 0.0), 1)
        post = tl.sum(tl.where(streams[None, :] == out_stream, hpost, 0.0), 1)
        if MIX_ONE:
            acc = tld.add_rn(tld.add_rn(tld.add_rn(
                tld.mul_rn(a0[:, None], s0), tld.mul_rn(a1[:, None], s1)),
                tld.mul_rn(a2[:, None], s2)), tld.mul_rn(a3[:, None], s3))
        else:
            acc = tld.fma(a3[:, None], s3, tld.fma(a2[:, None], s2,
                          tld.fma(a1[:, None], s1, tld.mul_rn(a0[:, None], s0))))
        result = tld.add_rn(acc, tld.mul_rn(post[:, None], y))
        tl.store(out_ptr + base + out_stream * HIDDEN, result, mask=mask)


def mix_site(s, y, head, chart, h0, neg_h0, g_eye, g_gate, *,
             tile=(4, 256, 4), mix_one=False, output=None):
    """Mix contiguous BF16 [T,4,H] streams using FP32 site coefficients.

    Head is [T,8] with h_post in the last four columns; chart is [T,4,4].
    h0, neg_h0 and g_eye are [4,4], g_gate is scalar. Coefficients are FP32.
    No state, model objects or scheduler decisions are retained.
    """
    tokens, streams, hidden = s.shape
    assert streams == 4 and s.is_cuda and s.dtype == torch.bfloat16 and s.is_contiguous()
    assert y.shape == (tokens, hidden) and y.dtype == s.dtype and y.is_contiguous()
    assert head.shape == (tokens, 8) and chart.shape == (tokens, 4, 4)
    assert all(value.dtype == torch.float32 and value.is_contiguous()
               for value in (head, chart, h0, neg_h0, g_eye, g_gate))
    if output is None:
        output = torch.empty_like(s)
    assert output.shape == s.shape and output.dtype == s.dtype and output.is_contiguous()
    block_m, block_c, warps = tile
    _site_mix[(triton.cdiv(tokens, block_m), triton.cdiv(hidden, block_c))](
        s, y, head, chart, h0, neg_h0, g_eye, g_gate, output,
        TOKENS=tokens, HIDDEN=hidden, BLOCK_M=block_m, BLOCK_C=block_c,
        MIX_ONE=mix_one, num_warps=warps, num_stages=1)
    return output

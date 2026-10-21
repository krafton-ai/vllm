"""Fuse BF16 zero padding and MXFP8 quantization without changing GEMM shape."""
import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_padded_rows(
    x_ptr, q_ptr, scale_ptr,
    rows: tl.constexpr, padded_rows: tl.constexpr, width: tl.constexpr,
    scale_width: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(
        x_ptr + row * width + cols,
        (row < rows) & (cols < width), other=0,
    ).to(tl.float32)
    values = tl.reshape(values, (BLOCK // 32, 32))
    scaled_max = tl.max(tl.abs(values), 1) * (1.0 / 448.0)
    bits = scaled_max.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    round_up = ((mantissa > 0) & (exponent != 254)
                & ~((exponent == 0) & (mantissa <= 0x400000)))
    exponent += tl.where(round_up, 1, 0)
    exponent = tl.where(scaled_max == float("inf"), 254, exponent)
    inverse_bits = tl.where(exponent == 254, 0x00400000, (254 - exponent) << 23)
    inverse = inverse_bits.to(tl.float32, bitcast=True)
    quantized = (values * inverse[:, None]).to(tl.float8e4nv)
    tl.store(q_ptr + row * width + cols, tl.reshape(quantized, (BLOCK,)),
             (row < padded_rows) & (cols < width))
    scale_cols = tl.program_id(1) * (BLOCK // 32) + tl.arange(0, BLOCK // 32)
    # FlashInfer's 128-row by 4-column scale-factor swizzle.
    scale_offset = ((row // 128) * (128 * scale_width)
                    + (scale_cols // 4) * 512 + (row % 32) * 16
                    + ((row % 128) // 32) * 4 + scale_cols % 4)
    tl.store(scale_ptr + scale_offset, exponent.to(tl.uint8), scale_cols < scale_width)


def quantize(x: torch.Tensor, padded_rows: int):
    """Match FlashInfer quantization of zero-padded, contiguous finite BF16 input.

    Preserve the existing padded GEMM dimensions and initialize scale padding.
    This is an inference-only helper; it does not provide an autograd backward.
    """
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("Expected a contiguous two-dimensional BF16 tensor")
    rows, width = x.shape
    if width % 32 or padded_rows < rows or padded_rows <= 0:
        raise ValueError("Invalid MXFP8 width or padded row count")
    scale_width = triton.cdiv(width // 32, 4) * 4
    scale_rows = triton.cdiv(padded_rows, 128) * 128
    q = torch.empty((padded_rows, width), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((scale_rows * scale_width,), dtype=torch.uint8, device=x.device)
    _quantize_padded_rows[(scale_rows, triton.cdiv(scale_width * 32, 256))](
        x, q, scales, rows, padded_rows, width, scale_width, 256, num_warps=4,
    )
    return q, scales

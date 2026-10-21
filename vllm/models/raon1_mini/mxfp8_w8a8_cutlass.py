# Copyright (c) 2026, KRAFTON. Zero-mismatch W8A8 backend.
"""RAON_MOE_MXFP8_W8A8_GEMM_MODE=4: swap-AB CUTLASS grouped MoE backend.

Kernel: vendored Sm100GroupedBlockScaledGemmKernel
(vendored/cutlass-ex92-grouped-blockscaled/, CUTLASS v4.5.0 example 92,
unmodified). Swap-AB: A = weights (M = output dim), B = activations
(N = tokens), which keeps the GEMM efficient for skinny per-expert decode
batches.

Call layout:
  * A = weights [M_out, K] k-major (stride (K, 1)), B = tokens [n_e, K]
    k-major, C = [M_out, n_e] m-major (stride (1, M_out)), so the rows of
    C^T are [token, channel] row-major (no transpose copy).
  * SF: sf_vec 32, mma layout (l=1, ceil(rows/128), ceil(sf_k/4), 32,4,4)
    contiguous; linear (row, ks) maps to the closed-form index:
    (row//128)*(512*rk) + (ks//4)*512 + (row%32)*16 + ((row//32)%4)*4 + (ks%4)
  * tiler (128,128), cluster (1,1), host_problem_shape_available=False
    (total_num_clusters := max_active_clusters, a grid constant, so no
    recompilation).

Bit-exactness with mode 0:
  * a1/weight quantization reuses mxfp8_w8a8_triton.quantize_mxfp8_rows /
    quantize_expert_weights unchanged; byte-identical GEMM inputs are a
    precondition for bit-identical outputs.
  * The fc1 GEMM uses c_dtype=Float32 (no bf16 round trip), fc2 uses
    BFloat16. SiTU + prob fold + a2 reuse situ_probfold_quant_mxfp8, which is
    row-local and therefore invariant to row placement.

Compact R-row layout: R = T * top_k is static in shape, so the SiTU/gather
grids are static. The data-dependent per-expert split lives in the contents
of device arrays (dim N column + B/C pointer columns). The grouped kernel
reads the pointers on device at run time, so CUDA graph replay is safe (the
addresses stay constant).

The routing/scatter glue runs as Triton kernels instead of ~25 small aten ops
per call, whose host dispatch otherwise dominates decode:
  * _swab_route_pack: single-CTA counting sort (in-block cumsum ranking is a
    stable sort, same placement as argsort(stable)). Writes counts/starts,
    the dim N column, the B/C pointer columns (base + starts * row bytes) and
    pos_for_slot/src_tok/src_slot/sf_row in one launch.
  * _swab_gather_a1: gathers a1 q/sf/prob into compact rows and writes the
    SF directly into the mma layout via the closed-form index.
  * _swab_scatter_sf: the same direct SF write for a2.
SF padding (rank >= n_e) may hold any value: the C store is masked by n_e.

Contract: off by default and fail-closed (no silent fallback). Weight refit
goes through invalidate_weight_cache(). The first JIT/allocation must happen
outside CUDA graph capture (raises otherwise). Stream =
torch.cuda.current_stream.
"""
import importlib.util
import os
import sys

import torch
import triton
import triton.language as tl

# Quantization helpers are imported from the mode-0 Triton module rather than
# copied, so the arithmetic cannot drift. No import cycle: mxfp8_w8a8_triton
# does not import this module at top level.
from .mxfp8_w8a8_triton import (
    _float_to_e8m0 as _tri_float_to_e8m0,
    _exp2f_rcp as _tri_exp2f_rcp,
    _MAX_NORM_RCP_C as _TRI_MAX_NORM_RCP_C,
    _sf_mma_addr,   # closed-form mma SF index (shared with the SiTU SF writer)
)

__all__ = ["expert_rows_swab", "invalidate_weight_cache"]

# Defaults to the in-tree vendored copy; override with RAON_SWAB_KERNEL_DIR.
_KERNEL_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "vendored", "cutlass-ex92-grouped-blockscaled")

_SF_VEC = 32
_MMA_TILER = (128, 128)
_CLUSTER = (1, 1)
# Experimental small-batch decode routing.
_SMALL_ROUTE = int(os.environ.get("RAON_SWAB_SMALL_ROUTE", "0"))
if _SMALL_ROUTE not in (0, 1, 2, 3):
    raise ValueError("RAON_SWAB_SMALL_ROUTE must be 0, 1 (pack), 2 (fused), or 3 (fused with C1 compaction)")
_SMALL_ROUTE_MAX_T = 8


def _parse_shape_map(env_name="RAON_SWAB_GEMM_SHAPE_MAP"):
    """RAON_SWAB_GEMM_SHAPE_MAP: per-c_dtype (tiler, cluster) override.
    Format "f32=tm,tn,cm,cn;bf16=tm,tn,cm,cn" (partial maps allowed). Unset
    keeps the default constants. Malformed values raise (fail-closed)."""
    raw = os.environ.get(env_name, "").strip()
    out = {}
    if not raw:
        return out
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        name, _, vals = part.partition("=")
        name = name.strip()
        if name not in ("f32", "bf16"):
            raise RuntimeError(
                f"{env_name}: unknown c_dtype '{name}' "
                "(f32|bf16 only; fail-closed)")
        try:
            nums = [int(v) for v in vals.split(",")]
        except ValueError:
            nums = []
        if len(nums) != 4 or any(n <= 0 for n in nums):
            raise RuntimeError(
                f"{env_name}: '{part}' must be tm,tn,cm,cn "
                "(four positive integers; fail-closed)")
        out[name] = ((nums[0], nums[1]), (nums[2], nums[3]))
    return out


_SHAPE_MAP = _parse_shape_map()
_PREFILL_SHAPE_MAP = _parse_shape_map("RAON_SWAB_PREFILL_GEMM_SHAPE_MAP")
_SMALL_SHAPE_MAP = _parse_shape_map("RAON_SWAB_SMALL_GEMM_SHAPE_MAP")

_PREWARM_DONE = False


def prewarm_shape_map_kernels(device=0, E=128, top_k=None, ffn=1088):
    """When a shape map is set, eagerly compile every mapped kernel at model
    build time, before worker threads are spawned. A cold JIT inside the first
    forward falls in a fork-unsafe phase of startup and can hang the engine.
    No-op when no shape map is set. With RAON_SWAB_SITU_EPILOGUE=1 the fc1
    (f32) entry compiles the fused epilogue kernel.
    """
    global _PREWARM_DONE
    if _PREWARM_DONE or not (_SHAPE_MAP or _PREFILL_SHAPE_MAP or _SMALL_SHAPE_MAP):
        return 0
    _PREWARM_DONE = True
    n = 0

    def _compile(name, experts, **kw):
        situ = ffn if (name == "f32" and _SITU_EPILOGUE) else None
        _compiled_kernel(name, experts, device, situ_ffn=situ, **kw)

    for prefill, mapping in ((False, _SHAPE_MAP), (True, _PREFILL_SHAPE_MAP)):
        for name in mapping:
            _compile(name, E, prefill=prefill)
            n += 1
    for name in _SMALL_SHAPE_MAP:
        _compile(name, E, small=True)
        n += 1
    if _SMALL_ROUTE == 3 and top_k is not None:
        for name in ("f32", "bf16"):
            _compile(name, top_k, small=True)
            n += 1
    print(f"[va] SHAPE_MAP prewarm: {n} CuTe kernel(s) compiled at model "
          f"build (freeze fix)", flush=True)
    return n


# RAON_SWAB_ROUTE_PACK_PREFILL=1: replace the aten routing for large R
# (prefill; argsort/bincount/fancy-index, ~15 ops) with three deterministic
# multi-CTA two-pass counting-sort kernels. The placement matches
# argsort(stable) exactly (block order preserves slot order; no atomics).
# Off by default (aten path).
_ROUTE_PACK_PREFILL = (
    os.environ.get("RAON_SWAB_ROUTE_PACK_PREFILL", "0") == "1")


def _parse_glue_tune():
    """RAON_SWAB_GLUE_TUNE_MAP: launch parameters for the prefill glue
    kernels. Format "qrows=block,warps;gather=block,warps;
    probfold=block,warps;scatter=block,warps" (partial maps allowed). Unset
    keeps the default constants. All kernels compute per row or per 32-element
    group, so block/warps are not expected to change results. Malformed values
    raise (fail-closed)."""
    raw = os.environ.get("RAON_SWAB_GLUE_TUNE_MAP", "").strip()
    out = {}
    if not raw:
        return out
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        name, _, vals = part.partition("=")
        name = name.strip()
        if name not in ("qrows", "gather", "probfold", "scatter"):
            raise RuntimeError(
                f"RAON_SWAB_GLUE_TUNE_MAP: unknown kernel key '{name}' "
                "(qrows|gather|probfold|scatter; fail-closed)")
        try:
            nums = [int(v) for v in vals.split(",")]
        except ValueError:
            nums = []
        if len(nums) != 2 or nums[0] % 32 or nums[0] <= 0 or nums[1] <= 0:
            raise RuntimeError(
                f"RAON_SWAB_GLUE_TUNE_MAP: '{part}' must be block,warps "
                "(positive integers, block a multiple of 32; fail-closed)")
        out[name] = (nums[0], nums[1])
    return out


_GLUE_TUNE = _parse_glue_tune()


def _set_glue_tune_for_bench(raw):
    """Bench-only hook; production code never calls it."""
    global _GLUE_TUNE
    if raw is None:
        os.environ.pop("RAON_SWAB_GLUE_TUNE_MAP", None)
    else:
        os.environ["RAON_SWAB_GLUE_TUNE_MAP"] = raw
    _GLUE_TUNE = _parse_glue_tune()
# Buffer availability is fixed from the env at import, separately from the
# dispatch flag (_ROUTE_PACK_PREFILL, which benches toggle at run time), so an
# arena created while the flag is off still has the blk buffers.
_ROUTE_PACK_AVAILABLE = _ROUTE_PACK_PREFILL
_ROUTE_BLK = 256

# RAON_SWAB_SITU_EPILOGUE=1 (default 0 = the GEMM13 fp32 + Triton SiTU chain
# above, byte-unchanged): GEMM13 runs mxfp8_gemm13_situ's kernel, whose
# epilogue applies SiTU + prob fold + a2 MXFP8 quantization to the fp32
# accumulator in registers and writes b2/SFB2 directly (no fp32 c1 tensor,
# no situ/scatter launches). The fc1 weight rows are permuted in place into
# 64 gate + 64 value rows per 128-row tile (_interleave_fc1), so this switch
# is process-wide and fixed at import, like the other mode-4 switches.
_SITU_EPILOGUE = os.environ.get("RAON_SWAB_SITU_EPILOGUE", "0") == "1"


def _set_situ_epilogue_for_bench(on):
    """Bench/test hook: toggle the fused GEMM13 epilogue in-process. Weights
    first used while ON are interleaved in place, so a comparison must give
    each path its own weight tensors."""
    global _SITU_EPILOGUE
    _SITU_EPILOGUE = bool(on)


def _fc1_interleave_rows(N13, device):
    """Source row of each interleaved fc1 row: tile t = rows [128t, 128t+128)
    holds gate rows [64t, 64t+64) then value rows ffn + [64t, 64t+64)."""
    ffn = N13 // 2
    r = torch.arange(N13, device=device)
    t, i = r // 128, r % 128
    return torch.where(i < 64, 64 * t + i, ffn + 64 * t + (i - 64))


def adopt_fc1_layout(ent):
    """Single entry point for every _WQ_CACHE producer (online quantization,
    artifact load, compact loader): bring the fc1 weights into the layout the
    active GEMM13 path consumes. No-op unless RAON_SWAB_SITU_EPILOGUE=1."""
    if _SITU_EPILOGUE and not ent.get("fc1_interleaved"):
        _interleave_fc1(ent["q13"], ent["s13_mma"])
        ent["fc1_interleaved"] = True
    return ent


def _interleave_fc1(q13, s13_mma):
    """Permute fc1 weight rows (e4m3 data + their SF in the mma layout) in
    place. A row permutation moves whole K-reductions; the GEMM value of every
    (channel, token) pair is unchanged."""
    E, N13, K = q13.shape
    if N13 % 128 or (N13 // 2) % 64 or K % 128:
        raise RuntimeError(f"RAON_SWAB_SITU_EPILOGUE: unsupported fc1 shape {tuple(q13.shape)}")
    perm = _fc1_interleave_rows(N13, q13.device)
    with torch.inference_mode():
        for e in range(E):
            q13[e].copy_(q13[e].index_select(0, perm))
        rk = _ceil_div(K // 32, 4)
        ks = torch.arange(K // 32, device=q13.device)

        def addr(rows):
            rows = rows[:, None]
            return ((rows // 128) * (512 * rk) + (ks // 4) * 512 + (rows % 32) * 16
                    + ((rows // 32) % 4) * 4 + (ks % 4)).reshape(-1)

        dst = addr(torch.arange(N13, device=q13.device))
        src = addr(perm)
        s13_mma[:, dst] = s13_mma[:, src].clone()


def _shape_for(c_dtype_name, prefill=False, small=False):
    # Large routed batches can use a different tile without slowing decode.
    default = _SHAPE_MAP.get(c_dtype_name, (_MMA_TILER, _CLUSTER))
    if small:
        return _SMALL_SHAPE_MAP.get(c_dtype_name, default)
    return _PREFILL_SHAPE_MAP.get(c_dtype_name, default) if prefill else default


def _set_shape_map_for_bench(raw):
    """Bench-only hook: replace the shape map in-process (rewrites the env
    and clears the compile cache, forcing a re-JIT). Production code never
    calls it."""
    global _SHAPE_MAP
    if raw is None:
        os.environ.pop("RAON_SWAB_GEMM_SHAPE_MAP", None)
    else:
        os.environ["RAON_SWAB_GEMM_SHAPE_MAP"] = raw
    _SHAPE_MAP = _parse_shape_map()
    _COMPILED.clear()

_DONOR = None            # vendored module (lazy import)
_COMPILED: dict = {}     # (dtype, expert count, tile, cluster) -> compiled kernel
_HW: dict = {}           # sm_count / max_active_clusters / tensormap_shape
_WQ_CACHE: dict = {}     # id(w13) -> prequantized weights + SF (mma layout)
_ARENAS: dict = {}       # (T, top_k, E, N13, K, ffn, dev) -> buffers
_PTRS: dict = {}         # (weights, arena) pair -> device arrays
# Arena pinning bound. _ARENAS keeps every arena forever because captured decode
# graphs hold their addresses (and _PTRS their pointer arrays) as kernel
# arguments. Its key includes T, the token count of the MoE call, and chunked
# prefill produces a new T almost every step, so pinning every arena runs the
# GPU out of memory. Only T <= this bound is pinned (decode graphs are at most
# max_num_seqs x (1 + speculative tokens) = 256 tokens); a larger T is eager
# prefill, which needs no stable address, so it uses one transient arena.
# The transient arena is a high-water mark: it is keyed without T, reused for
# every T up to its capacity and only reallocated (once, larger) when a bigger T
# arrives. vLLM's profile run executes max_num_batched_tokens first, so serving
# normally allocates it once at the full budget. Reallocating it whenever T
# changes (chunked prefill alternates T every step) costs a memset per step and
# fragments the caching allocator. Kernels read the row geometry (rmx128, rk*, slab*) from the arena and
# take the row count R from T, and rows/row-scale padding beyond R are never
# read into outputs (the same-T reuse already depends on this), so a larger arena serves a smaller T
# bit-identically. One invariant is not about rows: GEMM2 reads the SFB2 bytes
# for ks >= ffn/32 in the last group of four (K padding, multiplied with TMA
# zero-filled data), so they must stay zero. torch.zeros provides that at
# allocation and no writer stores ks >= ffn/32, for any T.
_ARENA_PIN_MAX_T = int(os.environ.get("RAON_MXFP8_ARENA_PIN_MAX_TOKENS", "512"))
_TRANSIENT: dict = {"key": None, "arena": None}


def _drop_transient_arena() -> None:
    old = _TRANSIENT["arena"]
    if old is not None:
        b1 = id(old["b1"])
        for k in [k for k in _PTRS if k[1] == b1]:
            del _PTRS[k]    # its pointer arrays reference the arena being freed
    _TRANSIENT["key"] = None
    _TRANSIENT["arena"] = None
_ANNOUNCED = False


def invalidate_weight_cache() -> int:
    """Re-quantize the cached weights in place after a weight refit.

    CUDA graph replay does not go through Python. Clearing _WQ_CACHE and
    _PTRS would (i) recreate the prequantized tensors in new storage and
    (ii) rebuild the device pointer arrays in _PTRS, which hold those
    addresses and are themselves kernel arguments, so captured decode graphs
    would keep reading freed memory. Instead each entry is kept and the new
    values are copy_'d into the same buffers. Weight addresses therefore never
    change and _PTRS must not be cleared (that would create new pointer arrays
    and leave the graph arguments stale).
    The cached tensors are inference tensors, so the copy runs under
    inference_mode. Entries are re-quantized one at a time, so the transient
    extra memory is at most one layer. Repeated calls are harmless.
    """
    if os.environ.get("RAON_MXFP8_WEIGHTS") and _WQ_CACHE:
        raise RuntimeError("Preconverted MXFP8 serving does not support weight refit; reconvert and restart")
    from . import mxfp8_w8a8_triton as TRI
    n = 0
    for _key, ent in list(_WQ_CACHE.items()):
        w13, w2 = ent["_refs"]
        nq13, ns13 = TRI.quantize_expert_weights(w13.detach())
        nq2, ns2 = TRI.quantize_expert_weights(w2.detach())
        E, N13, K = w13.shape
        _, Kw, ffn = w2.shape
        dev = w13.device

        def _mma_of(sv, rows, sf_k):
            rm, rk = _ceil_div(rows, 128), _ceil_div(sf_k, 4)
            lin_pad = torch.zeros(E, rm * 128, rk * 4, dtype=torch.uint8,
                                  device=dev)
            lin_pad[:, :rows, :sf_k] = sv
            mma = torch.zeros(E, rm * rk * 512, dtype=torch.uint8, device=dev)
            _sf_scatter_mma(lin_pad, mma)
            return mma

        ns13_mma = _mma_of(ns13, N13, K // 32)
        ns2_mma = _mma_of(ns2, Kw, ffn // 32)
        with torch.inference_mode():
            for k_, nv in (("q13", nq13), ("s13_mma", ns13_mma),
                           ("q2", nq2), ("s2_mma", ns2_mma)):
                old_t = ent[k_]
                if not (nv.shape == old_t.shape and nv.dtype == old_t.dtype
                        and nv.device == old_t.device):
                    raise RuntimeError(
                        f"invalidate_weight_cache: {k_} changed shape/dtype/device; "
                        "re-register and re-capture (fail-closed)")
                p0 = old_t.data_ptr()
                old_t.copy_(nv)
                assert old_t.data_ptr() == p0, f"invalidate_weight_cache: {k_} pointer moved"
            if ent.get("fc1_interleaved"):
                _interleave_fc1(ent["q13"], ent["s13_mma"])
        del nq13, ns13, nq2, ns2, ns13_mma, ns2_mma
        n += 1
    # _PTRS/_ARENAS: all addresses are unchanged; do not clear (see docstring).
    return n


def _ceil_div(a, b):
    return (a + b - 1) // b


def _load_donor():
    """Import the vendored kernel module under torch.device("cpu"),
    registering it in sys.modules before executing it."""
    global _DONOR
    if _DONOR is not None:
        return _DONOR
    path = os.path.join(
        os.environ.get("RAON_SWAB_KERNEL_DIR", _KERNEL_DIR_DEFAULT),
        "grouped_blockscaled_gemm.py")
    if not os.path.isfile(path):
        raise RuntimeError(
            f"RAON_MOE_MXFP8_W8A8_GEMM_MODE=4: vendored kernel not found: {path} "
            "(set RAON_SWAB_KERNEL_DIR; fail-closed)")
    with torch.device("cpu"):
        spec = importlib.util.spec_from_file_location(
            "raon_swab_gbs_donor", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(spec.name, None)
            raise
    _DONOR = mod
    return mod


# ---------------------------------------------------------------------------
# Triton glue kernels (replace host-dispatched aten ops)
# ---------------------------------------------------------------------------

@triton.jit
def _swab_route_pack(
    ids_ptr,                       # (R,) int — flat expert id
    dim1_ptr, dim2_ptr,            # (E,4) int32: writes the N column (col 1)
    pabc1_ptr, pabc2_ptr,          # (E,3) int64: writes the B/C columns (cols 1,2)
    b1_base, c1_base, b2_base, c2_base,   # int64 buffer base addresses
    rb_b1, rb_c1, rb_b2, rb_c2,           # row bytes
    pos_for_slot_ptr,              # (R,) int64: slot -> compact row
    src_tok_ptr, src_slot_ptr,     # (R,) int32: compact row -> token/slot
    sf_row_ptr,                    # (R,) int64: compact row -> e*RMX128+rank
    R, RMX128,
    E: tl.constexpr, TOPK: tl.constexpr, BLOCK: tl.constexpr,
):
    """Single-CTA counting sort. In-block cumsum ranking plus a running
    per-block base preserves order (same placement as torch.argsort(stable)).
    Placement does not affect output bits (all row ops are row-local)."""
    offs_e = tl.arange(0, E)
    counts = tl.zeros((E,), dtype=tl.int32)
    n_blk = tl.cdiv(R, BLOCK)
    for i0 in range(0, n_blk):
        idx = i0 * BLOCK + tl.arange(0, BLOCK)
        mask = idx < R
        e = tl.load(ids_ptr + idx, mask=mask, other=E).to(tl.int32)
        onehot = ((e[:, None] == offs_e[None, :]) & mask[:, None])
        counts += tl.sum(onehot.to(tl.int32), axis=0)
    starts = tl.cumsum(counts, axis=0) - counts          # exclusive
    tl.store(dim1_ptr + offs_e * 4 + 1, counts)
    tl.store(dim2_ptr + offs_e * 4 + 1, counts)
    s64 = starts.to(tl.int64)
    tl.store(pabc1_ptr + offs_e * 3 + 1, b1_base + s64 * rb_b1)
    tl.store(pabc1_ptr + offs_e * 3 + 2, c1_base + s64 * rb_c1)
    tl.store(pabc2_ptr + offs_e * 3 + 1, b2_base + s64 * rb_b2)
    tl.store(pabc2_ptr + offs_e * 3 + 2, c2_base + s64 * rb_c2)
    run = starts
    for i0 in range(0, n_blk):
        idx = i0 * BLOCK + tl.arange(0, BLOCK)
        mask = idx < R
        e = tl.load(ids_ptr + idx, mask=mask, other=E).to(tl.int32)
        onehot = ((e[:, None] == offs_e[None, :])
                  & mask[:, None]).to(tl.int32)
        cum = tl.cumsum(onehot, axis=0)                  # inclusive
        rank_blk = tl.sum(onehot * (cum - 1), axis=1)
        base_e = tl.sum(onehot * run[None, :], axis=1)
        start_e = tl.sum(onehot * starts[None, :], axis=1)
        dest = (base_e + rank_blk).to(tl.int64)
        rank_tot = (base_e + rank_blk - start_e).to(tl.int64)
        e64 = e.to(tl.int64)
        tl.store(pos_for_slot_ptr + idx, dest, mask=mask)
        tl.store(src_tok_ptr + dest, (idx // TOPK).to(tl.int32), mask=mask)
        tl.store(src_slot_ptr + dest, idx.to(tl.int32), mask=mask)
        tl.store(sf_row_ptr + dest, e64 * RMX128 + rank_tot, mask=mask)
        run += tl.sum(onehot, axis=0)


# Parallel routing (count -> scan -> place) with atomic counters, replacing
# the serial single-CTA _swab_route_pack for decode. Row placement within an
# expert depends on atomic order and is nondeterministic, but the output bits
# are not:
#   * each row's value depends only on its own input row and weights,
#   * SF addresses are computed together with the placement,
#   * the per-token finalize sum runs in pos order = expert order (a token's
#     top-k experts are distinct; rank within an expert only orders tokens),
# so the output is invariant to the placement permutation.
@triton.jit
def _swab_route_count(ids_ptr, counts_ptr, R, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < R
    e = tl.load(ids_ptr + idx, mask=mask, other=0).to(tl.int32)
    tl.atomic_add(counts_ptr + e, 1, mask=mask)


@triton.jit
def _swab_route_scan(
    counts_ptr, cursor_ptr,
    dim1_ptr, dim2_ptr, pabc1_ptr, pabc2_ptr,
    b1_base, c1_base, b2_base, c2_base,
    rb_b1, rb_c1, rb_b2, rb_c2,
    starts_ptr,
    E: tl.constexpr,
):
    """Single CTA over E elements: cumsum, dim/ptr writes, cursor init, and
    reset of counts to zero for the next call (the arena starts zeroed)."""
    offs_e = tl.arange(0, E)
    counts = tl.load(counts_ptr + offs_e)
    starts = tl.cumsum(counts, axis=0) - counts
    tl.store(dim1_ptr + offs_e * 4 + 1, counts)
    tl.store(dim2_ptr + offs_e * 4 + 1, counts)
    s64 = starts.to(tl.int64)
    tl.store(pabc1_ptr + offs_e * 3 + 1, b1_base + s64 * rb_b1)
    tl.store(pabc1_ptr + offs_e * 3 + 2, c1_base + s64 * rb_c1)
    tl.store(pabc2_ptr + offs_e * 3 + 1, b2_base + s64 * rb_b2)
    tl.store(pabc2_ptr + offs_e * 3 + 2, c2_base + s64 * rb_c2)
    tl.store(cursor_ptr + offs_e, starts)
    tl.store(starts_ptr + offs_e, starts)
    tl.store(counts_ptr + offs_e, tl.zeros((E,), dtype=tl.int32))


@triton.jit
def _swab_route_place(
    ids_ptr, cursor_ptr, starts_ptr,
    pos_for_slot_ptr, src_tok_ptr, src_slot_ptr, sf_row_ptr,
    R, RMX128,
    TOPK: tl.constexpr, BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < R
    e = tl.load(ids_ptr + idx, mask=mask, other=0).to(tl.int32)
    dest = tl.atomic_add(cursor_ptr + e, 1, mask=mask)
    start_e = tl.load(starts_ptr + e, mask=mask, other=0)
    rank = (dest - start_e).to(tl.int64)
    d64 = dest.to(tl.int64)
    tl.store(pos_for_slot_ptr + idx, d64, mask=mask)
    tl.store(src_tok_ptr + d64, (idx // TOPK).to(tl.int32), mask=mask)
    tl.store(src_slot_ptr + d64, idx.to(tl.int32), mask=mask)
    tl.store(sf_row_ptr + d64, e.to(tl.int64) * RMX128 + rank, mask=mask)


# Deterministic routing kernels for large R (prefill), replacing the aten
# argsort/bincount/fancy-index chain (~15 ops). Unlike the parallel kernels
# above they use no atomics: (block, expert) histogram -> per-block exclusive
# running sum (base_blk) -> stable in-block ranking. The placement ("expert
# ascending, then slot ascending within an expert") equals
# torch.argsort(flat_e, stable=True) exactly, so the metadata and all
# downstream inputs (gather, GEMM, finalize) match the aten path byte for byte.
@triton.jit
def _swab_route_count_blk(
    ids_ptr, counts_blk_ptr, R,
    E: tl.constexpr, BLOCK: tl.constexpr,
):
    blk = tl.program_id(0)
    idx = blk * BLOCK + tl.arange(0, BLOCK)
    mask = idx < R
    e = tl.load(ids_ptr + idx, mask=mask, other=E).to(tl.int32)
    offs_e = tl.arange(0, E)
    onehot = ((e[:, None] == offs_e[None, :]) & mask[:, None]).to(tl.int32)
    tl.store(counts_blk_ptr + blk * E + offs_e, tl.sum(onehot, axis=0))


@triton.jit
def _swab_route_scan_blk(
    counts_blk_ptr, base_blk_ptr,
    dim1_ptr, dim2_ptr, pabc1_ptr, pabc2_ptr,
    b1_base, c1_base, b2_base, c2_base,
    rb_b1, rb_c1, rb_b2, rb_c2,
    starts_ptr, n_blk,
    E: tl.constexpr,
):
    """Single CTA: per-expert totals, starts (exclusive cumsum), dim/ptr
    writes, and per-block bases (exclusive running sum over blocks)."""
    offs_e = tl.arange(0, E)
    total = tl.zeros((E,), dtype=tl.int32)
    for b in range(0, n_blk):
        total += tl.load(counts_blk_ptr + b * E + offs_e)
    starts = tl.cumsum(total, axis=0) - total
    tl.store(dim1_ptr + offs_e * 4 + 1, total)
    tl.store(dim2_ptr + offs_e * 4 + 1, total)
    s64 = starts.to(tl.int64)
    tl.store(pabc1_ptr + offs_e * 3 + 1, b1_base + s64 * rb_b1)
    tl.store(pabc1_ptr + offs_e * 3 + 2, c1_base + s64 * rb_c1)
    tl.store(pabc2_ptr + offs_e * 3 + 1, b2_base + s64 * rb_b2)
    tl.store(pabc2_ptr + offs_e * 3 + 2, c2_base + s64 * rb_c2)
    tl.store(starts_ptr + offs_e, starts)
    run = starts
    for b in range(0, n_blk):
        c = tl.load(counts_blk_ptr + b * E + offs_e)
        tl.store(base_blk_ptr + b * E + offs_e, run)
        run += c


@triton.jit
def _swab_route_place_blk(
    ids_ptr, base_blk_ptr, starts_ptr,
    pos_for_slot_ptr, src_tok_ptr, src_slot_ptr, sf_row_ptr,
    R, RMX128,
    E: tl.constexpr, TOPK: tl.constexpr, BLOCK: tl.constexpr,
):
    """Stable in-block ranking (onehot cumsum, as in _swab_route_pack) plus
    the base_blk offset gives the global stable placement. No atomics."""
    blk = tl.program_id(0)
    idx = blk * BLOCK + tl.arange(0, BLOCK)
    mask = idx < R
    e = tl.load(ids_ptr + idx, mask=mask, other=E).to(tl.int32)
    offs_e = tl.arange(0, E)
    onehot = ((e[:, None] == offs_e[None, :]) & mask[:, None]).to(tl.int32)
    cum = tl.cumsum(onehot, axis=0)                  # inclusive
    rank_blk = tl.sum(onehot * (cum - 1), axis=1)
    base = tl.load(base_blk_ptr + blk * E + offs_e)
    starts = tl.load(starts_ptr + offs_e)
    base_e = tl.sum(onehot * base[None, :], axis=1)
    start_e = tl.sum(onehot * starts[None, :], axis=1)
    dest = (base_e + rank_blk).to(tl.int64)
    rank_tot = (base_e + rank_blk - start_e).to(tl.int64)
    tl.store(pos_for_slot_ptr + idx, dest, mask=mask)
    tl.store(src_tok_ptr + dest, (idx // TOPK).to(tl.int32), mask=mask)
    tl.store(src_slot_ptr + dest, idx.to(tl.int32), mask=mask)
    tl.store(sf_row_ptr + dest, e.to(tl.int64) * RMX128 + rank_tot,
             mask=mask)


@triton.jit
def _swab_gather_a1(
    aq_ptr, as_ptr, w_ptr,         # a1 q (T,K) u8 view / sf (T,SFK) u8 / prob
    b1_ptr, sf_mma_ptr, probs_ptr,  # compact B1 u8 view / SFB1 mma / probs f32
    src_tok_ptr, src_slot_ptr, sf_row_ptr,
    K, SFK, RMX128, RK, SLAB,
    BLOCK_K: tl.constexpr, SF_BLOCK: tl.constexpr,
):
    """One program per compact row: copy the a1 q row, SF (mma layout), prob."""
    row = tl.program_id(0).to(tl.int64)
    t = tl.load(src_tok_ptr + row).to(tl.int64)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        offs = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        m = offs < K
        v = tl.load(aq_ptr + t * K + offs, mask=m)
        tl.store(b1_ptr + row * K + offs, v, mask=m)
    sfr = tl.load(sf_row_ptr + row)
    ks = tl.arange(0, SF_BLOCK)
    m2 = ks < SFK
    s = tl.load(as_ptr + t * SFK + ks, mask=m2)
    tl.store(sf_mma_ptr + _sf_mma_addr(sfr, ks, RMX128, RK, SLAB),
             s, mask=m2)
    slot = tl.load(src_slot_ptr + row).to(tl.int64)
    w = tl.load(w_ptr + slot).to(tl.float32)
    tl.store(probs_ptr + row, w)


@triton.jit
def _swab_qgather_a1(
    x_ptr, w_ptr,                  # activations x (T,K) bf16/fp32 / prob (R,)
    b1_ptr, sf_mma_ptr, probs_ptr,  # compact B1 e4m3 / SFB1 mma / probs f32
    src_tok_ptr, src_slot_ptr, sf_row_ptr,
    K, SFK, RMX128, RK, SLAB,
    stride_xm, stride_xk,
    BLOCK_K: tl.constexpr,
):
    """Fused a1 quantization + gather + SF write + prob in one launch.

    The arithmetic is identical to _mxfp8_quant_rows_kernel (same helper jit
    functions: amax -> _float_to_e8m0 -> _exp2f_rcp -> cvt.rn.satfinite).
    1x32 block quantization is block-local, so quantizing the T rows once and
    copying them to R rows is byte-identical to quantizing each of the R rows.
    The mma address uses the same closed form as _swab_gather_a1."""
    row = tl.program_id(0).to(tl.int64)
    t = tl.load(src_tok_ptr + row).to(tl.int64)
    sfr = tl.load(sf_row_ptr + row)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        offs = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        m = offs < K
        x = tl.load(x_ptr + t * stride_xm + offs * stride_xk,
                    mask=m, other=0.0).to(tl.float32)
        xr = tl.reshape(x, (BLOCK_K // 32, 32))
        amax = tl.max(tl.abs(xr), axis=1)
        e = _tri_float_to_e8m0(amax * _TRI_MAX_NORM_RCP_C)
        rcp = _tri_exp2f_rcp(e)
        q = (xr * rcp[:, None]).to(tl.float8e4nv)
        tl.store(b1_ptr + row * K + offs,
                 tl.reshape(q, (BLOCK_K,)), mask=m)
        soffs = k0 * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32)
        m2 = soffs < SFK
        tl.store(sf_mma_ptr + _sf_mma_addr(sfr, soffs, RMX128, RK, SLAB),
                 e.to(tl.uint8), mask=m2)
    slot = tl.load(src_slot_ptr + row).to(tl.int64)
    w = tl.load(w_ptr + slot).to(tl.float32)
    tl.store(probs_ptr + row, w)


@triton.jit
def _swab_finalize_combine(
    c2_ptr, pos_ptr, out_ptr, H,
    TOPK: tl.constexpr, KP2: tl.constexpr, BLOCK_H: tl.constexpr,
):
    """Finalize: compact fc2 rows (equivalent to (T,K,H)) -> (T,H). Sums in
    fp32 in expert-ascending order and casts to bf16 once, matching
    moe_align.combine_unweighted_fp32 element-wise (compact row indices are
    globally expert-ascending, so per-token pos order = expert order; ties
    keep stable slot order). Iterated min selection (TOPK passes, sentinel)."""
    t = tl.program_id(0).to(tl.int64)
    hb = tl.program_id(1)
    offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    m = offs < H
    kidx = tl.arange(0, KP2)
    kmask = kidx < TOPK
    BIG = 1 << 62
    pos = tl.load(pos_ptr + t * TOPK + kidx, mask=kmask, other=BIG)
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for _ in tl.static_range(TOPK):
        cur = tl.min(pos, axis=0)
        row = tl.load(c2_ptr + cur * H + offs, mask=m, other=0.0)
        acc += row.to(tl.float32)
        pos = tl.where(pos == cur, BIG, pos)
    tl.store(out_ptr + t * H + offs, acc.to(tl.bfloat16), mask=m)


@triton.jit
def _swab_scatter_sf(
    as_ptr, sf_mma_ptr, sf_row_ptr,
    SFK, RMX128, RK, SLAB,
    SF_BLOCK: tl.constexpr,
):
    """a2 SF (compact row order) -> mma layout via the closed-form index."""
    row = tl.program_id(0).to(tl.int64)
    sfr = tl.load(sf_row_ptr + row)
    ks = tl.arange(0, SF_BLOCK)
    m = ks < SFK
    s = tl.load(as_ptr + row * SFK + ks, mask=m)
    tl.store(sf_mma_ptr + _sf_mma_addr(sfr, ks, RMX128, RK, SLAB), s, mask=m)


def _sf_scatter_mma(lin_pad_view, mma_arena_view):
    """Padded linear SF (E, rm*128, rk*4) -> mma layout. Used only for
    weight prequantization (once per layer); equivalent to the closed form."""
    E, rows_pad, ks_pad = lin_pad_view.shape
    rm, rk = rows_pad // 128, ks_pad // 4
    v = lin_pad_view.view(E, rm, 4, 32, rk, 4).permute(0, 1, 4, 3, 2, 5)
    mma_arena_view.view(E, rm, rk, 32, 4, 4).copy_(v)


def _hw_info(cluster=None):
    """Hardware info with max_active_clusters per cluster shape (None means
    _CLUSTER). The tensormap shape depends only on kernel class constants and
    the SM count, not on the cluster (as in the donor's run())."""
    cluster = cluster if cluster is not None else _CLUSTER
    if not _HW:
        m = _load_donor()
        hw = m.utils.HardwareInfo()
        sm = hw.get_max_active_clusters(1)
        K = m.Sm100GroupedBlockScaledGemmKernel
        _HW.update(sm_count=sm, _hwobj=hw, _mac_by_cluster={},
                   tensormap_shape=(sm, K.num_tensormaps,
                                    K.bytes_per_tensormap // 8))
    key = (cluster[0], cluster[1])
    mac_map = _HW["_mac_by_cluster"]
    if key not in mac_map:
        mac_map[key] = _HW["_hwobj"].get_max_active_clusters(
            cluster[0] * cluster[1])
    # Value for the last queried cluster (shown in the startup log line)
    _HW["max_active_clusters"] = mac_map[key]
    return _HW


def _compiled_kernel(c_dtype_name, E, device, prefill=False, small=False,
                     situ_ffn=None):
    """Compile once per c_dtype (fc1=Float32, fc2=BFloat16). Raises if the
    first compile happens during CUDA graph capture (fail-closed; warm up
    outside capture). With situ_ffn set, fc1 compiles the fused SiTU epilogue
    kernel (tile/cluster follow the f32 entry)."""
    tiler, cluster = _shape_for(c_dtype_name, prefill=prefill, small=small)
    cache_key = (c_dtype_name, E, tiler, cluster)
    if situ_ffn is not None:
        cache_key += (("situ", situ_ffn),)
    ent = _COMPILED.get(cache_key)
    if ent is not None:
        return ent
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "RAON_MOE_MXFP8_W8A8_GEMM_MODE=4: first grouped kernel JIT during "
            "CUDA graph capture; warm up outside capture (fail-closed)")
    m = _load_donor()
    cutlass = m.cutlass
    hw = _hw_info(cluster)
    mac = hw["_mac_by_cluster"][(cluster[0], cluster[1])]
    ab, sf = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
    c_dtype = {"f32": cutlass.Float32, "bf16": cutlass.BFloat16}[c_dtype_name]

    # Initial dummy tensors (same as the donor's run())
    min_ab = 16
    min_c = (16 * 8 // c_dtype.width)
    min_c *= (16 + min_c - 1) // min_c
    min_sf = 16
    initial_abc = [
        m.create_tensor_and_stride(1, min_ab, min_ab, False, ab)[2],   # a k-major
        m.create_tensor_and_stride(1, min_ab, min_ab, False, ab)[2],   # b k-major
        m.create_tensor_and_stride(1, min_c, min_c, True, c_dtype)[2],  # c m-major
    ]
    initial_sf = [
        m.create_tensor_and_stride(1, min_sf, min_sf, False, sf)[2],
        m.create_tensor_and_stride(1, min_sf, min_sf, False, sf)[2],
    ]
    if situ_ffn is None:
        kern = m.Sm100GroupedBlockScaledGemmKernel(_SF_VEC, tiler, cluster)
    else:
        from .mxfp8_gemm13_situ import Sm100GroupedGemm13SituKernel
        kern = Sm100GroupedGemm13SituKernel(_SF_VEC, tiler, cluster, situ_ffn)

    # Placeholder arrays for compilation (only their layout is used; the real
    # arrays are passed at call time)
    ct = m.cutlass_torch.cute_tensor_like
    dim, dim_t = ct(torch.zeros(E, 4, dtype=torch.int32), cutlass.Int32,
                    is_dynamic_layout=False, assumed_align=16)
    strides, strides_t = ct(torch.zeros(E, 3, 2, dtype=torch.int32),
                            cutlass.Int32, is_dynamic_layout=False,
                            assumed_align=16)
    pabc, pabc_t = ct(torch.zeros(E, 3, dtype=torch.int64), cutlass.Int64,
                      is_dynamic_layout=False, assumed_align=16)
    psf, psf_t = ct(torch.zeros(E, 2, dtype=torch.int64), cutlass.Int64,
                    is_dynamic_layout=False, assumed_align=16)
    tmap, tmap_t = ct(torch.empty(hw["tensormap_shape"], dtype=torch.int64),
                      cutlass.Int64, is_dynamic_layout=False)
    stream = m.cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    extra = ()
    if situ_ffn is not None:
        # fc2 pointer arrays + b2/SFB2-arena/probs placeholders (dynamic
        # layouts: one compile serves every arena size).
        pabc2, _ = ct(torch.zeros(E, 3, dtype=torch.int64), cutlass.Int64,
                      is_dynamic_layout=False, assumed_align=16)
        psf2, _ = ct(torch.zeros(E, 2, dtype=torch.int64), cutlass.Int64,
                     is_dynamic_layout=False, assumed_align=16)
        extra = (pabc2, psf2) + _situ_out_tensors(
            torch.zeros(64, dtype=torch.uint8, device=device),
            torch.zeros(64, dtype=torch.uint8, device=device),
            torch.zeros(16, dtype=torch.float32, device=device))
    compiled = m.cute.compile(
        kern,
        initial_abc[0], initial_abc[1], initial_abc[2],
        initial_sf[0], initial_sf[1],
        E, dim, strides, pabc, psf, *extra,
        mac,                             # total_num_clusters
        tmap, mac, stream,
        options="--opt-level 2",
    )
    ent = (compiled, initial_abc, initial_sf,
           (dim, dim_t, strides, strides_t, pabc, pabc_t, psf, psf_t,
            tmap, tmap_t))
    _COMPILED[cache_key] = ent
    return ent


def _situ_out_tensors(b2_u8, sf2_u8, probs):
    """1-D dynamic-layout cute views of the fused epilogue outputs/input."""
    from cutlass.cute.runtime import from_dlpack
    return tuple(from_dlpack(t, assumed_align=16).mark_layout_dynamic()
                 for t in (b2_u8.reshape(-1), sf2_u8.reshape(-1), probs.reshape(-1)))


def _quantized_weights(w13, w2, layer_idx=None):
    """Prequantize weights (once per layer): mode-0 quantizers + SF mma scatter."""
    key = id(w13)
    ent = _WQ_CACHE.get(key)
    if ent is not None:
        return ent
    if os.environ.get("RAON_MXFP8_WEIGHTS"):
        from .mxfp8_artifact import load
        ent = load(f"layer-{layer_idx}-routed", [w13.shape, w2.shape], w13.device)
        ent["_refs"] = (w13, w2)
        _WQ_CACHE[key] = adopt_fc1_layout(ent)
        return ent
    from . import mxfp8_w8a8_triton as TRI
    E, N13, K = w13.shape
    _, Kw, ffn = w2.shape
    q13, s13 = TRI.quantize_expert_weights(w13.detach())
    q2, s2 = TRI.quantize_expert_weights(w2.detach())
    dev = w13.device

    def _mma_of(s, rows, sf_k):
        rm, rk = _ceil_div(rows, 128), _ceil_div(sf_k, 4)
        lin_pad = torch.zeros(E, rm * 128, rk * 4, dtype=torch.uint8,
                              device=dev)
        lin_pad[:, :rows, :sf_k] = s
        mma = torch.zeros(E, rm * rk * 512, dtype=torch.uint8, device=dev)
        _sf_scatter_mma(lin_pad, mma)
        return mma

    s13_mma = _mma_of(s13, N13, K // 32)
    s2_mma = _mma_of(s2, Kw, ffn // 32)
    ent = {"q13": q13, "s13_mma": s13_mma, "q2": q2, "s2_mma": s2_mma,
           "_refs": (w13, w2)}   # keep the originals alive so id(w13) stays valid
    adopt_fc1_layout(ent)
    _WQ_CACHE[key] = ent
    return ent


def _arena(T, top_k, E, N13, K, ffn, device):
    """Compact R-row buffers + fixed-size SF mma arenas, at constant addresses."""
    key = (T, top_k, E, N13, K, ffn, device.index, _SITU_EPILOGUE)
    ar = _ARENAS.get(key)
    if ar is not None:
        return ar
    shape = key[1:]
    capturing = torch.cuda.is_current_stream_capturing()
    if T > _ARENA_PIN_MAX_T and _TRANSIENT["key"] == shape and not capturing:
        if _TRANSIENT["arena"]["capacity"] >= T:
            return _TRANSIENT["arena"]
    if capturing:
        raise RuntimeError(
            "RAON_MOE_MXFP8_W8A8_GEMM_MODE=4: first arena allocation during "
            "capture; warm up outside capture (fail-closed)")
    if T > _ARENA_PIN_MAX_T:
        T = max(T, _TRANSIENT["arena"]["capacity"]) if _TRANSIENT["key"] == shape else T
    R = T * top_k
    sfk1, sfk2 = K // 32, ffn // 32
    rm = _ceil_div(T, 128)              # rows per expert <= T (top-k experts are distinct)
    rk1, rk2 = _ceil_div(sfk1, 4), _ceil_div(sfk2, 4)
    dev = device
    ar = {
        "capacity": T, "R": R, "rmx128": rm * 128,
        "rk1": rk1, "rk2": rk2,
        "slab1": rm * rk1 * 512, "slab2": rm * rk2 * 512,
        # compact R rows (GEMM B/C, SiTU in/out; valid rows only)
        "b1": torch.zeros(R, K, dtype=torch.float8_e4m3fn, device=dev),
        # Fused SiTU epilogue: no fp32 GEMM13 output (route kernels only form
        # never-dereferenced C pointers from this base).
        "c1": torch.zeros(0 if _SITU_EPILOGUE else R, N13, dtype=torch.float32,
                          device=dev),
        "probs": torch.zeros(R, dtype=torch.float32, device=dev),
        "b2": torch.zeros(R, ffn, dtype=torch.float8_e4m3fn, device=dev),
        "a2s": torch.zeros(R, sfk2, dtype=torch.uint8, device=dev),
        "c2": torch.zeros(R, K, dtype=torch.bfloat16, device=dev),
        # routing metadata (written by the route kernels)
        "pos_for_slot": torch.zeros(R, dtype=torch.int64, device=dev),
        "src_tok": torch.zeros(R, dtype=torch.int32, device=dev),
        "src_slot": torch.zeros(R, dtype=torch.int32, device=dev),
        "sf_row": torch.zeros(R, dtype=torch.int64, device=dev),
        # Parallel routing scratch. _swab_route_scan resets counts to zero
        # after use (zero-initialized here); addresses are constant.
        "route_counts": torch.zeros(E, dtype=torch.int32, device=dev),
        "route_cursor": torch.zeros(E, dtype=torch.int32, device=dev),
        "route_starts": torch.zeros(E, dtype=torch.int32, device=dev),
    }
    if _ROUTE_PACK_AVAILABLE:
        # Deterministic prefill routing scratch, allocated only when the env
        # switch is on. n_blk is the static upper bound for R.
        n_blk = _ceil_div(R, _ROUTE_BLK)
        ar.update({
            "route_counts_blk": torch.zeros(n_blk, E, dtype=torch.int32,
                                            device=dev),
            "route_base_blk": torch.zeros(n_blk, E, dtype=torch.int32,
                                          device=dev),
        })
    ar.update({
        # SF mma arenas (128-row blocks per expert, written directly)
        "sfb1_mma": torch.zeros(E, rm * rk1 * 512, dtype=torch.uint8,
                                device=dev),
        "sfb2_mma": torch.zeros(E, rm * rk2 * 512, dtype=torch.uint8,
                                device=dev),
    })
    if T <= _ARENA_PIN_MAX_T:
        _ARENAS[key] = ar
    else:
        _drop_transient_arena()
        _TRANSIENT["key"] = shape
        _TRANSIENT["arena"] = ar
    return ar


def _device_arrays(wq, ar, E, N13, K, ffn):
    """dim/strides/ptrs device arrays (fc1/fc2). The array addresses are
    constant; the route kernels write the B/C pointers and the dim N column
    each step."""
    key = (id(wq["q13"]), id(ar["b1"]), E)
    ent = _PTRS.get(key)
    if ent is not None:
        return ent
    m = _load_donor()
    cutlass = m.cutlass
    ct = m.cutlass_torch.cute_tensor_like
    hw = _hw_info()

    def _mk(dim_rows, strides_rows, pabc_rows, psf_rows):
        dim, dim_t = ct(torch.tensor(dim_rows, dtype=torch.int32),
                        cutlass.Int32, is_dynamic_layout=False,
                        assumed_align=16)
        strides, _s = ct(torch.tensor(strides_rows, dtype=torch.int32),
                         cutlass.Int32, is_dynamic_layout=False,
                         assumed_align=16)
        pabc, pabc_t = ct(torch.tensor(pabc_rows, dtype=torch.int64),
                          cutlass.Int64, is_dynamic_layout=False,
                          assumed_align=16)
        psf, _q = ct(torch.tensor(psf_rows, dtype=torch.int64),
                     cutlass.Int64, is_dynamic_layout=False,
                     assumed_align=16)
        tmap, _t = ct(torch.empty(hw["tensormap_shape"], dtype=torch.int64),
                      cutlass.Int64, is_dynamic_layout=False)
        assert dim_t.stride() == (4, 1) and pabc_t.stride() == (3, 1), \
            (dim_t.stride(), pabc_t.stride())   # indexing assumed by the route kernels
        return {"dim": dim, "dim_torch": dim_t, "strides": strides,
                "pabc": pabc, "pabc_torch": pabc_t, "psf": psf,
                "tmap": tmap, "_keep": (_s, _q, _t)}

    e = torch.arange(E)
    q13, s13m = wq["q13"], wq["s13_mma"]
    # B/C columns start at 0; the route kernels write them each step
    fc1 = _mk(
        [[N13, 0, K, 1]] * E,
        [[[K, 1], [K, 1], [1, N13]]] * E,
        torch.stack([
            q13.view(torch.uint8).data_ptr() + e * q13.stride(0),
            torch.zeros(E, dtype=torch.int64),
            torch.zeros(E, dtype=torch.int64),
        ], dim=1).tolist(),
        torch.stack([
            s13m.data_ptr() + e * s13m.stride(0),
            ar["sfb1_mma"].data_ptr() + e * ar["sfb1_mma"].stride(0),
        ], dim=1).tolist(),
    )
    q2, s2m = wq["q2"], wq["s2_mma"]
    fc2 = _mk(
        [[K, 0, ffn, 1]] * E,
        [[[ffn, 1], [ffn, 1], [1, K]]] * E,
        torch.stack([
            q2.view(torch.uint8).data_ptr() + e * q2.stride(0),
            torch.zeros(E, dtype=torch.int64),
            torch.zeros(E, dtype=torch.int64),
        ], dim=1).tolist(),
        torch.stack([
            s2m.data_ptr() + e * s2m.stride(0),
            ar["sfb2_mma"].data_ptr() + e * ar["sfb2_mma"].stride(0),
        ], dim=1).tolist(),
    )
    ent = {"fc1": fc1, "fc2": fc2}
    if _SITU_EPILOGUE:
        if not wq.get("fc1_interleaved"):
            raise RuntimeError("RAON_SWAB_SITU_EPILOGUE=1 needs interleaved fc1 "
                               "weights (cache populated while the switch was off)")
        if ar["R"] * ffn >= 2 ** 31:
            raise RuntimeError("RAON_SWAB_SITU_EPILOGUE: b2 index exceeds int32")
        ent["situ"] = _situ_out_tensors(ar["b2"].view(torch.uint8),
                                        ar["sfb2_mma"], ar["probs"])
    _PTRS[key] = ent
    return ent


def _launch(c_name, E, device, arrays, prefill=False, small=False):
    compiled, initial_abc, initial_sf, _ = _compiled_kernel(c_name, E, device, prefill=prefill, small=small)
    m = _DONOR
    stream = m.cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    compiled(initial_abc[0], initial_abc[1], initial_abc[2],
             initial_sf[0], initial_sf[1],
             arrays["dim"], arrays["strides"], arrays["pabc"],
             arrays["psf"], arrays["tmap"], stream)


def _launch_situ(E, device, both, ffn, prefill=False, small=False):
    """GEMM13 with the fused SiTU/prob/a2-quant epilogue: reads the fc1
    arrays, writes b2 rows and SFB2 through the fc2 pointer arrays."""
    compiled, initial_abc, initial_sf, _ = _compiled_kernel(
        "f32", E, device, prefill=prefill, small=small, situ_ffn=ffn)
    fc1, fc2 = both["fc1"], both["fc2"]
    stream = _DONOR.cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    compiled(initial_abc[0], initial_abc[1], initial_abc[2],
             initial_sf[0], initial_sf[1],
             fc1["dim"], fc1["strides"], fc1["pabc"], fc1["psf"],
             fc2["pabc"], fc2["psf"], *both["situ"],
             fc1["tmap"], stream)


_EP_FINALIZE_XCHECKED = False


def expert_rows_swab(x, topk_weights, topk_ids, w13, w2, layer_idx=None,
                     do_finalize=False, ep_finalize_local=None):
    """(T, top_k, H) prob-folded expert rows via the swap-AB CUTLASS grouped path.

    Bit-identical to mode 0 (fused_experts_mxw8a8): quantization, SiTU and
    the prob fold reuse the mode-0 kernels; only the two GEMMs are grouped.
    Uses the compact R-row layout and Triton glue kernels (see the module
    docstring).
    """
    global _ANNOUNCED
    if do_finalize and x.dtype != torch.bfloat16:
        raise RuntimeError(
            f"mode4 finalize requires bf16 output (x.dtype={x.dtype}; "
            "fail-closed)")
    if do_finalize and ep_finalize_local is not None:
        raise RuntimeError(
            "expert_rows_swab: do_finalize=True and ep_finalize_local are "
            "mutually exclusive (single-stage finalize cannot reproduce the "
            "two-stage EP combine)")
    if ep_finalize_local is not None and x.dtype != torch.bfloat16:
        raise RuntimeError(
            f"mode4 EP finalize requires bf16 output (x.dtype={x.dtype}; "
            "fail-closed)")
    from . import mxfp8_w8a8_triton as TRI

    T, K = x.shape
    E, N13, Kb = w13.shape
    ffn = N13 // 2
    top_k = topk_ids.size(1)
    R = T * top_k
    assert Kb == K and w2.shape == (E, K, ffn), (w13.shape, w2.shape)
    if K % 32 or ffn % 32:
        raise RuntimeError(f"K/ffn must be multiples of 32: {K}, {ffn}")
    if E & (E - 1):
        raise RuntimeError(f"route kernels require E to be a power of 2 (E={E}; fail-closed)")

    # Record te_precision_scope, the mode-4 equivalent of what mode 2 (trtllm)
    # records at install time (mxfp8_w8a8_trtllm.py, same API). Without it the
    # scope check treats routed fc1/fc2 as not realized. _record_scope is gated
    # by the SCOPE_RECORD env, skips non-integer layer_idx and is idempotent,
    # so re-recording after a refit is harmless.
    _first_install = id(w13) not in _WQ_CACHE
    wq = _quantized_weights(w13, w2, layer_idx)
    if _first_install:
        from . import mxfp8_align as MX
        MX._record_scope(layer_idx, ("routed_fc1", "routed_fc2"))
    ar = _arena(T, top_k, E, N13, K, ffn, x.device)
    arrays = _device_arrays(wq, ar, E, N13, K, ffn)

    if not _ANNOUNCED:
        _ANNOUNCED = True
        hw = _hw_info()
        print("[parity] RAON_MOE_MXFP8_W8A8_GEMM_MODE=4 (swap-AB CUTLASS "
              f"grouped) LIVE: E={E} tiler={_MMA_TILER} cluster={_CLUSTER} "
              f"clusters={hw['max_active_clusters']} compact-R rows + "
              "triton glue x3, glue "
              "math=mode0 kernels "
              f"small_route={_SMALL_ROUTE} small_route_max_t={_SMALL_ROUTE_MAX_T} "
              f"route_pack_prefill={_ROUTE_PACK_PREFILL} "
              f"situ_epilogue={_SITU_EPILOGUE} "
              f"shape_map={_SHAPE_MAP or 'default'} "
              f"small_shape_map={_SMALL_SHAPE_MAP or 'default'} "
              f"prefill_shape_map={_PREFILL_SHAPE_MAP or 'default'} "
              f"glue_tune={_GLUE_TUNE or 'default'}", flush=True)

    # --- routing + dim/ptr contents (all on device) -------------------------
    # Latency-critical decode sizes (R <= 8192) use the device routing
    # kernels. Large R (prefill) uses aten routing or the deterministic blk
    # kernels: the single-CTA pack loops R/BLOCK times serially and is slower
    # there, while host dispatch of ~15 aten ops is not dominant at that scale.
    flat_ids = topk_ids.reshape(-1)
    small_route = _SMALL_ROUTE if T <= _SMALL_ROUTE_MAX_T else 0
    grouped_experts = E
    if small_route >= 2:
        from .mxfp8_small_route import prepare
        if small_route == 3 and T == 1:
            source_arrays = arrays
            grouped_experts = top_k
            arrays = _device_arrays(wq, ar, grouped_experts, N13, K, ffn)
            prepare(x, topk_weights, topk_ids, ar, arrays, source_arrays=source_arrays)
        else:
            prepare(x, topk_weights, topk_ids, ar, arrays)
    elif small_route == 1:
        _swab_route_pack[(1,)](
            flat_ids, arrays["fc1"]["dim_torch"], arrays["fc2"]["dim_torch"],
            arrays["fc1"]["pabc_torch"], arrays["fc2"]["pabc_torch"],
            ar["b1"].data_ptr(), ar["c1"].data_ptr(),
            ar["b2"].data_ptr(), ar["c2"].data_ptr(),
            K, N13 * 4, ffn, K * 2,
            ar["pos_for_slot"], ar["src_tok"], ar["src_slot"], ar["sf_row"],
            R, ar["rmx128"], E=E, TOPK=top_k,
            BLOCK=triton.next_power_of_2(R), num_warps=4)
    elif R <= 8192:
        # Parallel count -> scan (1 CTA, E) -> place; faster than the serial
        # single-CTA _swab_route_pack.
        _swab_route_count[(_ceil_div(R, 1024),)](
            flat_ids, ar["route_counts"], R, BLOCK=1024, num_warps=4)
        _swab_route_scan[(1,)](
            ar["route_counts"], ar["route_cursor"],
            arrays["fc1"]["dim_torch"], arrays["fc2"]["dim_torch"],
            arrays["fc1"]["pabc_torch"], arrays["fc2"]["pabc_torch"],
            ar["b1"].data_ptr(), ar["c1"].data_ptr(),
            ar["b2"].data_ptr(), ar["c2"].data_ptr(),
            K, N13 * 4, ffn, K * 2,
            ar["route_starts"], E=E, num_warps=4)
        _swab_route_place[(_ceil_div(R, 1024),)](
            flat_ids, ar["route_cursor"], ar["route_starts"],
            ar["pos_for_slot"], ar["src_tok"], ar["src_slot"], ar["sf_row"],
            R, ar["rmx128"], TOPK=top_k, BLOCK=1024, num_warps=4)
    elif _ROUTE_PACK_PREFILL:
        # Deterministic kernels; metadata/placement match the aten path below
        # exactly. No atomics, so every run gives the same result.
        n_blk = _ceil_div(R, _ROUTE_BLK)
        _swab_route_count_blk[(n_blk,)](
            flat_ids, ar["route_counts_blk"], R,
            E=E, BLOCK=_ROUTE_BLK, num_warps=4)
        _swab_route_scan_blk[(1,)](
            ar["route_counts_blk"], ar["route_base_blk"],
            arrays["fc1"]["dim_torch"], arrays["fc2"]["dim_torch"],
            arrays["fc1"]["pabc_torch"], arrays["fc2"]["pabc_torch"],
            ar["b1"].data_ptr(), ar["c1"].data_ptr(),
            ar["b2"].data_ptr(), ar["c2"].data_ptr(),
            K, N13 * 4, ffn, K * 2,
            ar["route_starts"], n_blk, E=E, num_warps=4)
        _swab_route_place_blk[(n_blk,)](
            flat_ids, ar["route_base_blk"], ar["route_starts"],
            ar["pos_for_slot"], ar["src_tok"], ar["src_slot"], ar["sf_row"],
            R, ar["rmx128"], E=E, TOPK=top_k, BLOCK=_ROUTE_BLK, num_warps=4)
    else:
        flat_e = flat_ids.to(torch.int64)
        order = torch.argsort(flat_e, stable=True)
        sorted_e = flat_e[order]
        counts = torch.bincount(flat_e, minlength=E)
        starts = torch.cumsum(counts, 0) - counts
        aranger = torch.arange(R, device=x.device)
        rank = aranger - starts[sorted_e]
        ar["pos_for_slot"][order] = aranger
        ar["src_tok"].copy_((order // top_k).to(torch.int32))
        ar["src_slot"].copy_(order.to(torch.int32))
        ar["sf_row"].copy_(sorted_e * ar["rmx128"] + rank)
        counts32 = counts.to(torch.int32)
        arrays["fc1"]["dim_torch"][:, 1] = counts32
        arrays["fc2"]["dim_torch"][:, 1] = counts32
        arrays["fc1"]["pabc_torch"][:, 1] = ar["b1"].data_ptr() + starts * K
        arrays["fc1"]["pabc_torch"][:, 2] = (ar["c1"].data_ptr()
                                             + starts * (N13 * 4))
        arrays["fc2"]["pabc_torch"][:, 1] = (ar["b2"].data_ptr()
                                             + starts * ffn)
        arrays["fc2"]["pabc_torch"][:, 2] = (ar["c2"].data_ptr()
                                             + starts * (K * 2))

    # --- a1 quantization + gather + SF write + prob -------------------------
    if small_route >= 2:
        pass  # The fused routing kernel already wrote activations and scales.
    elif R <= 8192:
        # All four steps in one launch (mode-0 quantizer arithmetic; block-local,
        # so byte-identical to quantizing T rows and then copying).
        _swab_qgather_a1[(R,)](
            x, topk_weights.reshape(-1),
            ar["b1"], ar["sfb1_mma"], ar["probs"],
            ar["src_tok"], ar["src_slot"], ar["sf_row"],
            K, K // 32, ar["rmx128"], ar["rk1"], ar["slab1"],
            x.stride(0), x.stride(1),
            BLOCK_K=1024, num_warps=4,
        )
    else:
        qb, qw = _GLUE_TUNE.get("qrows", (512, 0))
        a1_q, a1_s = TRI.quantize_mxfp8_rows(
            x, block_k=qb, num_warps=qw or None)        # (T,K), (T,K/32)
        gb, gw = _GLUE_TUNE.get("gather", (1024, 4))
        _swab_gather_a1[(R,)](
            a1_q.view(torch.uint8), a1_s, topk_weights.reshape(-1),
            ar["b1"].view(torch.uint8), ar["sfb1_mma"], ar["probs"],
            ar["src_tok"], ar["src_slot"], ar["sf_row"],
            K, K // 32, ar["rmx128"], ar["rk1"], ar["slab1"],
            BLOCK_K=gb, SF_BLOCK=128, num_warps=gw,
        )

    # --- GEMM13 (grouped, C=fp32) -------------------------------------------
    if _SITU_EPILOGUE:
        # fp32 accumulator -> SiTU/prob/a2 quant in the epilogue (b2, SFB2).
        _launch_situ(grouped_experts, x.device, arrays, ffn,
                     prefill=R > 8192, small=T <= _SMALL_ROUTE_MAX_T)
    else:
        _launch("f32", grouped_experts, x.device, arrays["fc1"], prefill=R > 8192, small=T <= _SMALL_ROUTE_MAX_T)

    # --- SiTU + prob fold + a2 quantization (mode-0 kernel, R rows) ---------
    if _SITU_EPILOGUE:
        pass
    elif R <= 8192:
        # Write the SF directly into the mma arena, saving the scatter launch
        # (same values, only the store address differs).
        TRI.situ_probfold_quant_mxfp8(
            ar["c1"], ar["probs"], q_out=ar["b2"],
            sf_mma_out=ar["sfb2_mma"], sf_row=ar["sf_row"],
            sf_mma_geom=(ar["rmx128"], ar["rk2"], ar["slab2"]))
    else:
        pb, pw = _GLUE_TUNE.get("probfold", (2048, 0))
        TRI.situ_probfold_quant_mxfp8(ar["c1"], ar["probs"],
                                      q_out=ar["b2"], s_out=ar["a2s"],
                                      block_i=pb, num_warps=pw or None)
        sb, sw = _GLUE_TUNE.get("scatter", (64, 1))
        _swab_scatter_sf[(R,)](
            ar["a2s"], ar["sfb2_mma"], ar["sf_row"],
            ffn // 32, ar["rmx128"], ar["rk2"], ar["slab2"],
            SF_BLOCK=sb, num_warps=sw,
        )

    # --- GEMM2 (grouped, C=bf16) --------------------------------------------
    _launch("bf16", grouped_experts, x.device, arrays["fc2"], prefill=R > 8192, small=T <= _SMALL_ROUTE_MAX_T)

    # --- finalize: return (T, H) directly ------------------------------------
    # Matches moe_align.combine_unweighted_fp32 element-wise (expert-ascending
    # fp32 sum + one bf16 cast).
    if do_finalize:
        y = torch.empty(T, K, dtype=x.dtype, device=x.device)
        kp2 = 1
        while kp2 < top_k:
            kp2 *= 2
        _swab_finalize_combine[(T, _ceil_div(K, 1024))](
            ar["c2"], ar["pos_for_slot"], y, K,
            TOPK=top_k, KP2=kp2, BLOCK_H=1024, num_warps=4)
        return y

    # --- mode-4 EP2 fused finalize (RAON_MOE_COMBINE_EP_FINALIZE): folds the
    # inverse gather to (T,K,H) and the model's EP combine tail (argsort +
    # re-gather) into one kernel over the compact rows. Reuses
    # moe_ep_kernel.kernel_ep2_finalize_expertmajor with SRC=c2 (compact,
    # R x H bf16) and MAP=pos_for_slot (the inverse-gather index, so the loaded
    # bytes are identical); the accumulation order matches the existing EP
    # kernel. The first call cross-checks against [inverse gather +
    # combine_ep_partial_fp32] with torch.equal (fail-closed).
    if ep_finalize_local is not None:
        global _EP_FINALIZE_XCHECKED
        from . import moe_ep_kernel as MEK
        ids_c = topk_ids.contiguous()
        order = torch.argsort(ids_c, dim=-1, stable=True)
        fused = MEK.finalize_ep2_expertmajor(
            ar["c2"], ar["pos_for_slot"], ids_c, ep_finalize_local,
            order=order)
        if not _EP_FINALIZE_XCHECKED:
            _EP_FINALIZE_XCHECKED = True
            from . import moe_align as MA
            rows_ref = ar["c2"][ar["pos_for_slot"]].view(T, top_k, K)
            ref = MA.combine_ep_partial_fp32(
                rows_ref.contiguous(), ids_c, ep_size=2,
                num_local_experts=ep_finalize_local,
                out_dtype=torch.bfloat16)
            if not torch.equal(fused, ref):
                d = (fused.float() - ref.float()).abs()
                raise RuntimeError(
                    "RAON_MOE_COMBINE_EP_FINALIZE=1 (swab FIN0) live "
                    "cross-check FAILED: fused ep2-finalize kernel != "
                    f"reference tail on a {tuple(x.shape)} input -- "
                    f"{int((fused != ref).sum())}/{ref.numel()} elements, "
                    f"max|d|={d.max().item():.6e}. Refusing the fused "
                    "plumbing; set RAON_MOE_COMBINE_EP_FINALIZE=0.")
            _single = MA.combine_unweighted_fp32(
                rows_ref, ids_c, out_dtype=torch.bfloat16)
            _d = (fused.float() - _single.float()).abs()
            print(
                "[parity] RAON_MOE_COMBINE_EP_FINALIZE=1 LIVE: swab-FIN0 "
                "tail (compact-row unpermute + EP2 combine) absorbed into "
                f"ONE kernel on a {tuple(x.shape)} input -- cross-check "
                f"PASS (torch.equal vs reference tail, {fused.numel()} "
                f"elements); EP boundary live: vs single-stage contract "
                f"max|d|={_d.max().item():.6e} over "
                f"{int((_d > 0).sum())}/{_d.numel()} differing elements",
                flush=True)
        return fused

    # --- inverse gather: compact rows -> (T, top_k, H) ----------------------
    out = ar["c2"][ar["pos_for_slot"]].view(T, top_k, K)
    return out

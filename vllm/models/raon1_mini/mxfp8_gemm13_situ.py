# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2026, KRAFTON. Derived work: __call__ and kernel() below are
# copies of the vendored donor (cutlass-ex92-grouped-blockscaled,
# grouped_blockscaled_gemm.py lines 362-1681, CUTLASS v4.5.0) with ONLY the
# epilogue replaced. Mainloop, TMA, MMA, scheduler and pipelines are unchanged.
"""RAON_SWAB_SITU_EPILOGUE — swap-AB grouped MXFP8 GEMM13 with the SiTU +
router-prob fold + MXFP8 a2 quantization fused into the epilogue.

The fp32 accumulator never leaves the SM: each 128x(N) accumulator tile is
read from TMEM, staged through shared memory, and turned directly into the
e4m3 GEMM2 input rows (b2) plus their e8m0 scales written into the GEMM2 SFB
mma arena. No fp32 GEMM13 output tensor exists.

Layout contract (weights prepared by mxfp8_w8a8_cutlass._interleave_fc1):
each 128-row M tile t of the per-expert fc1 weight holds gate channels
[64t, 64t+64) in rows 0..63 and the matching value channels in rows
64..127. Row permutation does not change any per-row GEMM result (every
output element is the same K-reduction of its own weight row), so the
accumulator values are those of the unpermuted GEMM.

Numerics: the per-element arithmetic is the PTX instruction sequence the
production Triton kernel (_situ_probfold_quant_kernel with tanh_approx=1,
sig_via_tanh=1, prob_bf16=1, a2_scale_mode=0, a2_rcp_approx=0) compiles to,
emitted as inline PTX with explicit .rn rounding (no contraction):
  out = (100*tanh(g*0.25)) * (tanh(v*0.04) * ((1+tanh(g*0.5))*0.5)) * bf16(p)
  scale = TE float_to_e8m0(amax32(|out|) * fp32(1/448)),
  q = cvt.rn.satfinite.e4m3(out * exp2f_rcp(scale)).
The 1x32 block runs along FFN (= the M/TMEM-lane axis in swap-AB), so one
warp owns exactly one 32-channel block per token and the amax is a single
redux.sync.max.abs.f32 (exact; order free).
"""
from typing import Type, Tuple, Union

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass._mlir.dialects import arith, llvm

from .mxfp8_w8a8_cutlass import _load_donor

Sm100GroupedBlockScaledGemmKernel = _load_donor().Sm100GroupedBlockScaledGemmKernel


# Per-element arithmetic as explicit PTX. Each block reproduces the instruction
# sequence of the production Triton kernel (see the module docstring); .rn on
# every mul/add forbids FMA contraction, matching the uncontracted Triton PTX.
def _asm_f32(asm, constraints, *args):
    return cutlass.Float32(llvm.inline_asm(
        cutlass.Float32.mlir_type, [a.ir_value() for a in args], asm, constraints,
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


def _asm_i32(asm, constraints, *args):
    return cutlass.Int32(llvm.inline_asm(
        cutlass.Int32.mlir_type, [a.ir_value() for a in args], asm, constraints,
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


def _select(pred, a, b):
    """pred ? a : b on Float32 (no arithmetic, bit-exact)."""
    return cutlass.Float32(arith.select(pred.ir_value(), cutlass.Float32(a).ir_value(),
                                        cutlass.Float32(b).ir_value()))


def _bf16_round(p):
    """PROB_BF16: w.to(bfloat16).to(float32)."""
    return _asm_f32(
        "{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, $1;\n\tcvt.f32.bf16 $0, h;\n\t}",
        "=f,f", cutlass.Float32(p))


def _situ(g, v, wb):
    """(100*tanh(g*0.25)) * (tanh(v*0.04) * ((1 + tanh(g*0.5)) * 0.5)) * wb."""
    return _asm_f32(
        "{\n\t.reg .f32 tg, tv, ts;\n\t"
        "mul.rn.f32 tg, $1, 0f3E800000;\n\ttanh.approx.f32 tg, tg;\n\t"
        "mul.rn.f32 tv, $2, 0f3D23D70A;\n\ttanh.approx.f32 tv, tv;\n\t"
        "mul.rn.f32 ts, $1, 0f3F000000;\n\ttanh.approx.f32 ts, ts;\n\t"
        "add.rn.f32 ts, ts, 0f3F800000;\n\tmul.rn.f32 ts, ts, 0f3F000000;\n\t"
        "mul.rn.f32 tg, tg, 0f42C80000;\n\tmul.rn.f32 tv, tv, ts;\n\t"
        "mul.rn.f32 tg, tg, tv;\n\tmul.rn.f32 $0, tg, $3;\n\t}",
        "=f,f,f,f", cutlass.Float32(g), cutlass.Float32(v), cutlass.Float32(wb))


def _float_to_e8m0(amax):
    """TE float_to_e8m0(amax * fp32(1/448)) — mxfp8_w8a8_triton._float_to_e8m0."""
    return _asm_i32(
        "{\n\t.reg .f32 x;\n\t.reg .b32 b, ex, mn, r, t;\n\t.reg .pred pu, pz, pi, pn;\n\t"
        "mul.rn.f32 x, $1, 0f3B124925;\n\tmov.b32 b, x;\n\t"
        "shr.u32 ex, b, 23;\n\tand.b32 ex, ex, 255;\n\tand.b32 mn, b, 8388607;\n\t"
        "setp.ne.u32 pu, mn, 0;\n\tsetp.ne.and.u32 pu, ex, 254, pu;\n\t"
        "setp.eq.u32 pz, ex, 0;\n\tsetp.le.and.u32 pz, mn, 4194304, pz;\n\t"
        "selp.u32 r, 0, 1, pz;\n\tselp.u32 r, r, 0, pu;\n\tadd.u32 r, ex, r;\n\t"
        "setp.ne.u32 pn, mn, 0;\n\tselp.u32 t, 255, 254, pn;\n\t"
        "setp.eq.u32 pi, ex, 255;\n\tselp.u32 $0, t, r, pi;\n\t}",
        "=r,f", cutlass.Float32(amax))


def _exp2f_rcp(e):
    """TE exp2f_rcp: 2^(127-e); e==254 -> 2^-127 denormal, e==255 -> NaN bits."""
    return _asm_f32(
        "{\n\t.reg .b32 r;\n\t.reg .pred p;\n\t"
        "sub.u32 r, 254, $1;\n\tshl.b32 r, r, 23;\n\t"
        "setp.eq.u32 p, $1, 254;\n\tselp.b32 r, 4194304, r, p;\n\t"
        "setp.eq.u32 p, $1, 255;\n\tselp.b32 r, 2147483647, r, p;\n\t"
        "mov.b32 $0, r;\n\t}",
        "=f,r", cutlass.Int32(e))


def _quant_e4m3(out, rcp):
    """cvt.rn.satfinite.e4m3(out * rcp) in the low byte."""
    return _asm_i32(
        "{\n\t.reg .f32 y;\n\t.reg .b16 h;\n\t"
        "mul.rn.f32 y, $1, $2;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h, 0f00000000, y;\n\tcvt.u32.u16 $0, h;\n\t}",
        "=r,f,f", cutlass.Float32(out), cutlass.Float32(rcp))


class Sm100GroupedGemm13SituKernel(Sm100GroupedBlockScaledGemmKernel):
    def __init__(self, sf_vec_size, mma_tiler_mn, cluster_shape_mn, ffn):
        super().__init__(sf_vec_size, mma_tiler_mn, cluster_shape_mn)
        if mma_tiler_mn[0] != 128:
            raise ValueError("SiTU epilogue needs 1-CTA M=128 tiles (64 gate + 64 value rows)")
        if ffn % 64:
            raise ValueError(f"ffn must be a multiple of 64, got {ffn}")
        self.ffn = ffn

    @cute.jit
    def __call__(
        self,
        initial_a: cute.Tensor,
        initial_b: cute.Tensor,
        initial_c: cute.Tensor,
        initial_sfa: cute.Tensor,
        initial_sfb: cute.Tensor,
        group_count: cutlass.Constexpr[int],
        problem_shape_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        tensor_address_sfasfb: cute.Tensor,
        tensor_address_abc2: cute.Tensor,
        tensor_address_sfasfb2: cute.Tensor,
        out_q: cute.Tensor,
        out_sf: cute.Tensor,
        probs: cute.Tensor,
        total_num_clusters: cutlass.Constexpr[int],
        tensormap_cute_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        For grouped GEMM, tensor shapes, tensor strides, and tensor address are all provided
        by different tensors in global memory. The "initial" tensors only carry data type and
        majorness information.

        :param initial_a: Initial tensor A, used for data type and majorness information.
        :type initial_a: cute.Tensor
        :param initial_b: Initial tensor B, used for data type and majorness information.
        :type initial_b: cute.Tensor
        :param initial_c: Initial tensor C, used for data type and majorness information.
        :type initial_c: cute.Tensor
        :param initial_sfa: Initial tensor SFA, used for data type and majorness information.
        :type initial_sfa: cute.Tensor
        :param initial_sfb: Initial tensor SFB, used for data type and majorness information.
        :type initial_sfb: cute.Tensor
        :param group_count: The number of GEMM groups.
        :type group_count: cutlass.Constexpr[int]
        :param problem_shape_mnkl: Tensor containing the (M, N, K, L) shape for each group.
        :type problem_shape_mnkl: cute.Tensor
        :param strides_abc: Tensor containing the strides for A, B, and C for each group.
        :type strides_abc: cute.Tensor
        :param tensor_address_abc: Tensor containing the base addresses for A, B, and C for each group.
        :type tensor_address_abc: cute.Tensor
        :param tensor_address_sfasfb: Tensor containing the base addresses for SFA and SFB for each group.
        :type tensor_address_sfasfb: cute.Tensor
        :param total_num_clusters: Total number of clusters needed for all groups.
        :type total_num_clusters: cutlass.Constexpr[int]
        :param tensormap_cute_tensor: Tensor for storing tensormaps.
        :type tensormap_cute_tensor: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr[int]
        :param stream: CUDA stream for asynchronous execution.
        :type stream: cuda.CUstream
        :raises TypeError: If A and B data types do not match.
        """
        self.a_dtype = initial_a.element_type
        self.b_dtype = initial_b.element_type
        self.sf_dtype = initial_sfa.element_type
        self.c_dtype = initial_c.element_type
        self.is_nvfp4_output = self.c_dtype is cutlass.Float4E2M1FN
        self.a_major_mode = utils.LayoutEnum.from_tensor(initial_a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(initial_b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(initial_c)
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()
        # The epilogue needs every subtile to span the full 128-row M tile so
        # a gate row and its value row (r, r+64) are in the same subtile.
        epi_m = cute.size(self.epi_tile[0])
        self.epi_n = cute.size(self.epi_tile[1])
        if cutlass.const_expr(epi_m != 128 or self.epi_n % 2):
            raise ValueError(f"SiTU epilogue requires epi tile (128, even N), got {self.epi_tile}")
        if cutlass.const_expr(
            2 * self.epi_n * 128 > cute.cosize(self.c_smem_layout_staged.outer)
        ):
            raise ValueError("SiTU epilogue exchange buffer exceeds the C staging budget")

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            initial_a.shape, self.sf_vec_size
        )
        initial_sfa = cute.make_tensor(initial_sfa.iterator, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            initial_b.shape, self.sf_vec_size
        )
        initial_sfb = cute.make_tensor(initial_sfb.iterator, sfb_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            initial_a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            initial_b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFA
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            initial_sfa,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            initial_sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # Setup TMA store for C
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            initial_c,
            epi_smem_layout,
            self.epi_tile,
        )

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            total_num_clusters, self.cluster_shape_mn, max_active_clusters
        )

        self.buffer_align_bytes = 1024
        self.size_tensormap_in_i64 = (
            Sm100GroupedBlockScaledGemmKernel.num_tensormaps
            * Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap
            // 8
        )

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            tensormap_buffer: cute.struct.MemRange[
                cutlass.Int64, self.size_tensormap_in_i64
            ]
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sEpi: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, 2 * self.epi_n * 128],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            group_count,
            problem_shape_mnkl,
            strides_abc,
            tensor_address_abc,
            tensor_address_sfasfb,
            tensormap_cute_tensor,
            tensor_address_abc2,
            tensor_address_sfasfb2,
            out_q,
            out_sf,
            probs,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        group_count: cutlass.Constexpr,
        problem_sizes_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        ptrs_abc: cute.Tensor,
        ptrs_sfasfb: cute.Tensor,
        tensormaps: cute.Tensor,
        ptrs_abc2: cute.Tensor,
        ptrs_sfasfb2: cute.Tensor,
        mQ: cute.Tensor,
        mSF: cute.Tensor,
        mProbs: cute.Tensor,
    ):
        """
        GPU device kernel performing the grouped GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if warp_idx == self.tma_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfa)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfb)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: tensormap buffer, a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tensormap_smem_ptr = storage.tensormap_buffer.data_ptr()
        tensormap_a_smem_ptr = tensormap_smem_ptr
        tensormap_b_smem_ptr = (
            tensormap_a_smem_ptr
            + Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8
        )
        tensormap_sfa_smem_ptr = (
            tensormap_b_smem_ptr
            + Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8
        )
        tensormap_sfb_smem_ptr = (
            tensormap_sfa_smem_ptr
            + Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8
        )
        tensormap_c_smem_ptr = (
            tensormap_sfb_smem_ptr
            + Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8
        )

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar.ptr
        tmem_holding_buf_ptr = storage.tmem_holding_buf.ptr

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Tensor memory dealloc barrier init
        if use_2cta_instrs:
            if warp_idx == self.tma_warp_id:
                num_tmem_dealloc_threads = 32
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(
                        tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads
                    )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/SFA/SFB/C
        #
        # (2 buffers, EPI_N tokens, 128 channels) fp32 accumulator exchange
        sEpi = storage.sEpi.get_tensor(
            cute.make_layout(
                (2, self.epi_n, 128), stride=(self.epi_n * 128, 128, 1)
            )
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        #
        # Compute multicast mask for A/B/SFA/SFB buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #  TMA load scaled factor A partition_S/D
        sfa_cta_layout = a_cta_layout
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA load scaled factor B partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # Get tensormap buffer address
        #
        grid_dim = cute.arch.grid_dim()
        tensormap_workspace_idx = (
            bidz * grid_dim[1] * grid_dim[0] + bidy * grid_dim[0] + bidx
        )

        tensormap_manager = utils.TensorMapManager(
            utils.TensorMapUpdateMode.SMEM,
            Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap,
        )
        tensormap_a_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 0, None)].iterator
        )
        tensormap_b_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 1, None)].iterator
        )
        tensormap_sfa_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 2, None)].iterator
        )
        tensormap_sfb_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 3, None)].iterator
        )
        tensormap_c_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 4, None)].iterator
        )

        #
        # Persistent tile scheduling loop
        #
        # When the problem shapes are on device, we launch one CTA per SM.
        # The if condition later prevents the warps from extra CTAs from doing any work.
        tile_sched = utils.StaticPersistentGroupTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            grid_dim,
            self.cluster_tile_shape_mnk,
            utils.create_initial_search_state(),
            group_count,
            problem_sizes_mnkl,
        )
        initial_work_tile_info = tile_sched.initial_work_tile_info()

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id and initial_work_tile_info.is_valid_tile:
            #
            # Persistent tile scheduling loop
            #
            work_tile = initial_work_tile_info

            tensormap_init_done = cutlass.Boolean(False)
            # group index of last tile
            last_group_idx = cutlass.Int32(-1)

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            while work_tile.is_valid_tile:
                grouped_gemm_cta_tile_info = work_tile.group_search_result
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                is_k_tile_cnt_zero = cur_k_tile_cnt == 0
                # Do not load any data if cur_k_tile_cnt is 0
                if not is_k_tile_cnt_zero:
                    is_group_changed = cur_group_idx != last_group_idx
                    # skip tensormap update if we're working on the same group
                    if is_group_changed:
                        real_tensor_a = self.make_tensor_abc_for_tensormap_update(
                            cur_group_idx,
                            self.a_dtype,
                            (
                                grouped_gemm_cta_tile_info.problem_shape_m,
                                grouped_gemm_cta_tile_info.problem_shape_n,
                                grouped_gemm_cta_tile_info.problem_shape_k,
                            ),
                            strides_abc,
                            ptrs_abc,
                            0,  # 0 for tensor A
                        )
                        real_tensor_b = self.make_tensor_abc_for_tensormap_update(
                            cur_group_idx,
                            self.b_dtype,
                            (
                                grouped_gemm_cta_tile_info.problem_shape_m,
                                grouped_gemm_cta_tile_info.problem_shape_n,
                                grouped_gemm_cta_tile_info.problem_shape_k,
                            ),
                            strides_abc,
                            ptrs_abc,
                            1,  # 1 for tensor B
                        )
                        real_tensor_sfa = self.make_tensor_sfasfb_for_tensormap_update(
                            cur_group_idx,
                            self.sf_dtype,
                            (
                                grouped_gemm_cta_tile_info.problem_shape_m,
                                grouped_gemm_cta_tile_info.problem_shape_n,
                                grouped_gemm_cta_tile_info.problem_shape_k,
                            ),
                            ptrs_sfasfb,
                            0,  # 0 for tensor SFA
                        )
                        real_tensor_sfb = self.make_tensor_sfasfb_for_tensormap_update(
                            cur_group_idx,
                            self.sf_dtype,
                            (
                                grouped_gemm_cta_tile_info.problem_shape_m,
                                grouped_gemm_cta_tile_info.problem_shape_n,
                                grouped_gemm_cta_tile_info.problem_shape_k,
                            ),
                            ptrs_sfasfb,
                            1,  # 1 for tensor SFB
                        )
                        if not tensormap_init_done:
                            # wait tensormap initialization complete
                            self.tensormap_ab_init_barrier.arrive_and_wait()
                            tensormap_init_done = True

                        tensormap_manager.update_tensormap(
                            (
                                real_tensor_a,
                                real_tensor_b,
                                real_tensor_sfa,
                                real_tensor_sfb,
                            ),
                            (tma_atom_a, tma_atom_b, tma_atom_sfa, tma_atom_sfb),
                            (
                                tensormap_a_gmem_ptr,
                                tensormap_b_gmem_ptr,
                                tensormap_sfa_gmem_ptr,
                                tensormap_sfb_gmem_ptr,
                            ),
                            self.tma_warp_id,
                            (
                                tensormap_a_smem_ptr,
                                tensormap_b_smem_ptr,
                                tensormap_sfa_smem_ptr,
                                tensormap_sfb_smem_ptr,
                            ),
                        )

                    mma_tile_coord_mnl = (
                        grouped_gemm_cta_tile_info.cta_tile_idx_m
                        // cute.size(tiled_mma.thr_id.shape),
                        grouped_gemm_cta_tile_info.cta_tile_idx_n,
                        0,
                    )

                    #
                    # Slice to per mma tile index
                    #
                    # ((atom_v, rest_v), RestK)
                    tAgA_slice = tAgA[
                        (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                    ]
                    # ((atom_v, rest_v), RestK)
                    tBgB_slice = tBgB[
                        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                    ]

                    # ((atom_v, rest_v), RestK)
                    tAgSFA_slice = tAgSFA[
                        (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                    ]
                    # ((atom_v, rest_v), RestK)
                    tBgSFB_slice = tBgSFB[
                        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                    ]

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                    ab_producer_state.reset_count()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < cur_k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                    if is_group_changed:
                        tensormap_manager.fence_tensormap_update(tensormap_a_gmem_ptr)
                        tensormap_manager.fence_tensormap_update(tensormap_b_gmem_ptr)
                        tensormap_manager.fence_tensormap_update(tensormap_sfa_gmem_ptr)
                        tensormap_manager.fence_tensormap_update(tensormap_sfb_gmem_ptr)
                    #
                    # Tma load loop
                    #
                    for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                        # Conditionally wait for AB buffer empty
                        ab_pipeline.producer_acquire(
                            ab_producer_state, peek_ab_empty_status
                        )

                        # TMA load A/B/SFA/SFB
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, ab_producer_state.count)],
                            tAsA[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                ab_producer_state
                            ),
                            mcast_mask=a_full_mcast_mask,
                            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                tensormap_a_gmem_ptr,
                                cute.AddressSpace.generic,
                            ),
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_slice[(None, ab_producer_state.count)],
                            tBsB[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                ab_producer_state
                            ),
                            mcast_mask=b_full_mcast_mask,
                            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                tensormap_b_gmem_ptr,
                                cute.AddressSpace.generic,
                            ),
                        )
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_slice[(None, ab_producer_state.count)],
                            tAsSFA[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                ab_producer_state
                            ),
                            mcast_mask=sfa_full_mcast_mask,
                            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                tensormap_sfa_gmem_ptr,
                                cute.AddressSpace.generic,
                            ),
                        )
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB_slice[(None, ab_producer_state.count)],
                            tBsSFB[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                ab_producer_state
                            ),
                            mcast_mask=sfb_full_mcast_mask,
                            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                tensormap_sfb_gmem_ptr,
                                cute.AddressSpace.generic,
                            ),
                        )

                        # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                        ab_producer_state.advance()
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if ab_producer_state.count < cur_k_tile_cnt:
                            peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                ab_producer_state
                            )
                else:
                    if not tensormap_init_done:
                        # wait tensormap initialization complete
                        self.tensormap_ab_init_barrier.arrive_and_wait()
                        tensormap_init_done = True
                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx

            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id and initial_work_tile_info.is_valid_tile:
            #
            # Initialize tensormaps for A, B, SFA and SFB
            #
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_a, tensormap_a_smem_ptr, self.mma_warp_id
            )
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_b, tensormap_b_smem_ptr, self.mma_warp_id
            )
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_sfa, tensormap_sfa_smem_ptr, self.mma_warp_id
            )
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_sfb, tensormap_sfb_smem_ptr, self.mma_warp_id
            )
            # indicate tensormap initialization has finished
            self.tensormap_ab_init_barrier.arrive_and_wait()

            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            self.tmem_alloc_barrier.arrive_and_wait()

            #
            # Retrieving tensor memory ptr and make accumulator/SFA/SFB tensor
            #
            # Make accumulator tmem tensor
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf_ptr,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            #
            # Partition for S2T copy of SFA/SFB
            #
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            )
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
            )

            #
            # Persistent tile scheduling loop
            #
            work_tile = initial_work_tile_info

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            while work_tile.is_valid_tile:
                cur_group_idx = work_tile.group_search_result.group_idx
                problem_shape_k = work_tile.group_search_result.problem_shape_k

                # MMA warp is only interested in number of tiles along K dimension
                cur_k_tile_cnt = (
                    problem_shape_k + self.cluster_tile_shape_mnk[2] - 1
                ) // self.cluster_tile_shape_mnk[2]
                is_k_tile_cnt_zero = cur_k_tile_cnt == 0

                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < cur_k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta and not is_k_tile_cnt_zero:
                    acc_pipeline.producer_acquire(acc_producer_state)

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for k_tile in range(cur_k_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        #  Copy SFA/SFB from smem to tmem
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

                            # Set SFA/SFB tensor to tiled_mma
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kblock_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB[sf_kblock_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )

                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < cur_k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                #
                # Async arrive accumulator buffer full
                #
                if not is_k_tile_cnt_zero:
                    if is_leader_cta:
                        acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id and initial_work_tile_info.is_valid_tile:
            #
            # Alloc tensor memory buffer
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols,
                    tmem_holding_buf_ptr,
                    is_two_cta=use_2cta_instrs,
                )

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            self.tmem_alloc_barrier.arrive_and_wait()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf_ptr,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            #
            # Partition for epilogue (TMEM -> registers, as in the donor)
            #
            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
                self.epilog_tmem_copy_and_partition(
                    epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
                )
            )
            # Tile coordinates (m, n) of every register the t2r copy fills,
            # partitioned exactly like the accumulator (identity tensor).
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
            cC = cute.make_identity_tensor(
                (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1])
            )
            tCcC = thr_mma.partition_C(cC)
            cC_epi = cute.flat_divide(tCcC[((None, None), 0, 0)], epi_tile)
            tTR_cC = thr_copy_t2r.partition_D(cC_epi)
            tTR_cC = cute.group_modes(tTR_cC, 3, cute.rank(tTR_cC))

            # Phase-B ownership: warp w owns 32-channel block (w % 2) of the
            # tile's 64 FFN channels (lane = channel), and token columns
            # [h*EPI_N/2, (h+1)*EPI_N/2) with h = w // 2 (balanced: each warp
            # consumes gate/value rows the t2r of another warp produced).
            lane = tidx % 32
            blk = warp_idx % 2
            half_n = self.epi_n // 2
            col0 = (warp_idx // 2) * half_n
            ch = blk * 32 + lane
            ffn = self.ffn
            rk2 = (ffn // 32 + 3) // 4
            q_base = mQ.iterator.toint()
            sf_base = mSF.iterator.toint()

            work_tile = initial_work_tile_info

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            epi_count = cutlass.Int32(0)

            while work_tile.is_valid_tile:
                grouped_gemm_cta_tile_info = work_tile.group_search_result
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                is_k_tile_cnt_zero = cur_k_tile_cnt == 0
                problem_n = grouped_gemm_cta_tile_info.problem_shape_n
                tile_m = grouped_gemm_cta_tile_info.cta_tile_idx_m // cute.size(
                    tiled_mma.thr_id.shape
                )
                tile_n0 = (
                    grouped_gemm_cta_tile_info.cta_tile_idx_n
                    * self.cta_tile_shape_mnk[1]
                )
                # GEMM2 operand rows of this group: b2 row base (pointer
                # array of the fc2 problem) and its SFB mma slab.
                row0 = cutlass.Int32(
                    (ptrs_abc2[(cur_group_idx, 1)] - q_base) // ffn
                )
                sf_off = cutlass.Int32(ptrs_sfasfb2[(cur_group_idx, 1)] - sf_base)
                ks = tile_m * 2 + blk
                sf_ks = (ks // 4) * 512 + (ks % 4)
                q_col = tile_m * 64 + ch

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_consumer_state.index)
                ]

                #
                # Wait for accumulator buffer full
                #
                if not is_k_tile_cnt_zero:
                    acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                # Router probs of this warp's columns for the whole tile, issued
                # up front (lane c holds column col0 + c of every subtile;
                # out-of-range columns read a clamped valid row and are never
                # stored). PROB_BF16 rounding applied once per column.
                n_last = problem_n - 1
                pw = cute.make_rmem_tensor(subtile_cnt, cutlass.Float32)
                for s in cutlass.range_constexpr(subtile_cnt):
                    n_p = tile_n0 + s * self.epi_n + col0 + lane % half_n
                    n_p = cutlass.min(n_p, n_last)
                    pw[s] = _bf16_round(mProbs[row0 + n_p])

                for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                    buf = epi_count % 2
                    # Phase A: TMEM -> registers -> sEpi[buf, n_local, m]
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    tTR_cC_mn = tTR_cC[(None, None, None, subtile_idx)]
                    if not is_k_tile_cnt_zero:
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                    else:
                        tTR_rAcc.fill(0.0)
                    for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                        m_i, n_i = tTR_cC_mn[i]
                        sEpi[(buf, n_i - subtile_idx * self.epi_n, m_i)] = tTR_rAcc[i]
                    self.epilog_sync_barrier.arrive_and_wait()

                    # Phase B: SiTU + prob fold + 1x32 MXFP8 quant per token.
                    # All columns are computed branch-free first (independent
                    # chains the scheduler can interleave); stores follow.
                    # The 32-channel amax of each column is warp-uniform: lane c
                    # collects column c's amax and computes its e8m0 scale and
                    # reciprocal once; columns read them back by shuffle.
                    ov = cute.make_rmem_tensor(half_n, cutlass.Float32)
                    a_l = cutlass.Float32(0.0)
                    for c in cutlass.range_constexpr(half_n):
                        g = sEpi[(buf, col0 + c, ch)]
                        v = sEpi[(buf, col0 + c, ch + 64)]
                        wb = cute.arch.shuffle_sync(pw[subtile_idx], c)
                        ov[c] = _situ(g, v, wb)
                        amax = cute.arch.warp_redux_sync(ov[c], kind="fmax", abs=True)
                        a_l = _select(lane == c, amax, a_l)
                    e_l = _float_to_e8m0(a_l)
                    r_l = _exp2f_rcp(e_l)
                    n_base = tile_n0 + subtile_idx * self.epi_n + col0
                    q_ptr0 = (row0 + n_base) * ffn + q_col
                    for c in cutlass.range_constexpr(half_n):
                        q = _quant_e4m3(ov[c], cute.arch.shuffle_sync(r_l, c))
                        if n_base + c < problem_n:
                            mQ[q_ptr0 + c * ffn] = cutlass.Uint8(q)
                    n = n_base + lane
                    if lane < half_n and n < problem_n:
                        sf_idx = (
                            sf_off
                            + (n // 128) * (512 * rk2)
                            + sf_ks
                            + (n % 32) * 16
                            + ((n // 32) % 4) * 4
                        )
                        mSF[sf_idx] = cutlass.Uint8(e_l)
                    epi_count += 1

                #
                # Async arrive accumulator buffer empty
                #
                if not is_k_tile_cnt_zero:
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Dealloc the tensor memory buffer
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            self.epilog_sync_barrier.arrive_and_wait()
            if warp_idx == self.epilog_warp_id[0]:
                if use_2cta_instrs:
                    cute.arch.mbarrier_arrive(
                        tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1
                    )
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(
                    acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs
                )

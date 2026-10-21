"""Ordered EP combine: preserve FP32 slot sums and BF16 rank boundaries."""
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(X,IDS,ORDER,Y,H:tl.constexpr,K:tl.constexpr,EP:tl.constexpr,LOCAL:tl.constexpr,B:tl.constexpr):
 t=tl.program_id(0); h=tl.program_id(1)*B+tl.arange(0,B)
 acc=tl.full((B,),0,tl.float32)
 for r in range(EP):
  part=tl.full((B,),0,tl.float32)
  for j in range(K):
   slot=tl.load(ORDER+t*K+j)
   eid=tl.load(IDS+t*K+slot)
   value=tl.load(X+(t*K+slot)*H+h,h<H,other=0).to(tl.float32)
   part=tl.where(eid//LOCAL==r,part+value,part)
  acc=acc+part.to(tl.bfloat16).to(tl.float32)
 tl.store(Y+t*H+h,acc.to(tl.bfloat16),h<H)

@triton.jit
def kernel_fastpath_ep2(X,IDS,ORDER,Y,H:tl.constexpr,K:tl.constexpr,LOCAL:tl.constexpr,B:tl.constexpr):
 # RAON_MOE_COMBINE_EP_FASTPATH (default off): single-pass EP=2 variant.
 # `kernel` reloads X once per rank; this loads each slot once and keeps both
 # rank partial sums in registers. The addition order matches the reference
 # (_combine_ep_partial_fp32_loop) exactly: each part grows only in
 # ascending-expert j order (unselected slots are left bit-unchanged by
 # tl.where), then bf16(part).float() is accumulated for r=0,1 in order.
 # Graph-safe: no host sync.
 t=tl.program_id(0); h=tl.program_id(1)*B+tl.arange(0,B)
 part0=tl.full((B,),0,tl.float32)
 part1=tl.full((B,),0,tl.float32)
 for j in range(K):
  slot=tl.load(ORDER+t*K+j)
  eid=tl.load(IDS+t*K+slot)
  value=tl.load(X+(t*K+slot)*H+h,h<H,other=0).to(tl.float32)
  is0=(eid//LOCAL)==0
  part0=tl.where(is0,part0+value,part0)
  part1=tl.where(is0,part1,part1+value)
 acc=part0.to(tl.bfloat16).to(tl.float32)+part1.to(tl.bfloat16).to(tl.float32)
 tl.store(Y+t*H+h,acc.to(tl.bfloat16),h<H)

@triton.jit
def kernel_unweighted(X,IDS,ORDER,Y,H:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
 # RAON_MOE_COMBINE_UNW_FUSED (default off): kernel version of
 # combine_unweighted_fp32. Same contract: one fp32 accumulator over slots in
 # ascending-expert order, single cast at the end.
 t=tl.program_id(0); h=tl.program_id(1)*B+tl.arange(0,B)
 acc=tl.full((B,),0,tl.float32)
 for j in range(K):
  slot=tl.load(ORDER+t*K+j)
  acc=acc+tl.load(X+(t*K+slot)*H+h,h<H,other=0).to(tl.float32)
 tl.store(Y+t*H+h,acc.to(tl.bfloat16),h<H)

def combine_unweighted(x,ids,order=None):
 assert x.is_cuda and x.dtype==torch.bfloat16 and x.is_contiguous()
 assert ids.is_contiguous() and ids.shape==x.shape[:2]
 if order is None:order=torch.argsort(ids,dim=-1,stable=True)
 t,k,h=x.shape;y=torch.empty((t,h),dtype=x.dtype,device=x.device)
 kernel_unweighted[(t,triton.cdiv(h,256))](x,ids,order,y,h,k,256,enable_fp_fusion=False)
 return y

@triton.jit
def kernel_ep2_finalize_permuted(SRC,MAP,W,IDS,ORDER,Y,H:tl.constexpr,K:tl.constexpr,LOCAL:tl.constexpr,B:tl.constexpr):
 # RAON_MOE_COMBINE_EP_FINALIZE (default off): fuses the mode2 (trtllm) rows
 # path around the combine -- index_select unpermute, prob fold, (T,K,H)
 # materialization and the EP2 combine re-gather -- into one kernel that
 # returns (T,H) directly from permuted gemm2 rows. Per-element arithmetic
 # matches the unfused path:
 #   fold: bf16(fp32(bf16 row) * fp32(bf16(fp32 prob))), same as
 #         (rows.float()*probs).to(bf16); invalid slots get +0.0 as with
 #         torch.where.
 #   combine: same as kernel_fastpath_ep2.
 # Graph-safe: no host sync.
 t=tl.program_id(0); h=tl.program_id(1)*B+tl.arange(0,B)
 part0=tl.full((B,),0,tl.float32)
 part1=tl.full((B,),0,tl.float32)
 for j in range(K):
  slot=tl.load(ORDER+t*K+j)
  eid=tl.load(IDS+t*K+slot)
  src=tl.load(MAP+t*K+slot)
  ok=src>=0
  row=tl.load(SRC+tl.maximum(src,0).to(tl.int64)*H+h,(h<H)&ok,other=0).to(tl.float32)
  p=tl.load(W+t*K+slot).to(tl.bfloat16).to(tl.float32)
  v=tl.where(ok,((row*p).to(tl.bfloat16)).to(tl.float32),0.0)
  is0=(eid//LOCAL)==0
  part0=tl.where(is0,part0+v,part0)
  part1=tl.where(is0,part1,part1+v)
 acc=part0.to(tl.bfloat16).to(tl.float32)+part1.to(tl.bfloat16).to(tl.float32)
 tl.store(Y+t*H+h,acc.to(tl.bfloat16),h<H)

def finalize_ep2_permuted(src,map_flat,weights,ids,num_local,order=None):
 assert src.is_cuda and src.dtype==torch.bfloat16 and src.is_contiguous()
 assert ids.is_contiguous() and weights.dtype==torch.float32
 t,k=ids.shape
 h=src.shape[-1]
 assert map_flat.numel()==t*k and weights.shape==(t,k)
 map_flat=map_flat.contiguous();weights=weights.contiguous()
 if order is None:order=torch.argsort(ids,dim=-1,stable=True)
 y=torch.empty((t,h),dtype=src.dtype,device=src.device)
 kernel_ep2_finalize_permuted[(t,triton.cdiv(h,256))](src,map_flat,weights,ids,order,y,h,k,num_local,256,enable_fp_fusion=False)
 return y

@triton.jit
def kernel_ep2_finalize_expertmajor(SRC,MAP,IDS,ORDER,Y,H:tl.constexpr,K:tl.constexpr,LOCAL:tl.constexpr,B:tl.constexpr):
 # Expert-major variant of kernel_ep2_finalize_permuted for the sorted/mxfp8
 # path: rows are already in expert-major order with probs folded in by the
 # CuTe epilogue, so there is no prob fold and no invalid-slot branch.
 # MAP is the inverse of sort_idx (row for token-slot i is SRC[MAP[i]]; all
 # slots are valid). Replaces the out_flat.index_copy_ unpermute, the (T,K,H)
 # materialization and the combine re-gather. The loaded bytes equal
 # expert_out[t*K+slot] by definition and accumulation matches
 # kernel_fastpath_ep2, so results are bitwise identical (data movement only).
 # Graph-safe: no host sync (callers of this path are already eager-only
 # because of a counts .tolist()).
 t=tl.program_id(0); h=tl.program_id(1)*B+tl.arange(0,B)
 part0=tl.full((B,),0,tl.float32)
 part1=tl.full((B,),0,tl.float32)
 for j in range(K):
  slot=tl.load(ORDER+t*K+j)
  eid=tl.load(IDS+t*K+slot)
  src=tl.load(MAP+t*K+slot)
  v=tl.load(SRC+src.to(tl.int64)*H+h,h<H,other=0).to(tl.float32)
  is0=(eid//LOCAL)==0
  part0=tl.where(is0,part0+v,part0)
  part1=tl.where(is0,part1,part1+v)
 acc=part0.to(tl.bfloat16).to(tl.float32)+part1.to(tl.bfloat16).to(tl.float32)
 tl.store(Y+t*H+h,acc.to(tl.bfloat16),h<H)

def finalize_ep2_expertmajor(src,inv_map,ids,num_local,order=None):
 assert src.is_cuda and src.dtype==torch.bfloat16 and src.is_contiguous()
 assert ids.is_contiguous()
 t,k=ids.shape
 h=src.shape[-1]
 assert inv_map.numel()==t*k and src.shape[0]==t*k
 inv_map=inv_map.contiguous()
 if order is None:order=torch.argsort(ids,dim=-1,stable=True)
 y=torch.empty((t,h),dtype=src.dtype,device=src.device)
 kernel_ep2_finalize_expertmajor[(t,triton.cdiv(h,256))](src,inv_map,ids,order,y,h,k,num_local,256,enable_fp_fusion=False)
 return y

def _fastpath_enabled():
 import os
 return os.environ.get("RAON_MOE_COMBINE_EP_FASTPATH","0")=="1"

def combine(x,ids,ep=8,local=16,order=None):
 assert x.is_cuda and x.dtype==torch.bfloat16 and x.is_contiguous()
 assert ids.is_contiguous() and ids.shape==x.shape[:2]
 if order is None:order=torch.argsort(ids,dim=-1,stable=True)
 t,k,h=x.shape;y=torch.empty((t,h),dtype=x.dtype,device=x.device)
 if ep==2 and _fastpath_enabled():
  kernel_fastpath_ep2[(t,triton.cdiv(h,256))](x,ids,order,y,h,k,local,256,enable_fp_fusion=False)
  return y
 kernel[(t,triton.cdiv(h,256))](x,ids,order,y,h,k,ep,local,256,enable_fp_fusion=False)
 return y

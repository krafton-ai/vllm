# Raon-1 nano runtime sources

The Nano adapter is based on the Raon-1 mini implementation in
[`krafton-ai/vllm`](https://github.com/krafton-ai/vllm/tree/335a45e3f9d8a7f5f76c884734b5797ec9623fb3/vllm/models/raon1_mini).

- `mlp.py` contains Mini's dense `RaonMLP`; MoE is excluded.
- `mla_attention.py` and `mla_kv_cache.py` derive from Mini's standalone MLA.
  Nano retains its latent normalization and graph-stable scale buffers.
- `rda_base.py`, `rda_metadata.py`, and `rda_recurrent.py` derive from Mini's
  RDA primitives. Mini's speculative-decoding warmup, state relabeling, and
  split-prefill hooks are excluded. The Nano MTP adapter uses the retained
  speculative recurrence/conv state path; see `MTP.md` for validation status.
- `rda.py` preserves Nano's write gate and numerical adapters. `nano.py`
  preserves per-layer embeddings, tied weights, and output normalization.

Nano does not import or patch another model's Python implementation.
The MLA wrappers retain `torch.ops._C.fused_kimi_k3_mla_*` names because those
are compiled vLLM operator interfaces. `is_deepseek_mla` is likewise a vLLM
configuration interface. The shared `raon_1` tool parser uses xgrammar's
`qwen_xml` schema style. These interfaces do not select those models.

## Kernel provenance

`ops/kda/chunk*.py` preserves Nano's existing optional Triton prefill and
chunk-replay kernels from the same base revision's
`vllm/models/kimi_k3/nvidia/ops/third_party/kda/`. Mini no longer includes this
path. The kernels originate in flash-linear-attention; their copyright
headers and MIT license in `ops/kda/LICENSE` are retained.
`rda_recurrent.py` also retains Mini's flash-linear-attention attribution.

`kda_sgl_fused.py` and `kda_sgl_fused_kernel.cu` retain the SGLang/NVIDIA/Moonshot
kernel provenance in their headers. `phi_direct_gemm_vendored.py` retains
FlashInfer/NVIDIA attribution. `batch_invariant.py` retains the VeXact and
Thinking Machines Lab attribution.

The RDA training-contract references originally named
`megatron/core/ssm/kimi_delta_attention.py::_hf_fla_training_forward` and
`::_kda_cpr_core`. They document the source of the write/erase ratio and
recurrence arithmetic, rather than a runtime model dependency.

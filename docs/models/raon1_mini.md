# Raon-1-mini

Raon-1-mini combines recurrent Raon Delta Attention (RDA), NoPE MLA,
four-stream mHC residuals, and SiTU MoE layers. This implementation reuses
vLLM's native MLA, hybrid cache, quantization, and speculative decoding interfaces.

## Serving

Build from the model-support PR checkout. The default B300 deployment follows
[the public model card](https://huggingface.co/KRAFTON/Raon-1-mini-preview):

```bash
VLLM_KV_CACHE_LAYOUT=BLNHC vllm serve KRAFTON/Raon-1-mini-preview \
    --max-model-len 262144 --max-num-batched-tokens 16384 --max-num-seqs 128 \
    --gpu-memory-utilization 0.85 --kv-cache-dtype fp8_e4m3 \
    --enable-prefix-caching \
    --speculative-config '{"method":"dflash","model":"KRAFTON/Raon-1-mini-preview-dflash2","num_speculative_tokens":7,"draft_sample_method":"probabilistic","kv_cache_dtype":"auto"}' \
    --reasoning-parser raon_1 \
    --enable-auto-tool-choice --tool-call-parser raon_1 \
    --default-chat-template-kwargs '{"enable_thinking":true,"truncate_history_thinking":false}'
```

Use `KRAFTON/Raon-1-mini-preview-nvfp4` for the NVFP4 target. Both targets use
vLLM's native ModelOpt implementations. No model-specific Python plugin is
required. Fused kernels require the CUDA Toolkit and a C++ compiler on first
startup; subsequent starts reuse the compiled kernels.

The [deployment recipe](../../recipes/models/KRAFTON/Raon-1-mini-preview.yaml)
contains the B300 defaults and separate NVFP4 resource profiles for RTX 5090
and DGX Spark. Candidate validation is separate from the released fork's
published results. Use temperature 1.0, top_p 0.95 and top_k 20; set
presence_penalty to 0 for coding/math and 1.5 for other tasks.

## Implementation and validation status

This branch is under development. CPU configuration and parser tests do not
qualify model outputs or performance. GPU loading, numerical comparisons,
quality evaluation, and serving benchmarks must pass before public submission.

- RDA applies its separate write/erase gate ratio after the value convolution;
  the rolling convolution history retains unscaled values.
- mHC preserves four residual streams. The model exposes their concatenated
  intermediate states for the native DFlash2 interface. DFlash CUDA graphs
  size the staging buffer using the draft's declared `target_hidden_size`.
  Target prefill runs eagerly; decode uses CUDA graphs.
- MXFP8 experts use vLLM's ModelOpt loader. The checkpoint's E8M0 scale tensor
  suffix `weight_scale_inv` maps to the native `weight_scale` parameter.
  MXFP8 and NVFP4 reuse existing upstream quantization backends. Compatibility
  with SiTU, the backend selected on the target GPU, and numerical and
  performance parity with production must be qualified for each format.
- The `raon_1` parser handles reasoning markers and XML function calls,
  including streamed arguments and request-schema type conversion.
- Fused mHC/RDA kernels are selected automatically on SM10x/SM12x at TP1
  for the published model shape, BF16 convolution history, FP32 recurrent
  state in SD layout, and at most seven speculative tokens. Other
  configurations use the native path. `--additional-config '{"raon_fast":false}'`
  disables the fused kernels; explicitly requesting `true` rejects unsupported
  configurations. SM12x candidate qualification is still pending.
- NVFP4, FP8 KV cache, prefix caching, and DFlash2 are part of the qualification
  scope. Their availability in upstream vLLM does not establish Raon correctness.
- Pipeline parallelism, expert parallelism, and RecoverSSM are unsupported.

Published checkpoints:

- [MXFP8 target](https://huggingface.co/KRAFTON/Raon-1-mini-preview)
- [NVFP4 target](https://huggingface.co/KRAFTON/Raon-1-mini-preview-nvfp4)
- [DFlash2 draft](https://huggingface.co/KRAFTON/Raon-1-mini-preview-dflash2)

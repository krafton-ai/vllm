# Nano MTP (experimental)

Speculative decoding is optional and OFF by default. The target checkpoint and
its root `model.safetensors.index.json` remain unchanged. ON explicitly loads
`mtp/mtp_config.json` and `mtp/model.safetensors` from the same local model
folder or Hugging Face repository/revision. Private repositories use the normal
Hugging Face authentication, including the target's explicit `hf_token`.

Add these flags to the normal Nano server command:

```bash
--enforce-eager \
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

OFF omits `--speculative-config`. The initial supported configuration is BF16,
TP=PP=1, one draft token and eager execution. Other speculative methods,
multiple draft tokens, other model dtypes, CUDA graph execution, quantization
and `RAON_KDA_CPR=1` are rejected.

ON adds one MLA+mHC layer. vLLM's native MTP proposer shares the target's input
embedding and LM head. The auxiliary block normalizes the shifted token
embedding and target hidden state, concatenates and projects them, runs a
four-stream decoder, averages streams in FP32 and applies its own final norm.
The target reserves speculative convolution history and recurrent snapshots;
vLLM's existing sampler verifies proposals and selects accepted states.

## Validation and limits

CPU contract tests live in
`tests/config/test_raon1_nano_mtp_config.py`; the model-loading example is in
`tests/models/registry.py`. The 35 contract tests cover supported configurations,
checkpoint metadata compatibility, tensor mapping and the actual safetensors
reader API. CUDA qualification uses the real released target/auxiliary pair
on NVIDIA B300, rather than dummy weights.

The qualification ran 35 CPU contract tests and five CUDA kernel scenarios,
including forced accept/reject restoration and a 32-step packed/verification
recurrence comparison. Real-weight HF/native MTP comparisons passed the stated
BF16 tolerance for one-token and eight-token causal prefill. Mixed arrivals,
request cancellation/reuse and the 32,768-token input/output boundary completed
in both modes. This does not qualify contexts above 32K.

Greedy ON/OFF output is not guaranteed to be bitwise identical. In an initial
four-request, 128-output-token test, three requests matched exactly; one first
differed at output token 66. At the same input token and position, OFF's two
leading BF16 logits were tied at 20.875, while ON produced 21.0 and 20.875.
A broader forced-length (`ignore_eos=True`) matrix matched all tokens for
14 of 23 requests; exact token equality is therefore not a passed criterion.
The initial example's observed one-ULP difference changed the selected token. Treat exact token
parity separately from successful generation, cache restoration and quality.

CUDA graph, TP/PP > 1, multi-token drafts and quantized execution remain outside
the supported configuration. Contexts above the measured range, original
training-runtime logit parity, distributed serving and a full benchmark suite
need separate qualification. Stochastic sampling smoke tests do not establish
statistical distribution equivalence.

Native vLLM speculative-sampler restrictions on `min_p` and `logit_bias` also
apply. Hot reload of the auxiliary module is not qualified.

The auxiliary weights come from SFT while the target includes later RL
updates. Acceptance and speedup depend on the checkpoint pair, prompt, batch
size and runtime settings; neither is assumed from the presence of MTP weights.

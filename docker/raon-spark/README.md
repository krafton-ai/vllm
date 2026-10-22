# Raon on DGX Spark

This ARM64 image runs the Raon preview fork on GB10 (SM 12.1). It uses the
pinned CUDA 13.0 vLLM base below, the fork's Python sources, and the matching
upstream ARM64 binary wheel. Model weights and credentials are mounted at runtime.

## Build

From a clean checkout of this branch on the Spark:

```bash
curl -fL --retry 5 \
  'https://wheels.vllm.ai/53e211d2923192ea1f7442de93d59cfbb150e089/vllm-0.27.2rc1.dev151%2Bg53e211d29-cp38-abi3-manylinux_2_28_aarch64.whl' \
  -o docker/raon-spark/precompiled.whl
docker build -f docker/raon-spark/Dockerfile \
  --build-arg VLLM_REF="$(git rev-parse HEAD)" \
  -t ghcr.io/krafton-ai/raon-vllm-spark:local .
```

The Dockerfile verifies the wheel's SHA-256. The base is pinned to
`sha256:2a7cde230b59f3ce6cab33dd245ba6bee41aa87b38c9fe84f966ff24016813ce`.
Its PyTorch is `2.13.0+cu130`; its existing NumPy `2.2.6` and NCCL `2.30.7`
are retained. The base itself overrides PyTorch's NCCL dependency pin, so
`python -m pip check` reports that inherited override. Additional Python
packages are pinned in `overlay.lock`.

The image contains no weights, Hugging Face token, or registry credentials.
The Dockerfile-specific ignore file restricts the build context to source and
build inputs. The final stage excludes the source tree and downloaded wheel.

## Serve

Download the chosen checkpoint into your Hugging Face cache first, then run:

```bash
docker run --rm --gpus all --ipc=host \
  --hostname raon-spark -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/hf" \
  -v raon-spark-cache:/cache \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -e MODEL_ID=KRAFTON/Raon-1-mini-preview-nvfp4 \
  -e MAX_MODEL_LEN=131072 \
  ghcr.io/krafton-ai/raon-vllm-spark:local
```

For reproducible runs, set `MODEL_ID` to a mounted snapshot directory:

| Model | Hugging Face revision |
| --- | --- |
| `KRAFTON/Raon-1-mini-preview-nvfp4` | `0b0f06ac2b3d03abea62dd221fe7839fddf36dfe` |
| `KRAFTON/Raon-1-mini-preview` (MXFP8) | `da4cf162c8613993f2189ebb438a5f9a23f3e5ca` |
| `KRAFTON/Raon-1-mini-preview-dflash2` | `6450d1546d17ee1bb1cb942f316c7fb91cdd868a` |

Enable DFlash with `-e SPEC_TOKENS=7` and set `DRAFT_MODEL_ID` to the draft
snapshot directory. The OpenAI-compatible API serves the name
`Raon-1-mini-preview`. Additional vLLM CLI arguments can follow the image name.

The launcher defaults to a 32K window, eight sequences, a 2,048-token scheduler
batch, and a 3 GiB FP8 KV cache. `MAX_MODEL_LEN=131072` enables the measured
128K window. The KV budget is independent of the context window; concurrent
long requests may queue. `KV_CACHE_BYTES`, `MAX_NUM_SEQS`, and
`MAX_BATCHED_TOKENS` can be adjusted explicitly.

Keep `/cache` persistent and use a stable hostname to reuse compiled kernels.
The first launch compiles model-specific CUDA extensions. Host port publication
above exposes the API on the host's network interfaces; bind `127.0.0.1:8000:8000`
when only local access is needed.

## Spark compatibility

SM 12.x uses Triton MLA and FlashAttention prefill. The KDA vector-split
extension compiles for SM 12.1, and its required CUDA sources are included in
the Python wheel. JIT compilation uses CUDA's runtime headers before the
pip-distributed cuBLAS/cuSPARSE headers through `CPLUS_INCLUDE_PATH`.

MXFP8 defaults to the existing sliced FP8 GEMM (mode 3) on SM 12.x because
the mode-4 CUTLASS implementation requires SM 10.x/11.x instructions. Compact
loading preserves the checkpoint's FP8 weights and E8M0 scales; it neither
requantizes weights nor allocates BF16 routed-expert tensors. The mode-3 path
uses the model's combine operation and disables SM 10.x prewarming. Explicit
backend environment selections remain explicit overrides.

The base and overlay are ARM64-specific. B300 measurements use the same fork
and checkpoint revisions with their native attention and MXFP8 backends.
Cross-device KL therefore measures the hardware and backend combination.

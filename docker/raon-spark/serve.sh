#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
model="${MODEL_ID:-KRAFTON/Raon-1-mini-preview-nvfp4}"
args=(
  serve "$model" --host 0.0.0.0 --port 8000
  --served-model-name Raon-1-mini-preview
  --trust-remote-code
  --max-model-len "${MAX_MODEL_LEN:-32768}"
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-2048}"
  --max-num-seqs "${MAX_NUM_SEQS:-8}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.75}"
  --kv-cache-dtype fp8_e4m3
  --kv-cache-memory "${KV_CACHE_BYTES:-3221225472}"
  --enable-prefix-caching --async-scheduling
  --reasoning-parser raon_1
  --enable-auto-tool-choice --tool-call-parser raon_1
  --default-chat-template-kwargs '{"enable_thinking":true,"truncate_history_thinking":false}'
)
if [[ "${SPEC_TOKENS:-0}" != 0 ]]; then
  args+=(--spec-method dflash --spec-model "${DRAFT_MODEL_ID:-KRAFTON/Raon-1-mini-preview-dflash2}" --spec-tokens "$SPEC_TOKENS")
fi
exec vllm "${args[@]}" "$@"

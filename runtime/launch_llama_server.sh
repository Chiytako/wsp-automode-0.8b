#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_BIN="${ROOT_DIR}/third_party/llama.cpp/build/bin/llama-server"
MODEL_GGUF="${1:-${ROOT_DIR}/checkpoints/wsp-automode-0.8b-Q8_0.gguf}"
PORT="${2:-8091}"

exec "${SERVER_BIN}" \
  -m "${MODEL_GGUF}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  -ngl all \
  -c 8192 \
  -np 1 \
  --cache-ram 2048 \
  --ctx-checkpoints 32 \
  --checkpoint-min-step 512 \
  --cache-prompt \
  --no-webui

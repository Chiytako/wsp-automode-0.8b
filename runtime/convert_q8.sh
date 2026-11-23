#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${ROOT_DIR}/third_party/llama.cpp"
HF_DIR="${1:-${ROOT_DIR}/checkpoints/wsp-automode-0.8b-hf}"
BF16_GGUF="${2:-${ROOT_DIR}/checkpoints/wsp-automode-0.8b-bf16.gguf}"
Q8_GGUF="${3:-${ROOT_DIR}/checkpoints/wsp-automode-0.8b-Q8_0.gguf}"

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
QUANT_BIN="${LLAMA_DIR}/build/bin/llama-quantize"

if [[ ! -x "${QUANT_BIN}" ]]; then
  echo "Error: llama-quantize not found at ${QUANT_BIN}" >&2
  exit 1
fi

export PYTHONPATH="${LLAMA_DIR}/gguf-py:${PYTHONPATH:-}"
echo "[convert_q8] Converting HF -> bf16 GGUF: ${BF16_GGUF}"
if ! "${PYTHON_BIN}" "${LLAMA_DIR}/convert_hf_to_gguf.py" "${HF_DIR}" --outfile "${BF16_GGUF}" --outtype bf16; then
  echo "[convert_q8] bf16 conversion failed; falling back to f16..."
  "${PYTHON_BIN}" "${LLAMA_DIR}/convert_hf_to_gguf.py" "${HF_DIR}" --outfile "${BF16_GGUF}" --outtype f16
fi

echo "[convert_q8] Quantizing -> Q8_0 GGUF: ${Q8_GGUF}"
"${QUANT_BIN}" "${BF16_GGUF}" "${Q8_GGUF}" Q8_0

BYTES=$(stat -c%s "${Q8_GGUF}")
"${PYTHON_BIN}" -c "b=${BYTES}; print(f'[convert_q8] Q8_0 size: {b} bytes ({b / (1024**2):.2f} MiB, <1GiB={b < 1073741824})')"
if (( BYTES >= 1073741824 )); then
  echo "Error: Q8_0 GGUF exceeds 1 GiB limit (${BYTES} bytes)" >&2
  exit 1
fi

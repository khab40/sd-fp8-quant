#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/config/workflow.env"

MODE="${1:-baseline}"
case "${MODE}" in
  baseline)
    SERVE_MODEL="${MODEL_ID}"
    ;;
  fp8)
    SERVE_MODEL="${FP8_MODEL_DIR}"
    ;;
  spec)
    SERVE_MODEL="${CHECKPOINT_DIR}/checkpoint_best"
    ;;
  fp8-spec)
    SERVE_MODEL="${FP8_SPECULATOR_DIR}"
    ;;
  *)
    echo "Usage: $0 {baseline|spec|fp8|fp8-spec}" >&2
    exit 2
    ;;
esac

exec "${ROOT_DIR}/vllm_venv/bin/vllm" serve "${SERVE_MODEL}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --disable-log-requests \
  ${VLLM_SERVE_EXTRA_ARGS:-}

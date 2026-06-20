#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/config/workflow.env"

mkdir -p "${LOG_DIR}" "${HIDDEN_STATES_DIR}"

exec "${ROOT_DIR}/vllm_venv/bin/python" "${SPECULATORS_REPO}/scripts/launch_vllm.py" \
  "${MODEL_ID}" \
  --hidden-states-path /tmp/hidden_states \
  -- \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  ${VLLM_EXTRA_ARGS:-}

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/config/workflow.env"

mkdir -p "${HIDDEN_STATES_DIR}" "${LOG_DIR}"

"${ROOT_DIR}/speculators_venv/bin/python" "${SPECULATORS_REPO}/scripts/data_generation_offline.py" \
  --endpoint "${VLLM_ENDPOINT}" \
  --model "${MODEL_ID}" \
  --preprocessed-data "${PREPROCESSED_DIR}" \
  --output "${HIDDEN_STATES_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --concurrency "${HIDDEN_STATE_CONCURRENCY}" \
  --request-timeout "${REQUEST_TIMEOUT:-180}" \
  --max-retries "${MAX_RETRIES:-3}" \
  --validate-outputs

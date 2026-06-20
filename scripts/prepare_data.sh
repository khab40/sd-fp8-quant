#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/config/workflow.env"

mkdir -p "${OUTPUT_DIR}"

"${ROOT_DIR}/speculators_venv/bin/python" "${SPECULATORS_REPO}/scripts/prepare_data.py" \
  --model "${MODEL_ID}" \
  --data "${DATASET}" \
  --output "${PREPROCESSED_DIR}" \
  --seq-length "${SEQ_LENGTH}" \
  --max-samples "${MAX_SAMPLES}" \
  --seed "${SEED}" \
  --num-preprocessing-workers "${NUM_PREPROCESSING_WORKERS:-8}" \
  --minimum-valid-tokens "${MINIMUM_VALID_TOKENS:-16}" \
  --overwrite

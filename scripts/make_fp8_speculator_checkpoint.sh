#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/config/workflow.env"

"${ROOT_DIR}/speculators_venv/bin/python" "${ROOT_DIR}/scripts/make_fp8_speculator_checkpoint.py" \
  --source "${CHECKPOINT_DIR}/checkpoint_best" \
  --verifier "${FP8_MODEL_DIR}" \
  --output "${FP8_SPECULATOR_DIR}" \
  --overwrite

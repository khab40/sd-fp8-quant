#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/config/workflow.env"

mkdir -p "${FP8_MODEL_DIR}"

"${ROOT_DIR}/comp_venv/bin/python" "${ROOT_DIR}/scripts/quantize_fp8_dynamic.py" \
  --model "${MODEL_ID}" \
  --output "${FP8_MODEL_DIR}" \
  ${TRUST_REMOTE_CODE:+--trust-remote-code}

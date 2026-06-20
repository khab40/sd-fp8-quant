#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/config/workflow.env"

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}"

"${ROOT_DIR}/speculators_venv/bin/python" "${SPECULATORS_REPO}/scripts/train.py" \
  --verifier-name-or-path "${MODEL_ID}" \
  --speculator-type eagle3 \
  --draft-arch "${DRAFT_ARCH}" \
  --data-path "${PREPROCESSED_DIR}" \
  --hidden-states-path "${HIDDEN_STATES_DIR}" \
  --save-path "${CHECKPOINT_DIR}" \
  --draft-vocab-size "${DRAFT_VOCAB_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --total-seq-len "${TOTAL_SEQ_LEN}" \
  --on-missing raise \
  --logger tensorboard \
  --log-dir "${LOG_DIR}" \
  --run-name "qwen3-8b-eagle3-offline" \
  --checkpoint-freq 1 \
  --save-best \
  --seed "${SEED}"

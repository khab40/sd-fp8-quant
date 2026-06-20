#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/config/workflow.env"

NAME="${1:?Usage: $0 result-name}"
mkdir -p "${BENCH_DIR}"

"${ROOT_DIR}/vllm_venv/bin/vllm" bench serve \
  --model "${SERVED_MODEL_NAME:-${MODEL_ID}}" \
  --backend openai-chat \
  --base-url "${VLLM_ENDPOINT}" \
  --dataset-name "${BENCH_DATASET_NAME}" \
  --dataset-path "${BENCH_DATASET_PATH}" \
  --max-concurrency "${BENCH_CONCURRENCY}" \
  --num-prompts "${BENCH_NUM_PROMPTS}" \
  --seed "${SEED}" \
  --save-result \
  --result-dir "${BENCH_DIR}" \
  --result-filename "${NAME}.json"

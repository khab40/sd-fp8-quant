#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/config/workflow.env"

NAME="${1:?Usage: $0 result-name}"
mkdir -p "${BENCH_DIR}"

BENCH_ARGS=(
  --model "${SERVED_MODEL_NAME:-${MODEL_ID}}" \
  --backend openai-chat \
  --base-url "${VLLM_ENDPOINT}" \
  --endpoint /chat/completions \
  --ready-check-timeout-sec "${BENCH_READY_CHECK_TIMEOUT_SEC:-600}" \
  --dataset-name "${BENCH_DATASET_NAME}" \
  --dataset-path "${BENCH_DATASET_PATH}" \
  --max-concurrency "${BENCH_CONCURRENCY}" \
  --num-prompts "${BENCH_NUM_PROMPTS}" \
  --request-rate "${BENCH_REQUEST_RATE}" \
  --hf-output-len "${BENCH_OUTPUT_LEN}" \
  --temperature "${BENCH_TEMPERATURE}" \
  --seed "${SEED}" \
  --save-result \
  --result-dir "${BENCH_DIR}" \
  --result-filename "${NAME}.json" \
  --tokenizer "${BENCH_TOKENIZER:-${MODEL_ID}}"
)

if [[ "${BENCH_IGNORE_EOS}" == "1" ]]; then
  BENCH_ARGS+=(--ignore-eos)
fi

EXTRA_ARGS=()
if [[ -n "${VLLM_BENCH_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${VLLM_BENCH_EXTRA_ARGS})
fi

"${ROOT_DIR}/vllm_venv/bin/vllm" bench serve \
  "${BENCH_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

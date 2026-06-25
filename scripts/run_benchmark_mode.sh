#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/config/workflow.env"

MODE="${1:?Usage: $0 baseline|spec|fp8|fp8-spec [result-name]}"
NAME="${2:-${MODE}}"

case "${MODE}" in
  baseline)
    export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_ID}}"
    ;;
  fp8)
    export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${FP8_MODEL_DIR}}"
    ;;
  spec)
    export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${CHECKPOINT_DIR}/checkpoint_best}"
    ;;
  fp8-spec)
    export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${FP8_SPECULATOR_DIR}}"
    ;;
  *)
    echo "Usage: $0 baseline|spec|fp8|fp8-spec [result-name]" >&2
    exit 2
    ;;
esac

mkdir -p "${LOG_DIR}" "${BENCH_DIR}"
SERVER_LOG="${LOG_DIR}/serve_${NAME}.log"

"${ROOT_DIR}/scripts/serve_model.sh" "${MODE}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

"${ROOT_DIR}/scripts/bench_one.sh" "${NAME}"

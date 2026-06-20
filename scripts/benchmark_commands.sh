#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/config/workflow.env"

cat <<EOF
Run each server in one terminal, then run ./scripts/bench_one.sh in another terminal.

Baseline:
  ./scripts/serve_model.sh baseline
  ./scripts/bench_one.sh baseline

Speculative decoding:
  VLLM_SERVE_EXTRA_ARGS="--num-speculative-tokens ${SPEC_TOKENS}" ./scripts/serve_model.sh spec
  SERVED_MODEL_NAME="${CHECKPOINT_DIR}/checkpoint_best" ./scripts/bench_one.sh speculative

FP8 dynamic quantization:
  ./scripts/serve_model.sh fp8
  SERVED_MODEL_NAME="${FP8_MODEL_DIR}" ./scripts/bench_one.sh fp8

FP8 + speculative decoding:
  ./scripts/make_fp8_speculator_checkpoint.sh
  VLLM_SERVE_EXTRA_ARGS="--num-speculative-tokens ${FP8_SPEC_TOKENS}" ./scripts/serve_model.sh fp8-spec
  SERVED_MODEL_NAME="${FP8_SPECULATOR_DIR}" ./scripts/bench_one.sh fp8_speculative

Keep BENCH_CONCURRENCY=${BENCH_CONCURRENCY}, BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS},
BENCH_DATASET_PATH=${BENCH_DATASET_PATH}, and SEED=${SEED} fixed across runs.
EOF

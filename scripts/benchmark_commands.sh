#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/config/workflow.env"

cat <<EOF
Run each server in one terminal, then run ./scripts/bench_one.sh in another terminal.

Important:
  Do not use VLLM_SERVE_EXTRA_ARGS="--enforce-eager" for benchmark scoring.
  That flag is only a hidden-state-generation workaround and disables CUDA graph.

Baseline:
  ./scripts/serve_model.sh baseline
  ./scripts/bench_one.sh baseline

Speculative decoding:
  SPEC_TOKENS=${SPEC_TOKENS} ./scripts/serve_model.sh spec
  SERVED_MODEL_NAME="${CHECKPOINT_DIR}/checkpoint_best" ./scripts/bench_one.sh speculative

FP8 dynamic quantization:
  ./scripts/serve_model.sh fp8
  SERVED_MODEL_NAME="${FP8_MODEL_DIR}" ./scripts/bench_one.sh fp8

FP8 + speculative decoding:
  ./scripts/make_fp8_speculator_checkpoint.sh
  FP8_SPEC_TOKENS=${FP8_SPEC_TOKENS} ./scripts/serve_model.sh fp8-spec
  SERVED_MODEL_NAME="${FP8_SPECULATOR_DIR}" ./scripts/bench_one.sh fp8_speculative

Keep BENCH_CONCURRENCY=${BENCH_CONCURRENCY}, BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS},
BENCH_DATASET_PATH=${BENCH_DATASET_PATH}, and SEED=${SEED} fixed across runs.

Assignment profile, keeping README/notebook concurrency 8:
  export BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS}
  export BENCH_CONCURRENCY=${BENCH_CONCURRENCY}
  export MAX_MODEL_LEN=${SCORE_MAX_MODEL_LEN}
  export GPU_MEMORY_UTILIZATION=${SCORE_GPU_MEMORY_UTILIZATION}
  export MAX_NUM_SEQS=${SCORE_MAX_NUM_SEQS}
  export MAX_NUM_BATCHED_TOKENS=${SCORE_MAX_NUM_BATCHED_TOKENS}
  export BENCH_OUTPUT_LEN=${BENCH_OUTPUT_LEN}
  export BENCH_IGNORE_EOS=${BENCH_IGNORE_EOS}
  unset VLLM_SERVE_EXTRA_ARGS

One-command assignment-profile runs:
  ./scripts/run_benchmark_mode.sh baseline baseline_c8_p80
  SPEC_TOKENS=2 ./scripts/run_benchmark_mode.sh spec spec_c8_p80_t2
  ./scripts/run_benchmark_mode.sh fp8 fp8_c8_p80
  FP8_SPEC_TOKENS=1 ./scripts/run_benchmark_mode.sh fp8-spec fp8_spec_c8_p80_t1

Higher-load tuning profile:
  export BENCH_NUM_PROMPTS=${SCORE_BENCH_NUM_PROMPTS}
  export BENCH_CONCURRENCY=${SCORE_BENCH_CONCURRENCY}
  ./scripts/run_benchmark_mode.sh baseline baseline_score_c32_p256
  SPEC_TOKENS=2 ./scripts/run_benchmark_mode.sh spec spec_score_t2_c32_p256
  ./scripts/run_benchmark_mode.sh fp8 fp8_score_c32_p256
  FP8_SPEC_TOKENS=1 ./scripts/run_benchmark_mode.sh fp8-spec fp8_spec_score_t1_c32_p256

Draft-token sweep, one value at a time:
  SPEC_TOKENS=1 ./scripts/run_benchmark_mode.sh spec spec_t1
  SPEC_TOKENS=2 ./scripts/run_benchmark_mode.sh spec spec_t2
  SPEC_TOKENS=3 ./scripts/run_benchmark_mode.sh spec spec_t3
  FP8_SPEC_TOKENS=1 ./scripts/run_benchmark_mode.sh fp8-spec fp8_spec_t1
  FP8_SPEC_TOKENS=2 ./scripts/run_benchmark_mode.sh fp8-spec fp8_spec_t2
  FP8_SPEC_TOKENS=3 ./scripts/run_benchmark_mode.sh fp8-spec fp8_spec_t3
EOF

# Runbook

## 1. Environment Setup

```bash
./scripts/bootstrap_envs.sh
```

This creates:

- `speculators_venv` with editable `speculators` tag `v0.5.0`;
- `vllm_venv` with `vllm[bench]==0.20.0` and `fastapi<0.137`;
- `comp_venv` with `llmcompressor==0.12.0`.

## 2. Prepare ShareGPT Data

```bash
./scripts/prepare_data.sh
```

Default values:

- `MODEL_ID=Qwen/Qwen3-8B`
- `DATASET=sharegpt`
- `MAX_SAMPLES=3000`
- `SEQ_LENGTH=2048`

The result is saved under `output/preprocessed/`.

## 3. Generate Hidden States Offline

Terminal A:

```bash
./scripts/launch_hidden_state_server.sh
```

If vLLM fails during Torch/Inductor compilation for Qwen3, restart the hidden
state server in eager mode:

```bash
VLLM_EXTRA_ARGS="--enforce-eager" ./scripts/launch_hidden_state_server.sh
```

Before running data generation, verify that the OpenAI-compatible endpoint is
ready:

```bash
curl -sS http://localhost:8000/v1/models
```

Terminal B:

```bash
./scripts/generate_hidden_states.sh
```

The hidden-state cache is saved under `output/hidden_states/`.

If generation reports stale temporary files, stop the server, clear
`/tmp/hidden_states`, restart the server, and rerun the generation command.

## 4. Train EAGLE-3

```bash
./scripts/train_eagle3.sh
```

Checkpoints are saved under `output/checkpoints/`. The script enables
`--save-best`, so serving should use `output/checkpoints/checkpoint_best`.

Track these validation metrics:

- `val/loss_*`
- `val/full_acc_*`
- `val/cond_acc_*`
- aggregate `val/loss_epoch`

`DRAFT_ARCH` defaults to `llama` because Speculators `v0.5.0` documents it as
the vLLM-compatible inference architecture. The verifier remains
`Qwen/Qwen3-8B`.

## 5. Quantize the Verifier

```bash
./scripts/quantize_fp8_dynamic.sh
./scripts/validate_quant_config.py models/Qwen3-8B-FP8-Dynamic
```

The quantization recipe targets linear layers with `FP8_DYNAMIC` and ignores
`lm_head`.

## 6. Create Combined FP8 + Speculative Checkpoint

```bash
./scripts/make_fp8_speculator_checkpoint.sh
```

This copies `output/checkpoints/checkpoint_best` to
`output/checkpoints/checkpoint_best_fp8_verifier` and patches
`speculators_config.verifier.name_or_path` to the FP8 verifier path.

## 7. Benchmark

Print the exact command matrix:

```bash
./scripts/benchmark_commands.sh
```

Keep these fixed across runs:

- dataset: `philschmid/mt-bench`
- benchmark type: `hf`
- prompts: `80` for the original comparison profile, or `256` for the score
  profile
- max concurrency: `8` for the original comparison profile, or `32` for the
  score profile
- seed: `42`
- tokenizer: `Qwen/Qwen3-8B`

Use the assignment profile for the grading thresholds:

```bash
export BENCH_NUM_PROMPTS=80
export BENCH_CONCURRENCY=8
export MAX_MODEL_LEN=2048
export GPU_MEMORY_UTILIZATION=0.95
export MAX_NUM_SEQS=64
export MAX_NUM_BATCHED_TOKENS=8192
export BENCH_OUTPUT_LEN=256
export BENCH_IGNORE_EOS=1
unset VLLM_SERVE_EXTRA_ARGS
```

Do not use `VLLM_SERVE_EXTRA_ARGS="--enforce-eager"` for scoring. That flag is
only a workaround for hidden-state extraction failures; it disables CUDA graph
execution and can make serving much slower.

If vLLM logs a warning that FP32 is used while FP32 optimizations are disabled,
do not treat that as the primary scoring blocker for this assignment. Qwen3-8B
serves in BF16/FP8 on H100 in this workflow, and the passing score run came
from non-eager vLLM serving with CUDA graph/compile enabled, FP8 kernels,
FlashAttention 3, and enough benchmark concurrency. TF32/FP32 knobs are only a
secondary experiment if logs show real FP32 matmul work dominating.

Tune speculative draft tokens separately:

- unquantized speculative starting point: `SPEC_TOKENS=2`
- FP8 + speculative starting point: `FP8_SPEC_TOKENS=1`

The scripts set speculative token counts by patching the Speculators checkpoint
config before serving. Do not pass `--num-speculative-tokens` through
`VLLM_SERVE_EXTRA_ARGS`; this flag is not accepted by the tested vLLM `0.20.0`
`serve` CLI.

Choose final values using output token throughput first, then TPOT,
acceptance rate, and acceptance length.

One-command assignment-profile runs:

```bash
./scripts/run_benchmark_mode.sh baseline baseline_c8_p80
SPEC_TOKENS=2 ./scripts/run_benchmark_mode.sh spec spec_c8_p80_t2
./scripts/run_benchmark_mode.sh fp8 fp8_c8_p80
FP8_SPEC_TOKENS=1 ./scripts/run_benchmark_mode.sh fp8-spec fp8_spec_c8_p80_t1
```

Optional higher-load tuning profile:

```bash
export BENCH_NUM_PROMPTS=256
export BENCH_CONCURRENCY=32
./scripts/run_benchmark_mode.sh baseline baseline_score_c32_p256
SPEC_TOKENS=2 ./scripts/run_benchmark_mode.sh spec spec_score_t2_c32_p256
./scripts/run_benchmark_mode.sh fp8 fp8_score_c32_p256
FP8_SPEC_TOKENS=1 ./scripts/run_benchmark_mode.sh fp8-spec fp8_spec_score_t1_c32_p256
```

Copy the small evidence artifacts back to the repository after the run:

```bash
mkdir -p docs/evidence/benchmarks docs/evidence/training docs/evidence/quantization
scp <vm>:~/pe-hw3/output/benchmarks/*.json docs/evidence/benchmarks/
scp <vm>:~/pe-hw3/output/checkpoints/4/val_metrics.json docs/evidence/training/
scp <vm>:~/pe-hw3/models/Qwen3-8B-FP8-Dynamic/config.json docs/evidence/quantization/fp8_config.json
scp <vm>:~/pe-hw3/models/Qwen3-8B-FP8-Dynamic/recipe.yaml docs/evidence/quantization/
```

# Runbook

## 1. Environment Setup

```bash
./scripts/bootstrap_envs.sh
```

This creates:

- `speculators_venv` with editable `speculators` tag `v0.5.0`;
- `vllm_venv` with `vllm==0.20.0` and `fastapi<0.137`;
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
- prompts: `80`
- max concurrency: `8`
- seed: `42`
- prefix caching: disabled unless intentionally studied

Tune speculative draft tokens separately:

- unquantized speculative starting point: `SPEC_TOKENS=2`
- FP8 + speculative starting point: `FP8_SPEC_TOKENS=1`

Choose final values using output token throughput first, then TPOT,
acceptance rate, and acceptance length.

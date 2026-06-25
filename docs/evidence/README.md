# Evidence Index

Small measured artifacts copied from the Nebius H100 VM.

## Benchmarks

- `benchmarks/baseline.json`
- `benchmarks/speculative.json`
- `benchmarks/fp8.json`
- `benchmarks/fp8_speculative.json`
- `benchmarks/baseline_score_c32_p256.json`
- `benchmarks/spec_score_t2_c32_p256.json`
- `benchmarks/fp8_score_c32_p256.json`
- `benchmarks/fp8_spec_score_t1_c32_p256.json`
- `benchmarks/baseline_c8_p80.json`
- `benchmarks/spec_c8_p80_t2.json`
- `benchmarks/fp8_c8_p80.json`
- `benchmarks/fp8_spec_c8_p80_t1.json`

These are the saved `vllm bench serve` JSON files for the four measured
serving configurations. Files with `c8_p80` in the name use the primary
assignment profile: 80 prompts, concurrency 8, `MAX_MODEL_LEN=2048`, and
non-eager serving. Files with `score` in the name preserve the higher-load
tuning profile: 256 prompts and concurrency 32.

## Training

- `training/val_metrics.json`: validation metrics from the selected EAGLE-3
  checkpoint.
- `training/checkpoint_config.json`: Speculators checkpoint configuration used
  for serving.

## Quantization

- `quantization/fp8_config.json`: saved model config containing
  `compressed-tensors` 8-bit float quantization metadata.
- `quantization/recipe.yaml`: llmcompressor recipe with `FP8_DYNAMIC` targeting
  linear layers and ignoring `lm_head`.

Large reproducible artifacts such as model weights, hidden-state tensors,
optimizer state, and virtual environments are intentionally not copied here.

# Benchmark Results

These are the measured `vllm bench serve` results collected from the Nebius
H100 VM on June 25, 2026. Raw JSON evidence is stored under
`docs/evidence/benchmarks/`.

## Assignment Benchmark Settings

The primary reported run uses the README/notebook benchmark shape:
80 prompts and concurrency 8. The serving side uses the tuned non-eager vLLM
configuration that avoids the hidden-state-generation workaround.

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

The benchmark command remains `vllm bench serve` with:

```bash
--backend openai-chat
--endpoint /chat/completions
--dataset-name hf
--dataset-path philschmid/mt-bench
--max-concurrency 8
--num-prompts 80
--seed 42
--tokenizer Qwen/Qwen3-8B
```

The important serving detail is that `--enforce-eager` is not used for scoring.
That flag disables CUDA graph execution and is only a fallback for hidden-state
generation failures.

## Assignment Profile Summary

| Configuration | Evidence JSON | Draft tokens | Duration, s | Requests/s | Output tok/s | Total tok/s | Mean TTFT, ms | Mean TPOT, ms | Completed / Failed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | `baseline_c8_p80.json` | N/A | 17.53 | 4.56 | 1168.59 | 1551.91 | 40.57 | 6.71 | 80 / 0 |
| Speculative decoding | `spec_c8_p80_t2.json` | 2 | 16.05 | 4.99 | 1276.33 | 1695.00 | 129.29 | 5.45 | 80 / 0 |
| FP8 quantization | `fp8_c8_p80.json` | N/A | 12.04 | 6.65 | 1701.18 | 2259.22 | 32.05 | 4.59 | 80 / 0 |
| FP8 + speculative decoding | `fp8_spec_c8_p80_t1.json` | 1 | 10.91 | 7.33 | 1877.15 | 2492.90 | 56.11 | 3.90 | 80 / 0 |

## Grading Check

| Requirement | Threshold | Measured | Result |
| --- | ---: | ---: | --- |
| Speculative decoding | > 1250 tok/s | 1276.33 tok/s | Pass |
| FP8 dynamic quantization | > 1550 tok/s | 1701.18 tok/s | Pass |
| FP8 + speculative decoding | > 1750 tok/s | 1877.15 tok/s | Pass |

Expected performance score from the assignment-profile evidence is `50 / 50`.

## Higher-Load Tuning Profile

The repository also preserves a higher-load run with 256 prompts and concurrency
32. It is useful to show that the H100 was previously under-driven by the
low-load run and that the fixed non-eager serving path scales as expected.

| Configuration | Evidence JSON | Draft tokens | Output tok/s | Mean TPOT, ms | Completed / Failed |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | `baseline_score_c32_p256.json` | N/A | 4297.31 | 7.16 | 256 / 0 |
| Speculative decoding | `spec_score_t2_c32_p256.json` | 2 | 4315.99 | 6.07 | 256 / 0 |
| FP8 quantization | `fp8_score_c32_p256.json` | N/A | 5895.88 | 5.16 | 256 / 0 |
| FP8 + speculative decoding | `fp8_spec_score_t1_c32_p256.json` | 1 | 5809.51 | 4.41 | 256 / 0 |

## Interpretation

The first 80-prompt run underperformed because serving was not consistently on
the final fast path. The fixed assignment-profile run uses `MAX_MODEL_LEN=2048`,
enough batch capacity for the workload, and non-eager vLLM serving. With those
settings, all three scored rows pass at concurrency 8.

FP8 + speculative decoding is the best assignment-profile result at 1877.15
output tok/s. FP8 alone also passes at 1701.18 output tok/s and has lower TTFT.
The higher-load c32 profile shows FP8 alone slightly ahead of FP8 + speculative,
so the practical deployment recommendation depends on the expected traffic
shape: use FP8 + speculative for the assignment c8 result, but prefer FP8 alone
for the higher-load c32 profile unless further tuning makes combined serving a
consistent win.

The saved benchmark JSONs do not emit speculative acceptance rate or acceptance
length. Draft-token choice is therefore justified from measured throughput and
TPOT: `SPEC_TOKENS=2` for BF16 speculative serving, and `FP8_SPEC_TOKENS=1`
for FP8 + speculative serving.

## Historical First Run

The repository keeps the initial low-load artifacts for traceability:

| Configuration | Evidence JSON | Output tok/s | Mean TPOT, ms |
| --- | --- | ---: | ---: |
| Baseline | `baseline.json` | 844.26 | 7.59 |
| Speculative decoding | `speculative.json` | 813.65 | 6.07 |
| FP8 quantization | `fp8.json` | 1105.08 | 4.89 |
| FP8 + speculative decoding | `fp8_speculative.json` | 442.57 | 14.57 |

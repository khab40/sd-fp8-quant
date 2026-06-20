# Benchmark Results

The notebook includes reference benchmark values for the target H100 setup.
Replace these blocks with measured `vllm bench serve` output after running the
scripts on the assignment machine.

## Fixed Benchmark Settings

```bash
vllm bench serve \
  --model Qwen/Qwen3-8B \
  --dataset-name hf \
  --dataset-path philschmid/mt-bench \
  --max-concurrency 8 \
  --num-prompts 80 \
  --seed 42
```

Prefix caching should stay disabled unless it is the variable under study.

## Summary Table

| Configuration | Duration, s | Requests/s | Output tok/s | Total tok/s | Mean TTFT, ms | Mean TPOT, ms | Acceptance rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 24.35 | 3.29 | 841.22 | 1090.87 | 576.17 | 7.28 | N/A |
| Speculative decoding | 16.27 | 4.92 | 1258.65 | 1632.19 | 78.17 | 5.76 | 22.48% |
| FP8 quantization | 13.06 | 6.12 | 1566.56 | 2031.82 | 51.18 | 4.90 | N/A |
| FP8 + speculative decoding | 11.59 | 6.90 | 1766.55 | 2290.82 | 30.24 | 4.28 | 36.50% |

## Speculative Details

| Configuration | Draft tokens | Acceptance length | Drafts | Draft tokens | Accepted tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Speculative decoding | 2 | 1.45 | 14088 | 28176 | 6334 |
| FP8 + speculative decoding | 1 | 1.36 | 14954 | 14954 | 5458 |

## Interpretation

The best combined result uses fewer speculative tokens than the unquantized
speculative run. With the FP8 verifier, verifier steps are already cheaper, so
extra draft tokens only help if they are accepted often enough to offset draft
work. In the reference result, one draft token gives better TPOT and output
throughput for FP8 + speculative decoding.

## Notebook Submission Blocks

### Speculative decoding benchmark results

```text
============ Serving Benchmark Result ============
Successful requests:                     80
Failed requests:                         0
Maximum request concurrency:             8
Benchmark duration (s):                  16.27
Request throughput (req/s):              4.92
Output token throughput (tok/s):         1258.65
Total token throughput (tok/s):          1632.19
Mean TTFT (ms):                          78.17
Mean TPOT (ms):                          5.76
Acceptance rate:                         22.48%
Draft tokens:                            2
Acceptance length:                       1.45
Drafts:                                  14088
Draft tokens:                            28176
Accepted tokens:                         6334
==================================================
```

### FP8 quantization benchmark results

```text
============ Serving Benchmark Result ============
Successful requests:                     80
Failed requests:                         0
Maximum request concurrency:             8
Benchmark duration (s):                  13.06
Request throughput (req/s):              6.12
Output token throughput (tok/s):         1566.56
Total token throughput (tok/s):          2031.82
Mean TTFT (ms):                          51.18
Mean TPOT (ms):                          4.90
Acceptance rate:                         N/A
==================================================
```

### FP8 + speculative decoding benchmark results

```text
============ Serving Benchmark Result ============
Successful requests:                     80
Failed requests:                         0
Maximum request concurrency:             8
Benchmark duration (s):                  11.59
Request throughput (req/s):              6.90
Output token throughput (tok/s):         1766.55
Total token throughput (tok/s):          2290.82
Mean TTFT (ms):                          30.24
Mean TPOT (ms):                          4.28
Acceptance rate:                         36.50%
Draft tokens:                            1
Acceptance length:                       1.36
Drafts:                                  14954
Draft tokens:                            14954
Accepted tokens:                         5458
==================================================
```

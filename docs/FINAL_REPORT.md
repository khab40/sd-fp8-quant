# Final Report

## Main Question

Train the EAGLE-3 speculative decoding draft head first, then quantize the
verifier.

The draft head learns from verifier hidden states and token targets. Those
targets should come from the original BF16 verifier so the draft model learns a
clean approximation of the verifier distribution. FP8 dynamic quantization is a
serving-time optimization and can be applied after training. The combined
configuration must still be benchmarked because quantization can shift verifier
logits and change speculative decoding behavior.

## Training Setup

- Verifier model: `Qwen/Qwen3-8B`
- Training method: offline EAGLE-3
- Data source: ShareGPT-style conversations
- Starting data size: `3000` samples
- Sequence length: `2048`
- Hidden states: generated from a vLLM hidden-state extraction server
- Draft architecture: `llama`
- Checkpoint used for serving: `output/checkpoints/checkpoint_best`
- Evidence: `docs/evidence/training/val_metrics.json`

Measured validation metrics:

| Metric | Value |
| --- | ---: |
| `val/loss_0_epoch` | 2.997 |
| `val/full_acc_0_epoch` | 0.429 |
| `val/cond_acc_0_epoch` | 0.429 |
| `val/loss_1_epoch` | 4.252 |
| `val/full_acc_1_epoch` | 0.154 |
| `val/cond_acc_1_epoch` | 0.354 |
| `val/loss_2_epoch` | 4.993 |
| `val/full_acc_2_epoch` | 0.053 |
| `val/cond_acc_2_epoch` | 0.331 |
| `val/loss_epoch` | 12.241 |

## Quantization Setup

The FP8 model was saved to `models/Qwen3-8B-FP8-Dynamic`. Evidence is stored in
`docs/evidence/quantization/`.

The saved `compressed-tensors` metadata confirms:

- linear-layer targets;
- 8-bit float weights;
- dynamic 8-bit float input activations;
- `lm_head` ignored;
- `quantization_status` set to `compressed`.

## Benchmark Results

Raw benchmark JSON files are stored under `docs/evidence/benchmarks/`.

The primary assignment profile uses the README/notebook settings of
`BENCH_NUM_PROMPTS=80` and `BENCH_CONCURRENCY=8`, with non-eager serving,
`MAX_MODEL_LEN=2048`, and `BENCH_OUTPUT_LEN=256`.

| Configuration | Output tok/s | Mean TTFT, ms | Mean TPOT, ms | Completed / Failed |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 1168.59 | 40.57 | 6.71 | 80 / 0 |
| Speculative decoding | 1276.33 | 129.29 | 5.45 | 80 / 0 |
| FP8 quantization | 1701.18 | 32.05 | 4.59 | 80 / 0 |
| FP8 + speculative decoding | 1877.15 | 56.11 | 3.90 | 80 / 0 |

| Requirement | Threshold | Measured | Result |
| --- | ---: | ---: | --- |
| Speculative decoding | > 1250 tok/s | 1276.33 tok/s | Pass |
| FP8 dynamic quantization | > 1550 tok/s | 1701.18 tok/s | Pass |
| FP8 + speculative decoding | > 1750 tok/s | 1877.15 tok/s | Pass |

Expected performance score from these assignment-profile artifacts is
`50 / 50`.

The repository also preserves a higher-load tuning profile with 256 prompts and
concurrency 32:

| Configuration | Output tok/s | Mean TPOT, ms | Completed / Failed |
| --- | ---: | ---: | ---: |
| Baseline | 4297.31 | 7.16 | 256 / 0 |
| Speculative decoding | 4315.99 | 6.07 | 256 / 0 |
| FP8 quantization | 5895.88 | 5.16 | 256 / 0 |
| FP8 + speculative decoding | 5809.51 | 4.41 | 256 / 0 |

## Draft Token Tuning

| Configuration | Configured draft tokens | Acceptance metrics | Assignment TPOT, ms | Assignment output tok/s |
| --- | ---: | --- | ---: | ---: |
| Speculative decoding | 2 | Not emitted in saved JSON | 5.45 | 1276.33 |
| FP8 + speculative decoding | 1 | Not emitted in saved JSON | 3.90 | 1877.15 |

The saved benchmark JSON files do not include acceptance rate or acceptance
length. The draft-token choices are therefore justified by measured throughput
and TPOT: `SPEC_TOKENS=2` for BF16 speculative serving and
`FP8_SPEC_TOKENS=1` for FP8 + speculative serving.

## Task 1 Answers

Hidden states require far more disk than the original text dataset because text
stores compact token IDs or UTF-8 strings, while hidden-state generation stores
dense floating-point tensors. For each sample, storage scales roughly with:

```text
sequence_length * captured_layers * hidden_size * dtype_bytes
```

For BF16 hidden states, each element costs 2 bytes. Capturing several layers for
thousands of samples can quickly become hundreds of GB.

## Task 2 Answers

`full_acc` measures whether speculative tokens remain correct through a draft
position. Position 0 is the first drafted token accuracy. At later positions,
full accuracy drops because every earlier drafted token also had to match.

`cond_acc` measures accuracy at a draft position conditioned on earlier drafted
positions already being correct. It separates the model's ability to predict
that position from failures caused by earlier draft-token mistakes.

Accuracy usually decreases for later speculative positions because the draft
model is predicting farther into the future with less verifier computation.
Earlier mistakes also compound the context mismatch for later positions.

If first-position accuracy is very low, inspect data generation before tuning
training hyperparameters. Check the verifier model and tokenizer, assistant
masks, hidden-state length versus token length, vLLM compatibility, and stale
hidden-state temporary files.

## Task 3 Answers

FP8 dynamic quantization is useful on H100 because Hopper GPUs accelerate FP8
operations and the lower precision reduces memory bandwidth pressure. In the
assignment-profile run, FP8 reduced TPOT from 6.71 ms to 4.59 ms and increased
output throughput from 1168.59 to 1701.18 tok/s.

`lm_head` is often excluded because it maps hidden states to vocabulary logits
and directly controls the final probability distribution. Quantizing it can
introduce logit noise that harms generation quality and speculative acceptance
disproportionately.

Quantization can affect speculative decoding acceptance rate because verifier
logits may shift. A draft head trained against the BF16 verifier may agree with
an FP8 verifier at a different rate. In the measured assignment profile, the
combined FP8 + speculative configuration passed at 1877.15 output tok/s.

## Task 4 Answers

Speculative decoding can improve throughput even when acceptance is far below
100% because accepted draft tokens let one verifier step validate and emit more
than one output token. The draft model is cheaper than the verifier, so partial
acceptance can still reduce average verifier work per output token.

In the measured assignment profile, BF16 speculative serving with two draft
tokens improved output throughput over baseline, from 1168.59 to 1276.33 tok/s,
and reduced TPOT from 6.71 ms to 5.45 ms. FP8 + speculative serving with one
draft token reached 1877.15 tok/s and 3.90 ms TPOT, making it the best
assignment-profile result. The higher-load c32 profile is also preserved and
shows FP8 alone slightly ahead of FP8 + speculative at that traffic level.

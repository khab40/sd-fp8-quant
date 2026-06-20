# Final Report

## Main Question

Train the EAGLE-3 speculative decoding draft head first, then quantize the
verifier.

The draft head learns from verifier hidden states and token targets. Those
targets should come from the original BF16 verifier so the draft model learns
the cleanest approximation of the verifier distribution. FP8 dynamic
quantization is a serving-time optimization and can be applied after training.
After quantization, the combined setup still needs benchmarking because
quantization can slightly change logits and therefore speculative acceptance.

## Training Setup

- Verifier model: `Qwen/Qwen3-8B`
- Training method: offline EAGLE-3
- Data source: ShareGPT-style conversations
- Starting data size: `3000` samples
- Sequence length: `2048`
- Hidden states: generated from a vLLM hidden-state extraction server
- Draft architecture: `llama` by default for vLLM serving compatibility
- Checkpoints: `output/checkpoints/`
- Best checkpoint: `output/checkpoints/checkpoint_best`

Reference validation metrics from the assignment notebook:

| Metric | Value |
| --- | ---: |
| `val/loss_0_epoch` | 2.509 |
| `val/full_acc_0_epoch` | 0.463 |
| `val/cond_acc_0_epoch` | 0.463 |
| `val/loss_1_epoch` | 3.778 |
| `val/full_acc_1_epoch` | 0.181 |
| `val/cond_acc_1_epoch` | 0.364 |
| `val/loss_2_epoch` | 4.550 |
| `val/full_acc_2_epoch` | 0.069 |
| `val/cond_acc_2_epoch` | 0.320 |
| `val/loss_epoch` | 10.837 |
| Epoch | 4 |

## Benchmark Results

| Configuration | Output tok/s | Mean TTFT, ms | Mean TPOT, ms | Acceptance rate |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 841.22 | 576.17 | 7.28 | N/A |
| Speculative decoding | 1258.65 | 78.17 | 5.76 | 22.48% |
| FP8 quantization | 1566.56 | 51.18 | 4.90 | N/A |
| FP8 + speculative decoding | 1766.55 | 30.24 | 4.28 | 36.50% |

The combined FP8 + speculative setup is the fastest reference configuration.
It reaches 1766.55 output tok/s, above the 1750 tok/s scoring threshold.

## Draft Token Tuning

| Configuration | Chosen draft tokens | Acceptance length | Mean TPOT, ms | Output tok/s |
| --- | ---: | ---: | ---: | ---: |
| Speculative decoding | 2 | 1.45 | 5.76 | 1258.65 |
| FP8 + speculative decoding | 1 | 1.36 | 4.28 | 1766.55 |

The optimal draft-token count is not the same for both cases. The unquantized
verifier benefits from two draft tokens because verifier steps are relatively
expensive and accepted draft work replaces enough verifier work. With FP8, the
verifier is already faster; one draft token is better because extra draft work
has less room to pay for itself.

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
position. Position 0 is the ordinary first drafted token accuracy. At later
positions, full accuracy drops because every earlier drafted token also had to
match.

`cond_acc` measures accuracy at a draft position conditioned on the earlier
drafted positions being correct. It separates "can the model predict this
position?" from failures caused by earlier draft-token mistakes.

Accuracy usually decreases for later speculative positions because the draft
model must predict farther into the future with less verifier computation. Any
mistake compounds the context mismatch for later positions, so deeper draft
positions are harder.

If first-position accuracy is very low, inspect data generation before tuning
training hyperparameters. Check that tokenized lengths match hidden-state
lengths, the verifier model and tokenizer match, assistant masks are correct,
the hidden-state vLLM version is compatible, and stale hidden-state temporary
files were cleared.

## Task 3 Answers

FP8 dynamic quantization is useful on H100 because Hopper GPUs have strong FP8
support. Weight and activation FP8 reduces memory bandwidth pressure and can
increase serving throughput while preserving most model quality for many
post-training quantization workloads.

`lm_head` is often excluded because it maps hidden states to vocabulary logits
and is directly responsible for the final probability distribution. Quantizing
it can introduce logit noise that harms generation quality and speculative
acceptance disproportionately.

Quantization can affect speculative decoding acceptance rate because the
verifier's logits may shift. If the draft head was trained against the BF16
verifier but served against a quantized verifier, small distribution changes can
increase or decrease agreement. This is why the combined setup must be
benchmarked separately.

## Task 4 Answers

Speculative decoding can improve throughput even when acceptance is far below
100% because accepted draft tokens let the system emit multiple tokens per
verifier step. The draft model is cheaper than the verifier, so partial
acceptance can still reduce average verifier work per output token.

For the reference setup, the optimal speculative-token count is 2 for the BF16
verifier speculative run and 1 for FP8 + speculative. The final choice is based
on output throughput first, then acceptance length and TPOT. More draft tokens
are only useful when accepted often enough to offset the extra draft compute.

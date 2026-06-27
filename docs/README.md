# Documentation

In-depth concepts and methodology behind ternlight. For end-user usage, see
[`packages/ternlight/`](../packages/ternlight). For build and contribution
notes specific to one component, see the README in that directory.

## What's here

| Doc | What it covers |
|---|---|
| [overview.md](overview.md) | Project overview — what ternlight is, how the three technical choices stack, when to use it |
| [architecture.md](architecture.md) | System design — the `.bin` format, runtime model, packaging pipeline |
| [inference-engine.md](inference-engine.md) | Runtime internals — engine layout, BitLinear forward pass, tokenization path |
| [model-internals.md](model-internals.md) | Canonical math reference — forward pass, backprop, distillation dynamics |
| [eval-methodology.md](eval-methodology.md) | Quality scorecard methodology — each metric, how to reproduce it |

## Reading paths

**New visitor curious about the project**
Start with [overview.md](overview.md). End-to-end framing in a single read.

**Contributor touching the inference engine**
Read [architecture.md](architecture.md) → [inference-engine.md](inference-engine.md).
Use [model-internals.md](model-internals.md) as the canonical reference when
making sure runtime behavior matches training-time behavior.

**Researcher reproducing results**
Start at [eval-methodology.md](eval-methodology.md) for the scorecard, then
[`eval/quality/RESULTS.md`](../eval/quality/RESULTS.md) for the numbers.
[model-internals.md](model-internals.md) covers training-side math.

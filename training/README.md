# training

Python pipeline that produces the `.bin` model file consumed by the Wasm engine. Two stages:

```
training/
├── distill/    Stage 1 — distillation training (Python, GPU)
│   ├── prepare.py       build the .pt cache (MS MARCO + teacher embeddings)
│   ├── train.py         training entry point (fp32 baseline or QAT, by config)
│   ├── evaluate.py      go/no-go scorecard (Task 1, 2, 3)
│   ├── trainer.py       Trainer class (warmup → QAT controlled by config)
│   ├── model.py         StudentEncoder + attention + FFN + transformer block
│   ├── quantization.py  BitLinear swap, embedding ternarization, zero-frac monitor
│   ├── loss.py          distillation + contrastive
│   ├── data.py          TernDataset + collate
│   ├── config.py        pydantic schema + YAML loader
│   ├── configs/         per-tier configs (micro.yaml, micro-fp32.yaml, small.yaml, smoke.yaml)
│   ├── corpora/         eval data (general.jsonl, tech.jsonl)
│   └── tests/           model shapes, loss math, quant round-trip, smoke train
└── pack/       Stage 2 — bit-pack .pt → .bin (Python, CPU)
    ├── pack.py          read .pt, ternarize embedding, pack 2 bits/weight, write .bin
    ├── verify.py        round-trip a packed .bin and compare against the .pt
    └── tests/
```

## When to touch what

- **New training run** — edit `distill/configs/*.yaml`, run `distill/train.py`
- **New loss function** — `distill/loss.py`
- **Architecture change** — `distill/model.py`, then update `distill/configs/*.yaml`
- **Quantization change (training-time or post-train)** — `distill/quantization.py` (must stay in sync with `pack/pack.py` — they apply the same math at different stages)
- **Bit-packing / .bin format change** — `pack/pack.py`
- **Eval methodology change** — `distill/evaluate.py` for go/no-go; `eval/regression/` (at repo root) for ongoing quality regression

## Outputs

```
distill/runs/<run-name>/
    └── checkpoint_ep<N>.pt          float32 shadow weights + config + optimizer state
                                     (NOT committed — see .gitignore)
        ↓ pack.py
pack/out/model.bin                   packed ternary + projection (the shipped artifact)
                                     (also NOT committed — attached to GitHub Release)
        ↓ scripts/release-model.sh
github.com/.../releases/v0.1.0       per-version model binary download
        ↓ at npm publish time
packages/semantic/model.bin          bundled into the published package
```

## Math reference

What the model actually computes — forward pass, backprop, distillation dynamics — is documented in [../docs/training/model-internals.md](../docs/training/model-internals.md). Read that before changing the training code.

## Status

Pre-alpha. Source code migration from `tern-distill-prototype/poc/` (→ `distill/`) and `tern-distill-prototype/export/` (→ `pack/`) pending.

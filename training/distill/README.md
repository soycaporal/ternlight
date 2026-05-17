# training/distill

Stage 1 of the training pipeline: distill a small ternary student from a frozen teacher. Produces a `.pt` checkpoint that `training/pack/` consumes.

## Quickstart

```bash
# install uv if needed: curl -LsSf https://astral.sh/uv/install.sh | sh
cd training/distill
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 1. One-time: build the cached training dataset
python prep/prepare.py --config configs/micro.yaml

# 2. Train (uses GPU if available)
python train.py --config configs/micro.yaml

# 3. Run go/no-go eval suite on the final checkpoint
python evaluate.py --checkpoint runs/micro-qat-1M-60ep/checkpoint_ep60.pt
```

## Layout

```
distill/
├── prep/                  Phase 1 — data preparation (one-time setup)
│   ├── prepare.py             entry point: read config, build cache
│   └── ingest.py              helpers: loaders, dedup, tokenize, encode, split, manifest, stats
├── train.py               Phase 2/3 entry point: parse args, load config, run training
├── trainer.py             Trainer class (warmup → QAT controlled by config, not by file)
├── model.py               StudentEncoder + attention + FFN + transformer block
│                          (explicit Q/K/V Linear layers — BitLinear-reachable)
├── quantization.py        BitLinear swap, embedding ternarization, zero-frac health
├── loss.py                distillation (cosine) + contrastive (within-batch guardrail)
├── evaluate.py            Phase 4 entry point: load checkpoint, run Tasks 1/2/3, emit verdict
├── data.py                cross-phase: TernDataset, collate_fn, save_cache, load_cache
├── config.py              cross-phase: pydantic schemas + YAML loader
├── configs/               per-tier configs
│   ├── micro.yaml             d_model=256, full QAT run
│   ├── micro-fp32.yaml        float32 baseline (the ruler for QAT)
│   ├── small.yaml             d_model=384 fallback
│   └── smoke.yaml             tiny config for CI / quick local check
├── corpora/               local eval data (jsonl) — review/edit without touching Python
│   ├── general.jsonl
│   └── tech.jsonl
└── requirements.txt
```

`prep/` is the only sub-package — it's a discrete one-time setup phase with multiple files. The training files (model, trainer, loss, quantization) stay flat at root because they're tightly coupled to one another. `data.py` and `config.py` are cross-phase contracts.

## Training pipeline

The pipeline has four phases. Phases 2 and 3 are both invoked through `train.py`; the only difference is which config you pass.

| Phase | Action | Entry point | Key knobs |
|---|---|---|---|
| 1. Data prep | Multi-source mix → tokenize → teacher encode → `.pt` cache | `prep/prepare.py --config <yaml>` | source mix, sample count, teacher id |
| 2. Float32 baseline | Distillation in pure fp32. Establishes the architecture ceiling. | `train.py --config configs/micro-fp32.yaml` | epochs, batch size, LR |
| 3. QAT training | Float32 warmup → BitLinear ternary. Same distillation loss + contrastive guardrail. | `train.py --config configs/micro.yaml` | warmup_epochs, lambda schedule, loss weights |
| 4. Post-train eval | Apply ternary projection to embedding, run Tasks 1/2/3, emit GO/MARGINAL/NO-GO. | `evaluate.py --checkpoint <run>/checkpoint_ep<N>.pt` | quant_embedding (default on) |

Phase 2 is the **ruler**, phase 3 is the **ship**: phase 3 is what we deliver, phase 2 only exists so we can measure how much QAT costs vs. an unconstrained architecture.

## Post-training evaluation (Phase 4)

`evaluate.py` runs three tasks against a checkpoint. Critically, the embedding table is post-train ternarized first — `nn.Embedding` is untouched during QAT, but the shipped `.bin` is ternary, so eval has to apply the same projection to be honest.

| Task | What it tests | Source | Pass threshold |
|---|---|---|---|
| 1. Teacher Alignment | Per-sample cosine sim vs. teacher | 2,000 held-out MS MARCO queries | mean > 0.75 |
| 2. STS-B Ranking | Does the model rank pairs like humans? | `mteb/stsbenchmark-sts` test (1,379 pairs) | AUC > 0.80 |
| 3. Recall@3 | End-to-end retrieval | `corpora/general.jsonl`, `corpora/tech.jsonl` | min(general, tech) R@3 > 0.70 |

Decision table:
- Any **FAIL** → NO-GO; rethink architecture
- Any **MARGINAL** with no FAIL → retry with `configs/small.yaml` (d_model=384)
- All **PASS** → GO; proceed to `training/pack/` for the .bin export

## Math reference

Forward pass, backprop, loss formulation, why each design choice was made — all in [../../docs/training/model-internals.md](../../docs/training/model-internals.md). The Phase 2 postmortem ([../../docs/training/postmortem-bitlinear-asymmetry.md](../../docs/training/postmortem-bitlinear-asymmetry.md)) covers what the *real* BitLinear forward pass computes — relevant when modifying `quantization.py` because the same math has to apply identically in `training/pack/pack.py` and `engine/src/inference.rs`.

## Status

Pre-alpha — code migration from `tern-distill-prototype/poc/` pending.

# @tern — Monorepo Structure

> Captures the long-term repository structure for open source release. The current `tern-core/` working directory becomes the repo root (renamed to `tern/` or the final repo name). No code has been written against this structure yet — this is the target to migrate toward before Milestone 2 code begins.

---

## Why Monorepo

The training pipeline, Wasm engine, and JS packages are too interdependent to split across repos. The engine build output feeds directly into each JS package. The training pipeline produces the model binary that bundles into each package. For open source contributions, everything needs to be in one place.

---

## Top-Level Structure

```
tern/                               ← repo root (rename from tern-core)
│
├── packages/                       ← JS packages (pnpm workspace)
│   ├── semantic/                   @tern/semantic — embedding + similarity
│   ├── classify/                   @tern/classify — intent routing
│   ├── filter/                     @tern/filter — spam/toxicity
│   └── core/                       @tern/core — shared types, JS utils
│
├── engine/                         ← Rust crate → compiles to engine.wasm
│   ├── src/
│   ├── tests/                      engine parity tests (test_embed.js, test_qkv.js, ...)
│   │                               "does the Rust math match Python at the element level?"
│   ├── Cargo.toml
│   └── assets/
│       └── tokenizer.json          committed — BERT vocab, embedded at compile time
│
├── training/                       ← Python training pipeline
│   ├── distill/                    Stage 1 — distillation training (fp32 baseline + QAT)
│   │   ├── prepare.py              build the .pt cache (MS MARCO + teacher embeddings)
│   │   ├── train.py                training entry point (fp32 or QAT, by config)
│   │   ├── evaluate.py             Phase 1 go/no-go scorecard on the .pt checkpoint
│   │   ├── trainer.py              Trainer class (warmup → QAT by config)
│   │   ├── model.py                StudentEncoder + attention + FFN
│   │   ├── quantization.py         BitLinear swap, embedding ternarization, zero-frac health
│   │   ├── loss.py                 distillation + contrastive
│   │   ├── data.py                 TernDataset + collate
│   │   ├── config.py               pydantic schema + YAML loader
│   │   ├── configs/                per-tier configs (micro.yaml, micro-fp32.yaml, ...)
│   │   ├── corpora/                eval data (general.jsonl, tech.jsonl)
│   │   ├── tests/
│   │   └── requirements.txt
│   └── pack/                       Stage 2 — bit-pack .pt → .bin
│       ├── pack.py                 read .pt, ternarize, pack 2 bits/weight, write .bin
│       ├── verify.py               round-trip a packed .bin against the source .pt
│       └── tests/
│
├── eval/                           ← cross-cutting engine-quality evaluation
│   ├── regression/                 engine vs Phase 1 baselines on real eval tasks
│   │   ├── prepare_eval_data.py    one-time: cache MS MARCO, STS-B, retrieval corpora
│   │   ├── regression_test.js      runs the shipped engine against baselines
│   │   └── test_data/              cached reference data (gitignored if large)
│   ├── benchmarks/                 perf — latency (cold/warm), throughput, memory
│   │   ├── latency.js
│   │   └── memory.js
│   ├── compatibility/              target-runtime matrix (Node, Browser, CF Workers, Deno, Bun)
│   │   └── runtimes.yaml
│   ├── results/                    committed JSON outputs per release version
│   │   ├── v0.1.0.json
│   │   └── v0.1.1.json
│   └── REPORT.md                   human-readable scorecard, regenerated per release
│
├── models/                         ← model release registry (no binaries in git)
│   └── README.md                   points to GitHub Releases / HuggingFace Hub
│
├── docs/                           ← all current tern-core .md files move here
│   ├── tern-scoping.md
│   ├── tern-architecture.md
│   ├── tern-model-sizing.md
│   ├── tern-phase1-prototype.md
│   ├── tern-phase2-prototype.md
│   ├── tern-future-work.md
│   ├── tern-monorepo.md            ← this file
│   ├── training/
│   │   ├── design.md
│   │   ├── milestones.md
│   │   ├── setup.md
│   │   ├── implementation-guide.md
│   │   ├── model-internals.md      forward pass + backprop + distillation reference
│   │   ├── phase-1-conclusion.md
│   │   └── postmortem-bitlinear-asymmetry.md
│   └── eval/
│       └── methodology.md          how the scorecard is computed, what each metric means
│
├── notebooks/                      ← learning + exploration (not part of build)
│   └── 01-ternary-transformer/
│       ├── 01-attention-from-scratch.ipynb
│       ├── 02-bitlinear-layer.ipynb
│       ├── 03-full-model-architecture.ipynb
│       └── 04-distillation-training.ipynb
│
├── scripts/                        ← build orchestration
│   ├── build-engine.sh             cargo build → wasm-opt → copy to packages/*/
│   └── release-model.sh            push .bin to GitHub Releases
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                  lint + test JS packages
│   │   └── build-engine.yml        Rust → Wasm build check on PR
│   └── CONTRIBUTING.md
│
├── pnpm-workspace.yaml             JS monorepo workspace config
├── Cargo.toml                      Rust workspace root (members: engine/)
├── package.json                    root — tooling only, not published to npm
└── README.md
```

---

## Workspace Configs

**`pnpm-workspace.yaml`**
```yaml
packages:
  - 'packages/*'
```

**`Cargo.toml` (root)**
```toml
[workspace]
members = ["engine"]
```

**Python** — no workspace tooling yet. `training/distill/requirements.txt` is sufficient. A `packages/python/` directory with `pyproject.toml` gets added if/when a Python package ships.

---

## Build Flow

```
training/distill/
    ↓  train.py
    ↓  pack/pack.py
model.bin (~1.75MB micro)  →  GitHub Release asset
    ↓
    ↓                     engine/ (Rust)
    ↓                         ↓  cargo build --target wasm32-unknown-unknown
    ↓                     engine.wasm (~750KB)
    ↓                         ↓
    └──────────────────→  packages/semantic/
                              ├── index.js
                              ├── engine.wasm   ← from engine/ build output
                              └── model.bin     ← from GitHub Release, bundled at publish time
                                  ↓
                              npm publish @tern/semantic
```

---

## Evaluation & Quality Reporting

A separate top-level concern from training. Training-time eval (per-epoch loss, val/spearman) lives inside `training/distill/trainer.py` because it's about *training health*. Phase 1 go/no-go eval on the .pt checkpoint lives in `training/distill/evaluate.py`. Engine-quality eval (against the shipped Wasm artifact) lives in `eval/` because it's about *what we ship*.

### Four kinds of testing, each with one home

| Kind | Question it answers | Lives in |
|---|---|---|
| Engine parity tests | Does the Rust math match Python at the element level? | `engine/tests/` |
| Training-time eval | Is training going well? | `training/distill/trainer.py` (per-epoch val) |
| Phase 1 go/no-go eval | Does the .pt checkpoint pass the 3-task scorecard? | `training/distill/evaluate.py` |
| Engine quality eval | Does the shipped engine produce eval-quality embeddings? | `eval/regression/` |
| Engine perf / compat | How fast, how big, where does it run? | `eval/benchmarks/`, `eval/compatibility/` |
| Package integration | Does the JS API behave correctly? | `packages/*/tests/` |

The Phase 2 lessons (see [docs/training/postmortem-bitlinear-asymmetry.md](docs/training/postmortem-bitlinear-asymmetry.md)) are baked into this split. Engine parity tests alone aren't enough — they only validate against whatever reference you wrote. Engine quality eval against held-out tasks is the test that catches "the engine computes consistent but wrong math."

### The release scorecard

Every release publishes a multi-dimensional scorecard. No headline numbers without their gaps. No claims without methodology.

The six dimensions:

| Dimension | What gets measured | Why users care |
|---|---|---|
| **Quality** | Teacher alignment cosine sim, STS-B AUC + Spearman, MTEB subset scores, R@K on retrieval | "Is the model good?" |
| **Quantization gap** | Float32 baseline vs ternary, per-task and per-component (embedding / BitLinear / projection) | "What does the small size cost me?" |
| **Performance** | Latency (cold + warm), throughput, memory peak, per-target runtime | "Will it fit my latency / memory budget?" |
| **Size** | `.wasm` bytes, `.bin` bytes, total bundled, gzipped over the wire | "How much does my user download?" |
| **Compatibility** | Required Wasm features (SIMD, bulk memory), min Node / browser versions, OS notes | "Will it run where I need it?" |
| **Honest comparison** | Side-by-side with transformers.js + quantized MiniLM, ONNX Runtime Web, server APIs | "Why this over alternatives?" |

### Operating principles

- **Always publish the gap, not just the headline.** "Task 2 AUC = 0.84" alone is dishonest. "Task 2 AUC = 0.84 (vs 0.86 for full-precision teacher; 0.85 for transformers.js + MiniLM)" is honest. The OSS embedding ecosystem has a credibility problem with cherry-picked benchmarks; leading with honest comparison earns disproportionate trust.
- **Version-anchor everything.** `eval/results/v0.1.0.json` is committed to the repo at release tag time. Cross-version diffs surface regressions automatically.
- **Reproducible methodology.** Every metric in the scorecard has a script in `eval/` that produces it. PRs that add a metric must add the producing script. No metric exists in the scorecard without committed code that generates it.
- **Same harness, every release.** The scorecard regeneration is one command (`scripts/run-eval.sh`) that exercises `eval/regression/`, `eval/benchmarks/`, `eval/compatibility/` and updates `eval/REPORT.md` + `eval/results/v{X.Y.Z}.json`. Manual scorecard updates are a smell — fix the harness, don't paper over.
- **Distinguish engine quality from model quality.** A bad model on a correct engine looks the same in some metrics as a buggy engine on a great model. The scorecard separates these by always running the *same input* through both the Python reference (real model) and the engine, and reporting the engine vs reference parity alongside the absolute quality numbers.

### Why this matters for adoption

Open-source embedding projects live or die on two questions: "is the quality believable?" and "does it run in my environment?" The scorecard is the answer to both. Without it, adoption depends on the user manually running their own benchmarks — which most won't, so they'll just pick a project that publishes its own.

---

## Model File Strategy

The `.bin` model files (1.75–3MB depending on tier) **bundle inside the npm package**. This is the "zero config, no network call" product promise — the package works offline, in edge environments, and without a postinstall download step.

Binary files do not live in git. They are attached to GitHub Releases and pulled into the package at **publish time** by the maintainer, not at install time by the user.

```bash
# Run once per release by maintainer:
scripts/release-model.sh v0.1.0 model-micro.bin
# → gh release create v0.1.0-micro --attach model-micro.bin
```

Tiers ship as separate releases and can be independently versioned.

---

## Contributor Layers

The structure is intentionally layered so contributors only need the toolchain for their layer:

| Layer | Directory | Toolchain needed |
|---|---|---|
| JS packages / API | `packages/` | Node.js, pnpm |
| Wasm engine | `engine/` | Rust, wasm-pack |
| Training pipeline | `training/` | Python, PyTorch, GPU |
| Engine quality eval | `eval/` | Node.js, Python (for reference data prep) |
| Documentation | `docs/` | Markdown only |

A JS developer contributing to `@tern/semantic`'s API surface does not need Rust installed. An ML researcher improving distillation does not need to understand Wasm. The engine build output (`engine.wasm`) is committed as a build artifact so JS contributors don't need to rebuild it for routine work.

---

## Current State → Monorepo Mapping

| Current location | Target location |
|---|---|
| `tern-core/*.md` | `docs/` |
| `tern-core/tern-distill-prototype/*.md` | `docs/training/` (markdown docs) |
| `tern-core/tern-distill-prototype/poc/` | `training/distill/` (training code) |
| `tern-core/tern-distill-prototype/export/` | `training/pack/` (.bin bit-packing) |
| `tern-core/tern-distill-prototype/engine/src/` | `engine/src/` |
| `tern-core/tern-distill-prototype/engine/test_*.js` | `engine/tests/` (engine parity tests) |
| `tern-core/tern-distill-prototype/bridge/` | `eval/regression/` (regression suite + ref data) |
| `tern-core/01-ternary-transformer/` | `notebooks/` *(if kept)* |
| `refs/` *(local only)* | `refs/` at root *(or external, not committed)* |

**The rename from `tern-core` → `tern` is the only disruptive change.** Everything else is moving files into a cleaner hierarchy. This should be done before any training code is written to avoid re-pathing imports and config references later.

---

## Python Package (Future)

A Python package that wraps the same Wasm engine via `wasmtime-py` would live at `packages/python/`. This is a future work item — see `tern-future-work.md`. The monorepo structure accommodates it without changes.

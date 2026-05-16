# eval — engine quality + performance scorecard

Cross-cutting evaluation of the *shipped* engine (engine + bundled model + JS API). Distinct from:

- `engine/tests/` — element-level parity tests (does the Rust math match Python?)
- `training/distill/trainer.py` — training-time validation (is training going well? per-epoch val/spearman, zero-fraction health)
- `training/distill/evaluate.py` — Phase 1 go/no-go scorecard on the .pt checkpoint
- `packages/*/tests/` — JS API integration tests

## Layout

```
eval/
├── regression/       Engine vs Phase 1 baselines on real eval tasks
│                     "Does the shipped engine produce eval-quality embeddings?"
├── benchmarks/       Latency (cold + warm), throughput, memory
│                     "How fast and how big?"
├── compatibility/    Target-runtime matrix (Node, Browser, CF Workers, Deno, Bun)
│                     "Where does it run?"
├── results/          Committed JSON outputs per release version (v0.1.0.json, ...)
└── REPORT.md         Human-readable scorecard, regenerated per release
```

## Run all evals

```bash
bash scripts/run-eval.sh        # populates results/v<X.Y.Z>.json + regenerates REPORT.md
```

## The release scorecard

Six dimensions, every release. See [../docs/eval/methodology.md](../docs/eval/methodology.md) for the full methodology — what each metric measures, how to reproduce it, why we chose it.

| Dimension | What gets measured |
|---|---|
| Quality | Teacher alignment cosine sim, STS-B AUC + Spearman, MTEB subset, R@K |
| Quantization gap | Float32 baseline vs ternary, per-task and per-component |
| Performance | Latency (cold + warm), throughput, memory peak, per-target |
| Size | .wasm bytes, .bin bytes, total bundled, gzipped over the wire |
| Compatibility | Required Wasm features, min Node/browser, OS notes |
| Honest comparison | Side-by-side with transformers.js + MiniLM, ONNX, embedding APIs |

## Operating principles

1. **Always publish the gap, not just the headline.** Numbers without context mislead.
2. **Version-anchor everything.** `results/v<X.Y.Z>.json` is committed at release time.
3. **Reproducible methodology.** Every metric in the scorecard has a script in `eval/` that produces it.
4. **Same harness, every release.** `scripts/run-eval.sh` regenerates everything; manual scorecard updates are a smell.
5. **Distinguish engine from model.** Both engine bugs and model weaknesses show up as metric drops — separate them by reporting parity vs absolute quality side-by-side.

## Status

Pre-alpha. Migration from `tern-distill-prototype/bridge/` pending.
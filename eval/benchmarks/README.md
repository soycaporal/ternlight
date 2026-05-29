# eval/benchmarks

Smoke tests and performance measurements of the **shipped WASM artifact** (engine + `.bin`).

Distinct from `training/distill/evaluation.py`, which measures the `.pt` checkpoint via Python `unpack.UnpackedModel.forward()`. This directory measures the actual deliverable users load.

## What's here

```
benchmarks/
├── README.md                  this file
├── smoke.js                   semantic similarity sanity (10 sentence pairs)
├── perf.js                    cold-start + per-query latency + memory + bundle size
├── corpus/
│   └── smoke-pairs.json       the 10 sentence pairs as data
└── results/                   per-run JSON dumps — GITIGNORED (machine-specific)
```

## Running

The scripts assume the engine is already built. From repo root:

```bash
# Build the engine (skip if already built)
cd engine && wasm-pack build --target nodejs --features emb_int8 && cd ..

# Smoke test — eyeball check that similarity scores match expectations
node eval/benchmarks/smoke.js

# Perf baseline — JSON to stdout, summary to stderr
node eval/benchmarks/perf.js > eval/benchmarks/results/$(date +%Y%m%d)-$(git rev-parse --short HEAD)-emb_int8.json
```

To test a different build target, change the feature flag, re-pack the matching `.bin` to `engine/assets/model.bin`, and re-run.

## Where baselines live

Per-run JSON in `results/` is **gitignored** — machine-specific, regenerated freely.

Curated baseline numbers live in [`docs/tern-inference-engine.md` → Target-device perf](../../docs/tern-inference-engine.md). One row per (build, device, date). That's the source of truth for "how fast was this thing at point X."

## Why split smoke and perf

| | `smoke.js` | `perf.js` |
|---|---|---|
| Question | "Does the model do the right thing semantically?" | "How fast is the build?" |
| Output | Human-readable per-pair verdicts | JSON for machine diffing |
| Cadence | After every build, eyeball before shipping | After every optimization, compare to prior baseline |
| Audience | Whoever's reviewing the engine | Whoever's tracking perf over time |

## Out of scope (deferred)

- **Quality eval against held-out tasks** (Spearman, NDCG@10, recall@k) — lives in [`../regression/`](../regression/). The parity tests (`engine/tests/`) prove engine quality matches the `.pt` to ~1e-7, so the Python eval numbers carry over to the engine.
- **Per-runtime portability** (Cloudflare Workers, Deno, Bun, browser) — lives in [`../compatibility/`](../compatibility/) when implemented. Node.js is the first cut.
- **Cold-cache vs warm-cache** — Node measurements are warm-OS-cache by default; that's the user-relevant case.
- **Larger query corpus** — `perf.js` uses ~15 hardcoded strings for v1. A real workload corpus (e.g., SciFact's 300 test queries) belongs upstream as a data dependency, not embedded here.

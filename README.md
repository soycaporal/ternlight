<p align="center">
  <img src="docs/assets/banner.jpeg" alt="ternlight" width="960">
</p>

# ternlight

[![CI](https://github.com/soycaporal/ternlight/actions/workflows/ci.yml/badge.svg)](https://github.com/soycaporal/ternlight/actions/workflows/ci.yml)
[![Build Engine](https://github.com/soycaporal/ternlight/actions/workflows/build-engine.yml/badge.svg)](https://github.com/soycaporal/ternlight/actions/workflows/build-engine.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://ternlight-demo.vercel.app)

> **Lightening-fast semantic embeddings in a 7 MB WebAssembly bundle.**
> Engine + model + tokenizer shipped together. Embedding search on CPU — no API calls, no GPU.

**[Try the live demo](https://ternlight-demo.vercel.app)** - search 2k React docs entirely on-device

Distilled from [`all-MiniLM-L6`][minilm], with [BitNet b1.58][bitnet] style quantization-aware training. Three core design choices stack to fit an embedding model in 7 MB:

- **Ternary weights.** Every weight is one of three values: `-1`, `0`, or `+1`. Inference becomes add and subtract, no matmul operations. Quality holds because the model is trained for ternary weights from the start, not quantized after the fact.
- **One bundle.** Model and full BERT tokenizer pack into a single WASM file. `npm install` and you're done — no postinstall step, no runtime fetch.
- **SIMD inference engine.** Inference engine is written in Rust and compiled to WASM with SIMD. Add/subtract math leans on CPU vector instructions.

[minilm]: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
[bitnet]: https://arxiv.org/abs/2402.17764

<p align="center">
  <img src="eval/quality/charts/pareto.png" alt="Quality vs size — ternlight reaches 30× compression with a modest accuracy drop" width="720">
</p>

## Overview

A 2-layer Transformer encoder, trained with quantization-aware distillation and int4-quantized at the embedding layer.

| Spec         |                                                                         |
| ------------ | ----------------------------------------------------------------------- |
| Bundle       | 7 MB (engine + 4.6 MB model + 695 KB tokenizer, all inside one `.wasm`) |
| Output       | 384-dim L2-normalized vector                                            |
| Max input    | 128 tokens (~95 English words)                                          |
| Architecture | 2-layer Transformer · d_model=256 · 4 attention heads                   |
| Parameters   | ~9.5M                                                                   |
| Targets      | Node 18+, modern browsers with WASM SIMD, edge runtimes                 |
| License      | MIT                                                                     |

### Results 

Based on `emb_int4` quantized embedding - shipped build

| Metric                           |                  |
| -------------------------------- | ---------------: |
| Spearman vs MiniLM-L6 teacher    |        **0.835** |
| Quality retained vs fp32 student |          **95%** |
| Compression vs fp32 student      |         **8.2×** |
| Latency p50 (M-series Mac)       |        **~2 ms** |
| Throughput                       | **~500 emb/sec** |

---

## Install and usage

```bash
npm install ternlight
```

```js
import { embed, cosineSim, similar } from 'ternlight';

// One primitive: turn a string into a 384-dim L2-normalized vector
const v1 = embed("arctic terns migrate from pole to pole");
const v2 = embed("longest migration in the animal kingdom");

cosineSim(v1, v2);   // ~0.71 — same concept, different words

// Nearest-neighbor search over a corpus
const matches = similar("which seabird travels farthest", [
  "arctic terns migrate from pole to pole",
  "puffins dive underwater for fish",
  "how to debounce a search input",
], { topK: 2 });
// → [{ text: "arctic terns...", sim: 0.78 },
//    { text: "puffins...",      sim: 0.31 }]
```

> **Status:** v0.1 (pre-alpha). The engine works end-to-end and matches Phase 1 quality baselines. Packaging, performance polish, and public release are still in progress. Not yet on npm.

## Why this exists

On-device embedding unlocks:

- **Search-as-you-type.** Results appear before the user finishes typing. Faster than any network round-trip can be.
- **Privacy-sensitive apps.** Queries and documents never leave the device — no data-handling agreements, no leak risk.
- **Offline-first apps.** Browser extensions, Obsidian plugins, desktop apps.
- **Edge-runtime apps.** Cloudflare Workers, Deno Deploy, Vercel Edge. Embeddings co-locate with your request handler — no separate inference service to call.
- **Edge devices and IoT hardware.** Raspberry Pi, single-board computers, industrial gateways, kiosks. Add/subtract math runs efficiently on ARM cores — no GPU or NPU required.
- **Static sites.** Jekyll, Hugo, Astro. Ship the model with the bundle; semantic search works without a backend.

---

## Repository layout

```
ternlight/
├── packages/         JS packages (npm, pnpm workspace)
│   ├── semantic/     @tern/semantic — embedding API
│   └── core/         @tern/core — shared types
├── engine/           Rust → Wasm inference engine
├── training/         Python distillation pipeline
├── eval/             Engine quality + perf benchmarks
├── docs/             Design docs, architecture, postmortems
├── models/           Model release registry pointers
├── scripts/          Build orchestration
└── .github/          CI workflows
```

## Contributing

There's still tons of headroom for perf and quality improvements. Beyond stacked encoders, I'm curious about other modalities, specially generative use cases that fit the same tight constraints. JS, Rust, and ML contributors all welcome.

See [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).

---

<sub>Banner photo via <a href="https://macaulaylibrary.org/asset/637450290">Macaulay Library</a>, Cornell Lab of Ornithology.</sub>
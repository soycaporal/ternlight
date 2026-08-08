<p align="center">
  <img src="docs/assets/banner.jpeg" alt="ternlight" width="960">
</p>

# ternlight

[![CI](https://github.com/soycaporal/ternlight/actions/workflows/ci.yml/badge.svg)](https://github.com/soycaporal/ternlight/actions/workflows/ci.yml)
[![Build Engine](https://github.com/soycaporal/ternlight/actions/workflows/build-engine.yml/badge.svg)](https://github.com/soycaporal/ternlight/actions/workflows/build-engine.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://ternlight.dev/demo/)

**Lightning-fast semantic embeddings in a 5-7 MB WebAssembly bundle.** Engine + model + tokenizer in one file. Embedding search on CPU - no API calls, no GPU. **[Try the live demo](https://ternlight.dev/demo/)** - search 2k docs entirely on-device.

## Install and usage

Two tiers, same API - pick by size/quality trade ([full comparison](#overview)):

```bash
npm install @ternlight/base    # quality tier  - 7 MB wire, ~5 ms/embed
npm install @ternlight/mini    # small tier    - 5 MB wire, ~2.5 ms/embed
```

```js
import { embed, cosineSim, similar } from '@ternlight/base';

// One primitive: string → 384-dim L2-normalized Float32Array
cosineSim(embed('reset my password'), embed('I forgot my password'));   // 0.88

// Nearest-neighbor search over a corpus
similar('I want my money back', [
  'Refunds: how to get your money back',
  'Track the status of your delivery',
  'Update your billing address',
], { topK: 2 });
// → [{ text: 'Refunds: how to get your money back', sim: 0.70 },
//    { text: 'Update your billing address',         sim: 0.24 }]
```

Works in Node ≥ 18, browsers (via any bundler), Cloudflare Workers, Vercel Edge, Deno, and Bun - the package routes each environment to the right loader. Package docs: [`@ternlight/base`](packages/base/README.md) · [`@ternlight/mini`](packages/mini/README.md).

## Overview

Distilled from [`all-MiniLM-L6`][minilm] with [BitNet b1.58][bitnet]-style quantization-aware training. Three design choices stack to fit an embedding model in a few MB:

- **Ternary weights** - every weight is `-1`, `0`, or `+1`; inference is adds and subtracts. Quality holds because the model trains as a ternary model from the start.
- **One bundle** - model + BERT tokenizer + engine in a single `.wasm`. No postinstall step, no runtime fetch.
- **SIMD inference engine** - hand-written Rust compiled to WASM SIMD; the add/subtract math rides CPU vector instructions.

[minilm]: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
[bitnet]: https://arxiv.org/abs/2402.17764

<p align="center">
  <img src="eval/quality/charts/pareto.png" alt="Quality vs size - ternlight reaches 30× compression with a modest accuracy drop" width="720">
</p>

All numbers measured on the shipped int4 builds (M-series Mac, Node/V8):

|                                  | @ternlight/mini            | @ternlight/base            |
| -------------------------------- | -------------------------- | -------------------------- |
| **Wire size** (gzipped wasm)     | **5.0 MB**                 | **7.2 MB**                 |
| **Latency** (p50 per embed)      | **2.5 ms**                 | **5.1 ms**                 |
| Throughput (single-thread)       | ~400 emb/s                 | ~195 emb/s                 |
| Spearman vs MiniLM-L6 teacher    | 0.820                      | **0.844**                  |
| Retrieval (SciFact NDCG@10)      | 0.439                      | **0.465**                  |
| Architecture                     | 2-layer · d_model=256 · 4 heads | 2-layer · d_model=384 · 6 heads |
| Parameters                       | ~9.5M                      | ~15.4M                     |
| Output                           | 384-dim L2-normalized      | 384-dim L2-normalized      |
| Max input                        | 128 tokens (~95 words)     | 128 tokens (~95 words)     |
| Quantization                     | ternary weights · int4 embeddings | ternary weights · int4 embeddings |

## Why this exists

On-device embedding unlocks:

- **Search-as-you-type.** Results appear before the user finishes typing. Faster than any network round-trip can be.
- **Privacy-sensitive apps.** Queries and documents never leave the device - no data-handling agreements, no leak risk.
- **Offline-first apps.** Browser extensions, Obsidian plugins, desktop apps.
- **Edge-runtime apps.** Cloudflare Workers, Deno Deploy, Vercel Edge. Embeddings co-locate with your request handler - no separate inference service to call.
- **Edge devices and IoT hardware.** Raspberry Pi, single-board computers, industrial gateways, kiosks. Add/subtract math runs efficiently on ARM cores - no GPU or NPU required.
- **Static sites.** Jekyll, Hugo, Astro. Ship the model with the bundle; semantic search works without a backend.

---

## Repository layout

```
ternlight/
├── packages/         Published npm packages (pnpm workspace)
│   ├── base/         @ternlight/base - quality tier (d384)
│   └── mini/         @ternlight/mini - small/fast tier (d256)
├── engine/           Rust → Wasm inference engine
├── training/         Python distillation + QAT pipeline, packer
├── eval/             Engine quality + perf benchmarks
├── docs/             Design docs
├── models/           Model release registry pointers
├── scripts/          Build + release orchestration
└── .github/          CI workflows
```

Deeper reading: [project overview](docs/overview.md) · [architecture](docs/architecture.md) · [inference engine](docs/inference-engine.md) · [model internals](docs/model-internals.md) · [eval methodology](docs/eval-methodology.md).

## Bundling

The browser build ships WebAssembly as an ES-module asset (wasm-bindgen bundler target). Modern bundlers handle this natively; a couple need a one-line adjustment. Thanks to [@dmisdm](https://github.com/dmisdm) for [surfacing the Vite setup](https://github.com/soycaporal/ternlight/issues/3).

**Vite** - exclude the package from dependency pre-bundling so Vite's own pipeline instantiates the WASM:

```js
// vite.config.js
export default {
  optimizeDeps: {
    exclude: ['@ternlight/base'],   // or '@ternlight/mini'
  },
};
```

Requires **Vite ≥ 8.1**, which handles WASM ESM out of the box with no plugin ([release notes](https://vite.dev/blog/announcing-vite8-1)). On older Vite, also add [`vite-plugin-wasm`](https://www.npmjs.com/package/vite-plugin-wasm). Astro, SvelteKit, and Nuxt are Vite-based - same fix would apply.

**Next.js** - enable async WASM in the webpack config; the model runs client-side, so import it from a Client Component (or via `next/dynamic`):

```js
// next.config.js
module.exports = {
  webpack(config) {
    config.experiments = { ...config.experiments, asyncWebAssembly: true };
    return config;
  },
};
```

**webpack 5** - same experiment flag:

```js
// webpack.config.js
module.exports = {
  experiments: { asyncWebAssembly: true },
};
```

**Cloudflare Workers, Vercel Edge, Deno, Bun** - work with no extra config; the package routes each to the right loader.

## Contributing

There's still tons of headroom for perf and quality improvements. Beyond stacked encoders, I'm curious about other modalities, specially generative use cases that fit the same tight constraints. JS, Rust, and ML contributors all welcome.

See [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md).

## Acknowledgments

ternlight builds on three open-source efforts:

- **[BitNet b1.58](https://arxiv.org/abs/2402.17764)** (Ma et al., Microsoft Research, 2024) - the architectural research underlying ternary weight training.
- **[`bitlinear`](https://github.com/schneiderkamplab/bitlinear)** by [@schneiderkamplab](https://github.com/schneiderkamplab) - the reference PyTorch implementation of BitLinear. We use it directly during training (`bitlinear==2.4.6`) and the Rust inference engine mirrors its forward-pass math byte-for-byte.
- **[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)** - the teacher model the student is distilled from.

The Rust engine in [`engine/src/kernels.rs`](engine/src/kernels.rs) is an independent reimplementation of `bitlinear`'s `BitLinear.forward()` for the WASM target; parity tests guard against drift.

## License

MIT - see [LICENSE](LICENSE).

---

<sub>Banner photo via <a href="https://macaulaylibrary.org/asset/637450290">Macaulay Library</a>, Cornell Lab of Ornithology.</sub>

---

<p align="center"><em>In memory of Alex Movsessian - who held software to the highest standard, and treated everyone around him with kindness.</em></p>

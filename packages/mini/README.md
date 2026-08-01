# @ternlight/mini

**On-device semantic embeddings for JavaScript - the small, fast tier. 5 MB on the wire, ~2.5 ms per embedding, zero API calls.**

A 1.58-bit (BitNet-style ternary) sentence encoder compiled to WebAssembly. Give it text, get back a 384-dimensional unit vector for semantic search, FAQ matching, deduplication, or clustering — computed entirely on the user's CPU. No network, no GPU, no ML runtime, no model download at runtime: the model ships inside the wasm.

```bash
npm install @ternlight/mini
```

```js
import { embed, cosineSim, similar } from '@ternlight/mini';

// embed() → Float32Array(384), L2-normalized — compare any two with a dot product
cosineSim(embed('reset my password'), embed('I forgot my password'));   // 0.91

// or top-K semantic search over any list of strings
const results = similar('how do I reset my password', faqEntries, { topK: 3 });
// [{ text: 'Resetting a forgotten password', sim: 0.80 }, ...]
```

Works in Node ≥ 18, browsers (via any bundler), Cloudflare Workers, Vercel Edge, Deno, and Bun — one package, the right loader is picked automatically.

## mini vs base

`mini` is the size/speed tier; [`@ternlight/base`](https://www.npmjs.com/package/@ternlight/base) is the quality tier with the same API:

| | **@ternlight/mini** | @ternlight/base |
|---|---|---|
| Wire size (gzipped wasm) | **~5.0 MB** | ~7.2 MB |
| Embed latency (p50, M-series CPU) | **~2.5 ms** | ~5 ms |
| Teacher fidelity (Spearman) | 0.820 | 0.844 |
| Paraphrase handling | good | noticeably stronger |

Rule of thumb: browser bundles and latency-critical UI → `mini`; server-side, retrieval quality, or paraphrase-heavy matching → `base`. Switching later is a one-line import change.

## API

| Function | Description |
|---|---|
| `embed(text)` | → `Float32Array(384)`, unit-length. Sync, ~2.5 ms. Truncates at 128 tokens. |
| `cosineSim(a, b)` | Cosine similarity of two embeddings (a dot product — they're normalized). |
| `similar(query, corpus, { topK })` | Embed query + corpus, return top-K `{ text, sim }` sorted. |
| `engineInfo()` | Build/model info string — dimensions, quantization format. |
| `TernError` | Typed error (`INVALID_INPUT`, `DIM_MISMATCH`). |

For repeated searches, embed your corpus once and reuse the vectors:

```js
const index = docs.map((d) => ({ d, v: embed(d.text) }));
const q = embed(query);
index.sort((a, b) => cosineSim(q, b.v) - cosineSim(q, a.v));
```

## Bundler setup (browsers)

The wasm is imported as an ES module. Webpack 5 needs one flag; Vite needs the wasm plugin:

```js
// webpack.config.js
experiments: { asyncWebAssembly: true }

// vite.config.js
import wasm from 'vite-plugin-wasm';
export default { plugins: [wasm()] };
```

Node needs nothing — `require()` or `import` and go.

**No bundler at all?** (plain `<script type="module">`, hand-rolled setups) — use the `/web` subpath, which has an explicit `init()` that fetches the wasm:

```js
import init, { embed } from '@ternlight/mini/web';
await init();                    // fetches + compiles the wasm (once, then cached)
const v = embed('hello world');  // Float32Array(384)
```

Note: `/web` exposes the raw engine exports (`embed`, `tokenize`, `config_summary`) without the JS helper layer — `cosineSim` is a dot product one-liner if you need it.

## How it works

Three ideas stacked: **(1)** a small transformer student is distilled from [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) *while being trained as a ternary model* (QAT), so quantization costs almost nothing; **(2)** ternary weights pack to 2 bits each, putting the whole model + tokenizer + engine in one wasm file; **(3)** the forward pass is hand-written Rust compiled to WASM with explicit SIMD, so it runs at near-native speed in every JS runtime. Details in the [repo docs](https://github.com/soycaporal/ternlight/tree/main/docs).

## License

MIT

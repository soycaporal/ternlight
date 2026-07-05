# ternlight

> On-device semantic embeddings for JavaScript. Embed text, compare
> similarity, find nearest neighbors — locally, no API keys, no backend.

A 1.58-bit BitNet-style sentence embedder distilled from
`sentence-transformers/all-MiniLM-L6-v2`. The compiled engine and trained
model fit in **~5 MB of WASM** and run inference in **~2 ms per query** on
modern CPUs.

## Install

```bash
npm install ternlight
```

The package bundles the compiled WASM engine. The model weights and BERT
tokenizer are embedded directly into the `.wasm` — no postinstall download,
no asset fetching at runtime.

## Use

```js
const { embed, cosineSim, similar } = require('ternlight');

// 1. Embed a single string → 384-dim L2-normalized Float32Array
const v1 = embed("how do I reset my password");
const v2 = embed("forgot my password");

// 2. Compare two embeddings
cosineSim(v1, v2);  // → ~0.85

// 3. Nearest-neighbor search over a corpus
const corpus = [
  "I forgot my password and need to reset it",
  "where is my package shipment tracking",
  "how to cancel a recurring subscription",
];
const matches = similar("forgot password", corpus, { topK: 3 });
// → [{ text: "I forgot my password...", sim: 0.86 }, ...]
```

## API

### `embed(text: string): Float32Array`

Returns a 384-dim L2-normalized embedding. Output is on the unit hypersphere
— cosine similarity reduces to a dot product.

Input is tokenized via BERT WordPiece and truncated to **128 tokens (~95
English words)**. Longer text is silently truncated, so embed at sentence or
short-paragraph granularity, not full-document.

### `cosineSim(a: Float32Array, b: Float32Array): number`

Cosine similarity between two L2-normalized vectors. For typical text,
output is in `[0, 1]`.

### `similar(query, corpus, opts?): SimilarityResult[]`

Convenience: embed query, embed each corpus item, return top-K matches
sorted descending by similarity.

```ts
interface SimilarityResult {
  text: string;
  sim: number;
}
```

### `engineInfo(): string`

Debug helper that returns the loaded engine's configuration string. Useful
for confirming which build is in use:

```js
engineInfo();
// → "tern-engine v1 | embedding_format=int4 | vocab=30522 d_model=256 ..."
```

### `TernError`

All thrown errors are instances of `TernError`. The `code` property gives a
stable identifier:

- `INVALID_INPUT` — argument wasn't the expected type
- `DIM_MISMATCH` — vectors of different lengths were compared

## Reuse embeddings (recommended for any non-trivial corpus)

`similar()` re-embeds the corpus on every call, which is fine for small
demos but wasteful for repeated searches. For real use, embed the corpus
**once** upfront and reuse the vectors:

```js
const { embed, cosineSim } = require('ternlight');

const corpusEmbeds = corpus.map(embed);  // ~2 ms × N items, do this once

function search(query, k = 5) {
  const q = embed(query);
  return corpusEmbeds
    .map((v, i) => ({ text: corpus[i], sim: cosineSim(q, v) }))
    .sort((a, b) => b.sim - a.sim)
    .slice(0, k);
}
```

At ~500 embeddings/sec on a modern CPU, even a 10,000-item corpus is
~20 seconds to embed once and then sub-3 ms per query forever after.
Cache the vectors to disk for repeat runs.

## Performance

Measured on an M-series Mac, release build with WASM SIMD:

| Metric | Value |
| --- | --- |
| Per-call latency (p50) | ~2 ms |
| Throughput | ~500 embeddings/sec |
| Bundle | ~11 MB WASM (model + tokenizer + engine, all embedded) |
| Cold start | ~50 ms (Node `require()` + WASM compile) |

## Quality

The shipped model is distilled from `sentence-transformers/all-MiniLM-L6-v2`
via BitNet b1.58-style quantization-aware training, then post-training int4
quantized at the embedding layer.

Spearman rank correlation vs the MiniLM-L6 teacher on a held-out 100-query
MS MARCO test split, 1000 random pairs:

| Variant (linear weights ternary in all) | Bin size | Spearman | Pearson |
| --- | --- | --- | --- |
| Pre-QAT fp32 student (for reference) | 38.0 MB | 0.883 | 0.907 |
| `emb_int8` (8-bit embedding lookup) | 8.3 MB | 0.841 | 0.872 |
| **`emb_int4` (4-bit embedding, current ship)** | **4.6 MB** | **0.835** | **0.864** |
| `emb_ternary` (1.58-bit embedding) | 2.9 MB | 0.710 | 0.756 |

Full results in
[`eval/quality/RESULTS.md`](https://github.com/soycaporal/ternlight/blob/main/eval/quality/RESULTS.md).

The model is purpose-built for short-string similarity (queries, intents,
FAQs, product listings). It's not designed for long-document understanding
— inputs longer than ~95 English words get truncated to 128 tokens.

## How it works (one paragraph)

A 2-layer Transformer student (~9.5M parameters) is distilled from MiniLM-L6
during training. All linear layers use BitLinear with ternary weights
(`{-1, 0, +1}` plus a per-matrix fp32 scale), trained with the BitNet b1.58
straight-through estimator. After training, the token embedding table gets
post-training int4 quantization (16 levels per row + a per-row fp32 scale),
giving the smallest size cost for the embedding lookup that matters most.
The whole thing packs into a binary that's smaller than most product images.

For the full design, see
[`docs/overview.md`](https://github.com/soycaporal/ternlight/blob/main/docs/overview.md).

## Status

**v0.1, pre-alpha. Not yet published to npm.** Track progress and issues at
[github.com/soycaporal/ternlight](https://github.com/soycaporal/ternlight).

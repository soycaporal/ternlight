---
license: mit
language: en
base_model: sentence-transformers/all-MiniLM-L6-v2
base_model_relation: quantized
pipeline_tag: sentence-similarity
tags:
  - sentence-embeddings
  - sentence-similarity
  - semantic-search
  - on-device
  - wasm
  - webassembly
  - bitnet
  - ternary
  - quantization
  - distillation
  - edge-deployment
---

# ternlight

A 1.58-bit [BitNet][bitnet-paper]-style sentence embedding model distilled from
[`sentence-transformers/all-MiniLM-L6-v2`][teacher] via quantization-aware training,
with post-training int4 quantization at the embedding layer. Weights are ternary
`{-1, 0, +1}`, so inference is adds and subtracts rather than float matmuls, and the
whole model — engine, tokenizer, and weights — ships as a single WASM bundle that runs
on CPU with no API calls, no GPU, and no runtime download.

ternlight ships in **two tiers**, same API and same 384-dim output — pick by the
size/quality trade. It is designed for short-string semantic similarity (search queries,
intent classification, FAQ matching, product cards) deployed on-device (browser, Node,
edge runtimes, ARM single-board computers). It is *not* a frontier model; it trades
absolute quality for size and on-device deployability.

## Tiers

| Tier | Architecture | Params | Wire (gzip) | Latency (p50, CPU) | Throughput | Spearman vs teacher | SciFact NDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **base** ⭐ | 2-layer · d_model 384 · 6 heads | ~15.4M | 7.2 MB | 5.1 ms | ~195 emb/s | **0.844** | **0.465** |
| **mini** | 2-layer · d_model 256 · 4 heads | ~9.5M | 5.0 MB | 2.5 ms | ~400 emb/s | 0.820 | 0.439 |
| teacher (MiniLM-L6) | 6-layer · d_model 384 | ~22.7M | ~90 MB (fp32) | — | — | 1.000 (ref) | — |

- **base** — the quality tier. ~12× smaller on the wire than the fp32 teacher while
  retaining 0.84 rank-correlation. Use when quality matters more than the last 2 MB.
- **mini** — the small/fast tier. ~1.6× the throughput at 5 MB, a modest quality step down.

Both tiers: ternary linear weights + int4 embedding table, 384-dim L2-normalized output,
128-token max input, BERT WordPiece vocabulary (30,522, identical to the teacher).
Numbers measured on the shipped int4 builds (M-series Mac, Node single-threaded).

## How to use

ternlight runs via a [custom Rust→WASM inference engine][engine-source], not via the
`transformers` library. Two paths:

### Path 1 — via npm (recommended)

```bash
npm install @ternlight/base    # quality tier — 7 MB wire, ~5 ms/embed
npm install @ternlight/mini    # small tier   — 5 MB wire, ~2.5 ms/embed
```

```js
import { embed, cosineSim, similar } from '@ternlight/base';   // or '@ternlight/mini'

const v1 = embed("arctic terns migrate from pole to pole");
const v2 = embed("longest migration in the animal kingdom");
cosineSim(v1, v2);   // ~0.71 — semantically related, different wording

// Nearest-neighbor search over a corpus
similar("which seabird travels farthest", corpus, { topK: 5 });
```

The model and tokenizer are bundled into each npm package — no separate download. Works in
Node ≥ 18, browsers (via any bundler), Cloudflare Workers, Vercel Edge, Deno, and Bun.

### Path 2 — direct download

```python
from huggingface_hub import hf_hub_download

# pick a tier (adjust the paths to match this repo's file layout)
model_bin = hf_hub_download(repo_id="wenshutang/ternlight", filename="base/model-int4.bin")
tokenizer = hf_hub_download(repo_id="wenshutang/ternlight", filename="tokenizer.json")
```

The `.bin` files are a custom BitNet b1.58 format. See the [engine source][engine-source]
for the binary layout and reference forward pass if you want to implement a custom loader
in another language or runtime — there is no `transformers.AutoModel.from_pretrained()` path.

## Model details

| Property | base | mini |
| --- | --- | --- |
| Layers | 2-layer Transformer encoder | 2-layer Transformer encoder |
| d_model | 384 | 256 |
| Attention heads | 6 | 4 |
| Parameters | ~15.4M | ~9.5M |
| Output dimension | 384 (L2-normalized) | 384 (L2-normalized) |
| Max input | 128 tokens (~95 English words; longer inputs are silently truncated) | same |
| Vocabulary | 30,522 (BERT WordPiece, identical to teacher) | same |
| Linear weights | Ternary `{-1, 0, +1}` + per-matrix fp32 scale | same |
| Embedding table | 4-bit per-row PTQ + per-row fp32 scale | same |

## Training

Distilled from `sentence-transformers/all-MiniLM-L6-v2`:

1. **Distillation objective** — cosine/MSE loss between student and teacher 384-dim
   embeddings over ~1M sentence pairs (search queries, paraphrases, statements).
2. **BitNet b1.58 quantization-aware training** — all linear layers use ternary weights
   trained end-to-end with the straight-through estimator. Training under the quantization
   constraint *from the start* (rather than quantizing post-hoc) is what preserves fidelity:
   a naive post-training ternary quant of the same encoder drops sharply, so nearly all of
   the retained quality comes from the QAT, not the format.
3. **Post-training int4 quantization** — applied to the token embedding table after QAT,
   chosen via an ablation over int8/int4/ternary on the table. The embedding table dominates
   parameter count, so compressing it gives the largest size win for the smallest quality cost.

Training data (base): **mix_v3_1M** — ~1M pairs from MS MARCO, Quora duplicates, AllNLI,
GooAQ, and StackExchange duplicates (English-only, `seed=42`). See
[`configs/mix-v3-robust.yaml`][mixv3].

### Provenance

**base (`model-int4.bin`)**

| | |
| --- | --- |
| Training run | `robust-d384-mixv3-ep40` |
| Source checkpoint | `checkpoint_ep40.pt` |
| Source code commit | `178e227` |
| Data manifest | `mix_v3_1M` (seed 42) |
| Packed at | 2026-07-05 |
| Bin size | 7,492,312 bytes |
| SHA-256 | `68cc2c43…d2ee1db8` |
| Gate (2026-07-04) | test/spearman 0.851, ndcg@10 0.4665 (int8 form) |

**mini (`model-int4.bin`)** — ⚠️ *verify against the currently shipped `@ternlight/mini`
build; this sidecar predates the 2026-07-05 package rebuild.*

| | |
| --- | --- |
| Training run | `qat-resume-ep10-ep40` |
| Source checkpoint | `checkpoint_ep40.pt` |
| Source code commit | `dff16b1` |
| Packed at | 2026-06-03 |
| Bin size | 4,839,512 bytes |
| SHA-256 | `07d8cfdb…f2e5b6c98` |

Each `.bin` ships with a `.bin.json` sidecar containing full provenance for reproducibility.

## Evaluation

**Fidelity — Spearman rank correlation vs teacher.** Held-out queries, 1,000 deterministic
random pairs, `seed=42`. Spearman of 1.0 = ranks pair similarities identically to the teacher.

**Retrieval — SciFact NDCG@10.** Absolute retrieval quality on the BEIR SciFact task.

| Model | Spearman vs teacher | SciFact NDCG@10 |
| --- | ---: | ---: |
| MiniLM-L6 (teacher) | 1.000 (ref) | — |
| ternlight **base** ⭐ | **0.844** | **0.465** |
| ternlight **mini** | 0.820 | 0.439 |

Full methodology and reproduction scripts: [`eval/quality/RESULTS.md`][results-md].

## Intended use

**Designed for**:

- Short-string semantic similarity (queries, intents, FAQs, product titles, tags)
- On-device deployment — browsers, Node services, Cloudflare Workers, Deno Deploy,
  Vercel Edge, Raspberry Pi-class ARM single-board computers
- Cost-free embedding at any scale (no per-call API charges)
- Privacy-sensitive workloads where queries cannot leave the user's device

**Not designed for**:

- Long-document understanding (max input is 128 tokens — silently truncated above)
- Multilingual workloads (English-only, inherited from MiniLM-L6)
- Maximum absolute quality (use a frontier model like `text-embedding-3-large` or
  `voyage-3` if quality dominates over size and deployability)

## Limitations

- **English-only**: teacher, tokenizer (`bert-base-uncased`, no CJK vocab), and training
  data are English. Non-English text will not tokenize or embed sensibly. Multilingual is
  the most-requested feature and the pipeline is language-agnostic, but it is not done yet.
- **128-token cap**: text longer than 128 BERT WordPiece tokens is silently truncated.
  Embed at sentence or short-paragraph granularity, not full document.
- **Custom runtime required**: no `transformers` path. Use the npm packages or implement a
  custom loader from the binary format.
- **Inherited biases**: distilled from `all-MiniLM-L6-v2`; the same demographic and topical
  bias caveats from the sentence-transformers corpus apply.
- **v0.1**: the binary format and JS API may change before v1.0.

## License

MIT, matching the teacher model and the ternlight project. See [LICENSE][license].

## Citation

```bibtex
@software{ternlight2026,
  title  = {ternlight: a 1.58-bit BitNet sentence embedder in a few MB of WASM},
  author = {Tang, Wen Shu},
  year   = {2026},
  url    = {https://github.com/soycaporal/ternlight}
}
```

ternlight builds on:

- [BitNet b1.58][bitnet-paper] (Ma et al., 2024) — ternary weight training
- [`bitlinear`][bitlinear-repo] by [@schneiderkamplab][bitlinear-author] — the reference
  PyTorch implementation of BitLinear, used directly during training (`bitlinear==2.4.6`);
  the Rust inference engine mirrors its forward-pass math byte-for-byte
- [`sentence-transformers/all-MiniLM-L6-v2`][teacher] — teacher model

## Links

- **GitHub**: <https://github.com/soycaporal/ternlight>
- **Live demo**: <https://ternlight-demo.vercel.app>
- **npm**: `@ternlight/base` · `@ternlight/mini`

[teacher]: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
[bitnet-paper]: https://arxiv.org/abs/2402.17764
[bitlinear-repo]: https://github.com/schneiderkamplab/bitlinear
[bitlinear-author]: https://github.com/schneiderkamplab
[github]: https://github.com/soycaporal/ternlight
[engine-source]: https://github.com/soycaporal/ternlight/tree/main/engine
[results-md]: https://github.com/soycaporal/ternlight/blob/main/eval/quality/RESULTS.md
[mixv3]: https://github.com/soycaporal/ternlight/blob/main/training/distill/configs/mix-v3-robust.yaml
[license]: https://github.com/soycaporal/ternlight/blob/main/LICENSE

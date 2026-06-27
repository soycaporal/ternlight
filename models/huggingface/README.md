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
with post-training int4 quantization at the embedding layer. The shipped binary is
**4.6 MB**; the full WASM bundle (engine + tokenizer + model) is **7 MB** and runs
on CPU in ~2 ms per call.

ternlight is designed for short-string semantic similarity — search queries, intent
classification, FAQ matching, product cards — deployed on-device (browser, Node,
edge runtimes, ARM single-board computers). It is *not* a frontier model; it trades
absolute quality for size and on-device deployability.

## Model variants

| File | Bin size | Spearman vs teacher | Quality retained vs fp32 student |
| --- | ---: | ---: | ---: |
| **`model-int4.bin`** ⭐ | **4.6 MB** | **0.835** | **95%** |
| `model-embedding-int8.bin` | 8.3 MB | 0.841 | 95% |
| `model-ternary.bin` | 2.9 MB | 0.710 | 80% |

`model-int4.bin` is the shipped default. `int8` offers a slight quality bump at
~1.8× the size. `ternary` is the size-extreme variant — useful when bytes are at
absolute premium and you can tolerate the ~15 pt drop in pair-ranking quality.

All variants share the same architecture and tokenizer.

## How to use

ternlight runs via a [custom Rust→WASM inference engine][engine-source], not via the
`transformers` library. Two paths:

### Path 1 — via the `ternlight` npm package (recommended)

```bash
npm install ternlight
```

```js
import { embed, cosineSim, similar } from 'ternlight';

const v1 = embed("arctic terns migrate from pole to pole");
const v2 = embed("longest migration in the animal kingdom");

cosineSim(v1, v2);   // ~0.71 — semantically related, different wording

// Nearest-neighbor search over a corpus
const matches = similar("which seabird travels farthest", corpus, { topK: 5 });
```

The model and tokenizer are bundled into the npm package — no separate download.

### Path 2 — direct download

```python
from huggingface_hub import hf_hub_download

model_bin = hf_hub_download(repo_id="wenshutang/ternlight", filename="model-int4.bin")
tokenizer = hf_hub_download(repo_id="wenshutang/ternlight", filename="tokenizer.json")
```

The `.bin` files are a custom BitNet b1.58 format. See the
[engine source][engine-source] for the binary layout and reference forward pass
if you want to implement a custom loader (e.g., in another language or runtime).

## Model details

| Property | Value |
| --- | --- |
| Architecture | 2-layer Transformer encoder |
| Parameters | ~9.5M |
| Output dimension | 384 (L2-normalized) |
| Max input | 128 tokens (~95 English words; longer inputs are silently truncated) |
| d_model | 256 |
| Attention heads | 4 |
| FFN dim | 1024 |
| Vocabulary | 30,522 (BERT WordPiece, identical to teacher) |
| Linear weights | Ternary `{-1, 0, +1}` + per-matrix fp32 scale |
| Embedding weights (int4 variant) | 4-bit per-row PTQ + per-row fp32 scale |
| Embedding weights (int8 variant) | 8-bit per-row PTQ + per-row fp32 scale |
| Embedding weights (ternary variant) | Ternary, same scheme as linear weights |

## Training

Distilled from `sentence-transformers/all-MiniLM-L6-v2` in three stages:

1. **Distillation objective** — MSE loss between student and teacher 384-dim
   embeddings, plus an optional contrastive term.
2. **BitNet b1.58 quantization-aware training** — all linear layers use ternary
   weights trained end-to-end with the straight-through estimator. Training the
   model with the quantization constraint from the start (rather than quantizing
   post-hoc) preserves ~95% of the fp32 student's pair-ranking quality.
3. **Post-training int4 quantization (PTQ)** — applied to the token embedding
   table after QAT completes. The embedding table dominates parameter count, so
   compressing it aggressively gives the largest size win for the smallest
   quality cost.

Training data: ~1M sentences from MS MARCO and general web text. English-only.

### Provenance (`model-int4.bin`)

| | |
| --- | --- |
| Training run | `qat-resume-ep10-ep40` |
| Source checkpoint | `checkpoint_ep40.pt` |
| Source code commit | `dff16b1` |
| Packed at | 2026-06-03 |
| SHA-256 | `07d8cf...e5b6c98` |

Each `.bin` ships with a `.bin.json` sidecar containing the full provenance for
reproducibility checks.

## Evaluation

### Spearman rank correlation vs teacher

Held-out MS MARCO queries, 1,000 deterministic random pairs, `seed=42`. Spearman
of 1.0 = the candidate ranks pair similarities identically to the teacher.

| Variant | Bin size | Bits/param | Spearman | Pearson |
| --- | ---: | ---: | ---: | ---: |
| MiniLM-L6 (teacher) | 90.9 MB | 32.00 | 1.000 | 1.000 |
| Student fp32 (pre-QAT) | 38.0 MB | 32.00 | 0.883 | 0.907 |
| ternlight `emb_int8` | 8.3 MB | 7.37 | 0.841 | 0.872 |
| **ternlight `emb_int4`** ⭐ | **4.6 MB** | **4.08** | **0.835** | **0.864** |
| ternlight `emb_ternary` | 2.9 MB | 2.43 | 0.710 | 0.756 |

Full methodology and reproduction scripts:
[`eval/quality/RESULTS.md`][results-md].

### Performance (M-series Mac, Node single-threaded)

| Metric | Value |
| --- | ---: |
| Latency p50 | ~2 ms |
| Throughput | ~450 emb/sec (sentence-length input) |
| Cold start | ~112 ms (require + first inference) |
| Memory (RSS, post-warmup) | ~150 MB |

Throughput scales inversely with sequence length — ~900 emb/sec on short queries
(3-4 tokens), ~150 emb/sec on long paragraphs (~25 tokens). Methodology:
[`eval/benchmarks/perf.js`][perf-js].

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

- **English-only**: the tokenizer and training data are English. Performance on
  non-English text is undefined and likely poor.
- **128-token cap**: text longer than 128 BERT WordPiece tokens is silently
  truncated. Embed at sentence or short-paragraph granularity, not full document.
- **Custom runtime required**: no `transformers.AutoModel.from_pretrained()` path
  is provided. Use the [ternlight npm package][github] or implement a custom
  loader from the binary format.
- **Inherited biases**: ternlight is distilled from `all-MiniLM-L6-v2`, which
  inherits training-data biases from the sentence-transformers corpus. The same
  caveats around demographic and topical bias apply.
- **Pre-alpha (v0.1)**: the binary format and JS API may change before v1.0.

## License

MIT, matching the teacher model and the ternlight project. See
[LICENSE][license].

## Citation

If you use ternlight in published work, please cite:

```bibtex
@software{ternlight2026,
  title  = {ternlight: a 1.58-bit BitNet sentence embedder in 7 MB of WASM},
  author = {Tang, Wen Shu},
  year   = {2026},
  url    = {https://github.com/soycaporal/ternlight}
}
```

ternlight builds on:

- [BitNet b1.58][bitnet-paper] (Ma et al., 2024) — ternary weight training
- [`sentence-transformers/all-MiniLM-L6-v2`][teacher] — teacher model

## Links

- **GitHub**: <https://github.com/soycaporal/ternlight>
- **Live demo**: <https://ternlight-demo.vercel.app>
- **npm**: `npm install ternlight`

[teacher]: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
[bitnet-paper]: https://arxiv.org/abs/2402.17764
[github]: https://github.com/soycaporal/ternlight
[engine-source]: https://github.com/soycaporal/ternlight/tree/main/engine
[results-md]: https://github.com/soycaporal/ternlight/blob/main/eval/quality/RESULTS.md
[arch-md]: https://github.com/soycaporal/ternlight/blob/main/docs/architecture.md
[license]: https://github.com/soycaporal/ternlight/blob/main/LICENSE

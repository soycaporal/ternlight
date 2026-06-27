# Eval Methodology

> What each metric in the release scorecard measures, how to reproduce it, and why we chose it.

This is the reference document for `eval/REPORT.md` and `eval/results/v<X.Y.Z>.json`. Every metric we publish should be traceable here.

## Status

Pre-alpha — full methodology pending. The skeleton below mirrors the six dimensions of the release scorecard.

---

## 1. Quality

### Teacher alignment (mean per-query cosine similarity)

**What it measures:** how closely the student's embeddings align with the teacher (`all-MiniLM-L6-v2`) for the same input. Average of `cosine_sim(student(q), teacher(q))` across N held-out queries.

**Why this metric:** the most direct test of "did distillation work?" — directly compares student output to the target the loss was minimizing.

**Methodology:**
- Held-out queries: MS MARCO `train[25000:27000]` (2,000 queries — same slice as Phase 1's `eval.py`)
- Teacher embeddings precomputed once (`prepare_eval_data.py`), cached
- Student embeddings: run shipped engine on each query
- Both vectors are L2-normalized (unit), so cosine sim is a dot product

**Reproduce:**
```bash
node eval/regression/regression_test.js
```

### STS-B AUC

**What it measures:** can the model distinguish "similar" from "not similar" sentence pairs as scored by humans? Wilcoxon-Mann-Whitney form: probability that a randomly picked similar pair scores higher than a randomly picked dissimilar pair.

**Why this metric:** STS Benchmark is the de-facto standard for sentence embedding quality. AUC is robust to scale/offset and aligns with binary classification use cases (FAQ matching, intent routing).

**Methodology:**
- Dataset: `mteb/stsbenchmark-sts` test split (1,379 pairs)
- For each pair, embed both sentences, compute cosine sim
- Binarize human scores at threshold 4.0 (>=4 → "similar", <4 → "not similar")
- AUC = (count of (similar, dissimilar) pairs where similar > dissimilar) / (total such pairs)

### STS-B Spearman

**What it measures:** rank correlation between model similarity scores and human similarity scores. Sensitive to the full distribution, not just a threshold.

**Methodology:** same data as AUC. Spearman correlate model sims vs human scores.

### Recall@K

**What it measures:** for nearest-neighbor retrieval, fraction of queries where the correct match appears in the top-K results.

**Methodology:**
- Two corpora: 20 general queries + 20 tech queries (hardcoded — same as Phase 1 eval)
- Each query has one correct match in the corresponding corpus
- For each query, embed query + each corpus item, rank by cosine sim, check if correct match is in top-K

---

## 2. Quantization gap

**What it measures:** how much quality the ternary quantization costs vs full float32. Per-component breakdown helps diagnose which quantization step matters most.

**Components:**
- Embedding (ternary post-training, AbsMean) — compare with float32 embedding
- BitLinear weights (ternary, AbsMedian round-clamp) — compare with float32 BitLinear weights
- Activation int8 quantization — compare with float32 activations

**Methodology:** TBD. Approach: ablate each quantization step in the Python reference (`dump_embed.py`-style harness), re-run quality eval, report deltas.

---

## 3. Performance

### Cold start

**What it measures:** time from `import` to first `embed()` call returning. Includes Wasm instantiation, tokenizer init, model.bin parse + weight unpacking.

### Warm latency

**What it measures:** steady-state per-call time after the engine is warm. The number users see in production.

### Throughput

**What it measures:** strings/sec under sustained sequential load. Useful for offline indexing scenarios.

### Memory peak

**What it measures:** peak RSS during 100 sequential `embed()` calls. Surfaces memory leaks and per-call allocation pressure.

**Methodology:** TBD. Per-target via `eval/benchmarks/`.

---

## 4. Size

| Artifact | Why it matters |
|---|---|
| `engine.wasm` bytes | What the JS bundler ships |
| `engine.wasm` gzipped | What the user actually downloads |
| `model.bin` bytes | Largest single asset |
| `model.bin` gzipped | model.bin is mostly random ternary bytes — compresses poorly |
| Total npm install | What `du -sh node_modules/@tern/` shows |

---

## 5. Compatibility

For each target in `eval/compatibility/runtimes.yaml`:
- PASS = engine instantiates, `embed("hello world")` returns valid output, output matches canonical reference
- FAIL with reason

---

## 6. Honest comparison

Side-by-side with closest alternatives. For each comparison, document:
- The other library's version
- The model used (e.g., `Xenova/all-MiniLM-L6-v2` for transformers.js)
- The same inputs, the same metrics
- Bundle sizes and runtime characteristics

The point isn't to "win" on every axis — it's to give users an honest tradeoff so they can pick what fits their constraints.

---

## Schema for `eval/results/v<X.Y.Z>.json`

See [../../eval/results/README.md](../../eval/results/README.md).
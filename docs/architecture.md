# Architecture

> System design — components, data flow, model format, and runtime behavior.
> For end-to-end framing of the project, see [overview.md](overview.md).
> For training-time math (forward pass, backprop, distillation dynamics), see [model-internals.md](model-internals.md).

---

## 1. System Overview

The ternlight runtime is composed of three tightly coupled components:

```
┌──────────────────────────────────────────────┐
│  Node.js / JS Wrapper                        │
│  - Thin API surface (embed, similarity,      │
│    classify) — no tokenization logic here    │
│  - Passes raw strings directly into Wasm     │
└─────────────────────┬────────────────────────┘
                      │ string input / f32 vector output
┌─────────────────────▼────────────────────────┐
│  Wasm Engine (Rust)                          │
│  - HuggingFace `tokenizers` crate (compiled  │
│    in) — BERT WordPiece, vocab embedded      │
│  - Hardcoded computation graph               │
│  - Branchless bitwise ternary math           │
│  - SIMD-accelerated additions/subtractions   │
└─────────────────────┬────────────────────────┘
                      │ linear memory map
┌─────────────────────▼────────────────────────┐
│  Model Binary (.bin)                         │
│  - 24-byte structural header                 │
│  - Sequential bit-packed ternary weight      │
│    matrices (4 weights per byte)             │
└──────────────────────────────────────────────┘
```

---

## 2. Core Architecture Pillars

Three technical choices stack to fit a capable embedding model into a 7 MB WASM bundle that runs on CPU.

### Pillar 1: Quantization-aware training (QAT) with ternary weights

All Linear layers in the student are **BitLinear layers** — weights are constrained to three values: `{-1, 0, +1}`. The model is trained for ternary weights from the start ([BitNet b1.58][bitnet-paper] quantization-aware training).

At inference time this means:

- No floating-point matrix multiplication — only additions and subtractions
- Weights pack at ~1.58 bits per parameter (log₂(3)); packing overhead brings this to ~2 bits
- Quality stays within ~95% of the full-precision baseline (see [`eval/quality/RESULTS.md`](../eval/quality/RESULTS.md))

The model is an encoder - it produces a single fixed-size embedding vector per input, not autoregressive next token generation.

### Pillar 2: Bit-packing — model + tokenizer in one WASM bundle

Weights serialize at ~2 bits per parameter (four weights per byte), with the embedding layer  further compressed via 4-bit per-row PTQ. The whole model fits into a binary file you can embed *inside* the `.wasm` itself:

- The model `.bin` embeds at compile time via Rust
- Similarly, the HuggingFace `tokenizers` crate compiles into the same `.wasm` - tokenization happens inside Wasm (not JS bindings)
- The BERT WordPiece vocabulary embeds at compile time via the same mechanism - no separate vocab file ships
- Completely self contained, no postinstall, no runtime fetch - `npm install` and you're done

The resulting `.wasm` is ~7 MB total: 4.6 MB packed model + 695 KB tokenizer + ~1.7 MB engine code.

### Pillar 3: SIMD inference engine in Rust → WASM

The engine is not a generic inference framework. It is a **hardcoded computation graph** compiled from Rust to WebAssembly.

- Allocates a single contiguous block of memory at startup
- Maps the model `.bin` sequentially into that memory (no deserialization)
- Executes each layer in order using branchless bitwise operations
- Uses 128-bit WASM SIMD lanes for vectorized add/subtract over bit-packed rows

The ternary matmul is simplified to sign-conditioned add/subtract that maps directly onto CPU vector instructions. The engine is structurally coupled to the specific layer shapes defined in the `.bin` header.

```rust
#[wasm_bindgen]
pub fn embed(text: &str) -> Vec<f32> {
    let encoding = TOKENIZER.encode(text, false).unwrap();
    let ids = encoding.get_ids();
    // → forward pass → embedding vector
}
```

[bitnet-paper]: https://arxiv.org/abs/2402.17764

---

## 3. Model Format: The `.bin` File

The exported model is a single binary file with a minimal header:

```
Offset  Size    Field
──────────────────────────────────────────────
0       4B      Magic number (0x5445524E — "TERN")
4       2B      Format version
6       2B      d_model
8       2B      n_layers
10      2B      n_heads
12      2B      ffn_dim
14      2B      vocab_size
16      2B      max_seq_len
18      2B      Reserved
20      4B      Total weight bytes (excluding header)
──────────────────────────────────────────────
24      N bytes Bit-packed weight matrices (sequential)
```

Weight matrices are stored in layer order: embedding table, then for each layer - Q, K, V, O projections, FFN up, FFN down, layer norm scales. All values are 2-bit encoded with four weights per byte.

---

## 4. Training & Distillation Pipeline

Training uses traditional PyTorch + GPU infrastructure, separate from the WASM runtime.

### Phase A: Distillation Training (Python / GPU)

1. **Teacher model:** A high-quality sentence transformer (e.g., `all-MiniLM-L6-v2`) generates soft embedding targets for the training corpus.
2. **Student model:** A 2-layer BitLinear transformer defined in PyTorch. Uses float32 shadow weights during training to enable backpropagation.
3. **Quantization-Aware Training (QAT):** The forward pass uses the sign function to project shadow weights to `{-1, 0, +1}` (with a zero-band threshold). Gradients flow through the shadow weights via the straight-through estimator.
4. **Loss:** Align student vectors with teacher vectors via cosine embedding loss.

### Phase B: Export & Bit-Packing (Python Script)

1. **Discard training state:** Float32 shadow weights and all optimizer states are deleted.
2. **Materialize ternary weights:** Shadow weights are projected to `{-1, 0, +1}` and stored as integers.
3. **Pack:** Every four ternary values are packed into one byte using 2-bit encoding (`00` = 0, `01` = +1, `10` = -1, `11` = unused/padding).
4. **Write:** The 24-byte header is prepended and the file is written as a raw `.bin`.

### Phase C: Inference (Wasm)

`embed(text: &str) -> Vec<f32>` is the entry point. Inside the engine:

1. **Tokenize** — BERT WordPiece via the `tokenizers` crate.
2. **Embedding lookup** — each token ID indexes the (int4-quantized) embedding table; per-row scales restore the fp32 activation magnitude.
3. **Forward pass** — 2 transformer layers (attention + FFN, ternary weights throughout).
4. **Mean-pool and L2-normalize** → 384-dim unit vector.

---

## 5. Shipped Model Configuration

The shipped student is a single architecture; only the embedding quantization varies across variants.

| Hyperparameter | Value |
|---|---|
| d_model | 256 |
| n_layers | 2 |
| n_heads | 4 (d_k = 64) |
| ffn_dim | 1024 (4× d_model) |
| vocab_size | 30,522 (BERT WordPiece) |
| max_seq_len | 128 |
| Total params | ~9.5M |
| Output dim | 384 (L2-normalized) |

### Variants

| Variant | Embedding quantization | Bin size | Bundle (engine + tokenizer + bin) |
|---|---|---:|---:|
| **`emb_int4`** ⭐ | 4-bit per-row PTQ + per-row fp32 scale | **4.6 MB** | **~7 MB** |
| `emb_int8` | 8-bit per-row + per-row fp32 scale | 8.3 MB | ~11 MB |
| `emb_ternary` | Packed ternary + per-row fp32 scale | 2.9 MB | ~5 MB |
| `emb_fp32` | fp32 row-major | 38 MB | ~40 MB (parity reference, not shipped) |

All variants share the same WASM engine binary. The engine reads dimensional constants from the `.bin` header at startup and allocates memory accordingly.

---

## 6. Runtime Performance Model

A single `embed()` call runs **~218M operations** per input string. The compute splits cleanly between ternary weight matmuls and a small float-multiply tail.

**Ternary add/subtract: ~201M ops (~92%).** Every learned matrix is bit-packed weights, so every weight matmul reduces to add/sub:

| Stage | Per 2 layers |
|---|---:|
| Q/K/V/O projections | ~33.6M |
| FFN (up + down, 256 ↔ 1024) | ~134.4M |
| Embedding scale + readout | ~33.6M |
| **Total** | **~201.6M** |

**Float multiply: ~17M ops (~8%).** Bounded to operations over *activations* (which can't be ternarized) plus per-token non-linearities:

| Stage | Ops | Why float |
|---|---:|---|
| Attention scores (Q @ K.T, attn × V) | ~16.8M | Both operands are float activations |
| Softmax, scaling, LayerNorm × 5, GELU × 2 | ~780K | Transcendentals + per-token statistics |

The dominant share is add/sub (no multiply), which maps directly to SIMD lanes. The remaining 8% is float work over activations, not weights, so it can't be ternarized.

### Why ternary add/sub is fast on CPU

A ternary matmul inner loop looks like:

```
for each weight:
    if weight == +1:  accumulator += input[i]
    if weight == -1:  accumulator -= input[i]
    if weight ==  0:  skip
```

- **No multiply unit needed.** Float add is 1 CPU cycle, float multiply is 3–5 cycles. Ternary matmul is 3–5× cheaper per operation than float matmul.
- **Branch-free implementation.** The weight encodes a sign bit - the add/subtract decision can be computed without branching: `accumulator += input[i] * weight` where weight is -1, 0, or +1.
- **The zero weights (skip) are free sparsity.** At ~45% zero fraction (from the scaled training run), nearly half the operations are skipped entirely. Effective op count is closer to ~120M than 218M.

### Cache behavior — the key advantage at this model size

The shipped int4 model is **4.6 MB**. Modern CPUs have:

```
L1 cache:   ~128 KB  — holds current layer's activations
L2 cache:   ~4–12 MB — holds the ENTIRE model
L3 cache:   ~32 MB+  — irrelevant, everything fits in L2
```

For comparison, `all-MiniLM-L6-v2` at fp32 is ~90 MB — it would constantly evict L2 cache. ternlight's model fits entirely in L2 cache from the first call onward. Every weight read is a cache hit.

### Measured latency

`emb_int4` on M4 Max, Node 20, WASM SIMD enabled:

| Metric | Value |
|---|---|
| Latency p50 | ~2 ms |
| Latency p95 | ~4 ms |
| Cold start | ~112 ms (require + first inference) |
| Sustained throughput | ~450 emb/sec (sentence-length input) |

Per-build benchmark history lives in [`eval/benchmarks/results/`](../eval/benchmarks/results/).

---

## 7. Build Pipeline Summary

```
Training corpus (MS MARCO + general English text)
    ↓
Teacher embeddings (MiniLM-L6 / GPU)
    ↓
QAT student training (PyTorch + `bitlinear==2.4.6`)
  └── tokenizer: HuggingFace `tokenizers` Python bindings
      (same Rust core as the WASM build — structural symmetry)
    ↓
Weight export + bit-packing (Python)
    ↓
model-int4.bin (4.6 MB)
    ↓
    ↓  ←── wasm-pack build --target nodejs --features emb_int4
    ↓       Cargo.toml includes `tokenizers` crate
    ↓       BERT vocab + model.bin embedded via include_bytes!()
    ↓
npm package (`ternlight`)
│   index.js (thin JS wrapper)                              ~10 KB
│   pkg/tern_engine_bg.wasm  (engine + tokenizer + model)   ~7 MB
└── pkg/tern_engine.js (wasm-bindgen glue)                  ~13 KB
                                                           ─────────
                                                           ~7 MB total
```

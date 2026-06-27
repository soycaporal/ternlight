# @tern — Technical Architecture Document

> For product context, use cases, and tier definitions, see [tern-scoping.md](tern-scoping.md).
> For model sizing rationale and parameter breakdowns, see [tern-model-sizing.md](tern-model-sizing.md).

---

## 1. System Overview

The @tern runtime is composed of three tightly coupled components:

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

### Pillar 1: Ternary Transformer (Micro-Architecture)

The student model is a 2-layer transformer encoder where all Linear layers are replaced with **BitLinear layers**. These layers constrain all learned weights to three states: `{-1, 0, +1}`.

At inference time this means:
- No floating-point matrix multiplication — only integer additions and subtractions
- Weights pack at ~1.58 bits per parameter (log₂(3)); practical packing overhead brings this to ~2 bits
- Four weights fit into a single byte using 2-bit encoding

The model is an encoder only — it produces a single fixed-size embedding vector per input, not autoregressive token predictions.

### Pillar 2: Hardcoded Wasm Engine

The engine is not a generic inference framework. It is a **hardcoded computation graph** compiled from Rust to WebAssembly. It:

- Includes the HuggingFace `tokenizers` Rust crate as a Cargo dependency — tokenization happens inside Wasm, not in JS
- Embeds the BERT WordPiece vocabulary at compile time via `include_bytes!()` — no separate vocab file ships
- Allocates a single contiguous block of linear memory at startup
- Maps the model `.bin` file sequentially into that memory (no deserialization)
- Executes each layer in order using branchless bitwise operations
- Uses SIMD where available for vectorized additions over bit-packed rows

No floating-point matrix multiplication is performed at inference time. The engine has no plugin system, no dynamic dispatch, and no model-agnostic abstraction — it is structurally coupled to the specific layer shapes defined in the header.

```toml
# Cargo.toml
[dependencies]
tokenizers = "0.19"
```

```rust
#[wasm_bindgen]
pub fn embed(text: &str) -> Vec<f32> {
    let encoding = TOKENIZER.encode(text, false).unwrap();
    let ids = encoding.get_ids();
    // → forward pass → embedding vector
}
```

**License:** The HuggingFace `tokenizers` crate is Apache 2.0.

### Pillar 3: Thin JS Bridge

The Node.js wrapper is intentionally minimal — tokenization no longer lives here:
- **API surface:** `embed(text)`, `similarity(a, b)`, and `classify(text, labels[])` pass raw strings directly into Wasm and receive float32 vectors back
- **No tokenization logic in JS** — the Wasm engine handles the full pipeline from raw string to embedding vector
- **Memory handoff:** Output embedding vector is read back from Wasm linear memory as a `Float32Array`

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

Weight matrices are stored in layer order: embedding table, then for each layer — Q, K, V, O projections, FFN up, FFN down, layer norm scales. All values are 2-bit encoded with four weights per byte.

---

## 4. Training & Distillation Pipeline

The training pipeline is strictly separated from the runtime. PyTorch and GPU infrastructure are training-time concerns only — nothing from the training environment ships in the package.

### Phase A: Distillation Training (Python / GPU)

1. **Teacher model:** A high-quality sentence transformer (e.g., `all-MiniLM-L6-v2`) generates soft embedding targets for the training corpus.
2. **Student model:** A 2-layer BitLinear transformer defined in PyTorch. Uses float32 shadow weights during training to enable gradient computation.
3. **Quantization-Aware Training (QAT):** The forward pass uses the sign function to project shadow weights to `{-1, 0, +1}` (with a zero-band threshold). Gradients flow through the shadow weights via the straight-through estimator.
4. **Loss:** Cosine embedding loss between student and teacher output vectors, trained on a focused English/tech-domain corpus matching the target use case.

### Phase B: Export & Bit-Packing (Python Script)

1. **Discard training state:** Float32 shadow weights and all optimizer states are deleted.
2. **Materialize ternary weights:** Shadow weights are projected to `{-1, 0, +1}` and stored as integers.
3. **Pack:** Every four ternary values are packed into one byte using 2-bit encoding (`00` = 0, `01` = +1, `10` = -1, `11` = unused/padding).
4. **Write:** The 24-byte header is prepended and the file is written as a raw `.bin`.

### Phase C: Inference (Node.js / Wasm)

1. `fs.readFileSync` loads the `.bin` into a Node.js Buffer.
2. The Buffer is written into Wasm linear memory at offset 0.
3. The JS wrapper passes the raw input string to the Wasm `embed()` function.
4. Inside Wasm, the embedded `tokenizers` crate tokenizes the string using BERT WordPiece (vocab compiled in at build time).
5. The Wasm engine performs the forward pass (embedding lookup → 2× attention + FFN → mean pooling) and writes the output embedding vector to a known memory offset.
6. The JS wrapper reads the output vector back as a `Float32Array` and returns it to the caller.

---

## 5. Target Model Configuration (Micro Tier)

The micro tier is the default and primary build target. See [tern-model-sizing.md](tern-model-sizing.md) for full sizing rationale.

| Hyperparameter | Value |
|---|---|
| d_model | 256 |
| n_layers | 2 |
| n_heads | 8 (d_k = 32) |
| ffn_dim | 1024 (4× d_model) |
| vocab_size | 10,000 |
| max_seq_len | 128 |
| Total params | ~7M |
| Packed model size | ~1.75MB |

### Tier Configurations

| Tier | d_model | n_layers | Params | Packed |
|---|---|---|---|---|
| nano | 128 | 2 | ~3M | ~750KB |
| **micro** *(default)* | **256** | **2** | **~7M** | **~1.75MB** |
| base | 384 | 2 | ~12M | ~3.0MB |

All tiers share the same Wasm engine binary. The engine reads dimensional constants from the `.bin` header at startup and allocates memory accordingly.

### Wasm Binary Size Breakdown

The `tokenizers` crate adds to the compiled Wasm size. Estimated breakdown:

| Component | Estimated Size |
|---|---|
| Inference engine (Rust) | ~400KB |
| `tokenizers` crate (compiled) | ~200–300KB |
| BERT vocab embedded (`include_bytes!`) | ~115KB |
| **Total Wasm binary** | **~715–815KB** |

> **Risk:** The `tokenizers` crate pulls in `regex`, `unicode-normalization`, and `serde`. These compile cleanly to Wasm but tree-shaking is not guaranteed. A test compile should be done early in Phase 2 to get a real binary size number before committing to the vocab-embedded approach.

---

## 6. Runtime Performance Model

A single `embed()` call processes one string through the full forward pass on CPU. This section breaks down where the cycles go and what to be mindful of when building the Wasm engine.

### Operation budget per embed() call

**Ternary matmul — add/subtract only, no float multiply:**

| Operation | Dimensions | Ops per layer | Notes |
|---|---|---|---|
| Q, K, V projections | 128 × 256 × 256 × 3 | ~25.2M | Pure add/sub |
| Output projection | 128 × 256 × 256 | ~8.4M | Pure add/sub |
| FFN up (256→1024) | 128 × 1024 × 256 | ~33.6M | Pure add/sub |
| FFN down (1024→256) | 128 × 256 × 1024 | ~33.6M | Pure add/sub |
| **Per layer** | | **~100.8M** | |
| **Two layers** | | **~201.6M** | |

**Float multiply (unavoidable — not covered by ternary):**

| Operation | Ops | Why it can't be ternary |
|---|---|---|
| Attention scores (Q @ K.T) | ~8.4M | Q, K are float activations, not weights |
| Attention × V | ~8.4M | Same — activation × activation |
| Score scaling (÷ sqrt(d_head)) | ~130K | Single divide per score |
| Softmax (exp, div) | ~130K | Transcendental functions |
| LayerNorm (mean, var, div) × 5 | ~260K | Statistical normalization |
| GELU activation × 2 | ~260K | Non-linear activation |
| **Total float multiply** | **~17M** | ~8% of total ops |

**Summary: ~218M total ops per call.** ~92% are ternary add/subtract, ~8% are float multiply (mostly in attention score computation).

### Why ternary add/sub is fast on CPU

A ternary matmul inner loop looks like:

```
for each weight:
    if weight == +1:  accumulator += input[i]
    if weight == -1:  accumulator -= input[i]
    if weight ==  0:  skip
```

- **No multiply unit needed.** Float add is 1 CPU cycle, float multiply is 3–5 cycles. Ternary matmul is 3–5× cheaper per operation than float matmul.
- **Branch-free implementation.** The weight encodes a sign bit — the add/subtract decision can be computed without branching: `accumulator += input[i] * weight` where weight is literally -1, 0, or +1. The multiply by ±1 is optimized away by the compiler.
- **The zero weights (skip) are free sparsity.** At ~45% zero fraction (from the scaled training run), nearly half the operations are skipped entirely. Effective op count is closer to ~120M than 218M.

### Cache behavior — the key advantage at this model size

The entire packed model is ~2.7MB. Modern CPUs have:

```
L1 cache:   ~128KB   — holds current layer's activations
L2 cache:   ~4–12MB  — holds the ENTIRE model
L3 cache:   ~32MB+   — irrelevant, everything fits in L2
```

For comparison, `all-MiniLM-L6-v2` at float32 is ~88MB — it constantly evicts L2 and hits L3/RAM. @tern's model fits entirely in L2 cache from the first call onward. Every weight read is a cache hit.

**What this means in practice:**

- First call may be ~10ms (cold cache, loading .bin into L2)
- Subsequent calls ~3–8ms (everything in L2, no RAM access)
- The bottleneck shifts from memory bandwidth to raw ALU throughput

### What to be mindful of when building the Wasm engine

**Memory layout matters.** Weights should be stored contiguously in the order they're accessed during the forward pass. If the engine reads W_q, then jumps to norm1, then back to W_k, it defeats sequential cache prefetch. The .bin format already stores weights in forward-pass order for this reason.

**Unpack close to use.** Two strategies for reading 2-bit packed weights:

1. **Unpack entire matrix to float32, then matmul.** Simpler code but doubles memory: a 256×256 ternary matrix is 16KB packed, 256KB unpacked. At 12 matrices that's 3MB of temporary float32 — blows L1 and pressures L2.

2. **Unpack per-row during matmul.** Read one packed row (64 bytes for 256 weights), unpack to a float32 row buffer (1KB), compute the dot product, reuse the buffer. Only 1KB of temporary memory. Cache-friendly.

Strategy 2 is strongly preferred. The unpacking cost (~256 shift+mask ops per row) is negligible compared to the matmul ops. This keeps the hot loop in L1.

**Attention score computation is the float-heavy part.** Q @ K.T and attn @ V are full float32 matmuls over activation tensors (not weights). These are 128×64 × 64×128 per head — small matrices that fit in L1. Not a bottleneck, but this is where SIMD (future work) would help most.

**GELU is expensive per-op but rare.** Only called twice (once per FFN). Can be approximated with the fast tanh version: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))` — avoids the `erf()` call.

### Estimated latency targets

| Environment | Estimated latency | Notes |
|---|---|---|
| Native Rust (M4 Max) | ~1–2ms | Baseline reference |
| Wasm in Node.js (V8) | ~5–15ms | Wasm sandbox overhead |
| Wasm + SIMD (V8) | ~2–5ms | Future optimization |
| Cloudflare Workers | ~5–20ms | Depends on cold/warm |
| Browser (Chrome/Firefox) | ~5–15ms | Comparable to Node.js |

These are estimates based on op count and typical Wasm overhead factors. Real numbers will come from Phase 2 benchmarks. The scoping doc targets 1–5ms — achievable with SIMD, likely 5–15ms without.

---

## 7. Similarity Scoring

The API exposes two scoring mechanisms depending on use case:

**Cosine similarity (default for `similarity()` and `classify()`)**
Computed in JS from the float32 output vectors. Standard dot product normalized by magnitudes. Returns a value in `[-1, 1]`; for semantic tasks typically `[0, 1]`.

**Hamming distance (optional, for `@tern/search` batch operations)**
The float32 embedding is binarized (sign function) and compared using bitwise XOR. O(n) over the number of bits. Useful for large-scale nearest-neighbor search where cosine similarity over full float vectors would be too slow.

---

## 7. Build Pipeline Summary

```
Training corpus (English/tech text)
    ↓
Teacher embeddings (MiniLM / GPU)
    ↓
QAT student training (PyTorch)
  └── tokenizer: HuggingFace `tokenizers` Python bindings
      (same Rust core as the Wasm build — structural symmetry)
    ↓
Weight export + bit-packing (Python)
    ↓
.bin model file (~1.75MB micro)
    ↓
    ↓  ←── cargo build --target wasm32-unknown-unknown --release
    ↓       Cargo.toml includes `tokenizers` crate
    ↓       BERT vocab embedded via include_bytes!()
    ↓
npm package (@tern/semantic)
│   index.js (thin JS wrapper, no tokenizer logic)  ~20KB
│   engine.wasm  (inference + tokenizer + vocab)    ~750KB
└── model.bin                                       ~1.75MB
                                                    ─────────
                                                    ~2.5MB total (micro tier)
```

# Tern Inference Engine — Design

> Rust crate at [`engine/`](../engine/), compiled to WebAssembly via `wasm-pack`. Loads a packed `.bin` produced by [`training/pack/`](../training/distill/), tokenizes input text, runs the BitLinear-faithful forward pass, returns a 384-dim L2-normalized embedding.

---

## Overview

| | |
|---|---|
| **Input** | UTF-8 string (any length; tokenizer truncates to 128) |
| **Output** | `Float32Array(384)`, L2-normalized |
| **WASM target** | `wasm32-unknown-unknown` (no WASI — runs in Node.js, browsers, Cloudflare Workers, Vercel Edge) |
| **Build tool** | `wasm-pack` + `wasm-opt -Oz` |
| **Format coupling** | Single-format per build (Cargo feature picks the embedding loader at compile time — no runtime dispatch) |

---

## Design principles

- **One build = one format.** Each WASM artifact bakes in exactly one embedding loader. No runtime format dispatch — bundles stay tight, hot paths stay branchless, test matrix stays small.
- **BitLinear weights are always ternary.** That's the project's reason for existing. Only the embedding-table precision varies across builds.
- **Parity-first.** Engine output must match the Python reference at [`training/pack/unpack.py`](../training/pack/unpack.py) (`UnpackedModel.forward()`) within per-format tolerance. Inference-level parity, not byte-level.
- **Static artifacts, baked at compile time.** Tokenizer vocab + model weights are embedded into the WASM via `include_bytes!` — no runtime file I/O, no network at inference time.

---

## Architecture

```
engine/
├── Cargo.toml         features: emb_int8 | emb_ternary | emb_fp32 (mutually exclusive)
├── src/
│   ├── lib.rs         #[wasm_bindgen] public API
│   ├── format.rs      .bin header parse, format-tag dispatch, sha256 verification
│   ├── tokenizer.rs   HF tokenizers crate, OnceLock<Tokenizer>, embedded BERT vocab
│   ├── model.rs       WeightLayout precomputed once, byte-offset lookups into MODEL_BYTES
│   ├── kernels.rs     bitlinear_forward + embedding_lookup_{fp32,int8,ternary,int4}
│   └── inference.rs   end-to-end forward: embedding → transformer layers → final LN → mean pool → fp32 projection → l2 normalize
├── assets/
│   ├── tokenizer.json BERT vocab (~455 KB, committed)
│   └── model.bin      pack output (NOT committed, fetched per release)
└── tests/             per-stage + end-to-end parity vs unpack.UnpackedModel.forward()
```

---

## Build targets

Hierarchy locked by [Phase 4 Stage A](tern-training-pipeline.md#stage-a-results-2026-05-24): int8 matches fp32 quality within noise → fp32 is Pareto-dominated as a ship target. Three feature-gated builds:

| Target | Role | Embedding format | Cargo feature | Bundle (WASM + `.bin`) |
|---|---|---|---|---|
| `tern-engine-emb-int8` | **primary ship** | int8 per-row + fp32 scale | `emb_int8` | ~9.5 MB |
| `tern-engine-emb-ternary` | size-constrained alt ship | packed ternary {-1, 0, +1} + per-row fp32 scale | `emb_ternary` | ~4.0 MB |
| `tern-engine-emb-fp32` | parity reference only — **NOT for user ship** | fp32 row-major | `emb_fp32` | ~32.5 MB |

`emb_int4` is reserved in the wire format (id = 3) but not currently a build target. Adding it requires only a feature gate + a lookup kernel; no format-version bump.

Exactly one feature per build, enforced at compile time via `compile_error!` in `lib.rs`.

---

## Tokenizer

HuggingFace `tokenizers` crate with the `unstable_wasm` feature, `fancy-regex` backend (pure Rust, no C deps).

```toml
tokenizers = { version = "0.19", default-features = false, features = ["unstable_wasm", "fancy-regex"] }
```

Vocab is BERT-base-uncased, loaded once via `OnceLock<Tokenizer>`, embedded into the WASM at compile time:

```rust
static TOKENIZER_BYTES: &[u8] = include_bytes!("../assets/tokenizer.json");
```

No parity test against Python is required — the same Rust crate underpins both training-time tokenization (via Python `transformers`) and engine-time tokenization. Spot-check ~10 strings during integration.

---

## Asset strategy

| Asset | Size | Committed | Why |
|---|---|---|---|
| `assets/tokenizer.json` | ~455 KB | **yes** | Engine can't compile without it; size is negligible; part of the model contract |
| `assets/model.bin` | 2–8 MB per format | **no** | Versioned per training run; fetched per release |
| `assets/model.bin.json` (sidecar) | ~500 B | **no** | Travels with the `.bin`; informational, engine doesn't read it |

Rule: commit small static assets the engine can't compile without; fetch large/versioned artifacts.

---

## `.bin` wire format (v1)

The packer writes a single `.bin` per build target. The engine reads it linearly during load; the only random access at inference time is into the embedding section.

```
header (32 bytes, little-endian):
  magic                 4 bytes   "TERN" (0x5445524E)
  format_version        2 bytes   uint16 — currently 1
  embedding_format      1 byte    0=fp32, 1=int8_per_row, 2=ternary_packed, 3=int4_per_row
  weights_format        1 byte    0=ternary_packed (only option in v1)
  vocab_size            4 bytes   uint32
  d_model               2 bytes   uint16
  n_layers              1 byte
  n_heads               1 byte
  ffn_dim               2 bytes   uint16
  output_dim            2 bytes   uint16
  max_seq_len           2 bytes   uint16
  reserved             10 bytes   zero-padding to 32-byte alignment

embedding section:
  layout depends on embedding_format byte — see "Embedding layouts" below

per-layer transformer block (× n_layers):
  ln_1.weight                       d_model fp32
  ln_1.bias                         d_model fp32
  attn.q/k/v.weight                 packed ternary, each is d_model × d_model
  attn.q/k/v.weight_scale           fp32 per matrix
  attn.out.weight                   packed ternary, d_model × d_model
  attn.out.weight_scale             fp32
  attn.out.bias                     d_model fp32
  ln_2.weight, ln_2.bias            d_model fp32 each
  ffn.up.weight                     packed ternary, d_model × ffn_dim
  ffn.up.weight_scale               fp32
  ffn.up.bias                       ffn_dim fp32
  ffn.down.weight                   packed ternary, ffn_dim × d_model
  ffn.down.weight_scale             fp32
  ffn.down.bias                     d_model fp32

final layer norm:
  ln_final.weight                   d_model fp32
  ln_final.bias                     d_model fp32

output projection (fp32, NOT ternary):
  out.weight                        d_model × output_dim × 4 bytes fp32 row-major
  out.bias                          output_dim × 4 bytes fp32

trailing:
  sha256                            32 bytes — hash of all preceding bytes
```

Q/K/V matrices carry no bias (matches the source architecture). Output projection stays fp32 by design — see [postmortem-bitlinear-asymmetry.md](training/postmortem-bitlinear-asymmetry.md). LayerNorm parameters stay fp32 (tiny, not worth quantizing). Per-matrix `w_scale` for each BitLinear is mandatory.

### Embedding layouts

All quantized formats carry **per-row fp32 scales** — one scale per vocab row. Per-row scaling preserves dynamic range per token; do not collapse to a single global scale.

**`emb_fp32` (format_id = 0)**
```
weights:  vocab_size × d_model × 4 bytes, row-major fp32
```
~31 MB for our config (vocab=30522, d_model=256).

**`emb_int8` (format_id = 1)**
```
weights:  vocab_size × d_model bytes, int8
scales:   vocab_size × 4 bytes, fp32 (one per row)
```
~7.9 MB.

**`emb_ternary` (format_id = 2)**
```
weights:  vocab_size × ceil(d_model × 2 / 8) bytes packed
          (2 bits per element: 00=zero, 01=+1, 10=-1, 11=reserved)
scales:   vocab_size × 4 bytes, fp32 (one per row)
```
~2.1 MB. (Tighter 1.585-bit packing is captured as future work — see [tern-future-work.md §16](tern-future-work.md).)

**`emb_int4` (format_id = 3)** — reserved, no current build target
```
weights:  vocab_size × (d_model / 2) bytes packed
          (4 bits per element, signed symmetric [-7, +7]; lower nibble = element 2k, upper = 2k+1)
scales:   vocab_size × 4 bytes, fp32 (one per row)
```
~4.0 MB.

---

## Public API (`#[wasm_bindgen]`)

| Function | Signature | Purpose |
|---|---|---|
| `embed(text)` | `&str → Vec<f32>` (length = output_dim = 384) | Primary entry point; returns L2-normalized embedding |
| `tokenize(text)` | `&str → Vec<u32>` | Debug — exposes token IDs for parity tests |
| `config_summary()` | `→ String` | Debug — human-readable header dump |

JS bridge stays thin — `similarity(a, b)` and `classify(text, labels)` live above the WASM boundary as pure JS over the returned float32 vectors.

---

## Build

```bash
rustup target add wasm32-unknown-unknown

wasm-pack build --target nodejs --features emb_int8     # primary ship
wasm-pack build --target nodejs --features emb_ternary  # alt ship
wasm-pack build --target nodejs --features emb_fp32     # parity reference (not for users)
```

Post-build, `wasm-opt -Oz` is applied. The per-feature sweep is driven by `scripts/build-engine.sh`.

---

## Verification — parity contract

For every build target, a parity test asserts that the engine's `embed(text)` output matches the Python reference (`UnpackedModel.from_bin(...).forward(...)` from [`training/pack/unpack.py`](../training/pack/unpack.py)) within a per-format tolerance, after L2-normalization.

| Build | Per-dim tolerance |
|---|---|
| `emb_fp32` | 1e-5 (effectively bit-exact) |
| `emb_int8` | 1e-4 |
| `emb_ternary` | 5e-3 |
| `emb_int4` | 5e-4 (placeholder until first measurement) |

**The contract is inference-level, not byte-level.** Byte equality is necessary but not sufficient — a buggy forward pass would pass a byte-level test while shipping a different model. Parity tests run the full forward pass on a fixed input set and compare against the Python reference.

Failure modes the parity test must catch:
- Endianness drift in header or weight reads
- Sign-table mismatch in ternary unpack
- Nibble-order mismatch in int4 unpack
- Off-by-one in per-row scale indexing
- Wrong `activation_scale × weight_scale` combination order in BitLinear
- LayerNorm epsilon mismatch
- Output projection accidentally ternarized (silent quality loss)

Reference dumps and test harness live in [`engine/tests/`](../engine/tests/).

---

## Future format work

Tighter packing (5 ternary values per byte → ~18% smaller `.bin`), transport-layer compression, sparse embedding rows, fp16 output projection, and other deferred byte-savings are captured in [tern-future-work.md §16](tern-future-work.md). They're format-level changes that would require a `format_version` bump; v1 stays as specified above.

---

## Target-device perf

Benchmark harness lives in [`eval/benchmarks/`](../eval/benchmarks/) — separate from training-time eval (which measures the `.pt`); this measures the shipped `.wasm` artifact. Per-run JSON dumps in `eval/benchmarks/results/` are gitignored and machine-specific; the curated baselines below are the source of truth for "how fast was this thing at point X."

### Baselines

| Date | Commit | Build | Device | Cold start | p50 | p99 | RSS | `.wasm` size | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-05-28 | 436579f | `emb_int8` scalar | M4 Max (darwin-arm64) | 693 ms | 499 ms | 529 ms | 146 MB | 11.0 MB | First baseline. Scalar Rust forward, per-call buffer alloc. |
| 2026-05-29 | 1f2ea19 | `emb_int8` + A1 padding skip | M4 Max (darwin-arm64) | 171 ms | 38.2 ms | 72.1 ms | 138 MB | 11.0 MB | [Phase A1] All per-row ops iterate `n_active` rows instead of `seq_len`. **13× faster at p50**, byte-identical smoke output. Already inside the 50–100 ms UI target band. See [tern-runtime-perf.md → Decision log](tern-runtime-perf.md#decision-log). |
| 2026-05-29 | a0b9bd7 | `emb_int8` + A1 + A3 pre-unpack | M4 Max (darwin-arm64) | 151 ms | **3.4 ms** | 6.5 ms | 140 MB | 11.0 MB | [Phase A3] BitLinear weights unpacked from 2-bit packed → `i8` at engine init; matmul inner loop becomes branchless `i8 × i8` MAC. **11.2× faster than A1**, byte-identical embedding vector. Cumulative speedup vs baseline: 147× at p50. Likely picking up V8 NEON auto-vec; Phase B will lock that in portably. |
| 2026-05-29 | d6adfe0 | `emb_int8` + A1 + A3 + B2 SIMD | M4 Max (darwin-arm64) | 150 ms | **1.82 ms** | 3.51 ms | 141 MB | 10.99 MB | [Phase B2] Explicit WASM `simd128` intrinsics in the BitLinear matmul inner loop (`v128_load` + `i16x8_extmul_*` + `i32x4_extadd_pairwise`). **1.87× faster than A3** on V8, bit-identical vector. SIMD opcodes are now baked into the `.wasm` itself, so any compliant runtime (Wasmtime, JavaScriptCore, etc.) executes SIMD — no dependency on host auto-vec. Cumulative speedup vs baseline: **274× at p50**. |

Coverage we'll want eventually but don't have yet:
- `emb_ternary` baseline on the same device (one rebuild + re-run)
- Mid-tier x86 laptop baseline (Node.js, generic Linux)
- Generic browser WASM baseline (headless Chromium via Playwright)
- Mobile baseline (lowest priority — depends on actual deployment surface)

### What moves these numbers in future iterations

The optimization strategy lives in [tern-runtime-perf.md](tern-runtime-perf.md) — that doc owns the ranked menu of levers, the SIMD design, and the per-iteration decision log. High-level pointers:

- **Runtime (forward-pass) optimization** — padding skip, `wasm-opt -O3`, pre-unpack weights, SIMD, buffer reuse. See [tern-runtime-perf.md → Phases A / B / C](tern-runtime-perf.md). Expected stacked impact: 5–10× p50 reduction.
- **Format-level size optimization** — [§16 tight ternary packing](tern-future-work.md#16-bin-format-tighter-packing--compression-options) reduces `.wasm` size for `emb_ternary` builds (~18% smaller `.bin`); no effect on `emb_int8`. No runtime impact.
- **Transport-layer gzip** ([§16.2](tern-future-work.md)) — reduces *download* size; on-device numbers unchanged.

### Smoke test (semantic similarity sanity)

Complementary to perf: [`eval/benchmarks/smoke.js`](../eval/benchmarks/smoke.js) runs ~10 sentence pairs through the engine and prints cosine scores. Catches "model loaded but produces semantic garbage" failure modes that aggregate quality metrics (Spearman, NDCG@10) would mask. Run after every build, before perf measurement.



## Notes (ignore)

| Step | What                                                                                                            | Risk                                                                                 |
| ---- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 1    | Confirm `tokenizers` + `unstable_wasm` still compiles cleanly on current Rust/wasm-pack                         | Low — tern-core proved it works, but the crate may have moved versions               |
| 2    | Write `format.rs` — header parse, validate magic + version, format-tag dispatch                                 | Low (mirrors `format.py`)                                                            |
| 3    | Write `tokenizer.rs` — HF crate + `include_bytes!` + `OnceLock`                                                 | Low (copy from tern-core's tokenizer.rs:1-46 verbatim, adjust for our crate version) |
| 4    | Write `model.rs` — `WeightLayout` walking the v1 spec, computing offsets, format-aware embedding section sizing | Medium (multi-format layout is new)                                                  |
| 5    | Write `kernels.rs::bitlinear_forward` — mirror `unpack.bitlinear_forward` exactly                               | **High** (postmortem-class — needs careful parity verification)                      |
| 6    | Write `kernels.rs::embedding_lookup_*` — feature-gated, per-format                                              | Medium                                                                               |
| 7    | Write `inference.rs::embed` — full transformer forward                                                          | Medium                                                                               |
| 8    | Set up parity tests against `UnpackedModel.forward()` outputs                                                   | Low (mechanics are well-trodden)                                                     |
| 9    | Build all 3 ship targets via `wasm-pack --features <emb_X>`, run all parity tests                               | Gate — must pass before shipping anything                                            |
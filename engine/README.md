# engine — Rust → Wasm inference engine

> For contributors building or modifying the WASM engine. If you're *using* ternlight in an app, see [`packages/ternlight/`](../packages/ternlight) instead.

The math half of ternlight. Reads the trained model from `assets/model.bin`, tokenizes input via the embedded BERT tokenizer, runs the full BitLinear-faithful forward pass, and returns a 384-dim L2-normalized embedding.

## Build

```bash
wasm-pack build --target nodejs
```

Output lands in `pkg/`. `scripts/build-engine.sh` (at the repo root) runs this plus `wasm-opt -Oz` and copies the artifacts into `packages/semantic/`.

## Layout

```
engine/
├── Cargo.toml
├── src/
│   ├── lib.rs           wasm-bindgen exports — public surface (embed, tokenize)
│   ├── tokenizer.rs     HuggingFace BERT tokenizer, lazy-init, embedded vocab
│   ├── model.rs         .bin format parser, layout offsets, weight readers
│   └── inference.rs     forward pass — embedding lookup, attention, FFN, BitLinear
├── tests/               engine parity tests vs Python reference dumps
└── assets/
    ├── tokenizer.json   BERT vocab — committed, embedded at compile time
    └── model.bin        trained weights — NOT committed, fetched per release
```

## Math reference

Forward pass details (every step, with code line refs) live in
[../docs/model-internals.md](../docs/model-internals.md). The
`bitlinear_forward` function in `inference.rs` mirrors `BitLinear.forward`
from the training-time library exactly — keeping training-time and
runtime arithmetic in lockstep is required for quality to transfer.
Parity tests in `tests/` enforce this.

## Tests

```bash
node tests/test_embed.js
```

Tests in `tests/` validate engine output against Python reference dumps. The
references must be derived from the *real* training-time model — see the
postmortem doc for why hand-written references aren't sufficient.

## Status

v0.1, pre-alpha. Engine produces eval-quality embeddings, all regression tasks
pass against Phase 1 baselines. Performance is the active workstream — current
~570 ms per call, target ~50 ms with SIMD + cached weight unpacking.
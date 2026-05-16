# training/pack

Stage 2 of the training pipeline: bit-pack a `.pt` checkpoint into the packed `.bin` format the Wasm engine consumes.

## What it does

Reads the float32 shadow weights from the checkpoint, applies post-training quantization (AbsMean for the embedding, AbsMedian round-clamp for BitLinear weights — matches what BitLinear does at runtime), packs ternary weights at 2 bits each, captures per-matrix `w_scale` values, and writes the `.bin` file.

```bash
python pack.py --checkpoint ../distill/runs/micro-qat-150k-100ep/checkpoint_ep100.pt \
               --output out/model.bin
```

## Layout

```
pack/
├── pack.py        entry point: .pt → .bin
├── verify.py      round-trip a packed .bin and compare against the source .pt
└── tests/         pytest — ternary round-trip, byte-level layout, w_scale parity
```

## .bin format

Documented in [../../docs/tern-architecture.md](../../docs/tern-architecture.md) (system-level) and in `pack.py`'s docstrings (byte-level). Key points:

- 24-byte header with magic, format version, model dims, output_dim, weight bytes
- Embedding table (ternary, packed 2 bits/weight)
- Per layer: each BitLinear matrix followed by its `w_scale` (f32) and optional bias
- Final layer norm (f32)
- Output projection (f32, NOT ternary — see postmortem for why)

Format version 2 — see [../../docs/training/postmortem-bitlinear-asymmetry.md](../../docs/training/postmortem-bitlinear-asymmetry.md) for the v1 → v2 migration that added per-BitLinear `w_scale`.

## Why this stage is "pack" and not "export"

What happens here is specifically bit-packing: ternary weights → 2 bits each, plus serialization into a fixed binary layout. "Export" is generic format-conversion jargon that doesn't tell a contributor what's actually going on; `pack/` does.

## Cross-stage invariant

The post-training quantization math in `pack.py` **must** match the math `training/distill/quantization.py` applies during evaluation. If they drift, the `.bin` ships a different model than `evaluate.py` measured. The bitlinear-asymmetry postmortem is the cautionary tale here.

## Status

Pre-alpha — code migration from `tern-distill-prototype/export/` pending.

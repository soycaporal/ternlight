# training/pack

Phase 5 of the training pipeline: pack a `.pt` checkpoint into the binary `.bin` the WASM engine consumes.

Spec: [docs/tern-inference-engine.md](../../docs/tern-inference-engine.md) is the canonical wire-format reference. This README covers usage and code layout; see the engine doc for byte-level details.

## What it does

Reads the float32 shadow weights from a QAT checkpoint, applies post-training quantization for both BitLinear weights (AbsMedian round-clamp, matches `bitlinear==2.4.6` exactly) and the embedding table (per-row PTQ — int8 / int4 / ternary, or fp32 untouched), writes a single-format `.bin` plus a sidecar JSON manifest.

```
python pack.py \
    --ckpt ../distill/runs/qat-resume-ep10-22ed6bc/checkpoint_ep40.pt \
    --embedding-format int8 \
    --output out/model-int8.bin
```

One `.pt` ckpt produces *N* `.bin` files — one per build target. Run pack.py once per format:

```
python pack.py --ckpt <ckpt> --embedding-format ternary --output out/model-ternary.bin
python pack.py --ckpt <ckpt> --embedding-format int8    --output out/model-int8.bin
python pack.py --ckpt <ckpt> --embedding-format int4    --output out/model-int4.bin
python pack.py --ckpt <ckpt> --embedding-format fp32    --output out/model-fp32.bin
```

The pack format and engine build target must agree — a `.bin` packed for int8 will fail to load in a ternary-built engine (the parser rejects mismatched `embedding_format` byte at load time).

## Layout

```
pack/
├── README.md
├── format.py       header struct + format-tag constants — single source of truth for the wire format
├── encoders.py     per-section encoders (4 embedding formats + bitlinear + layernorm + projection)
├── unpack.py       Python reference reader + engine-equivalent forward pass — the parity reference
├── pack.py         entry point: load .pt → quantize → emit sections → sha256 → sidecar
├── verify.py       pack → unpack → forward parity vs the source .pt — the postmortem-class test
└── manifest.py     sidecar JSON writer (provenance, sha256, source ckpt link)
```

## The parity contract

Before any `.bin` ships, [verify.py](verify.py) must pass for that format:

```
python verify.py --ckpt <ckpt> --embedding-format int8
```

What it asserts: `max |source_model(x) − unpacked_model(x)|` ≤ tolerance, where both forward passes run on the same test inputs. Tolerances per format are defined in [verify.py:_TOLERANCE](verify.py) and mirror the table in [tern-inference-engine.md](../../docs/tern-inference-engine.md#verification--parity-contract).

**This is inference-level parity, not byte-level.** The prior tern-core project shipped a byte-level verifier that missed five BitLinear forward-pass divergences (parameterless LN, per-token int8 activation quant, `w_scale` rescaling, AbsMedian quant formula, exact GELU vs tanh-approx), costing ~23 pts on teacher cosine. The contract here is: source model's forward must match unpacked model's forward.

## Cross-stage invariants

These can drift silently and cause postmortem-class bugs. Treat them as locked:

1. **BitLinear quant formula** (`encoders._bitlinear_quant_params`) must match `bitlinear==2.4.6`'s `BitLinear.forward` exactly. The library is pinned in `training/distill/requirements.txt` for this reason.
2. **Engine-side BitLinear forward** (`unpack.bitlinear_forward`) must match the same library's forward exactly. The Rust engine, when written, must match `unpack.bitlinear_forward` — this is the Python reference.
3. **Output projection stays fp32.** It was trained as plain `nn.Linear`, not `BitLinear`; ternarizing it costs real quality. See [docs/training/postmortem-bitlinear-asymmetry.md](../../docs/training/postmortem-bitlinear-asymmetry.md).
4. **Per-row embedding scales are the chosen design.** Deliberate divergence from tern-core's global-scale POC. Documented as such in the wire format spec; do not "simplify" back.
5. **Q/K/V have no bias; W_out, fc1, fc2 do.** Encoders take `bias: Tensor | None`; the wire format depends on presence/absence being correct per matrix.

## Notes on the embedding PTQ helpers

Three of the embedding PTQ functions live (for now) in different places:

- `int8_quantize_embedding_` → [../distill/ternary_qat.py](../distill/ternary_qat.py) (used by eval-time `evaluation.py` too)
- `ternarize_embedding_` (global-scale) → [../distill/ternary_qat.py](../distill/ternary_qat.py) (used by eval, kept for Stage A reproducibility)
- `_ternary_per_row_embedding_inplace` and `_int4_quantize_embedding_inplace` → [pack.py](pack.py) (pack-only for v1)

**Open task:** the eval-time `ternarize_embedding_` uses a single global scale while the packer uses per-row scales. Stage A's ternary numbers were measured with the global-scale version; the actual shipped model (per-row scales) may have slightly different quality. Re-run Stage A under per-row ternary before treating v1 packer output as the canonical ship artifact.

## Status

v1, pre-flight. Code is written and structured for the parity test to be the gating check. Until `verify.py` passes for every format on the ep40 QAT ckpt, nothing here ships to the engine side.

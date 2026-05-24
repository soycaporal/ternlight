"""Pack a `.pt` checkpoint into ternlight's `.bin` v1.

Usage:
  python pack.py \\
      --ckpt ../distill/runs/qat-resume-ep10-22ed6bc/checkpoint_ep40.pt \\
      --embedding-format int8 \\
      --output out/model-int8.bin

Pipeline:
  1. Load the `.pt` checkpoint + reconstruct model architecture from its config
  2. Apply BitLinear swap (so the model in memory matches what the trainer used)
  3. Apply embedding PTQ matching the requested format — in place
  4. Walk module tree, emit sections in the order defined by tern-inference-engine.md
  5. SHA256 over the body, append as trailing 32 bytes
  6. Write sidecar `.bin.json` if --manifest is provided (or alongside by default)

What this file does NOT do:
  - Multi-format emit in one invocation — call pack.py once per format
  - Quality eval — that's evaluation.py's job, after pack
  - Engine inference — that's the Rust engine
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import torch

# pack/ is meant to be executed from training/distill/'s venv; we ensure the
# distill modules (model, ternary_qat) are importable so we can rebuild the
# architecture and apply the QAT swap.
_PACK_DIR = Path(__file__).parent
_DISTILL_DIR = _PACK_DIR.parent / "distill"
sys.path.insert(0, str(_DISTILL_DIR))
sys.path.insert(0, str(_PACK_DIR))

import ternary_qat
from model import StudentEncoder

from encoders import (
    encode_embedding,
    encode_bitlinear,
    encode_layernorm,
    encode_projection_fp32,
)
from format import (
    EMB_BY_NAME,
    Header,
    HEADER_SIZE,
    SHA256_SIZE,
)
from manifest import write_manifest


# ─────────────────────────────────────────────────────────────────────────────

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _apply_embedding_ptq(model: StudentEncoder, embedding_format_name: str) -> None:
    """Mutate model.embedding.weight in place to match the requested format.

    The packer encodes from the *quantized* tensor; the encoder's job is the
    storage layout (bit-packing, scale separation), not the quant math. We do
    the quant math here, in fp32, so the round-trip (pack → unpack → forward)
    matches the source model's forward exactly.

    For fp32 format, no PTQ is applied — the embedding stays as-trained.
    """
    if embedding_format_name == "fp32":
        return
    if embedding_format_name == "int8":
        stats = ternary_qat.int8_quantize_embedding_(model)
        print(f"  int8 PTQ applied: scale ∈ [{stats['scale_min']:.4f}, {stats['scale_max']:.4f}]")
        return
    if embedding_format_name == "int4":
        # Same shape as int8 but range [-7, 7]
        _int4_quantize_embedding_inplace(model)
        return
    if embedding_format_name == "ternary":
        _ternary_per_row_embedding_inplace(model)
        return
    raise ValueError(f"unknown embedding_format: {embedding_format_name}")


def _int4_quantize_embedding_inplace(model: StudentEncoder) -> None:
    """Per-row int4 PTQ — mirrors encoders.encode_embedding_int4's quant.

    NOT a permanent home — should eventually move to training/distill/ternary_qat.py
    alongside int8_quantize_embedding_(). Left here for now to keep this PR
    self-contained.
    """
    with torch.no_grad():
        w = model.embedding.weight
        emb = w[1:]
        scales = (emb.abs().amax(dim=1) / 7.0).clamp(min=1e-8)
        q = (emb / scales.unsqueeze(1)).round().clamp(-7, 7)
        w[1:] = q * scales.unsqueeze(1)
    print(f"  int4 PTQ applied: scale ∈ [{scales.min().item():.4f}, {scales.max().item():.4f}]")


def _ternary_per_row_embedding_inplace(model: StudentEncoder) -> None:
    """Per-row ternary PTQ — mirrors encoders.encode_embedding_ternary's quant.

    Deliberate upgrade over training/distill/ternary_qat.ternarize_embedding_()
    which uses a single global scale. Eval-time results (Stage A) were measured
    with the global-scale version; results under per-row scales should be
    re-measured before the v1 packer is treated as the canonical ship format.

    Stays in pack.py until we decide to update the eval-time function too.
    """
    with torch.no_grad():
        w = model.embedding.weight
        emb = w[1:]
        scales = emb.abs().mean(dim=1).clamp(min=1e-12)
        thresh = 0.5 * scales.unsqueeze(1)
        signs = torch.sign(emb)
        mag = emb.abs()
        ternary = signs * (mag > thresh).to(emb.dtype)
        # Store dequantized values back (so subsequent forward passes see them)
        w[1:] = ternary * scales.unsqueeze(1)
    n_zero = (ternary == 0).float().mean().item()
    print(f"  ternary PTQ applied (per-row): zero_frac={n_zero:.3f}  scales ∈ [{scales.min().item():.4f}, {scales.max().item():.4f}]")


# ─────────────────────────────────────────────────────────────────────────────

def pack(
    ckpt_path: Path,
    output_path: Path,
    embedding_format_name: str,
    manifest_path: Path | None = None,
    eval_scorecard_path: str | None = None,
    source_data_manifest: str | None = None,
) -> None:
    """Pack a `.pt` ckpt into a `.bin` of the requested embedding format."""
    print(f"→ Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    src_cfg = ckpt["config"]
    src_run = src_cfg.get("run_name", "unknown")
    src_epoch = ckpt["epoch"]
    is_qat = bool(src_cfg.get("enable_qat", False))
    if not is_qat:
        raise ValueError(f"ckpt is not from a QAT run (enable_qat={is_qat}). "
                         f"Packing fp32 baseline ckpts isn't supported — there's no BitLinear to encode.")
    print(f"  source: {src_run}  epoch={src_epoch}  qat={is_qat}")

    # 1) Build architecture from the ckpt's own config (avoids drift)
    model = StudentEncoder(
        vocab_size = src_cfg["vocab_size"],
        d_model    = src_cfg["d_model"],
        n_layers   = src_cfg["n_layers"],
        n_heads    = src_cfg["n_heads"],
        ffn_dim    = src_cfg["ffn_dim"],
        output_dim = src_cfg["output_dim"],
        dropout    = src_cfg["dropout"],
    )

    # 2) Load weights BEFORE swapping nn.Linear → BitLinear
    model.load_state_dict(ckpt["model_state"])

    # 3) Apply QAT swap + lambda=1 (so module tree matches the trained shape)
    n_swapped = ternary_qat.swap(model)
    ternary_qat.set_lambda(model, 1.0)
    print(f"  swapped {n_swapped} nn.Linear → BitLinear (lambda=1)")

    # 4) Apply embedding PTQ in place — required so encoded bytes match the
    #    quantized model the engine will reconstruct
    _apply_embedding_ptq(model, embedding_format_name)
    model.eval()

    # 5) Build the byte stream section by section
    print(f"→ Encoding sections...")
    parts: list[bytes] = []

    # Header
    embedding_format = EMB_BY_NAME[embedding_format_name]
    header = Header(
        embedding_format = embedding_format,
        vocab_size       = src_cfg["vocab_size"],
        d_model          = src_cfg["d_model"],
        n_layers         = src_cfg["n_layers"],
        n_heads          = src_cfg["n_heads"],
        ffn_dim          = src_cfg["ffn_dim"],
        output_dim       = src_cfg["output_dim"],
        max_seq_len      = src_cfg.get("max_seq_len", 128),
    )
    parts.append(header.pack())
    assert len(parts[-1]) == HEADER_SIZE

    # Embedding
    parts.append(encode_embedding(model.embedding.weight.detach(), embedding_format))

    # Per layer
    for li, layer in enumerate(model.layers):
        parts.append(encode_layernorm(layer.norm1))
        # Q/K/V — no bias
        parts.append(encode_bitlinear(layer.attn.W_q.weight.detach(), None))
        parts.append(encode_bitlinear(layer.attn.W_k.weight.detach(), None))
        parts.append(encode_bitlinear(layer.attn.W_v.weight.detach(), None))
        # W_out — has bias
        parts.append(encode_bitlinear(layer.attn.W_out.weight.detach(), layer.attn.W_out.bias.detach()))
        parts.append(encode_layernorm(layer.norm2))
        # FFN — both have bias
        parts.append(encode_bitlinear(layer.ff.fc1.weight.detach(), layer.ff.fc1.bias.detach()))
        parts.append(encode_bitlinear(layer.ff.fc2.weight.detach(), layer.ff.fc2.bias.detach()))

    # Final LN
    parts.append(encode_layernorm(model.norm))

    # Output projection (fp32, NOT ternary — see postmortem)
    parts.append(encode_projection_fp32(model.projection.weight.detach(), model.projection.bias.detach()))

    # 6) Concatenate body + append trailing sha256
    body = b"".join(parts)
    sha256 = hashlib.sha256(body).digest()
    blob = body + sha256
    print(f"  body bytes:    {len(body):,}")
    print(f"  total bytes:   {len(blob):,}  (+{SHA256_SIZE} sha256)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(blob)
    print(f"✓ Wrote {output_path}")

    # 7) Sidecar manifest
    if manifest_path is None:
        manifest_path = output_path.with_suffix(output_path.suffix + ".json")
    write_manifest(
        bin_path             = output_path,
        manifest_path        = manifest_path,
        embedding_format     = embedding_format,
        training_run_id      = f"{src_run}-ep{src_epoch}",
        code_commit          = _git_commit(),
        source_ckpt_path     = str(ckpt_path),
        eval_scorecard_path  = eval_scorecard_path,
        source_data_manifest = source_data_manifest,
    )
    print(f"✓ Wrote {manifest_path}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack a QAT .pt ckpt into a .bin (v1)")
    parser.add_argument("--ckpt", type=Path, required=True, help="path to QAT .pt checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="path to write .bin")
    parser.add_argument("--embedding-format", choices=("fp32", "int8", "int4", "ternary"),
                        required=True, help="embedding precision for this .bin")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="sidecar JSON path (default: <output>.json)")
    parser.add_argument("--eval-scorecard", type=str, default=None,
                        help="optional path/URI to the eval scorecard for this ckpt")
    parser.add_argument("--source-data-manifest", type=str, default=None,
                        help="optional path/URI to the prep cache manifest")
    args = parser.parse_args()
    pack(
        ckpt_path             = args.ckpt,
        output_path           = args.output,
        embedding_format_name = args.embedding_format,
        manifest_path         = args.manifest,
        eval_scorecard_path   = args.eval_scorecard,
        source_data_manifest  = args.source_data_manifest,
    )

"""Pack → unpack → inference parity test.

The postmortem-class test. tern-core's POC verifier only asserted byte-level
round-trip equivalence, and missed five BitLinear forward-pass divergences
that cost ~23 pts teacher cosine and ~10 pts STS-B before they were caught.
This file's job is to never let that class of bug ship undetected.

What it does:
  1. Load the source `.pt` checkpoint + reconstruct it through the SAME path
     pack.py uses (load → swap → set_lambda(1) → apply embedding PTQ).
  2. Pack it to a `.bin` (in a temp file).
  3. Unpack that `.bin` back into UnpackedModel.
  4. Run BOTH the source model AND the unpacked model forward on the same
     input batch from the test cache.
  5. Assert per-format tolerance on the L2-normalized output embeddings.

What it does NOT do:
  - Quality eval (Spearman, NDCG@10) — that's evaluation.py's job
  - Multi-format sweep — call this once per format
"""

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

# Reuse distill venv + add pack/ for our modules
_PACK_DIR = Path(__file__).parent
_DISTILL_DIR = _PACK_DIR.parent / "distill"
sys.path.insert(0, str(_DISTILL_DIR))
sys.path.insert(0, str(_PACK_DIR))

import ternary_qat
from data  import TernDataset, collate_fn, load_cache
from model import StudentEncoder

from pack    import pack as run_pack
from unpack  import UnpackedModel


# Per-format tolerance on L2-normalized output, max abs diff across all elements.
# Aligned with tern-inference-engine.md "Verification — parity contract" table.
_TOLERANCE = {
    "fp32":    1e-5,
    "int8":    1e-4,
    "int4":    5e-4,
    "ternary": 5e-3,
}


def _load_source_model(ckpt_path: Path, embedding_format_name: str) -> StudentEncoder:
    """Rebuild the source-side model exactly as pack.py prepares it.

    Same load → swap → set_lambda → embedding PTQ pipeline. This is what the
    `.bin` is supposed to be a faithful representation of.
    """
    # Imports inside to avoid Pylance namespace confusion with pack's helpers
    from pack import _apply_embedding_ptq
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = ckpt["config"]
    model = StudentEncoder(
        vocab_size = cfg["vocab_size"],
        d_model    = cfg["d_model"],
        n_layers   = cfg["n_layers"],
        n_heads    = cfg["n_heads"],
        ffn_dim    = cfg["ffn_dim"],
        output_dim = cfg["output_dim"],
        dropout    = cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    ternary_qat.swap(model)
    ternary_qat.set_lambda(model, 1.0)
    _apply_embedding_ptq(model, embedding_format_name)
    return model.eval()


def _get_eval_inputs(cache_dir: Path, cache_name: str, n_samples: int = 32, batch_size: int = 32):
    """Pull a small batch from the test split for the parity check.

    The parity check doesn't need many samples — it's testing forward-pass
    correctness, not quality. 32 sequences cover enough sequence lengths /
    padding patterns to catch most off-by-one bugs.
    """
    splits, _manifest = load_cache(cache_dir, cache_name)
    dataset = TernDataset(splits["test"])
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )
    batch = next(iter(loader))
    return batch["input_ids"][:n_samples], batch["attention_mask"][:n_samples]


def verify(
    ckpt_path: Path,
    embedding_format_name: str,
    cache_dir: Path,
    cache_name: str,
    n_samples: int = 32,
    keep_bin: bool = False,
) -> bool:
    """Returns True if parity holds within tolerance, False otherwise."""
    tol = _TOLERANCE[embedding_format_name]
    print(f"╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  Parity verify: embedding_format={embedding_format_name}  tolerance={tol}")
    print(f"╚══════════════════════════════════════════════════════════════════╝")

    with tempfile.TemporaryDirectory(prefix="ternlight-verify-") as tmpdir:
        bin_path = Path(tmpdir) / f"model-{embedding_format_name}.bin"

        # 1) Pack
        print(f"\n[1/4] Pack →")
        run_pack(
            ckpt_path             = ckpt_path,
            output_path           = bin_path,
            embedding_format_name = embedding_format_name,
        )

        # 2) Unpack
        print(f"\n[2/4] Unpack ← {bin_path}")
        ref_model = UnpackedModel.from_bin(bin_path)
        print(f"  unpacked: vocab={ref_model.header.vocab_size}  d_model={ref_model.header.d_model}  "
              f"n_layers={ref_model.header.n_layers}  output_dim={ref_model.header.output_dim}")

        # 3) Load source model the same way pack.py prepared it
        print(f"\n[3/4] Load source model (matching pack-time preparation)")
        src_model = _load_source_model(ckpt_path, embedding_format_name)

        # 4) Forward both on the same inputs, compare
        print(f"\n[4/4] Forward parity ({n_samples} samples from test split)")
        input_ids, attention_mask = _get_eval_inputs(cache_dir, cache_name, n_samples=n_samples)
        with torch.no_grad():
            src_out = src_model(input_ids, attention_mask)       # [N, output_dim], L2-normalized
            ref_out = ref_model.forward(input_ids, attention_mask)
        max_abs_diff = (src_out - ref_out).abs().max().item()
        mean_abs_diff = (src_out - ref_out).abs().mean().item()
        # Cosine between source and ref output as a second signal
        cos = F.cosine_similarity(src_out, ref_out, dim=-1).mean().item()

        ok = max_abs_diff <= tol
        verdict = "PASS ✓" if ok else "FAIL ✗"
        print(f"\n  max  |Δ| = {max_abs_diff:.2e}    tolerance = {tol:.2e}")
        print(f"  mean |Δ| = {mean_abs_diff:.2e}")
        print(f"  cosine(src, ref) = {cos:.6f}  (expected ≈ 1.0)")
        print(f"\n  {verdict}  embedding_format={embedding_format_name}")

        if keep_bin:
            keep_path = Path("/tmp") / bin_path.name
            keep_path.write_bytes(bin_path.read_bytes())
            print(f"\n  (kept .bin at {keep_path})")

        return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parity-verify pack → unpack → forward")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--embedding-format", choices=("fp32", "int8", "int4", "ternary"),
                        required=True)
    parser.add_argument("--cache-dir",  type=Path, default=Path(_DISTILL_DIR) / "cache")
    parser.add_argument("--cache-name", type=str,  default="msmarco_mix_1M")
    parser.add_argument("--n-samples",  type=int,  default=32)
    parser.add_argument("--keep-bin",   action="store_true",
                        help="keep the temp .bin at /tmp for inspection")
    args = parser.parse_args()
    ok = verify(
        ckpt_path             = args.ckpt,
        embedding_format_name = args.embedding_format,
        cache_dir             = args.cache_dir,
        cache_name            = args.cache_name,
        n_samples             = args.n_samples,
        keep_bin              = args.keep_bin,
    )
    sys.exit(0 if ok else 1)

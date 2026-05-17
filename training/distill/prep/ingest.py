"""Phase 1 helpers — pure functions that build the prepared dataset.

Only `prep/prepare.py` imports from this file. Each function does one thing and
returns its output; orchestration (logging, ordering, side effects) lives in
`prepare.py`.

Function groups:
    Source loading    HuggingFaceLoader, build_loaders
    Mixing & dedup    mix_sources, deduplicate
    Tokenize/encode   tokenize_batch, teacher_encode
    Splitting         stratified_split
    Provenance        build_manifest, git_commit
    Sanity stats      compute_stats, print_stats_dashboard
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import subprocess
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from config import DataConfig, SourceSpec


# ── Source loading ───────────────────────────────────────────────────────────

class HuggingFaceLoader:
    """Pulls raw text strings from one HF dataset via `load_dataset()`.

    Assumes `text_field` points at a string column. Rows where the field is
    missing or empty are skipped.
    """

    def __init__(self, spec: SourceSpec):
        self.spec = spec

    def load(self, n: int, seed: int) -> list[tuple[str, str]]:
        """Returns [(text, source_name), ...] of length up to n.

        Samples without replacement when the dataset is larger than n.
        Skips empty/whitespace-only strings.
        """
        rng = random.Random(seed)

        ds = load_dataset(
            self.spec.hf_dataset,
            self.spec.hf_config,
            split=self.spec.split,
        )

        total = len(ds)
        indices = (
            list(range(total)) if n >= total
            else rng.sample(range(total), n)
        )

        out: list[tuple[str, str]] = []
        for i in indices:
            text = self._extract_text(ds[i])
            if text:
                out.append((text, self.spec.name))
        return out

    def _extract_text(self, row: dict) -> str | None:
        val = row.get(self.spec.text_field)
        if not isinstance(val, str):
            return None
        return val.strip() or None


def build_loaders(specs: list[SourceSpec]) -> list[tuple[HuggingFaceLoader, float]]:
    """Return [(loader, weight), ...] in input order."""
    return [(HuggingFaceLoader(spec), spec.weight) for spec in specs]


# ── Mixing, dedup ────────────────────────────────────────────────────────────

def mix_sources(
    loaders:       list[tuple[HuggingFaceLoader, float]],
    total_samples: int,
    seed:          int,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Pull weighted N from each loader, concat, shuffle.

    Returns (mixed_samples, per_source_counts). Per-source seed is derived from
    the master seed so the same config reproduces the same mix.
    """
    rng = random.Random(seed)

    mixed: list[tuple[str, str]] = []
    counts: dict[str, int] = {}
    for i, (loader, weight) in enumerate(loaders):
        target = int(round(total_samples * weight))
        loaded = loader.load(target, seed + i + 1)
        counts[loader.spec.name] = len(loaded)
        mixed.extend(loaded)
        print(f"  [{loader.spec.name:30s}] requested {target:>8,}  got {len(loaded):>8,}")

    rng.shuffle(mixed)
    return mixed, counts


def deduplicate(
    samples: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], int]:
    """Exact-text dedup, case-insensitive after whitespace strip.

    Keeps the first occurrence. Returns (deduped, n_removed).
    """
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for text, source in samples:
        key = text.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((text, source))
    return deduped, len(samples) - len(deduped)


# ── Tokenize, teacher-encode ─────────────────────────────────────────────────

def tokenize_batch(
    texts:    list[str],
    tokenizer,
    max_len:  int,
) -> dict[str, torch.Tensor]:
    """Returns {input_ids, attention_mask}, both LongTensor (N, max_len)."""
    return tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )


def teacher_encode(
    texts:       list[str],
    teacher_id:  str,
    revision:    str = "main",
    batch_size:  int = 256,
    device:      str | None = None,
) -> torch.Tensor:
    """Returns (N, 384) FloatTensor — L2-normalized teacher embeddings."""
    if device is None:
        device = _best_device()

    teacher = SentenceTransformer(teacher_id, revision=revision, device=device)
    teacher.eval()

    with torch.no_grad():
        embs = teacher.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_tensor=True,
        )
    return embs.float().cpu()


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Splitting ────────────────────────────────────────────────────────────────

def stratified_split(
    samples:    list[dict],
    val_ratio:  float,
    test_ratio: float,
    seed:       int,
) -> dict[str, list[dict]]:
    """Per-source split → merge → shuffle within splits.

    Each source's samples are independently shuffled and split into
    [test | val | train] in that order. Splits are then concatenated across
    sources and shuffled, so a batch sees mixed sources.
    """
    rng = random.Random(seed)

    by_source: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_source[s["source"]].append(s)

    splits = {"train": [], "val": [], "test": []}
    for source, group in by_source.items():
        rng.shuffle(group)
        n = len(group)
        n_test = int(round(n * test_ratio))
        n_val  = int(round(n * val_ratio))
        splits["test"].extend(group[:n_test])
        splits["val"].extend(group[n_test:n_test + n_val])
        splits["train"].extend(group[n_test + n_val:])

    for k in splits:
        rng.shuffle(splits[k])
    return splits


# ── Provenance ───────────────────────────────────────────────────────────────

def build_manifest(
    cfg:             DataConfig,
    source_counts:   dict[str, int],
    n_dedup_removed: int,
    split_counts:    dict[str, int],
    code_commit:     str,
) -> dict:
    """JSON-serializable manifest written alongside the cache."""
    return {
        "code_commit":     code_commit,
        "packed_at":       datetime.now(timezone.utc).isoformat(),
        "config":          cfg.model_dump(mode="json"),
        "source_counts":   source_counts,
        "n_dedup_removed": n_dedup_removed,
        "split_counts":    split_counts,
    }


def git_commit() -> str:
    """Current HEAD SHA, or 'unknown' if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ── Sanity stats ─────────────────────────────────────────────────────────────

def compute_stats(samples: list[dict], tokenizer) -> dict:
    """Per-source counts, token-length percentiles, teacher-emb norm stats."""
    by_source: dict[str, int] = defaultdict(int)
    for s in samples:
        by_source[s["source"]] += 1

    lengths = np.array([s["attention_mask"].sum().item() for s in samples])
    norms   = np.array([s["teacher_emb"].norm().item()   for s in samples])

    return {
        "n_samples":     len(samples),
        "by_source":     dict(by_source),
        "length_p50":    int(np.percentile(lengths, 50)),
        "length_p95":    int(np.percentile(lengths, 95)),
        "length_max":    int(lengths.max()),
        "emb_norm_mean": float(norms.mean()),
        "emb_norm_std":  float(norms.std()),
    }


def print_stats_dashboard(stats: dict) -> None:
    n = stats["n_samples"]
    print()
    print(f"  n_samples     : {n:,}")
    print(f"  by_source     :")
    for src, count in stats["by_source"].items():
        print(f"    {src:30s}  {count:>10,}  ({100 * count / n:5.1f}%)")
    print(f"  token length  : p50={stats['length_p50']}  p95={stats['length_p95']}  max={stats['length_max']}")
    print(f"  teacher norm  : mean={stats['emb_norm_mean']:.4f}  std={stats['emb_norm_std']:.4f}  (expect mean≈1.0)")
    print()

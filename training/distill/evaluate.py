"""Phase 4 — ckpt-level quality eval.

Runs three (eventually four) tasks against a QAT checkpoint and writes a
scorecard. Honest measurement: BEFORE eval we ternarize the embedding table
in-place (the shipped .bin will have ternary embeddings; eval must reflect
that, not the fp32 shadow), and we run BitLinear in λ=1 (full ternary)
forward mode.

Tasks
-----
- test_split:   held-out test partition from prep cache. Mirror of val,
                but on data the model has never seen during training.
- stsb:         STS Benchmark (mteb/stsbenchmark-sts). Industry-standard
                sentence-similarity reference. Spearman + Pearson.
- retrieval:    Small BEIR task (BeIR/scifact). NDCG@10. Does it actually
                retrieve well? — the product question.
- qat_health:   Zero-fraction + embed-distribution diagnostics on the
                ternarized model. Catches subtle collapse on held-out data.

Quantization gap
----------------
If `baseline_ckpt_path` is provided, the same tasks run a second time against
the fp32 baseline (Phase 2 ep25). Deltas are written into the scorecard:
`{task}/{metric}_delta_qat_vs_fp32`. This is the load-bearing number for
Phase 3's value proposition.

Usage
-----
    python evaluate.py --config configs/micro-eval.yaml

Output
------
    runs/<run_name>-<sha>/
      scorecard.json   machine-readable; committed to repo
      scorecard.md     human-readable; paste-able into PRs/READMEs
"""

import argparse
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import evaluate as hf_evaluate
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

import ternary_qat
from config  import EvalConfig, load_config
from data    import TernDataset, collate_fn, load_cache
from model   import StudentEncoder


# ── Device + git helpers (lifted from train.py — small enough to duplicate) ──

def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ── Checkpoint loading ────────────────────────────────────────────────────────

@dataclass
class EvalModel:
    """Bundles the loaded model with provenance info from the ckpt."""
    model:            nn.Module
    is_qat:           bool         # was this ckpt trained with BitLinear?
    src_epoch:        int          # ckpt["epoch"]
    src_run_name:     str          # ckpt["config"]["run_name"]
    src_config:       dict         # ckpt["config"] verbatim
    embedding_ternarized: bool     # did we apply ternarize_embedding_()?


def load_for_eval(ckpt_path: Path, device: str) -> EvalModel:
    """Load a ckpt and prepare it for honest eval.

    For QAT ckpts: swap nn.Linear → BitLinear, set λ=1, ternarize the embedding.
    For fp32 ckpts: load as-is (no swap, no embedding ternarization).

    Why ternarize embedding here, not during training: the embedding table is
    NOT touched by QAT — BitLinear only replaces nn.Linear. But the shipped
    .bin will have a ternary embedding (82% of params, otherwise blows the
    size budget). Eval-time ternarization makes the scorecard honest about
    what we'll ship.
    """
    print(f"→ Loading ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    src_cfg = ckpt["config"]
    src_epoch = ckpt["epoch"]
    src_run = src_cfg.get("run_name", "unknown")
    is_qat = bool(src_cfg.get("enable_qat", False))
    print(f"  source run: {src_run}  epoch={src_epoch}  qat={is_qat}")

    # Build architecture from the ckpt's recorded config — this guarantees
    # we don't mismatch d_model / n_layers / etc.
    model = StudentEncoder(
        vocab_size = src_cfg["vocab_size"],
        d_model    = src_cfg["d_model"],
        n_layers   = src_cfg["n_layers"],
        n_heads    = src_cfg["n_heads"],
        ffn_dim    = src_cfg["ffn_dim"],
        output_dim = src_cfg["output_dim"],
        dropout    = src_cfg["dropout"],
    )

    # Load weights BEFORE the swap. BitLinear inherits nn.Linear's state_dict
    # keys ('weight', 'bias'), so this loads cleanly for both fp32 and QAT.
    model.load_state_dict(ckpt["model_state"])

    embedding_ternarized = False
    if is_qat:
        n_swapped = ternary_qat.swap(model)
        ternary_qat.set_lambda(model, 1.0)
        print(f"  swapped {n_swapped} nn.Linear → BitLinear (lambda=1)")
        stats = ternary_qat.ternarize_embedding_(model)
        embedding_ternarized = True
        print(f"  ternarized embedding: scale={stats['scale']:.4f}  zero_frac={stats['zero_fraction']:.3f}")

    model.to(device).eval()
    return EvalModel(
        model=model, is_qat=is_qat, src_epoch=src_epoch,
        src_run_name=src_run, src_config=src_cfg,
        embedding_ternarized=embedding_ternarized,
    )


# ── Task 1: held-out test split ───────────────────────────────────────────────

def eval_test_split(em: EvalModel, ecfg: EvalConfig, device: str) -> dict[str, float]:
    """Forward pass over the test split. Reports Spearman of pairwise student
    similarities vs pairwise teacher similarities + collapse diagnostics.

    Same metric shape as `trainer._eval_epoch` so we can compare apples to
    apples against training-time val numbers.
    """
    print(f"\n→ Task 1/3: held-out test split")
    splits, manifest = load_cache(ecfg.cache_dir, ecfg.cache_name)
    loader = DataLoader(
        TernDataset(splits["test"]),
        batch_size=ecfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    spearman = hf_evaluate.load("spearmanr")

    student_sims: list[float] = []
    teacher_sims: list[float] = []
    student_emb_chunks: list[torch.Tensor] = []
    total_distill = 0.0
    n_batches = 0

    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            teacher_emb    = batch["teacher_emb"].to(device)

            student_emb = em.model(input_ids, attention_mask)

            # Distillation loss as a single scalar — same definition as training
            cos_sim = torch.nn.functional.cosine_similarity(student_emb, teacher_emb, dim=-1)
            total_distill += (1.0 - cos_sim).mean().item()
            n_batches += 1

            # Pairwise cosine within batch (both L2-normalized)
            s_sim = (student_emb @ student_emb.T).cpu()
            t_sim = (teacher_emb @ teacher_emb.T).cpu()
            idx = torch.triu_indices(s_sim.size(0), s_sim.size(1), offset=1)
            student_sims.extend(s_sim[idx[0], idx[1]].tolist())
            teacher_sims.extend(t_sim[idx[0], idx[1]].tolist())

            student_emb_chunks.append(student_emb.cpu())

    elapsed = time.time() - t0
    result = spearman.compute(predictions=student_sims, references=teacher_sims)

    all_embs = torch.cat(student_emb_chunks, dim=0)
    embed_std_mean = all_embs.std(dim=0).mean().item()
    max_offdiag = max(student_sims) if student_sims else 0.0

    metrics = {
        "test/spearman":              result["spearmanr"],
        "test/distill_loss":          total_distill / n_batches,
        "test/embed_std_mean":        embed_std_mean,
        "test/embed_max_offdiag_cos": max_offdiag,
        "test/n_samples":             len(splits["test"]),
        "test/elapsed_seconds":       round(elapsed, 1),
    }
    print(f"  test/spearman = {metrics['test/spearman']:.4f}  ({elapsed:.1f}s)")
    return metrics


# ── Task 2: STS Benchmark ─────────────────────────────────────────────────────

def eval_stsb(em: EvalModel, ecfg: EvalConfig, device: str) -> dict[str, float]:
    """STS-B from HuggingFace — sentence-pair similarity vs human labels.

    Returns Spearman + Pearson between predicted cos(emb(s1), emb(s2)) and
    the gold similarity score (0-5 scale).

    TODO: implement.
      - Load: datasets.load_dataset(ecfg.stsb_dataset, split="test")
      - Tokenize each sentence with bert-base-uncased (matches training prep)
      - Forward pass, get student embeddings (already L2-normalized)
      - Compute cosine per pair
      - scipy.stats.spearmanr + pearsonr against gold

    Why this metric: STS-B is the canonical sentence-embedding benchmark.
    A number here is what gets external trust — "tern scores X on STS-B".
    """
    print(f"\n→ Task 2/3: STS-B  [TODO — stubbed]")
    return {
        "stsb/spearman": float("nan"),
        "stsb/pearson":  float("nan"),
        "stsb/_status":  "not_implemented",
    }


# ── Task 3: retrieval (small BEIR task) ───────────────────────────────────────

def eval_retrieval(em: EvalModel, ecfg: EvalConfig, device: str) -> dict[str, float]:
    """Retrieval NDCG@10 on a small BEIR corpus. The product question:
    does it actually retrieve semantically similar docs?

    TODO: implement.
      - Load: datasets.load_dataset(ecfg.retrieval_dataset, ...) — corpus, queries, qrels
        SciFact: ~1.4k corpus / 300 queries / qrels — fits comfortably in memory.
      - Encode all corpus docs (one batched forward pass)
      - Encode all queries
      - For each query: cosine vs all corpus → top-10
      - Compute NDCG@10 against qrels. Either `ranx`/`pytrec_eval`, or
        hand-roll since the dataset is small.

    Why this metric: STS-B is similarity, this is *retrieval*. They can
    diverge — a model can score well on STS-B but rank poorly under cosine.
    For a "ships as semantic search library" demo, retrieval is the harder
    test.
    """
    print(f"\n→ Task 3/3: retrieval ({ecfg.retrieval_dataset})  [TODO — stubbed]")
    return {
        "retrieval/ndcg@10":     float("nan"),
        "retrieval/recall@10":   float("nan"),
        "retrieval/_status":     "not_implemented",
    }


# ── Bucket C: QAT health (cheap, informative) ─────────────────────────────────

def eval_qat_health(em: EvalModel) -> dict[str, float]:
    """Layer-wise zero fractions on the ternarized weights.

    Only meaningful for QAT ckpts. Health thresholds (from POC):
      20-40%: healthy — all three states {-1, 0, +1} in use
      40-60%: watch  — model leaning sparse, may still recover
      >60%:   collapse — layer is dying
    """
    print(f"\n→ Bucket C: QAT health")
    if not em.is_qat:
        print("  fp32 ckpt — no zero fractions to report")
        return {}

    zf = ternary_qat.zero_fractions(em.model)
    if not zf:
        return {}

    metrics = {
        "qat/zero_frac_avg": sum(zf.values()) / len(zf),
        "qat/zero_frac_max": max(zf.values()),
        "qat/zero_frac_min": min(zf.values()),
        "qat/n_bitlinear":   len(zf),
    }
    print(f"  qat/zero_frac_avg = {metrics['qat/zero_frac_avg']:.3f}  "
          f"max = {metrics['qat/zero_frac_max']:.3f}")
    return metrics


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_all_tasks(em: EvalModel, ecfg: EvalConfig, device: str, prefix: str = "") -> dict[str, float]:
    """Run every enabled task on a single model, return a flat metrics dict.

    `prefix` (e.g. "fp32_") is prepended to all metric keys when this is the
    baseline pass — so the qat run is `test/spearman` and the baseline is
    `fp32_test/spearman`, side by side in the scorecard.
    """
    out: dict[str, float] = {}
    if ecfg.eval_test_split:  out.update(eval_test_split(em, ecfg, device))
    if ecfg.eval_stsb:        out.update(eval_stsb(em, ecfg, device))
    if ecfg.eval_retrieval:   out.update(eval_retrieval(em, ecfg, device))
    if ecfg.eval_qat_health:  out.update(eval_qat_health(em))
    if prefix:
        out = {f"{prefix}{k}" if not k.startswith(prefix) else k: v for k, v in out.items()}
    return out


def compute_gaps(qat_metrics: dict[str, float], fp32_metrics: dict[str, float]) -> dict[str, float]:
    """For metrics present in both qat and fp32 runs, emit a `_delta_qat_vs_fp32`
    column. Higher = QAT is better; lower = QAT cost quality.

    fp32 keys are prefixed with 'fp32_'. We diff against the same metric name
    minus the prefix.
    """
    gaps: dict[str, float] = {}
    for fp32_key, fp32_val in fp32_metrics.items():
        if not fp32_key.startswith("fp32_"):
            continue
        bare_key = fp32_key[len("fp32_"):]
        if bare_key in qat_metrics and isinstance(qat_metrics[bare_key], (int, float)) and isinstance(fp32_val, (int, float)):
            if not (qat_metrics[bare_key] != qat_metrics[bare_key] or fp32_val != fp32_val):  # NaN check
                gaps[f"{bare_key}_delta_qat_vs_fp32"] = qat_metrics[bare_key] - fp32_val
    return gaps


# ── Pretty stdout print ───────────────────────────────────────────────────────
#
# Format intentionally kept loose — we don't know yet what we want to commit as
# the canonical scorecard. Print, look, iterate. File-writers / JSON dumps come
# later if/when there's demand.

def print_results(
    qat_metrics:  dict[str, float],
    fp32_metrics: dict[str, float],
    gaps:         dict[str, float],
    qat_em:       EvalModel,
    fp32_em:      Optional[EvalModel],
) -> None:
    print(f"\n{'='*60}")
    print(f"  Phase 4 eval results")
    print(f"{'='*60}")
    print(f"  QAT:  {qat_em.src_run_name}  ep{qat_em.src_epoch}  (embedding_ternarized={qat_em.embedding_ternarized})")
    if fp32_em is not None:
        print(f"  fp32: {fp32_em.src_run_name}  ep{fp32_em.src_epoch}")

    # Quality — show qat, fp32, delta side by side when baseline present
    print(f"\n  Quality")
    for k in ("test/spearman", "test/distill_loss",
              "stsb/spearman", "stsb/pearson",
              "retrieval/ndcg@10", "retrieval/recall@10"):
        qat_v  = qat_metrics.get(k)
        fp32_v = fp32_metrics.get(f"fp32_{k}")
        delta  = gaps.get(f"{k}_delta_qat_vs_fp32")
        line = f"    {k:34s}  qat={_fmt(qat_v)}"
        if fp32_v is not None:
            line += f"  fp32={_fmt(fp32_v)}  Δ={_fmt(delta, signed=True)}"
        print(line)

    # QAT health (QAT only — fp32 has no zero fractions)
    print(f"\n  QAT health")
    for k in ("qat/zero_frac_avg", "qat/zero_frac_max", "qat/zero_frac_min",
              "qat/n_bitlinear", "test/embed_std_mean", "test/embed_max_offdiag_cos"):
        v = qat_metrics.get(k)
        if v is not None:
            print(f"    {k:34s}  {_fmt(v)}")

    # Stubs / run metadata
    stubs = [k for k in qat_metrics if k.endswith("/_status")]
    if stubs:
        print(f"\n  Stubbed (not implemented yet)")
        for k in stubs:
            print(f"    {k:34s}  {qat_metrics[k]}")


def _fmt(v, signed: bool = False) -> str:
    if v is None:                     return "—"
    if isinstance(v, str):            return v
    if isinstance(v, float):
        if v != v:                    return "NaN"  # NaN check
        return f"{v:+.4f}" if signed else f"{v:.4f}"
    return str(v)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path: Path) -> None:
    cfg = load_config(config_path)
    if cfg.eval is None:
        raise ValueError(f"Config {config_path} has no `eval:` section.")
    ecfg = cfg.eval

    device = best_device()
    print(f"Device: {device}\n")

    # ── Load QAT ckpt ─────────────────────────────────────────────────────────
    qat_em = load_for_eval(ecfg.ckpt_path, device)

    # ── W&B (flexible view; not a format commitment) ──────────────────────────
    short_sha = git_commit()[:7]
    full_run_name = f"{ecfg.run_name}-{short_sha}"
    wandb.init(
        project = cfg.wandb_project,
        group   = cfg.wandb_group,
        job_type= "eval",
        name    = full_run_name,
        tags    = cfg.wandb_tags,
        config  = {
            **ecfg.model_dump(mode="json"),
            "code_commit":       git_commit(),
            "device":            device,
            "qat_src_run":       qat_em.src_run_name,
            "qat_src_epoch":     qat_em.src_epoch,
        },
    )

    # ── Run all tasks on the QAT model ────────────────────────────────────────
    print(f"\n{'='*60}\n  Evaluating QAT ckpt (ep{qat_em.src_epoch})\n{'='*60}")
    qat_metrics = run_all_tasks(qat_em, ecfg, device)

    # ── Optional: same tasks on fp32 baseline for gap computation ─────────────
    fp32_em = None
    fp32_metrics: dict[str, float] = {}
    if ecfg.baseline_ckpt_path is not None:
        print(f"\n{'='*60}\n  Evaluating fp32 baseline\n{'='*60}")
        fp32_em = load_for_eval(ecfg.baseline_ckpt_path, device)
        fp32_metrics = run_all_tasks(fp32_em, ecfg, device, prefix="fp32_")

    # ── Gaps + stdout + W&B log (no committed file format yet) ────────────────
    gaps = compute_gaps(qat_metrics, fp32_metrics)
    print_results(qat_metrics, fp32_metrics, gaps, qat_em, fp32_em)

    wandb.log({**qat_metrics, **fp32_metrics, **gaps})
    wandb.finish()

    print(f"\n✓ Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 — eval a QAT ckpt")
    parser.add_argument("--config", type=Path, required=True, help="path to YAML config")
    args = parser.parse_args()
    main(args.config)

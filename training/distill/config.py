"""Cross-phase configuration schemas. Loaded from YAML.

One `Config` object drives every phase — `prep/prepare.py`, `train.py`,
`evaluate.py` all read from the same YAML file. Each phase consumes only the
sub-section it needs (`data`, `train`, `eval`).

TrainConfig and EvalConfig are stubbed so existing YAML can include their
blocks ahead of time; flesh them out when implementing those phases.
"""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ── Phase 1: data prep ────────────────────────────────────────────────────────

class SourceSpec(BaseModel):
    """One HuggingFace dataset to draw from. Loaded via `load_dataset(hf_dataset, hf_config, split)`."""
    name:       str                          # for the manifest (e.g. "ms_marco")
    hf_dataset: str                          # "ms_marco" or "sentence-transformers/..."
    hf_config:  Optional[str] = None         # e.g. "v2.1" for MS MARCO, "pair" for quora-duplicates
    split:      str = "train"
    text_field: str = "query"                # column holding the text
    weight:     float                        # fraction of total_samples; sum across sources must be 1.0


class DataConfig(BaseModel):
    sources: list[SourceSpec]
    total_samples: int                 # 1_000_000 full, 1_000 smoke
    max_len: int = 128
    teacher_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    teacher_revision: str = "main"
    tokenizer_id: str = "bert-base-uncased"
    teacher_batch_size: int = 256
    seed: int = 42
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    cache_dir: Path = Path("cache")
    cache_name: str = "msmarco_mix_1M"

    @model_validator(mode="after")
    def _weights_sum_to_one(self):
        total = sum(s.weight for s in self.sources)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"source weights must sum to 1.0; got {total}")
        return self


# ── Phase 2/3: training ───────────────────────────────────────────────────────

class TrainConfig(BaseModel):
    """Drives Phase 2 (fp32 baseline) and Phase 3 (QAT). Same Trainer for both;
    `enable_qat` is the switch.

    Architecture defaults match the POC's locked micro tier
    (d_model=256, 2 layers, 4 heads, ffn 1024, output 384).
    """

    # Architecture
    vocab_size:        int   = 30_522        # bert-base-uncased
    d_model:           int   = 256
    n_layers:          int   = 2
    n_heads:           int   = 4
    ffn_dim:           int   = 1_024
    output_dim:        int   = 384           # MiniLM-L6-v2 teacher dim
    dropout:           float = 0.1

    # Data cache reference (consumes prep's output)
    cache_dir:         Path  = Path("cache")
    cache_name:        str

    # Optimization
    epochs:            int   = 20
    batch_size:        int   = 64
    lr:                float = 1e-4
    weight_decay:      float = 0.01
    lr_warmup_ratio:   float = 0.10
    grad_clip:         float = 1.0

    # QAT (Phase 3 only — fp32 baseline leaves these as defaults)
    enable_qat:        bool  = False
    qat_warmup_epochs: int   = 0
    contrastive_w:     float = 0.0

    # Warm-start (optional — load weights from another run's checkpoint before training)
    # Used in Phase 3 to start QAT from a converged fp32 baseline checkpoint.
    # Loads model_state only; optimizer / scheduler / RNG are reset.
    init_from:         Path | None = None

    # Full-state resume (optional — continue a prior run from its last checkpoint).
    # Loads model + optimizer + scheduler + global_step. The loop picks up at
    # epoch == ckpt["epoch"] and runs to `epochs`. Distinct from init_from: this
    # is for resuming the SAME run shape (architecture, scheduler horizon),
    # init_from is for cross-recipe warm-starts.
    resume_from:       Path | None = None

    # Run management
    run_name:          str
    runs_dir:          Path  = Path("runs")
    save_every:        int   = 5
    log_every_n_steps: int   = 100   # per-step train loss/LR/grad-norm to W&B; 0 = disable


# ── Phase 4: eval ─────────────────────────────────────────────────────────────

class EvalConfig(BaseModel):
    """Phase 4 — ckpt-level quality evaluation.

    Produces three buckets of metrics on a single QAT ckpt:
      - Quality:          test/spearman, stsb/{spearman,pearson}, retrieval/ndcg@10
      - Quantization gap: same three vs fp32 baseline ckpt
      - QAT health:       qat/zero_frac_{avg,max}, embed/{std_mean,max_offdiag_cossim}

    Efficiency metrics (bits-per-weight, packed size, inference latency) are
    deferred to Phase 5 / root `eval/`. Those apply to the shipped .bin, not
    the .pt — before pack they're estimates, after pack they're facts.
    """

    # Checkpoint under test (the QAT model we want to score)
    ckpt_path:           Path
    # Optional fp32 baseline (Phase 2 ep25) for quantization-gap deltas
    baseline_ckpt_path:  Path | None = None

    # Data — test split comes from the prep cache (same one training used)
    cache_dir:           Path = Path("cache")
    cache_name:          str
    # External benchmark datasets pulled from HuggingFace
    stsb_dataset:        str = "mteb/stsbenchmark-sts"
    retrieval_dataset:   str = "BeIR/scifact"

    # Eval-time hyperparams (can be larger than train; no backward pass)
    batch_size:          int = 128

    # Task switches — disable any to skip
    eval_test_split:     bool = True
    eval_stsb:           bool = True
    eval_retrieval:      bool = True
    eval_qat_health:     bool = True

    # W&B run name (no committed file format yet — results go to stdout + wandb.log)
    run_name:            str


# ── Top-level ─────────────────────────────────────────────────────────────────

class Config(BaseModel):
    # W&B (shared across phases — levels 1-3: project, run, job_type)
    wandb_project: str = "ternlight-micro"
    wandb_group:   str
    wandb_tags:    list[str] = Field(default_factory=list)

    # Per-phase — all optional so a config can scope to one phase
    data:  Optional[DataConfig]  = None      # Phase 1
    train: Optional[TrainConfig] = None      # Phase 2/3
    eval:  Optional[EvalConfig]  = None      # Phase 4


def load_config(path: Path) -> Config:
    """Load and validate a YAML config."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)

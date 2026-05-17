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


# ── Phase 2/3: training (stub) ────────────────────────────────────────────────

class TrainConfig(BaseModel):
    """Stub — fleshed out when implementing train.py / trainer.py."""
    model_config = ConfigDict(extra="allow")


# ── Phase 4: eval (stub) ──────────────────────────────────────────────────────

class EvalConfig(BaseModel):
    """Stub — fleshed out when implementing evaluate.py."""
    model_config = ConfigDict(extra="allow")


# ── Top-level ─────────────────────────────────────────────────────────────────

class Config(BaseModel):
    # W&B (shared across phases — levels 1-3: project, run, job_type via tags)
    wandb_project: str = "ternlight-micro"
    wandb_group: str
    wandb_tags: list[str] = Field(default_factory=list)

    # Per-phase
    data:  DataConfig
    train: Optional[TrainConfig] = None
    eval:  Optional[EvalConfig]  = None


def load_config(path: Path) -> Config:
    """Load and validate a YAML config."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)

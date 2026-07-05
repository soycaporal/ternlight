"""Loss functions for distillation training.

Phase 2 (fp32 baseline) uses `distillation_loss` only.
Phase 3 (QAT) adds `contrastive_loss` as a guardrail against quantization-
induced embedding collapse — see docs/tern-training-pipeline.md.
"""

import torch
import torch.nn.functional as F


def distillation_loss(
    student_emb: torch.Tensor,   # (B, output_dim) — L2 normalized
    teacher_emb: torch.Tensor,   # (B, output_dim) — L2 normalized
) -> torch.Tensor:
    """Cosine distillation loss.

    Both inputs are unit vectors, so cosine similarity ∈ [-1, 1] and
    `1 - cosine_similarity` ∈ [0, 2]. Perfect alignment → 0. Orthogonal → 1.

    The mean over the batch is what we backprop through. The teacher
    embedding is a fixed target; no gradient flows into it.
    """
    cos_sim = F.cosine_similarity(student_emb, teacher_emb, dim=-1)
    return (1.0 - cos_sim).mean()


def relational_loss(
    student_emb: torch.Tensor,   # (B, output_dim) — L2 normalized
    teacher_emb: torch.Tensor,   # (B, output_dim) — L2 normalized
) -> torch.Tensor:
    """Relational (similarity-matrix) distillation — match the teacher's
    within-batch pairwise geometry.

    `distillation_loss` is pointwise: it pulls each sample toward its own
    teacher vector and says nothing about how sample i relates to sample j.
    Pairwise-ranking metrics (val/spearman, retrieval) score exactly those
    relations. This term matches the batch Gram matrix — every within-batch
    pairwise cosine — between student and teacher, so the student learns the
    teacher's similarity *structure*, not just its vectors.

    The diagonal is excluded: both sides are L2-normalized, so self-similarity
    is 1 on both and would only dilute the mean.

    Lineage: relational knowledge distillation (Park et al., RKD); see
    docs-local/tern-distillation-improvements.md — Recommendation 1.
    """
    s_sim = student_emb @ student_emb.T
    t_sim = teacher_emb @ teacher_emb.T
    off_diag = ~torch.eye(s_sim.size(0), dtype=torch.bool, device=s_sim.device)
    return (s_sim - t_sim)[off_diag].pow(2).mean()


def contrastive_loss(student_emb: torch.Tensor) -> torch.Tensor:
    """Within-batch repulsion — penalize high similarity between *different*
    samples in the same batch.

    Used in Phase 3 (QAT) as a guardrail. Under ternary quantization, the
    model has reduced expressive capacity and can collapse the embedding
    space — many inputs mapped to the same region, each near its teacher
    target but indistinguishable from siblings at retrieval time. This term
    penalizes that pattern.

    Not a true contrastive loss (no known positive/negative pairs). It's a
    constraint that says "different inputs should land at different points."
    Real contrastive learning using anchor/positive pairs is a v2 lever —
    see docs/tern-training-pipeline.md.

    Input `student_emb` is L2-normalized, so `student_emb @ student_emb.T`
    is the cosine similarity matrix.
    """
    sim_matrix = student_emb @ student_emb.T
    return sim_matrix.fill_diagonal_(0).pow(2).mean()

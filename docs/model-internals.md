# Model Internals — Forward Pass, Backprop, and Distillation

> Canonical technical reference for what the student model actually computes,
> how gradients flow during training, and why the distillation dynamics work
> the way they do. Companion to [phase-1-conclusion.md](phase-1-conclusion.md)
> (high-level review) and [postmortem-bitlinear-asymmetry.md](postmortem-bitlinear-asymmetry.md)
> (the inference math we got wrong the first time).

---

## Why this doc exists

Before this document, the deep math lived in three places: `model_scratch.py` comments, `train_qat.py` comments, and the `refs/bitlinear/` source. Reconstructing "what is the actual forward pass" required reading all three plus chasing through library defaults. That ambiguity is exactly what produced the [BitLinear asymmetry bug](postmortem-bitlinear-asymmetry.md). This doc is the single source of truth so that doesn't happen again.

**Audience:** anyone returning to the project, anyone implementing the model in a new target (Wasm, mobile, etc.), anyone modifying the loss or training process.

**Out of scope:** the Wasm engine implementation (covered in [export/GUIDE.md](export/GUIDE.md)), packaging/bundling (`tern-architecture.md`), the eval suite design (`phase-1-conclusion.md` Section "What We Measure").

---

## 1. Forward Pass — Every Step

The model takes a batch of token IDs (after tokenization by `bert-base-uncased`) and produces L2-normalized 384-dim embeddings. Configuration: `d_model=256`, `n_layers=2`, `n_heads=4`, `ffn_dim=1024`, `output_dim=384`, `max_len=128`.

### Pipeline at a glance

```
input_ids               (batch, 128)                                ← token IDs from tokenizer
    ↓ embedding lookup
x                       (batch, 128, 256)                           ← float32
    ↓ TransformerLayer × 2:
    │   ┌─ residual ──────────────────────────────┐
    │   │  norm1                  (batch, 128, 256)│
    │   │   ↓                                      │
    │   │  MultiHeadSelfAttention (Q, K, V, W_out)│
    │   │   ↓                                      │
    │   │  dropout                                 │
    │   └────────────────── + ────────────────────┘
    │   ┌─ residual ──────────────────────────────┐
    │   │  norm2                                   │
    │   │   ↓                                      │
    │   │  FeedForward (fc1, GELU, fc2)            │
    │   │   ↓                                      │
    │   │  dropout                                 │
    │   └────────────────── + ────────────────────┘
    ↓
final norm              (batch, 128, 256)
    ↓ mean pool (mask-aware)
pooled                  (batch, 256)
    ↓ projection (Linear 256→384, float32)
projected               (batch, 384)
    ↓ L2 normalize
output                  (batch, 384)                                ← unit vectors
```

Defined in [`StudentEncoder.forward`](poc/model_scratch.py#L240-L270).

### Step 1 — Embedding lookup

```
x = embedding_table[input_ids]
```

- `embedding_table`: `nn.Embedding(30522, 256)` with `padding_idx=0` — shape (30522, 256), float32 during training, ternary-quantized post-training (`quantize_embedding_to_ternary` in `eval.py`).
- `input_ids`: shape (batch, 128), values in `[0, 30522)`. Token 0 = `[PAD]`.
- Output: shape (batch, 128, 256). Padding positions get the all-zero row 0 because `padding_idx=0` zeroes that row's gradient.

Training: float32 lookups. At eval time after `quantize_embedding_to_ternary`, the embedding table contains only `{-1.0, 0.0, +1.0}` values (still stored as float32 but representing 2-bit values). The shipped `.bin` packs them at 2 bits each.

### Step 2 — Pre-LN attention block (×2 layers)

[`TransformerLayer.forward`](poc/model_scratch.py#L162-L171):

```python
x = x + dropout(attn(norm1(x), mask))
x = x + dropout(ff(norm2(x)))
```

This is the **Pre-LN** variant — LayerNorm is applied *before* the sublayer, not after. More stable for training than the Post-LN layout used in original BERT.

Each `norm1` is a standard `nn.LayerNorm(d_model)` with **learnable** affine parameters (gamma, beta), eps=1e-5. Per-token normalization over the feature dimension.

Sublayer 2a: **multi-head self-attention** (`MultiHeadSelfAttention.forward`, [model_scratch.py:70-109](poc/model_scratch.py#L70-L109)).

```
input_normed: (batch, 128, 256)

# Q, K, V projections (each is a BitLinear during QAT, see Section 1.3)
Q = W_q(input_normed)          # (batch, 128, 256)
K = W_k(input_normed)
V = W_v(input_normed)

# Reshape into heads: (batch, n_heads=4, 128, d_head=64)
Q, K, V = split_heads(Q, K, V)

# Scaled dot-product attention
scores = Q @ K.transpose(-2, -1) / sqrt(d_head)        # (batch, 4, 128, 128)
scores = mask_padding(scores, attention_mask)          # set padding cols to -inf
attn   = softmax(scores, dim=-1)                       # (batch, 4, 128, 128)
attn   = dropout(attn)                                 # only at training time
heads  = attn @ V                                      # (batch, 4, 128, 64)

# Merge heads back: (batch, 128, 256)
out = merge_heads(heads)
out = W_out(out)                                       # (batch, 128, 256)
```

`W_q`, `W_k`, `W_v` have **no bias** (matches `model_scratch.py:63-65`). `W_out` has bias.

Multi-head reshape note: physically, Q/K/V are kept as (batch, 128, 256) in memory and the heads dimension is just an *interpretation* of the last axis when computing attention. The Wasm engine uses index arithmetic instead of physical reshape — same math, no memory copies.

Sublayer 2b: **feed-forward network** (`FeedForward.forward`, [model_scratch.py:135-136](poc/model_scratch.py#L135-L136)).

```
input_normed: (batch, 128, 256)
hidden = fc1(input_normed)     # (batch, 128, 1024)  — expand 4×
hidden = GELU(hidden)          # nonlinearity, exact erf-based at training
hidden = dropout(hidden)
out    = fc2(hidden)           # (batch, 128, 256)   — compress back
```

`fc1` and `fc2` both have bias. The 4× expansion (`d_model → 4·d_model → d_model`) is the BERT default.

### Step 3 — What BitLinear actually does

After `replace_modules(model, match_name=r"^(?!.*projection)")` at the start of QAT, every `nn.Linear` in the attention and FFN (12 total: 6 per transformer layer × 2 layers) becomes a `BitLinear`. The projection is excluded by the negative-lookahead regex.

`BitLinear.forward` ([refs/bitlinear/bitlinear/bitlinear.py:65-79](refs/bitlinear/bitlinear/bitlinear.py)):

```python
def forward(self, x):
    # 1. Internal LayerNorm — parameter-less normalization
    x_norm = torch.layer_norm(x, [in_features])                # eps=1e-5, no affine

    # 2. Activation int8 quantization (per-token AbsMax)
    x_scale = 128 / x_norm.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    x_quant = round_clamp(x_norm * x_scale, [-128, 127])       # f32 representing int8

    # 3. Weight ternary quantization (AbsMedian)
    w_scale = 1 / weight.abs().median().clamp_(min=1e-5)       # scalar per matrix
    w_quant = round_clamp(weight * w_scale, [-1, 1])           # {-1, 0, +1}

    # 4. Quantized matmul + bias
    y_quant = F.linear(x_quant, w_quant, bias)                 # x_quant @ w_quant.T + bias

    # 5. Rescale to natural magnitude
    return y_quant / (w_scale * x_scale)
```

The `round_clamp` operation has a special gradient form (Section 2.1 below).

**Key observation:** the internal LayerNorm has no learnable parameters — it's pure normalize-only. Combined with the explicit `norm1`/`norm2` (which DO have affine params), the effective input to the matmul is "explicit-LN-then-strip-affine-and-renormalize." Mathematically the second LN is approximately a no-op when the first is well-trained (input already mean=0 var=1), but it's not exactly idempotent because of (a) numerical precision and (b) the explicit norm's gamma/beta shifting the distribution slightly.

**5 specific facts about BitLinear that are easy to miss** (source of the [postmortem bug](postmortem-bitlinear-asymmetry.md)):

1. The internal LayerNorm exists. It's the default. It's parameter-less.
2. Activations get int8 quantized via `round_clamp`. They are NOT passed as raw float32 to the matmul.
3. Weights use AbsMedian, not AbsMean. Different cutoff for snapping to zero.
4. The output is divided by `(w_scale · x_scale)`. Without this, magnitudes are off by a per-matrix learned factor.
5. The bias is stored in *post-rescale* space — it's added before the division and gets divided too. Implementations that "store bias separately and add later" need to compensate.

### Step 4 — Final layer norm

```
x = self.norm(x)        # nn.LayerNorm(d_model), per-token, with learnable affine
```

Same architecture as `norm1`/`norm2`. Applied to the output of the second transformer layer's residual.

### Step 5 — Mean pool over real tokens

```python
mask   = attention_mask.unsqueeze(-1).float()           # (batch, 128, 1)
pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
```

Padding positions contribute zero to the sum and aren't counted in the divisor. The `.clamp(min=1e-9)` guards against the degenerate "all padding" case (which shouldn't happen because `[CLS]` and `[SEP]` always exist).

Output: (batch, 256) — one pooled vector per sentence in the batch.

### Step 6 — Projection (256 → 384)

```python
projected = self.projection(pooled)     # nn.Linear(256, 384), float32
```

**Critical: this layer is float32, NOT BitLinear.** Excluded from `replace_modules` by the regex `^(?!.*projection)`. Why:

- The projection bridges the student's internal 256-dim space and the teacher's 384-dim space. Quantization noise on this layer directly corrupts the loss signal during training.
- It has very few parameters (256 × 384 = ~98k) so keeping it float32 is cheap.
- Most of the student's "learning to align with the teacher" happens here — it's the coordinate-frame adapter (Section 3.3 below).

The shipped `.bin` includes this layer as float32 (~395 KB). The Wasm engine applies it via `f32_linear()` in `inference.rs`.

### Step 7 — L2 normalize

```python
output = F.normalize(projected, dim=-1)
```

Divides each row by its L2 norm, producing unit vectors. After this, all embeddings live on the unit hypersphere. Cosine similarity between two L2-normalized vectors is just their dot product:

```
cos(a, b) = (a · b) / (||a|| · ||b||)  =  a · b   when ||a|| = ||b|| = 1
```

This makes downstream cosine-sim computation a single dot product rather than three operations.

---

## 2. Backprop — How Gradients Flow

Training uses `torch.autograd` to compute gradients automatically. The interesting parts are (a) how gradients flow through the quantization in BitLinear, (b) how losses combine, and (c) the dropout/eval distinction.

### 2.1 The QAT trick — straight-through estimator

The fundamental tension: ternary quantization (`round_clamp`) has zero gradient almost everywhere. Without intervention, no gradient would reach the weights through the BitLinear forward, and the model couldn't learn.

The fix is the **straight-through estimator (STE)**: the forward pass uses quantized values, but the backward pass treats the quantization as identity. Gradients flow through as if the operation were a no-op.

In code ([bitlinear.py:11-12](refs/bitlinear/bitlinear/bitlinear.py)):

```python
def round_clamp(input, range, lambda_=1):
    return lambda_ * (input.round().clamp(range[0], range[1]) - input).detach() + input
```

Let's trace what happens during training:

```
Forward (lambda_ = 1):
  output = 1 * (round_clamp(input) - input).detach() + input
         = round_clamp(input)                              # detach() makes this a constant
                                                            # so the +input cancels in value but not grad

Backward (lambda_ = 1):
  d/dx output = 0 (from the .detach() term) + 1 (from + input) = 1
                                                            # gradient flows straight through!
```

So at `lambda_ = 1.0`:
- Forward: behaves as `round_clamp(input)` — quantized
- Backward: gradient is identity — STE in effect

At `lambda_ = 0.0`:
- Forward: `0 * (...) + input = input` — unquantized
- Backward: still identity, but the forward is now also a pass-through

### 2.2 The warmup phase

Training starts with `set_lambda_(model, 0.0)` ([train_qat.py:182](poc/train_qat.py#L182)). For the first 5 epochs the model trains in pure float32 — BitLinear behaves as `nn.Linear` for both forward and backward.

At epoch 6, [train_qat.py:229](poc/train_qat.py#L229) calls `set_lambda_(model, 1.0)` — quantization activates. There's typically a small loss spike (visible in WandB's `train/loss`) as the model adjusts to the quantized forward pass. By epoch ~10 the loss recovers.

Why warmup helps: starting with random weights AND aggressive quantization simultaneously is brittle. Warmup gets the float32 weights into a basin where their *signs* are well-determined (which is what ternary quantization actually preserves), then activating quantization adds noise the model can absorb.

### 2.3 Shadow weights

During QAT, the optimizer updates the **float32** weights (the "shadow weights") even though the forward pass uses ternary versions. The float32 values store the model's actual learned state — quantization is applied each forward pass.

This is how:

```
weight (f32 shadow) ──┬──→ forward: round_clamp(weight * w_scale) → ternary
                      │
                      └──← backward: gradient lands here, optimizer updates this
```

At the end of training, the shadow weights are saved in the checkpoint. At export, they're snapped to ternary one final time and written as 2-bit packed values to `.bin`. The shadow weights themselves are discarded.

This is why `eval.py` calls `replace_modules(model, ...)` — the saved checkpoint has float32 shadow weights, and BitLinear's forward needs to apply quantization at inference time using those values.

### 2.4 Loss → gradient flow

The full loss is:

```python
loss = W_DISTILL * distillation_loss + W_CONTRASTIVE * contrastive_loss
     = 1.0 * (1 - cos(student, teacher)).mean()
     + 0.15 * student_sim_matrix.fill_diagonal_(0).pow(2).mean()
```

Gradient sources, in order of magnitude:

1. **Distillation gradient** (`W_DISTILL = 1.0`)
   - For each sample: ∂loss/∂student_emb = -teacher_emb / batch_size (approximately, since d/dx cos(x, y) when ||x||=1 involves projecting out the radial component)
   - Flows back through L2 normalize → projection → final norm → transformer layers → embedding
   - Pulls student_emb toward teacher_emb
2. **Contrastive gradient** (`W_CONTRASTIVE = 0.15`)
   - For each off-diagonal pair (i, j): ∂loss/∂student_emb_i ∝ student_sim[i,j] * student_emb_j
   - Pushes high-similarity pairs apart
   - Acts as a "spreading force" that prevents the embedding space from collapsing to a small region

The contrastive term is intentionally small (0.15× the distillation weight) because at high values it competes with teacher alignment — the student would prefer to spread embeddings apart than match the teacher. At 0.15 it's a guardrail, not a primary signal.

### 2.5 Per-layer gradient observations

What each part of the model learns from these gradients:

- **Projection layer (`self.projection`)** — receives the most direct loss signal. It's the last learnable layer before the L2 norm and loss. With only 256×384 + 384 = ~98k params, it's tiny but information-dense. Most of "match teacher's coordinate frame" learning happens here.

- **Final layer norm (`self.norm`)** — adjusts the distribution of the pooled vector before the projection. Its affine parameters (gamma, beta) shift mean/scale to whatever the projection prefers as input.

- **Transformer layers** — receive backprop through residuals + attention + FFN. The Pre-LN architecture means gradients flow through the residual paths (`x = x + sublayer(...)`) without being attenuated by sublayer LayerNorms. Each BitLinear's STE means gradients flow through quantization as identity.

- **Embedding table** — receives gradients only at positions corresponding to non-padding token IDs. `padding_idx=0` ensures `[PAD]`'s row gets no gradient. Each token ID's row updates only on samples where that token appears.

### 2.6 Optimizer + scheduler

[train_qat.py:185-193](poc/train_qat.py#L185-L193):

- **AdamW** with lr=2e-4, weight_decay=0.01 (defaults for transformers)
- **Linear warmup** for the first 10% of total steps, then linear decay to 0
- **Gradient clipping** at 1.0 — prevents exploding gradients during the QAT activation spike
- The optimizer updates float32 shadow weights even when QAT is active

### 2.7 Dropout and eval mode

`dropout=0.1` is applied at three places per transformer layer:
- After attention output (in `TransformerLayer.forward`)
- After FFN output (in `TransformerLayer.forward`)
- Inside `FeedForward.forward` between fc1+GELU and fc2

`model.train()` enables dropout. `model.eval()` disables it. The Wasm engine does not implement dropout — at inference time it's always off, so there's nothing to do.

---

## 3. Distillation Dynamics

### 3.1 The teacher's role

Teacher: `sentence-transformers/all-MiniLM-L6-v2` — 22M parameters, 384-dim output, MTEB-leading among small models. Loaded with `normalize_embeddings=True` so its outputs are unit vectors.

The teacher's job is to be a **frozen oracle**: for any input string, it produces a 384-dim "this is what the right embedding looks like" target. It receives no gradient and is never updated. After training, it's discarded — only the trained student ships.

Why the teacher exists at all: training a high-quality embedding model from scratch is expensive (large datasets, long training, careful curriculum). Distillation lets us train a much smaller student (~9M params) using the teacher's outputs as targets, which is far cheaper and produces a model that inherits much of the teacher's semantic understanding.

[prepare.py](poc/prepare.py) caches the teacher's embeddings to disk as a one-time precompute. The training loop then reads them from disk — the teacher itself never runs during training, only its cached outputs are consumed.

### 3.2 Why cosine similarity loss

```python
distillation_loss = (1 - cos(student, teacher)).mean()
```

Three reasons cosine sim is the right choice over alternatives:

1. **Both vectors are unit vectors** (L2-normalized). MSE on raw vectors would over-weight scale differences that don't exist in this setup.
2. **Embedding spaces are cone-shaped** — meaning is encoded in *direction*, not magnitude. Cosine sim is the natural metric for direction matching.
3. **Bounded in [0, 2]** — well-behaved for optimization, no need for gradient scaling.

The loss equals 0 when student and teacher are perfectly aligned, 1 when orthogonal, 2 when opposite. Typical training curve: starts ~1 (random), ends ~0.18 (cos sim ~0.82) for our scaled run.

### 3.3 Why the projection layer was learned

**The asymmetry problem:** student outputs 256-dim, teacher outputs 384-dim. The cosine sim loss requires both vectors to live in the same space. Three ways to bridge this:

- **(A) Project student up to 384** — what we did. A learned `Linear(256 → 384)` adapts the student's internal representation into the teacher's coordinate frame.
- **(B) Project teacher down to 256** — would require either fixed (non-learned) reduction (loses information) or a learned reducer (adds parameters that don't ship).
- **(C) Train both in the same dim from the start** — would require setting `d_model = 384`, increasing student parameter count by ~2.3× (Section 14 of [tern-future-work.md](../tern-future-work.md) discusses this).

Option A is the conventional choice for distilling small students from larger teachers. The projection layer is a learned coordinate adapter — its job is "given the student's encoding, predict where the teacher would have placed this input." Most of the student's distillation learning happens here.

**Implication for the engine:** the projection ships in the `.bin` (was originally dropped, then added back per the Phase 2 work). It applies *after* the encoder body and *before* L2 normalize. See [postmortem-bitlinear-asymmetry.md](postmortem-bitlinear-asymmetry.md) for the dim-reduction trade-offs we considered.

### 3.4 Why the contrastive guardrail (W=0.15)

[train_qat.py:106-111](poc/train_qat.py#L106-L111):

```python
sim_matrix = student_emb @ student_emb.T          # (batch, batch)
contrastive = sim_matrix.fill_diagonal_(0).pow(2).mean()
```

This penalizes high pairwise similarity *within* a batch (excluding self-similarity). It's a **diversity-promoting** loss.

The failure mode it prevents: distillation alone has no penalty for mode collapse. If the student maps every input to roughly the same vector (matching teacher poorly but uniformly), the average cosine distance is still finite. The model can settle into a low-entropy equilibrium where embeddings cluster in a tiny region of the hypersphere.

Adding the contrastive term creates a small "spreading force" — the student is mildly penalized for outputting similar vectors for different inputs. This pushes the embedding cloud toward filling the hypersphere more uniformly.

**Why 0.15 specifically:** larger weights compete with distillation. At W=0.5+, the student starts preferring spread over teacher matching, hurting the primary signal. At W<0.05, the contrastive force is too weak to prevent clustering during long runs. 0.15 was empirically chosen — see WandB run comparisons in [phase-1-conclusion.md](phase-1-conclusion.md).

### 3.5 Why variance loss was removed

An earlier version included a third term that penalized low embedding-component variance directly. It was removed because:

- It was redundant with the contrastive loss (both push for non-collapse)
- It had a different gradient profile that interacted badly with quantization activation at epoch 6
- The model trained more stably with just two terms

If you reintroduce it, expect to retune the QAT warmup carefully.

### 3.6 The data — MS MARCO and what bias it creates

Training corpus: 150k queries from MS MARCO v2.1, train split (queries 0:150000). MS MARCO is a large-scale information retrieval dataset of real Bing queries. Its statistical profile:

- **Almost entirely questions** (~80% interrogative phrasing)
- **Wide topical coverage** but consumer-leaning
- **Short** (typical 5-15 tokens)
- **Sparse on paraphrase pairs** — most queries are unique phrasings

This gave the student strong intuition for question-style intent matching but weaker handling of paraphrases without lexical overlap. The Phase 2 spot-checks showed:

- `"reset my password"` ↔ `"forgot my password"` → 0.80 (overlap on "password")
- `"cancel subscription"` ↔ `"how do I unsubscribe"` → 0.24 (no overlap)

The teacher knows these are similar; the student never saw the example pairs to learn the equivalence. [tern-future-work.md Section 13](../tern-future-work.md) details corpus diversification as the highest-impact future improvement.

### 3.7 Validation methodology — what gets measured during training

[train.py:117-168](poc/train.py#L117-L168) runs `eval_epoch` after each training epoch on a 10% val split:

- **`val/loss`** — distillation loss on val set. Tracks generalization.
- **`val/spearman`** — within-batch pairwise structural Spearman. Computes student pairwise similarity matrix vs teacher pairwise similarity matrix, flattens upper triangle, Spearman correlates. Measures whether the student preserves the *ordering* of similarities, not just individual alignment.

The `val/spearman = 0.813` reported in [phase-1-conclusion.md](phase-1-conclusion.md) is from this metric. **It's NOT the same as `eval.py`'s Task 1 mean cosine sim (also 0.812).** The two metrics happened to land at similar values by coincidence; they measure different things on different data slices. (This caused a stretch of debugging in the Phase 2 regression test — see the postmortem timeline.)

---

## 4. Cross-References

- **Where the math lives in code:**
  - Forward pass: [poc/model_scratch.py](poc/model_scratch.py)
  - BitLinear forward: [refs/bitlinear/bitlinear/bitlinear.py:65-79](refs/bitlinear/bitlinear/bitlinear.py)
  - Quantization measures: [refs/bitlinear/bitlinear/measures.py](refs/bitlinear/bitlinear/measures.py)
  - Internal LayerNorm: [refs/bitlinear/bitlinear/norms.py](refs/bitlinear/bitlinear/norms.py)
  - Distillation loss: [poc/train.py:86-106](poc/train.py#L86-L106)
  - Contrastive loss: [poc/train_qat.py:94-119](poc/train_qat.py#L94-L119)
  - QAT activation: [poc/train_qat.py:217-231](poc/train_qat.py#L217-L231)
  - Eval-time embedding quantization: [poc/eval.py:41-66](poc/eval.py#L41-L66)

- **Engine-side mirror (Wasm inference):**
  - `bitlinear_forward` in [engine/src/inference.rs](engine/src/inference.rs) — must match BitLinear.forward exactly
  - `f32_linear` for the projection layer in the same file

- **Related docs:**
  - [phase-1-conclusion.md](phase-1-conclusion.md) — high-level review of what was built
  - [postmortem-bitlinear-asymmetry.md](postmortem-bitlinear-asymmetry.md) — what we learned about BitLinear the hard way
  - [tern-architecture.md](../tern-architecture.md) — system architecture
  - [tern-phase1-prototype.md](../tern-phase1-prototype.md) — Phase 1 spec

---

## 5. Glossary

- **BitLinear** — drop-in replacement for `nn.Linear` from `schneiderkamplab/bitlinear`. Implements ternary weights + int8 activations + scale rescaling.
- **Distillation** — training a smaller student model to mimic a larger teacher's outputs.
- **QAT (Quantization-Aware Training)** — training where the forward pass uses quantized values but gradients flow through as if unquantized (via STE). Lets the model adapt to quantization noise during training, producing weights that are robust to quantization at inference.
- **STE (Straight-Through Estimator)** — gradient trick where a non-differentiable forward operation (e.g., rounding) has its gradient set to identity in the backward pass. Defined for BitLinear in `round_clamp`.
- **Shadow weights** — full-precision (f32) weights stored during QAT. The optimizer updates these; the forward pass quantizes them on the fly.
- **AbsMedian / AbsMean / AbsMax** — measures used by BitLinear to compute quantization scales. Different choice → different ternary cutoff.
- **Pre-LN vs Post-LN** — Pre-LN applies LayerNorm *before* each sublayer (more stable for training). Post-LN applies it after. Our model uses Pre-LN.
- **Projection (in this codebase)** — the float32 `Linear(256 → 384)` layer at the top of the student that adapts its internal coordinate frame to the teacher's. NOT the Q/K/V/W_out "projections" in attention (those are also called projections in transformer literature; we say "BitLinear" for those to avoid confusion).

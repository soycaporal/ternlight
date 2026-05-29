//! End-to-end forward pass. Mirrors `training/distill/model.py::StudentEncoder::forward`
//! and `training/pack/unpack.py::UnpackedModel::forward`.
//!
//! Flow:
//!   tokenize(text)
//!     → embedding_lookup(input_ids)                                  [T, d_model]
//!     → for each layer:
//!         x_norm = parametric_LN(x, ln1_w, ln1_b)
//!         Q/K/V  = bitlinear_forward(x_norm, …, bias=None)           [T, d_model]
//!         scaled-dot-product attention (per head) + mask + softmax
//!         attn_out = bitlinear_forward(out, W_out, …, bias)
//!         x      = x + attn_out                                       residual
//!         x_norm = parametric_LN(x, ln2_w, ln2_b)
//!         h      = bitlinear_forward(x_norm, fc1, …, bias)            [T, ffn_dim]
//!         h      = GELU(h)
//!         x      = x + bitlinear_forward(h, fc2, …, bias)             residual
//!     → final_LN(x, ln_final_w, ln_final_b)                           [T, d_model]
//!     → mean pool over non-padding positions                          [d_model]
//!     → fp32 projection (d_model → output_dim)                        [output_dim]
//!     → L2 normalize                                                  [output_dim]
//!
//! Per-call allocations (v1): scratch buffers sized for max seq_len (~2 MB
//! per query at T=128, d_model=256, ffn_dim=1024). Only the first `n_active`
//! rows are written/read on any given query — see Phase A1 in
//! `docs/tern-runtime-perf.md`. Buffer reuse across calls is C1 in the same doc.

use crate::kernels;
use crate::model::{self, LoadedModel};
use crate::tokenizer;

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

pub fn embed(text: &str) -> Vec<f32> {
    let m = model::get();
    let h = &m.layout.header;

    let d_model    = h.d_model    as usize;
    let n_layers   = h.n_layers   as usize;
    let n_heads    = h.n_heads    as usize;
    let ffn_dim    = h.ffn_dim    as usize;
    let output_dim = h.output_dim as usize;
    let d_head     = d_model / n_heads;
    let seq_len    = tokenizer::MAX_SEQ_LEN;

    // ── 1) Tokenize ─────────────────────────────────────────────────────────
    let input_ids = tokenizer::tokenize(text);

    // [Phase A1] Skip padded positions in the forward pass. The tokenizer
    // right-pads with PAD_TOKEN_ID, so the real tokens occupy 0..n_active and
    // the rest is padding the attention mask would zero out anyway. Every
    // downstream per-row op (LN, BitLinear, attention, mean pool) iterates
    // only over 0..n_active — for typical UI queries this is 3–5× less work.
    let n_active = input_ids.iter()
        .filter(|&&id| id != tokenizer::PAD_TOKEN_ID)
        .count();

    // ── 2) Embedding lookup → [n_active, d_model] ───────────────────────────
    // Buffer is sized for max seq_len; we only initialize the active prefix.
    let mut x = vec![0.0f32; seq_len * d_model];
    do_embedding_lookup(&input_ids[..n_active], m, &mut x[..n_active * d_model]);

    // ── Scratch buffers reused across layers ────────────────────────────────
    // Sized at max seq_len so we don't re-alloc per query; only the first
    // n_active rows are written/read on any given call.
    let mut x_norm        = vec![0.0f32; seq_len * d_model];
    let mut q_buf         = vec![0.0f32; seq_len * d_model];
    let mut k_buf         = vec![0.0f32; seq_len * d_model];
    let mut v_buf         = vec![0.0f32; seq_len * d_model];
    let mut attn_out      = vec![0.0f32; seq_len * d_model];
    let mut attn_residual = vec![0.0f32; seq_len * d_model];
    let mut ffn_hidden    = vec![0.0f32; seq_len * ffn_dim];
    let mut ffn_out       = vec![0.0f32; seq_len * d_model];
    let mut scores        = vec![0.0f32; n_heads * seq_len * seq_len];

    // ── 3) Per-layer transformer blocks ─────────────────────────────────────
    for layer_idx in 0..n_layers {
        let l_layout = &m.layout.layers[layer_idx];
        let lw       = &m.weights.layers[layer_idx];

        // Pre-LN attention
        parametric_layer_norm(&x, &lw.ln1_w, &lw.ln1_b, n_active, d_model, &mut x_norm);

        // Q / K / V (no bias)
        let wq_packed = &m.body[l_layout.w_q.weights_offset..l_layout.w_q.weights_offset + l_layout.w_q.weights_bytes];
        let wk_packed = &m.body[l_layout.w_k.weights_offset..l_layout.w_k.weights_offset + l_layout.w_k.weights_bytes];
        let wv_packed = &m.body[l_layout.w_v.weights_offset..l_layout.w_v.weights_offset + l_layout.w_v.weights_bytes];
        kernels::bitlinear_forward(&x_norm, wq_packed, lw.wq_scale, None, n_active, d_model, d_model, &mut q_buf);
        kernels::bitlinear_forward(&x_norm, wk_packed, lw.wk_scale, None, n_active, d_model, d_model, &mut k_buf);
        kernels::bitlinear_forward(&x_norm, wv_packed, lw.wv_scale, None, n_active, d_model, d_model, &mut v_buf);

        // Scaled-dot-product attention. After A1 the mask is unnecessary —
        // K and V span only the active prefix, so there are no padding keys
        // to attend to in the first place.
        // Memory layout: Q/K/V are [n_active, d_model] flat, interpreted as
        // [n_active, n_heads, d_head] — element (t, h, d) at index
        // `t * d_model + h * d_head + d`. The padded tail of each buffer is
        // never read.
        scaled_dot_product_attention(
            &q_buf, &k_buf, &v_buf,
            n_heads, d_head, d_model, n_active,
            &mut scores, &mut attn_out,
        );

        // W_out (with bias) — projects concatenated heads back to d_model
        let wout_packed = &m.body[l_layout.w_out.weights_offset..l_layout.w_out.weights_offset + l_layout.w_out.weights_bytes];
        kernels::bitlinear_forward(
            &attn_out, wout_packed, lw.wout_scale, Some(&lw.wout_bias),
            n_active, d_model, d_model, &mut attn_residual,
        );

        // Residual add: x += attn_residual (active prefix only)
        for i in 0..(n_active * d_model) { x[i] += attn_residual[i]; }

        // Pre-LN FFN
        parametric_layer_norm(&x, &lw.ln2_w, &lw.ln2_b, n_active, d_model, &mut x_norm);

        // fc1 (with bias) → [n_active, ffn_dim]
        let fc1_packed = &m.body[l_layout.fc1.weights_offset..l_layout.fc1.weights_offset + l_layout.fc1.weights_bytes];
        kernels::bitlinear_forward(
            &x_norm, fc1_packed, lw.fc1_scale, Some(&lw.fc1_bias),
            n_active, d_model, ffn_dim, &mut ffn_hidden,
        );

        // GELU (exact erf form — matches `F.gelu(approximate='none')`).
        // Only the active prefix is touched; trailing positions stay 0.
        gelu_inplace(&mut ffn_hidden[..n_active * ffn_dim]);

        // fc2 (with bias) → [n_active, d_model]
        let fc2_packed = &m.body[l_layout.fc2.weights_offset..l_layout.fc2.weights_offset + l_layout.fc2.weights_bytes];
        kernels::bitlinear_forward(
            &ffn_hidden, fc2_packed, lw.fc2_scale, Some(&lw.fc2_bias),
            n_active, ffn_dim, d_model, &mut ffn_out,
        );

        // Residual add: x += ffn_out (active prefix only)
        for i in 0..(n_active * d_model) { x[i] += ffn_out[i]; }
    }

    // ── 4) Final parametric LN ──────────────────────────────────────────────
    let mut x_final = vec![0.0f32; seq_len * d_model];
    parametric_layer_norm(
        &x, &m.weights.ln_final_w, &m.weights.ln_final_b,
        n_active, d_model, &mut x_final,
    );

    // ── 5) Mean pool over the active prefix ─────────────────────────────────
    let mut pooled = vec![0.0f32; d_model];
    for t in 0..n_active {
        let row = &x_final[t * d_model..(t + 1) * d_model];
        for d in 0..d_model { pooled[d] += row[d]; }
    }
    let inv_n = 1.0 / (n_active as f32).max(1e-9);
    for d in 0..d_model { pooled[d] *= inv_n; }

    // ── 6) fp32 projection (NOT ternary) ────────────────────────────────────
    let mut projected = vec![0.0f32; output_dim];
    fp32_linear(
        &pooled,
        &m.weights.projection_w, &m.weights.projection_b,
        d_model, output_dim,
        &mut projected,
    );

    // ── 7) L2 normalize ─────────────────────────────────────────────────────
    let norm_sq: f32 = projected.iter().map(|&v| v * v).sum();
    let inv_norm = 1.0 / norm_sq.sqrt().max(1e-12);
    for v in projected.iter_mut() { *v *= inv_norm; }

    projected
}

// ─────────────────────────────────────────────────────────────────────────────
// Embedding-lookup dispatch — one #[cfg] block fires per build
// ─────────────────────────────────────────────────────────────────────────────

fn do_embedding_lookup(ids: &[u32], m: &LoadedModel, out: &mut [f32]) {
    let layout  = &m.layout;
    let d_model = layout.header.d_model as usize;
    let body    = m.body;

    #[cfg(feature = "emb_fp32")]
    {
        let table = &body[layout.embedding_weights_offset
                        ..layout.embedding_weights_offset + layout.embedding_weights_bytes];
        kernels::embedding_lookup(ids, table, d_model, out);
    }

    #[cfg(feature = "emb_int8")]
    {
        let weights_bytes = &body[layout.embedding_weights_offset
                                ..layout.embedding_weights_offset + layout.embedding_weights_bytes];
        // SAFETY: u8 and i8 have identical size + alignment; we're reinterpreting
        // the byte view as a signed-int view of the same bytes.
        let weights: &[i8] = unsafe {
            core::slice::from_raw_parts(weights_bytes.as_ptr() as *const i8, weights_bytes.len())
        };
        kernels::embedding_lookup(ids, weights, &m.weights.embedding_scales, d_model, out);
    }

    #[cfg(any(feature = "emb_ternary", feature = "emb_int4"))]
    {
        let packed = &body[layout.embedding_weights_offset
                         ..layout.embedding_weights_offset + layout.embedding_weights_bytes];
        kernels::embedding_lookup(ids, packed, &m.weights.embedding_scales, d_model, out);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parametric LayerNorm (with learnable scale + shift)
// ─────────────────────────────────────────────────────────────────────────────
//
// y[i] = ((x[i] - mean) / sqrt(var + eps)) * weight[i] + bias[i]
// var is biased (divide by N), matching `torch.nn.LayerNorm`.

const LN_EPS: f32 = 1e-5;

fn parametric_layer_norm(
    x:       &[f32],
    weight:  &[f32],
    bias:    &[f32],
    n_rows:  usize,
    n_cols:  usize,
    out:     &mut [f32],
) {
    debug_assert_eq!(weight.len(), n_cols);
    debug_assert_eq!(bias.len(),   n_cols);
    let inv_n = 1.0 / n_cols as f32;
    for row in 0..n_rows {
        let x_row   = &x[row * n_cols..(row + 1) * n_cols];
        let out_row = &mut out[row * n_cols..(row + 1) * n_cols];
        let mean: f32 = x_row.iter().sum::<f32>() * inv_n;
        let var:  f32 = x_row.iter().map(|&v| { let d = v - mean; d * d }).sum::<f32>() * inv_n;
        let inv_std = 1.0 / (var + LN_EPS).sqrt();
        for i in 0..n_cols {
            out_row[i] = (x_row[i] - mean) * inv_std * weight[i] + bias[i];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scaled-dot-product multi-head attention with padding mask
// ─────────────────────────────────────────────────────────────────────────────
//
// Memory layout (no physical permutation):
//   Q/K/V are `[T, d_model]` flat; element (t, h, d) at `t*d_model + h*d_head + d`.
//   scores is `[n_heads, T, T]`; element (h, t1, t2) at `h*T*T + t1*T + t2`.
//   attn_out (this function's result) matches Q/K/V layout: `[T, d_model]` flat.

fn scaled_dot_product_attention(
    q:        &[f32],
    k:        &[f32],
    v:        &[f32],
    n_heads:  usize,
    d_head:   usize,
    d_model:  usize,
    seq_len:  usize,           // n_active after A1 — all positions are real
    scores:   &mut [f32],      // [n_heads × seq_len × seq_len] scratch (max-sized buffer)
    attn_out: &mut [f32],      // [seq_len × d_model] result (max-sized buffer)
) {
    let scale_factor = 1.0 / (d_head as f32).sqrt();
    let tt = seq_len * seq_len;

    // 1) scores[h, t1, t2] = (Q[t1, h, :] · K[t2, h, :]) / sqrt(d_head)
    for h in 0..n_heads {
        for t1 in 0..seq_len {
            for t2 in 0..seq_len {
                let q_base = t1 * d_model + h * d_head;
                let k_base = t2 * d_model + h * d_head;
                let mut dot = 0.0f32;
                for d in 0..d_head {
                    dot += q[q_base + d] * k[k_base + d];
                }
                scores[h * tt + t1 * seq_len + t2] = dot * scale_factor;
            }
        }
    }

    // 2) Softmax over t2 (stable, max-subtract). No key-masking needed after
    //    A1 — the caller pre-truncates Q/K/V to the active prefix.
    for h in 0..n_heads {
        for t1 in 0..seq_len {
            let row_off = h * tt + t1 * seq_len;
            let mut max_val = f32::NEG_INFINITY;
            for t2 in 0..seq_len {
                let v = scores[row_off + t2];
                if v > max_val { max_val = v; }
            }
            let mut sum = 0.0f32;
            for t2 in 0..seq_len {
                let e = libm::expf(scores[row_off + t2] - max_val);
                scores[row_off + t2] = e;
                sum += e;
            }
            let inv_sum = 1.0 / sum.max(1e-9);
            for t2 in 0..seq_len {
                scores[row_off + t2] *= inv_sum;
            }
        }
    }

    // 3) attn_out[t1, h, d] = sum_{t2} scores[h, t1, t2] * V[t2, h, d]
    for t1 in 0..seq_len {
        for h in 0..n_heads {
            let s_base   = h * tt + t1 * seq_len;
            let out_base = t1 * d_model + h * d_head;
            for d in 0..d_head {
                let mut acc = 0.0f32;
                for t2 in 0..seq_len {
                    let v_val = v[t2 * d_model + h * d_head + d];
                    acc += scores[s_base + t2] * v_val;
                }
                attn_out[out_base + d] = acc;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GELU (exact erf form — `F.gelu(approximate='none')`)
// ─────────────────────────────────────────────────────────────────────────────
//
// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
//
// Uses libm::erff to match PyTorch's default (which is the canonical formula).
// Tanh approximation would diverge from training-time math.

fn gelu_inplace(x: &mut [f32]) {
    let inv_sqrt2: f32 = core::f32::consts::FRAC_1_SQRT_2;
    for v in x.iter_mut() {
        *v = 0.5 * (*v) * (1.0 + libm::erff(*v * inv_sqrt2));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// fp32 projection (the output head — NOT ternary)
// ─────────────────────────────────────────────────────────────────────────────
//
// Standard `F.linear(input, weights, bias)` for the [d_model → output_dim] head.
// Weights are row-major [out_features × in_features].

fn fp32_linear(
    input:        &[f32],
    weights:      &[f32],   // [out_features × in_features] row-major
    bias:         &[f32],   // [out_features]
    in_features:  usize,
    out_features: usize,
    out:          &mut [f32],
) {
    debug_assert_eq!(input.len(),   in_features);
    debug_assert_eq!(weights.len(), out_features * in_features);
    debug_assert_eq!(bias.len(),    out_features);
    debug_assert_eq!(out.len(),     out_features);

    for j in 0..out_features {
        let w_row = &weights[j * in_features..(j + 1) * in_features];
        let mut acc = bias[j];
        for i in 0..in_features {
            acc += input[i] * w_row[i];
        }
        out[j] = acc;
    }
}

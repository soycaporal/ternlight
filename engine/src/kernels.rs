//! Numerical kernels: BitLinear forward + embedding lookup (feature-gated).
//!
//! `bitlinear_forward` is the load-bearing math — it MUST match
//! `training/pack/unpack.py::bitlinear_forward` within the per-format
//! tolerance documented in `docs/tern-inference-engine.md`. The Python
//! function is the reference; the Rust function below is the implementation.
//!
//! Embedding lookup is feature-gated: each build compiles in exactly one
//! `embedding_lookup` matching its embedding-format Cargo feature.
//!
//! Performance note: this is the v1 scalar implementation. SIMD acceleration
//! lives in `docs/tern-future-work.md` §6 (WASM SIMD). Per-call allocations
//! (LN buffer, x_quant buffer) are acceptable at the current performance
//! target; revisit if cold-start or per-query latency becomes the constraint.

// ═════════════════════════════════════════════════════════════════════════════
// BitLinear forward — NOT feature-gated (BitLinear weights are always ternary)
// ═════════════════════════════════════════════════════════════════════════════

const BITLINEAR_EPS:          f32 = 1e-5;
const ACTIVATION_RANGE_MAX:   f32 = 128.0;   // max(|range|) for activation_range = (-128, 127)

/// Engine-equivalent BitLinear forward, mirroring `bitlinear==2.4.6`'s
/// `BitLinear.forward()` with the model's training-time defaults:
///   weight_range = (-1, 1), weight_measure = AbsMedian
///   activation_range = (-128, 127), activation_measure = AbsMax (per-token)
///   norm = parameterless LayerNorm (default eps = 1e-5)
///   strategy = round_clamp, lambda = 1 (inference)
///
/// Math reference: `training/pack/unpack.py::bitlinear_forward`.
///
/// # Arguments
/// - `x`:           `[n_rows × in_features]` fp32 input rows
/// - `w_packed`:    `[out_features × (in_features / 4)]` bytes, 2-bit packed ternary
///                  (codes: 00=zero, 01=+1, 10=-1, 11=reserved). Lower bits = lower index.
/// - `w_scale`:     fp32 scale from the packer (one per matrix; AbsMedian-derived)
/// - `bias`:        optional fp32 bias vector of length `out_features`. None for Q/K/V.
/// - `n_rows`:      number of input rows to process (sequence length)
/// - `in_features`: must be a multiple of 4
/// - `out_features`
/// - `out`:         `[n_rows × out_features]` output buffer, fp32
pub fn bitlinear_forward(
    x:            &[f32],
    w_packed:     &[u8],
    w_scale:      f32,
    bias:         Option<&[f32]>,
    n_rows:       usize,
    in_features:  usize,
    out_features: usize,
    out:          &mut [f32],
) {
    // x and out may be larger than n_rows * stride — callers (post Phase A1)
    // pass max-sized scratch buffers but only ask us to process the active
    // prefix. We only touch the first n_rows * stride elements either way.
    debug_assert!(x.len()   >= n_rows * in_features);
    debug_assert!(out.len() >= n_rows * out_features);
    debug_assert_eq!(in_features % 4, 0, "in_features must be a multiple of 4 for 2-bit packing");
    let bytes_per_w_row = in_features / 4;
    debug_assert_eq!(w_packed.len(), out_features * bytes_per_w_row);

    // Scratch buffers reused across output features within a row. Per-row
    // allocations are the v1 trade-off; revisit if per-query latency is tight.
    let mut x_norm  = vec![0f32; in_features];
    let mut x_quant = vec![0i8;  in_features];

    for row_idx in 0..n_rows {
        let x_row   = &x[row_idx * in_features..(row_idx + 1) * in_features];
        let out_row = &mut out[row_idx * out_features..(row_idx + 1) * out_features];

        // 1) Parameterless LayerNorm over the last dim.
        layer_norm_(x_row, &mut x_norm);

        // 2) Per-token AbsMax → x_scale.
        let mut max_abs: f32 = BITLINEAR_EPS;
        for &v in x_norm.iter() {
            let av = v.abs();
            if av > max_abs { max_abs = av; }
        }
        let x_scale: f32 = ACTIVATION_RANGE_MAX / max_abs;

        // 3) Quantize activations to int8 (round + clamp; STE inactive at inference).
        for i in 0..in_features {
            let q = (x_norm[i] * x_scale).round().clamp(-128.0, 127.0);
            x_quant[i] = q as i8;
        }

        // 4) Matmul (int8 × ternary → int32) + optional bias (fp32) + rescale.
        //    Bias is added in pre-rescale space, matching F.linear semantics in the
        //    Python reference: y = (matmul + bias) / (w_scale * x_scale).
        let rescale = 1.0 / (w_scale * x_scale);

        for j in 0..out_features {
            let w_row = &w_packed[j * bytes_per_w_row..(j + 1) * bytes_per_w_row];
            let mut acc: i32 = 0;
            for (byte_idx, &byte) in w_row.iter().enumerate() {
                // 4 codes per byte, little-endian: element 4k in bits [1:0], 4k+1 in [3:2], etc.
                let base = byte_idx * 4;
                for sub in 0..4 {
                    let code = (byte >> (sub * 2)) & 0b11;
                    let xv = x_quant[base + sub] as i32;
                    match code {
                        0b01 => acc += xv,   // +1
                        0b10 => acc -= xv,   // -1
                        _    => {},          // 0 or reserved/pad
                    }
                }
            }
            let mut y = acc as f32;
            if let Some(b) = bias {
                y += b[j];
            }
            out_row[j] = y * rescale;
        }
    }
}

/// Parameterless LayerNorm: `(x - mean) / sqrt(var + eps)`, biased variance.
/// Matches `torch.layer_norm(x, [n])` with default eps=1e-5.
fn layer_norm_(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    debug_assert!(n > 0);
    let inv_n = 1.0 / n as f32;
    let mean: f32 = input.iter().sum::<f32>() * inv_n;
    let var:  f32 = input.iter().map(|&x| { let d = x - mean; d * d }).sum::<f32>() * inv_n;
    let inv_std = 1.0 / (var + BITLINEAR_EPS).sqrt();
    for i in 0..n {
        output[i] = (input[i] - mean) * inv_std;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Embedding lookup — feature-gated, one variant per build
// ═════════════════════════════════════════════════════════════════════════════
//
// Signature is intentionally NOT uniform across formats — each has different
// input slices (fp32 table; or i8 weights + fp32 scales; or packed bytes +
// scales). Callers from inference.rs choose the right call shape via the
// active feature flag.

/// fp32 embedding lookup. Pure gather — read 4 × d_model bytes per token id
/// as little-endian f32 and copy into `out`.
#[cfg(feature = "emb_fp32")]
pub fn embedding_lookup(
    ids:     &[u32],
    table:   &[u8],          // raw fp32 bytes, row-major
    d_model: usize,
    out:     &mut [f32],
) {
    debug_assert_eq!(out.len(), ids.len() * d_model);
    let row_bytes = d_model * 4;
    for (i, &id) in ids.iter().enumerate() {
        let src = &table[(id as usize) * row_bytes..(id as usize + 1) * row_bytes];
        let dst = &mut out[i * d_model..(i + 1) * d_model];
        for (k, chunk) in src.chunks_exact(4).enumerate() {
            dst[k] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
    }
}

/// int8 embedding lookup with per-row fp32 scale. `dst[k] = (row[k] as f32) * scale`.
#[cfg(feature = "emb_int8")]
pub fn embedding_lookup(
    ids:     &[u32],
    weights: &[i8],          // [vocab_size × d_model] row-major
    scales:  &[f32],         // [vocab_size]
    d_model: usize,
    out:     &mut [f32],
) {
    debug_assert_eq!(out.len(), ids.len() * d_model);
    debug_assert_eq!(weights.len(), scales.len() * d_model);
    for (i, &id) in ids.iter().enumerate() {
        let id_u = id as usize;
        let row   = &weights[id_u * d_model..(id_u + 1) * d_model];
        let scale = scales[id_u];
        let dst   = &mut out[i * d_model..(i + 1) * d_model];
        for k in 0..d_model {
            dst[k] = (row[k] as f32) * scale;
        }
    }
}

/// Ternary embedding lookup with per-row fp32 scale.
/// Codes: 00=zero, 01=+1, 10=-1, 11=reserved/pad. Lower bits = lower index.
/// `bytes_per_row = d_model / 4` (2 bits per element, 4 per byte).
#[cfg(feature = "emb_ternary")]
pub fn embedding_lookup(
    ids:     &[u32],
    packed:  &[u8],          // [vocab_size × (d_model / 4)] row-major
    scales:  &[f32],         // [vocab_size]
    d_model: usize,
    out:     &mut [f32],
) {
    debug_assert_eq!(out.len(), ids.len() * d_model);
    debug_assert_eq!(d_model % 4, 0);
    let bytes_per_row = d_model / 4;
    debug_assert_eq!(packed.len(), scales.len() * bytes_per_row);
    for (i, &id) in ids.iter().enumerate() {
        let id_u = id as usize;
        let row   = &packed[id_u * bytes_per_row..(id_u + 1) * bytes_per_row];
        let scale = scales[id_u];
        let dst   = &mut out[i * d_model..(i + 1) * d_model];
        for (byte_idx, &byte) in row.iter().enumerate() {
            let base = byte_idx * 4;
            for sub in 0..4 {
                let code = (byte >> (sub * 2)) & 0b11;
                dst[base + sub] = match code {
                    0b00 => 0.0,
                    0b01 =>  scale,
                    0b10 => -scale,
                    _    => 0.0,    // reserved/pad — should not appear in valid `.bin`
                };
            }
        }
    }
}

/// int4 embedding lookup with per-row fp32 scale.
/// Two values per byte: lower nibble = element 2k, upper = 2k+1.
/// Signed symmetric range stored as [-7, +7] (-8 excluded per packer); sign-extend the nibble.
#[cfg(feature = "emb_int4")]
pub fn embedding_lookup(
    ids:     &[u32],
    packed:  &[u8],          // [vocab_size × (d_model / 2)] row-major
    scales:  &[f32],         // [vocab_size]
    d_model: usize,
    out:     &mut [f32],
) {
    debug_assert_eq!(out.len(), ids.len() * d_model);
    debug_assert_eq!(d_model % 2, 0);
    let bytes_per_row = d_model / 2;
    debug_assert_eq!(packed.len(), scales.len() * bytes_per_row);
    for (i, &id) in ids.iter().enumerate() {
        let id_u = id as usize;
        let row   = &packed[id_u * bytes_per_row..(id_u + 1) * bytes_per_row];
        let scale = scales[id_u];
        let dst   = &mut out[i * d_model..(i + 1) * d_model];
        for (byte_idx, &byte) in row.iter().enumerate() {
            let low_n  = (byte       & 0x0F) as i8;
            let high_n = ((byte >> 4) & 0x0F) as i8;
            // Sign-extend from 4 bits: values 0..7 stay, values 8..15 map to -8..-1.
            // Packer only emits [-7, 7], so we never observe -8 in practice.
            let low_s  = if low_n  < 8 { low_n  } else { low_n  - 16 };
            let high_s = if high_n < 8 { high_n } else { high_n - 16 };
            dst[byte_idx * 2]     = (low_s  as f32) * scale;
            dst[byte_idx * 2 + 1] = (high_s as f32) * scale;
        }
    }
}

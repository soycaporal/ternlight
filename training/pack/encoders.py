"""Per-section encoders for ternlight `.bin` v1.

Each function returns the bytes that go into the corresponding section of the
wire format. The orchestration (which section in which order, header writing,
sha256) lives in pack.py — this file is just "tensor → bytes" per format.

Quantization formulas:
  - BitLinear weights:  AbsMedian round-clamp, range (-1, 1). Matches
                        `bitlinear==2.4.6`'s forward exactly. See [_bitlinear_quant_params].
  - Embedding (int8):   per-row symmetric quant, scale = max(|row|) / 127.
  - Embedding (int4):   per-row symmetric quant, scale = max(|row|) / 7,
                        signed range [-7, 7] (excludes -8 for sign symmetry).
  - Embedding (ternary):per-row AbsMean threshold (not a single global scale —
                        explicit departure from tern-core POC).
  - LayerNorm + projection: fp32 row-major, no quantization.

Padding row (vocab index 0) handling: the model uses padding_idx=0; the
training-time init zeros that row. We preserve that — the padding row is
written as all-zero values plus zero scale, regardless of format.
"""

import struct
import torch


# ═════════════════════════════════════════════════════════════════════════════
# Embedding encoders
# ═════════════════════════════════════════════════════════════════════════════

def encode_embedding_fp32(W: torch.Tensor) -> bytes:
    """W: [vocab_size, d_model] fp32 → row-major bytes."""
    assert W.dim() == 2 and W.dtype == torch.float32, f"expected [V, D] fp32, got {tuple(W.shape)} {W.dtype}"
    return W.detach().contiguous().cpu().numpy().tobytes()


def encode_embedding_int8(W: torch.Tensor) -> bytes:
    """Per-row symmetric int8 PTQ.

    Layout:
      [vocab_size × d_model] int8 weights, row-major
      [vocab_size] fp32 scales (one per row)

    Padding row → zeros + zero scale.
    """
    assert W.dim() == 2 and W.dtype == torch.float32
    vocab, d_model = W.shape
    with torch.no_grad():
        scales = (W.abs().amax(dim=1) / 127.0).clamp(min=1e-8)  # [vocab]
        scales[0] = 0.0  # padding row stays zero
        # Quantize using safe scales (avoid div-by-zero on padding row)
        safe_scales = scales.clamp(min=1e-12).unsqueeze(1)
        q = (W / safe_scales).round().clamp(-128, 127).to(torch.int8)
        q[0].zero_()
    weights_bytes = q.contiguous().cpu().numpy().tobytes()
    scales_bytes  = scales.contiguous().cpu().numpy().astype("float32").tobytes()
    return weights_bytes + scales_bytes


def encode_embedding_int4(W: torch.Tensor) -> bytes:
    """Per-row symmetric int4 PTQ. Two values per byte, lower nibble first.

    Symmetric range [-7, +7] (excludes -8) to keep zero exactly representable
    and the dequant kernel branch-free.

    Layout:
      [vocab_size × (d_model / 2)] packed int4 weights, row-major
      [vocab_size] fp32 scales (one per row)

    Requires d_model to be even (always true for our architectures).
    """
    assert W.dim() == 2 and W.dtype == torch.float32
    vocab, d_model = W.shape
    assert d_model % 2 == 0, f"int4 packing requires even d_model; got {d_model}"
    with torch.no_grad():
        scales = (W.abs().amax(dim=1) / 7.0).clamp(min=1e-8)
        scales[0] = 0.0
        safe_scales = scales.clamp(min=1e-12).unsqueeze(1)
        q = (W / safe_scales).round().clamp(-7, 7).to(torch.int8)
        q[0].zero_()
        # Map signed [-7,7] → 4-bit two's complement [0..15] for packing
        # Equivalent: q_unsigned = q & 0x0F
        q_u = (q & 0x0F).to(torch.uint8)
        # Pack pairs: lower nibble = element 2k, upper = element 2k+1
        low  = q_u[:, 0::2]
        high = q_u[:, 1::2]
        packed = (low | (high << 4))
    weights_bytes = packed.contiguous().cpu().numpy().tobytes()
    scales_bytes  = scales.contiguous().cpu().numpy().astype("float32").tobytes()
    return weights_bytes + scales_bytes


def encode_embedding_ternary(W: torch.Tensor) -> bytes:
    """Per-row ternary PTQ. Packs 4 weights/byte (2 bits each, little-endian).

    Per-row scale = mean(|row|) (AbsMean). Threshold = 0.5 × scale.
    Quantized values: {-1, 0, +1}. Encoded as 2 bits each:
      0b00 = zero, 0b01 = +1, 0b10 = -1, 0b11 = reserved/pad.

    Layout:
      [vocab_size × ceil(d_model × 2 / 8)] packed weights, row-major
      [vocab_size] fp32 scales (one per row)

    Padding row → zeros + zero scale.

    NOTE: this is a deliberate upgrade over tern-core's global-scale
    ternarization. Per-row scales preserve dynamic range per token, matching
    the int8 path's PTQ shape. Do not "simplify" back to global.
    """
    assert W.dim() == 2 and W.dtype == torch.float32
    vocab, d_model = W.shape
    bytes_per_row = (d_model * 2 + 7) // 8
    assert d_model % 4 == 0, f"ternary packing easier when d_model multiple of 4; got {d_model}"
    with torch.no_grad():
        scales = W.abs().mean(dim=1)                            # [vocab] AbsMean per row
        scales[0] = 0.0
        thresh = (0.5 * scales).clamp(min=1e-12).unsqueeze(1)   # [vocab, 1]
        signs  = torch.sign(W)                                  # [vocab, d_model]
        mag    = W.abs()
        ternary = signs * (mag > thresh).to(W.dtype)            # values in {-1, 0, +1}
        ternary[0].zero_()
        # Encode to 2-bit codes: 0→0b00, +1→0b01, -1→0b10
        codes = torch.zeros_like(ternary, dtype=torch.uint8)
        codes[ternary > 0] = 0b01
        codes[ternary < 0] = 0b10
        # Pack 4 codes per byte, little-endian within byte:
        #   element 0 → bits [1:0], element 1 → bits [3:2], etc.
        codes_reshaped = codes.view(vocab, -1, 4)
        packed = (codes_reshaped[:, :, 0]
                  | (codes_reshaped[:, :, 1] << 2)
                  | (codes_reshaped[:, :, 2] << 4)
                  | (codes_reshaped[:, :, 3] << 6))
    weights_bytes = packed.contiguous().cpu().numpy().tobytes()
    scales_bytes  = scales.contiguous().cpu().numpy().astype("float32").tobytes()
    return weights_bytes + scales_bytes


def encode_embedding(W: torch.Tensor, embedding_format: int) -> bytes:
    """Dispatch by format. Imports from format.py to stay aligned with the spec."""
    from format import EMB_FP32, EMB_INT8, EMB_TERNARY, EMB_INT4
    if embedding_format == EMB_FP32:    return encode_embedding_fp32(W)
    if embedding_format == EMB_INT8:    return encode_embedding_int8(W)
    if embedding_format == EMB_TERNARY: return encode_embedding_ternary(W)
    if embedding_format == EMB_INT4:    return encode_embedding_int4(W)
    raise ValueError(f"unknown embedding_format: {embedding_format}")


# ═════════════════════════════════════════════════════════════════════════════
# BitLinear encoder
# ═════════════════════════════════════════════════════════════════════════════

def _bitlinear_quant_params(W: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, float]:
    """Replicates the math `bitlinear==2.4.6` does internally on every forward.

    Source: bitlinear/bitlinear.py:14-15 (`scale()` function) and BitLinear.forward
            with default weight_range=1.58 → range (-1, 1), AbsMedian measure.

    Formula:
      w_scale = max(|range|) / AbsMedian(W).clamp(min=eps)
              = 1.0       / median(|W|).clamp(min=eps)
      w_quant = round_clamp(W * w_scale, (-1, 1))    # ternary {-1, 0, +1}

    Returns (w_quant: int8 tensor in {-1, 0, +1}, w_scale: python float).
    """
    median = W.detach().abs().median().clamp(min=eps)
    w_scale = 1.0 / float(median)
    w_quant = (W * w_scale).round().clamp(-1, 1).to(torch.int8)
    return w_quant, w_scale


def encode_bitlinear(W: torch.Tensor, bias: torch.Tensor | None) -> bytes:
    """Encode a BitLinear matrix to bytes.

    Layout:
      [out_features × in_features × 2 bits, packed] ternary weights, row-major within row,
        4 weights per byte (lower bits = lower index, same as embedding ternary packing)
      [4 bytes] w_scale fp32
      [out_features × 4 bytes] bias fp32  (only if bias is not None — Q/K/V skip this)

    Caller must know from the wire-format spec which matrices have biases:
      - W_q, W_k, W_v          → no bias
      - W_out, fc1, fc2        → bias

    in_features must be a multiple of 4 (always true for our config: d_model=256,
    ffn_dim=1024).
    """
    assert W.dim() == 2 and W.dtype == torch.float32, f"expected [out, in] fp32, got {tuple(W.shape)} {W.dtype}"
    out_features, in_features = W.shape
    assert in_features % 4 == 0, f"in_features must be multiple of 4; got {in_features}"

    w_quant, w_scale = _bitlinear_quant_params(W)
    # Map ternary values to 2-bit codes: 0→0b00, +1→0b01, -1→0b10
    codes = torch.zeros_like(w_quant, dtype=torch.uint8)
    codes[w_quant > 0] = 0b01
    codes[w_quant < 0] = 0b10
    # Pack 4 codes per byte
    codes_reshaped = codes.view(out_features, -1, 4)
    packed = (codes_reshaped[:, :, 0]
              | (codes_reshaped[:, :, 1] << 2)
              | (codes_reshaped[:, :, 2] << 4)
              | (codes_reshaped[:, :, 3] << 6))
    weights_bytes = packed.contiguous().cpu().numpy().tobytes()
    scale_bytes   = struct.pack("<f", w_scale)
    bias_bytes    = b""
    if bias is not None:
        assert bias.dim() == 1 and bias.numel() == out_features, \
            f"bias shape mismatch: expected [{out_features}], got {tuple(bias.shape)}"
        bias_bytes = bias.detach().contiguous().cpu().numpy().astype("float32").tobytes()
    return weights_bytes + scale_bytes + bias_bytes


# ═════════════════════════════════════════════════════════════════════════════
# LayerNorm + projection encoders (fp32, no quantization)
# ═════════════════════════════════════════════════════════════════════════════

def encode_layernorm(ln: torch.nn.LayerNorm) -> bytes:
    """LN: weight + bias, both fp32, both d_model-shaped."""
    w_bytes = ln.weight.detach().contiguous().cpu().numpy().astype("float32").tobytes()
    b_bytes = ln.bias.detach().contiguous().cpu().numpy().astype("float32").tobytes()
    return w_bytes + b_bytes


def encode_projection_fp32(W: torch.Tensor, bias: torch.Tensor) -> bytes:
    """Output projection: fp32 row-major weight + fp32 bias. NOT ternary."""
    assert W.dim() == 2 and W.dtype == torch.float32
    assert bias.dim() == 1 and bias.numel() == W.shape[0]
    w_bytes = W.detach().contiguous().cpu().numpy().astype("float32").tobytes()
    b_bytes = bias.detach().contiguous().cpu().numpy().astype("float32").tobytes()
    return w_bytes + b_bytes

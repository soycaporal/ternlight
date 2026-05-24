"""Python reference reader for ternlight `.bin` v1.

Mirrors the read path the Rust engine implements. Used by verify.py to confirm
that a packed `.bin` produces the same forward-pass output as the source `.pt`.
Also serves as the engine's behavior specification — when the Rust engine is
written, its forward pass must match this Python implementation within the
tolerance documented in tern-inference-engine.md.

This is NOT a production inference engine. It's slow (pure PyTorch, no
quantized matmul), it doesn't optimize bundle size, and it doesn't run on
WASM. Its only job is to be obviously-correct so the engine can be checked
against it.
"""

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from format import (
    EMB_FP32, EMB_INT8, EMB_TERNARY, EMB_INT4,
    HEADER_SIZE, SHA256_SIZE,
    Header,
)


# ═════════════════════════════════════════════════════════════════════════════
# Section decoders — reverse of encoders.py
# ═════════════════════════════════════════════════════════════════════════════

def decode_embedding_fp32(buf: bytes, off: int, vocab: int, d_model: int) -> tuple[torch.Tensor, int]:
    n_bytes = vocab * d_model * 4
    arr = torch.frombuffer(buf, dtype=torch.float32, count=vocab * d_model, offset=off).clone()
    return arr.view(vocab, d_model), off + n_bytes


def decode_embedding_int8(buf: bytes, off: int, vocab: int, d_model: int) -> tuple[torch.Tensor, int]:
    n_w_bytes = vocab * d_model
    n_s_bytes = vocab * 4
    q = torch.frombuffer(buf, dtype=torch.int8, count=vocab * d_model, offset=off).clone().view(vocab, d_model)
    off += n_w_bytes
    scales = torch.frombuffer(buf, dtype=torch.float32, count=vocab, offset=off).clone()
    off += n_s_bytes
    # Dequant: row i → q[i].float() * scales[i]
    W = q.float() * scales.unsqueeze(1)
    return W, off


def decode_embedding_int4(buf: bytes, off: int, vocab: int, d_model: int) -> tuple[torch.Tensor, int]:
    assert d_model % 2 == 0
    bytes_per_row = d_model // 2
    n_w_bytes = vocab * bytes_per_row
    n_s_bytes = vocab * 4
    packed = torch.frombuffer(buf, dtype=torch.uint8, count=vocab * bytes_per_row, offset=off).clone().view(vocab, bytes_per_row)
    off += n_w_bytes
    scales = torch.frombuffer(buf, dtype=torch.float32, count=vocab, offset=off).clone()
    off += n_s_bytes
    # Unpack nibbles
    low_nib  = packed & 0x0F          # element 2k
    high_nib = (packed >> 4) & 0x0F   # element 2k+1
    # Reconstruct signed int4: values were stored as q & 0x0F where q ∈ [-7, 7]
    # So nibble n maps to:  n if n < 8 else n - 16  (standard two's complement 4-bit)
    def sign_extend(n: torch.Tensor) -> torch.Tensor:
        n_i = n.to(torch.int8)
        return torch.where(n_i < 8, n_i, n_i - 16)
    low_s  = sign_extend(low_nib)
    high_s = sign_extend(high_nib)
    # Interleave back to [vocab, d_model]
    q = torch.zeros((vocab, d_model), dtype=torch.int8)
    q[:, 0::2] = low_s
    q[:, 1::2] = high_s
    W = q.float() * scales.unsqueeze(1)
    return W, off


def decode_embedding_ternary(buf: bytes, off: int, vocab: int, d_model: int) -> tuple[torch.Tensor, int]:
    assert d_model % 4 == 0
    bytes_per_row = d_model // 4
    n_w_bytes = vocab * bytes_per_row
    n_s_bytes = vocab * 4
    packed = torch.frombuffer(buf, dtype=torch.uint8, count=vocab * bytes_per_row, offset=off).clone().view(vocab, bytes_per_row)
    off += n_w_bytes
    scales = torch.frombuffer(buf, dtype=torch.float32, count=vocab, offset=off).clone()
    off += n_s_bytes
    # Extract 4 codes per byte (2 bits each)
    code0 = packed       & 0b11
    code1 = (packed >> 2) & 0b11
    code2 = (packed >> 4) & 0b11
    code3 = (packed >> 6) & 0b11
    # Codes → values: 0b00→0, 0b01→+1, 0b10→-1, 0b11→reserved (should not appear)
    def codes_to_ternary(c: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(c, dtype=torch.float32)
        out[c == 0b01] =  1.0
        out[c == 0b10] = -1.0
        return out
    ternary = torch.zeros((vocab, d_model), dtype=torch.float32)
    ternary[:, 0::4] = codes_to_ternary(code0)
    ternary[:, 1::4] = codes_to_ternary(code1)
    ternary[:, 2::4] = codes_to_ternary(code2)
    ternary[:, 3::4] = codes_to_ternary(code3)
    # Per-row scale gives the final embedding value
    W = ternary * scales.unsqueeze(1)
    return W, off


def decode_embedding(buf: bytes, off: int, header: Header) -> tuple[torch.Tensor, int]:
    vocab, d_model = header.vocab_size, header.d_model
    if header.embedding_format == EMB_FP32:    return decode_embedding_fp32(buf, off, vocab, d_model)
    if header.embedding_format == EMB_INT8:    return decode_embedding_int8(buf, off, vocab, d_model)
    if header.embedding_format == EMB_TERNARY: return decode_embedding_ternary(buf, off, vocab, d_model)
    if header.embedding_format == EMB_INT4:    return decode_embedding_int4(buf, off, vocab, d_model)
    raise ValueError(f"unknown embedding_format: {header.embedding_format}")


def decode_bitlinear(buf: bytes, off: int, in_features: int, out_features: int,
                     has_bias: bool) -> tuple[torch.Tensor, float, torch.Tensor | None, int]:
    """Returns (w_quant_int8_ternary, w_scale_float, bias_or_None, new_offset)."""
    assert in_features % 4 == 0
    bytes_per_row = in_features // 4
    n_w_bytes = out_features * bytes_per_row
    packed = torch.frombuffer(buf, dtype=torch.uint8, count=out_features * bytes_per_row, offset=off).clone().view(out_features, bytes_per_row)
    off += n_w_bytes
    # Extract 4 codes per byte
    code0 = packed       & 0b11
    code1 = (packed >> 2) & 0b11
    code2 = (packed >> 4) & 0b11
    code3 = (packed >> 6) & 0b11
    def codes_to_ternary(c: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(c, dtype=torch.int8)
        out[c == 0b01] =  1
        out[c == 0b10] = -1
        return out
    w_quant = torch.zeros((out_features, in_features), dtype=torch.int8)
    w_quant[:, 0::4] = codes_to_ternary(code0)
    w_quant[:, 1::4] = codes_to_ternary(code1)
    w_quant[:, 2::4] = codes_to_ternary(code2)
    w_quant[:, 3::4] = codes_to_ternary(code3)
    # w_scale
    w_scale, = struct.unpack_from("<f", buf, off)
    off += 4
    # Optional bias
    bias = None
    if has_bias:
        bias = torch.frombuffer(buf, dtype=torch.float32, count=out_features, offset=off).clone()
        off += out_features * 4
    return w_quant, w_scale, bias, off


def decode_layernorm(buf: bytes, off: int, d_model: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    w = torch.frombuffer(buf, dtype=torch.float32, count=d_model, offset=off).clone()
    off += d_model * 4
    b = torch.frombuffer(buf, dtype=torch.float32, count=d_model, offset=off).clone()
    off += d_model * 4
    return w, b, off


def decode_projection(buf: bytes, off: int, in_features: int, out_features: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    n_w = in_features * out_features
    W = torch.frombuffer(buf, dtype=torch.float32, count=n_w, offset=off).clone().view(out_features, in_features)
    off += n_w * 4
    b = torch.frombuffer(buf, dtype=torch.float32, count=out_features, offset=off).clone()
    off += out_features * 4
    return W, b, off


# ═════════════════════════════════════════════════════════════════════════════
# Engine-side forward pass — the reference implementation
# ═════════════════════════════════════════════════════════════════════════════
#
# Replicates `bitlinear==2.4.6`'s BitLinear.forward exactly with:
#   weight_range = 1.58 → (-1, 1)
#   weight_measure = AbsMedian (already used at pack time, scale stored)
#   activation_range = 8 → (-128, 127)
#   activation_measure = AbsMax (per-token, keepdim=True)
#   norm = parameterless torch.layer_norm
#   strategy = round_clamp (lambda=1 → just round + clamp in inference)
#
# Critical: this MUST match the math the .pt's BitLinear forward does.
# The bitlinear-asymmetry postmortem is what happens when it doesn't.

_BITLINEAR_EPS = 1e-5
_ACTIVATION_RANGE_MAX = 128.0   # max(|range|) for activation_range = (-128, 127)


def bitlinear_forward(x: torch.Tensor, w_quant: torch.Tensor, w_scale: float,
                      bias: torch.Tensor | None) -> torch.Tensor:
    """Engine-equivalent BitLinear forward in pure PyTorch.

    Args:
        x:       [..., in_features] fp32 activations
        w_quant: [out_features, in_features] int8 in {-1, 0, +1}
        w_scale: scalar float (from packer)
        bias:    [out_features] fp32 or None

    Returns:
        [..., out_features] fp32
    """
    in_features = x.shape[-1]
    # 1) Parameterless LayerNorm (default eps from torch)
    x_norm = torch.layer_norm(x, [in_features])
    # 2) Per-token activation scale (AbsMax over last dim, keepdim)
    x_max = x_norm.abs().max(dim=-1, keepdim=True).values.clamp(min=_BITLINEAR_EPS)
    x_scale = _ACTIVATION_RANGE_MAX / x_max
    # 3) Quantize activations (round + clamp; STE inactive in inference)
    x_quant = (x_norm * x_scale).round().clamp(-128, 127)
    # 4) Matmul + bias (in F.linear convention: x @ w.T + bias)
    y_quant = F.linear(x_quant, w_quant.float(), bias)
    # 5) Rescale back to fp32 activations
    return y_quant / (w_scale * x_scale)


# ═════════════════════════════════════════════════════════════════════════════
# UnpackedModel — top-level forward
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class _LayerBlocks:
    """All tensors for one TransformerLayer, unpacked."""
    ln1_w: torch.Tensor;       ln1_b: torch.Tensor
    Wq_quant: torch.Tensor;    Wq_scale: float
    Wk_quant: torch.Tensor;    Wk_scale: float
    Wv_quant: torch.Tensor;    Wv_scale: float
    Wout_quant: torch.Tensor;  Wout_scale: float;  Wout_bias: torch.Tensor
    ln2_w: torch.Tensor;       ln2_b: torch.Tensor
    fc1_quant: torch.Tensor;   fc1_scale: float;   fc1_bias: torch.Tensor
    fc2_quant: torch.Tensor;   fc2_scale: float;   fc2_bias: torch.Tensor


class UnpackedModel:
    """In-memory model rebuilt from a `.bin`. Forward pass is the engine reference."""

    def __init__(
        self,
        header: Header,
        embedding: torch.Tensor,          # [vocab, d_model] fp32 (already dequantized)
        layers: list[_LayerBlocks],
        ln_final_w: torch.Tensor, ln_final_b: torch.Tensor,
        proj_W: torch.Tensor, proj_b: torch.Tensor,  # fp32, NOT ternary
    ):
        self.header = header
        self.embedding = embedding
        self.layers = layers
        self.ln_final_w = ln_final_w
        self.ln_final_b = ln_final_b
        self.proj_W = proj_W
        self.proj_b = proj_b

    @property
    def d_model(self) -> int:   return self.header.d_model
    @property
    def n_heads(self) -> int:   return self.header.n_heads
    @property
    def d_head(self) -> int:    return self.d_model // self.n_heads

    # ── Forward pass — mirror of StudentEncoder.forward + BitLinear math ────

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """input_ids: [B, T] long. attention_mask: [B, T] {0,1}. Returns [B, output_dim] L2-normalized."""
        x = self.embedding[input_ids]                       # [B, T, d_model]
        for layer in self.layers:
            x = self._transformer_layer(x, layer, attention_mask)
        # Final LN (parametric)
        x = F.layer_norm(x, [self.d_model], weight=self.ln_final_w, bias=self.ln_final_b)
        # Mean pool (matches StudentEncoder.forward)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)
        # Output projection (fp32, NOT ternary)
        projected = F.linear(pooled, self.proj_W, self.proj_b)
        return F.normalize(projected, dim=-1)

    def _transformer_layer(self, x: torch.Tensor, L: _LayerBlocks,
                           attention_mask: torch.Tensor | None) -> torch.Tensor:
        # Pre-LN attention
        x_norm = F.layer_norm(x, [self.d_model], weight=L.ln1_w, bias=L.ln1_b)
        attn_out = self._attention(x_norm, L, attention_mask)
        x = x + attn_out
        # Pre-LN FFN
        x_norm = F.layer_norm(x, [self.d_model], weight=L.ln2_w, bias=L.ln2_b)
        ff_out = self._feedforward(x_norm, L)
        return x + ff_out

    def _attention(self, x: torch.Tensor, L: _LayerBlocks,
                   attention_mask: torch.Tensor | None) -> torch.Tensor:
        B, T, _ = x.shape
        # Q/K/V have no bias
        Q = bitlinear_forward(x, L.Wq_quant, L.Wq_scale, None).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = bitlinear_forward(x, L.Wk_quant, L.Wk_scale, None).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = bitlinear_forward(x, L.Wv_quant, L.Wv_scale, None).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # Scaled dot-product attention
        scale = self.d_head ** 0.5
        scores = (Q @ K.transpose(-2, -1)) / scale
        if attention_mask is not None:
            pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        # W_out has bias
        return bitlinear_forward(out, L.Wout_quant, L.Wout_scale, L.Wout_bias)

    def _feedforward(self, x: torch.Tensor, L: _LayerBlocks) -> torch.Tensor:
        h = bitlinear_forward(x, L.fc1_quant, L.fc1_scale, L.fc1_bias)
        h = F.gelu(h)
        return bitlinear_forward(h, L.fc2_quant, L.fc2_scale, L.fc2_bias)

    # ── Load from `.bin` ────────────────────────────────────────────────────

    @classmethod
    def from_bin(cls, path: str | Path) -> "UnpackedModel":
        buf = Path(path).read_bytes()
        if len(buf) < HEADER_SIZE + SHA256_SIZE:
            raise ValueError(f"file too small: {len(buf)} bytes")
        # Verify trailing sha256
        body = buf[:-SHA256_SIZE]
        expected_hash = buf[-SHA256_SIZE:]
        actual_hash = hashlib.sha256(body).digest()
        if actual_hash != expected_hash:
            raise ValueError(f"sha256 mismatch: file has been tampered with or truncated")
        # Parse header
        header = Header.unpack(body)
        off = HEADER_SIZE
        # Embedding (already dequantized to fp32 tensor)
        embedding, off = decode_embedding(body, off, header)
        # Per layer
        layers: list[_LayerBlocks] = []
        for _ in range(header.n_layers):
            ln1_w, ln1_b, off = decode_layernorm(body, off, header.d_model)
            Wq_q, Wq_s, _, off       = decode_bitlinear(body, off, header.d_model, header.d_model, has_bias=False)
            Wk_q, Wk_s, _, off       = decode_bitlinear(body, off, header.d_model, header.d_model, has_bias=False)
            Wv_q, Wv_s, _, off       = decode_bitlinear(body, off, header.d_model, header.d_model, has_bias=False)
            Wo_q, Wo_s, Wo_b, off    = decode_bitlinear(body, off, header.d_model, header.d_model, has_bias=True)
            ln2_w, ln2_b, off = decode_layernorm(body, off, header.d_model)
            fc1_q, fc1_s, fc1_b, off = decode_bitlinear(body, off, header.d_model, header.ffn_dim, has_bias=True)
            fc2_q, fc2_s, fc2_b, off = decode_bitlinear(body, off, header.ffn_dim, header.d_model, has_bias=True)
            layers.append(_LayerBlocks(
                ln1_w, ln1_b,
                Wq_q, Wq_s, Wk_q, Wk_s, Wv_q, Wv_s,
                Wo_q, Wo_s, Wo_b,
                ln2_w, ln2_b,
                fc1_q, fc1_s, fc1_b,
                fc2_q, fc2_s, fc2_b,
            ))
        # Final LN
        ln_final_w, ln_final_b, off = decode_layernorm(body, off, header.d_model)
        # Output projection (fp32 — NOT ternary)
        proj_W, proj_b, off = decode_projection(body, off, header.d_model, header.output_dim)
        # Bytes consumed should exactly equal body length
        if off != len(body):
            raise ValueError(f"trailing bytes after parse: consumed {off}, body has {len(body)}")
        return cls(header, embedding, layers, ln_final_w, ln_final_b, proj_W, proj_b)

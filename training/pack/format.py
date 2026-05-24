"""Wire format constants + Header struct for ternlight's `.bin` v1.

Canonical spec lives in [docs/tern-inference-engine.md](../../docs/tern-inference-engine.md).
This file is the single source of truth for byte-level layout. Encoders/unpackers
import constants from here; do NOT redefine magic / version / format tags elsewhere.

Header is 32 bytes, little-endian. Everything after the header is sequential
(no random access). SHA256 of all preceding bytes lives in the trailing 32 bytes.
"""

import struct
from dataclasses import dataclass


# ── Constants ────────────────────────────────────────────────────────────────

MAGIC          = b"TERN"                # 4 bytes
FORMAT_VERSION = 1                      # uint16 — ternlight v1 (tern-core POC versions superseded)

# embedding_format byte values
EMB_FP32       = 0
EMB_INT8       = 1
EMB_TERNARY    = 2
EMB_INT4       = 3
EMB_NAMES      = {EMB_FP32: "fp32", EMB_INT8: "int8", EMB_TERNARY: "ternary", EMB_INT4: "int4"}
EMB_BY_NAME    = {v: k for k, v in EMB_NAMES.items()}

# weights_format byte values
WEIGHTS_TERNARY = 0                     # only option in v1

HEADER_SIZE = 32                        # bytes; fixed across v1
SHA256_SIZE = 32                        # trailing hash

# Header struct: little-endian
#   magic         4s   "TERN"
#   version       H    uint16
#   emb_fmt       B    uint8
#   weights_fmt   B    uint8
#   vocab_size    I    uint32
#   d_model       H    uint16
#   n_layers      B    uint8
#   n_heads       B    uint8
#   ffn_dim       H    uint16
#   output_dim    H    uint16
#   max_seq_len   H    uint16
#   reserved      10s  zero-padding
# Total: 4+2+1+1+4+2+1+1+2+2+2+10 = 32 bytes
_HEADER_STRUCT = struct.Struct("<4sHBBIHBBHHH10s")
assert _HEADER_STRUCT.size == HEADER_SIZE, f"header struct size {_HEADER_STRUCT.size} ≠ {HEADER_SIZE}"


@dataclass
class Header:
    """In-memory representation of the `.bin` header. Encode/decode via pack()/unpack()."""

    embedding_format: int                # one of EMB_*
    weights_format:   int = WEIGHTS_TERNARY
    vocab_size:       int = 0
    d_model:          int = 0
    n_layers:         int = 0
    n_heads:          int = 0
    ffn_dim:          int = 0
    output_dim:       int = 0
    max_seq_len:      int = 128
    format_version:   int = FORMAT_VERSION

    def pack(self) -> bytes:
        if self.embedding_format not in EMB_NAMES:
            raise ValueError(f"unknown embedding_format: {self.embedding_format}")
        if self.weights_format != WEIGHTS_TERNARY:
            raise ValueError(f"unsupported weights_format: {self.weights_format} (v1 only allows ternary)")
        return _HEADER_STRUCT.pack(
            MAGIC,
            self.format_version,
            self.embedding_format,
            self.weights_format,
            self.vocab_size,
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.ffn_dim,
            self.output_dim,
            self.max_seq_len,
            b"\x00" * 10,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "Header":
        if len(data) < HEADER_SIZE:
            raise ValueError(f"need ≥{HEADER_SIZE} bytes, got {len(data)}")
        magic, version, emb_fmt, w_fmt, vocab, d_model, n_layers, n_heads, ffn, out_dim, max_seq, _reserved \
            = _HEADER_STRUCT.unpack(data[:HEADER_SIZE])
        if magic != MAGIC:
            raise ValueError(f"bad magic: {magic!r} (expected {MAGIC!r})")
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported format version {version} (expected {FORMAT_VERSION})")
        if emb_fmt not in EMB_NAMES:
            raise ValueError(f"unknown embedding_format byte: {emb_fmt}")
        if w_fmt != WEIGHTS_TERNARY:
            raise ValueError(f"unsupported weights_format: {w_fmt}")
        return cls(
            embedding_format = emb_fmt,
            weights_format   = w_fmt,
            vocab_size       = vocab,
            d_model          = d_model,
            n_layers         = n_layers,
            n_heads          = n_heads,
            ffn_dim          = ffn,
            output_dim       = out_dim,
            max_seq_len      = max_seq,
            format_version   = version,
        )

    @property
    def embedding_format_name(self) -> str:
        return EMB_NAMES[self.embedding_format]

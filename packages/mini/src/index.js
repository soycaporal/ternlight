// ternlight — public API.
//
// Thin JS wrapper around the WASM engine. The engine does the real work
// (tokenization, BitLinear forward pass, returning a normalized embedding);
// this file adds ergonomic helpers (cosine similarity, top-K search) and
// stable error types.
//
// The compiled engine lives in ../pkg-node/, populated by scripts/build-engine.sh
// from engine/pkg/ (the wasm-pack nodejs-target output). The model weights
// and BERT tokenizer are baked into the .wasm itself via Rust's
// `include_bytes!`, so this package has no runtime asset fetches.

const { embed: _embed, config_summary } = require('../pkg-node/tern_engine.js');

class TernError extends Error {
  constructor(message, code) {
    super(message);
    this.name = 'TernError';
    this.code = code;
  }
}

/**
 * Embed text → 384-dim L2-normalized Float32Array.
 * Pure CPU inference via WASM. Synchronous. No network calls.
 *
 * Input is tokenized via BERT WordPiece and truncated to 128 tokens
 * (~95 English words). Longer text is silently truncated.
 */
function embed(text) {
  if (typeof text !== 'string') {
    throw new TernError(
      'embed(text): text must be a string',
      'INVALID_INPUT',
    );
  }
  return _embed(text);
}

/**
 * Cosine similarity between two embeddings.
 *
 * Since ternlight embeddings are L2-normalized, this reduces to a dot product
 * — no per-call sqrt or division.
 */
function cosineSim(a, b) {
  if (a.length !== b.length) {
    throw new TernError(
      `vector length mismatch: ${a.length} vs ${b.length}`,
      'DIM_MISMATCH',
    );
  }
  let dot = 0;
  const len = a.length;
  for (let i = 0; i < len; i++) dot += a[i] * b[i];
  return dot;
}

/**
 * Convenience: embed query + each corpus item, return top-K matches sorted
 * descending by similarity.
 *
 * For repeated searches over the same corpus, embed it once upfront and call
 * cosineSim() yourself — see the README "Reuse embeddings" pattern.
 */
function similar(query, corpus, opts = {}) {
  const topK = opts.topK ?? 5;
  const q = embed(query);
  return corpus
    .map((text) => ({ text, sim: cosineSim(q, embed(text)) }))
    .sort((a, b) => b.sim - a.sim)
    .slice(0, topK);
}

/**
 * Debug helper: returns a string describing the loaded engine's configuration
 * (format version, embedding format, dimensions, vocab size). Useful for
 * confirming which build of the engine is actually loaded.
 */
function engineInfo() {
  return config_summary();
}

module.exports = { embed, cosineSim, similar, engineInfo, TernError };

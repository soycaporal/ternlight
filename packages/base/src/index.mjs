// ternlight — public ESM entry. Wraps the WASM engine.
//
// The compiled engine lives in ../pkg-node/ as wasm-pack's nodejs-target output
// (CommonJS). We bridge to it via createRequire — this keeps a single WASM
// artifact and exposes both CJS (src/index.js) and ESM (this file) consumers
// from the same package, routed via the "exports" field in package.json.

import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { embed: _embed, config_summary } = require('../pkg-node/tern_engine.js');

export class TernError extends Error {
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
export function embed(text) {
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
export function cosineSim(a, b) {
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
export function similar(query, corpus, opts = {}) {
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
export function engineInfo() {
    return config_summary();
}

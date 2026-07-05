/**
 * A 384-dim L2-normalized text embedding.
 */
export type Embedding = Float32Array;

/**
 * One row from a nearest-neighbor search.
 */
export interface SimilarityResult {
  /** The original input string. */
  text: string;
  /** Cosine similarity to the query, in `[-1, 1]`. */
  sim: number;
}

/**
 * Common error class for ternlight. The `code` property gives a stable
 * identifier for catching specific error categories programmatically:
 *
 *   - `INVALID_INPUT` — argument was not the expected type
 *   - `DIM_MISMATCH`  — vectors of different lengths were compared
 */
export class TernError extends Error {
  code?: string;
  constructor(message: string, code?: string);
}

/**
 * Embed text → 384-dim L2-normalized embedding.
 *
 * Pure on-device CPU inference via WASM. Synchronous in Node. No network call.
 *
 * Input is tokenized via BERT WordPiece (vocab size 30522) and truncated to
 * 128 tokens (~95 English words). Longer text is silently truncated.
 *
 * @param text — Input string.
 * @returns 384-dimensional `Float32Array` on the unit hypersphere.
 *
 * @example
 *   const v = embed("how do I reset my password");
 *   v.length; // 384
 */
export function embed(text: string): Embedding;

/**
 * Cosine similarity between two embeddings.
 *
 * Since ternlight embeddings are L2-normalized, this reduces to a dot product
 * (no per-call sqrt or division).
 *
 * @returns Scalar in `[-1, 1]`. For typical text, output is in `[0, 1]`.
 *
 * @example
 *   const a = embed("forgot password");
 *   const b = embed("password reset");
 *   cosineSim(a, b); // ~0.85
 */
export function cosineSim(a: Embedding, b: Embedding): number;

/**
 * Convenience: embed query + each corpus item, return top-K matches sorted
 * descending by similarity.
 *
 * For repeated searches over the same corpus, embed it once upfront and call
 * `cosineSim()` yourself — see the README for that pattern.
 *
 * @param query - Query string.
 * @param corpus - Array of candidate strings.
 * @param opts.topK - Number of results to return (default 5).
 */
export function similar(
  query: string,
  corpus: string[],
  opts?: { topK?: number },
): SimilarityResult[];

/**
 * Debug helper: returns a string describing the loaded engine's configuration
 * (format version, embedding format, dimensions, vocab size). Useful for
 * confirming which build of the engine is actually loaded.
 *
 * @example
 *   engineInfo();
 *   // → "tern-engine v1 | embedding_format=int4 | vocab=30522 d_model=256 ..."
 */
export function engineInfo(): string;

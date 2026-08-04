/* tslint:disable */
/* eslint-disable */

/**
 * Debug helper: returns a human-readable summary of the loaded model's
 * header (format version, embedding format, dimensions, vocab size).
 */
export function config_summary(): string;

/**
 * Primary entry point: text → 384-dim L2-normalized embedding.
 *
 * Tokenizes via the embedded BERT vocab, runs the full forward pass against
 * the embedded `.bin`, returns the output vector.
 */
export function embed(text: string): Float32Array;

/**
 * Debug helper: returns token IDs the tokenizer produced. Used by per-stage
 * parity tests to confirm tokenization matches the Python `tokenizers` library.
 */
export function tokenize(text: string): Uint32Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly config_summary: () => [number, number];
    readonly embed: (a: number, b: number) => [number, number];
    readonly tokenize: (a: number, b: number) => [number, number];
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;

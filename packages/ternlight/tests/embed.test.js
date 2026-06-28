// Basic smoke test that the package's public API actually works end-to-end.
// Run with: node --test packages/ternlight/tests/
//
// Requires that `pkg/` has been populated by scripts/build-engine.sh.

const test = require('node:test');
const assert = require('node:assert/strict');

const { embed, cosineSim, similar, engineInfo, TernError } = require('../src/index.js');

test('embed returns a 384-dim Float32Array', () => {
    const v = embed('hello world');
    assert.ok(v instanceof Float32Array, 'embed must return a Float32Array');
    assert.equal(v.length, 384, 'embedding dim must be 384');
});

test('embed output is L2-normalized', () => {
    const v = embed('the quick brown fox');
    let norm = 0;
    for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm);
    assert.ok(Math.abs(norm - 1) < 1e-3, `expected ||v|| ≈ 1, got ${norm}`);
});

test('cosineSim of a vector with itself is ~1', () => {
    const v = embed('forgot my password');
    const sim = cosineSim(v, v);
    assert.ok(Math.abs(sim - 1) < 1e-4, `expected self-similarity ≈ 1, got ${sim}`);
});

test('cosineSim ranks semantically similar pairs higher than unrelated', () => {
    const v1 = embed('how do I reset my password');
    const v2 = embed('forgot my password');
    const v3 = embed('chocolate cake recipe');
    const close = cosineSim(v1, v2);
    const far   = cosineSim(v1, v3);
    assert.ok(close > far, `expected ${close} > ${far}`);
});

test('similar returns top-K matches sorted by similarity descending', () => {
    const corpus = [
        'I forgot my password and need to reset it',
        'where is my package shipment tracking',
        'how to cancel a recurring subscription',
    ];
    const matches = similar('forgot password', corpus, { topK: 2 });
    assert.equal(matches.length, 2, 'should return topK results');
    assert.equal(
        matches[0].text,
        'I forgot my password and need to reset it',
        'most similar match should be the password-reset string',
    );
    assert.ok(
        matches[0].sim > matches[1].sim,
        'results should be sorted descending by sim',
    );
});

test('embed throws TernError on non-string input', () => {
    assert.throws(
        () => embed(42),
        (err) => err instanceof TernError && err.code === 'INVALID_INPUT',
    );
});

test('cosineSim throws TernError on dim mismatch', () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([1, 2, 3, 4]);
    assert.throws(
        () => cosineSim(a, b),
        (err) => err instanceof TernError && err.code === 'DIM_MISMATCH',
    );
});

test('engineInfo returns a non-empty config string', () => {
    const info = engineInfo();
    assert.equal(typeof info, 'string');
    assert.ok(info.length > 0);
    assert.ok(info.includes('embedding_format='), 'should mention embedding_format');
});

// ── Edge cases ────────────────────────────────────────────────────────────

test('embed handles empty string without crashing', () => {
    const v = embed('');
    assert.ok(v instanceof Float32Array);
    assert.equal(v.length, 384);
    // Empty input still produces a unit vector (from [CLS]/[SEP] alone).
    let norm = 0;
    for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
    assert.ok(Math.abs(Math.sqrt(norm) - 1) < 1e-3);
});

test('embed silently truncates input longer than 128 tokens', () => {
    // ~500 words, far past the 128-token cap. Should not throw or return
    // garbage — just truncate and produce a valid embedding.
    const longText = ('the quick brown fox jumps over the lazy dog. '.repeat(50)).trim();
    const v = embed(longText);
    assert.ok(v instanceof Float32Array);
    assert.equal(v.length, 384);
    let norm = 0;
    for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
    assert.ok(Math.abs(Math.sqrt(norm) - 1) < 1e-3, 'truncated output must still be unit-norm');
});

test('embed is deterministic — same input produces identical output', () => {
    const text = 'kubernetes pod stuck in CrashLoopBackOff';
    const v1 = embed(text);
    const v2 = embed(text);
    assert.equal(v1.length, v2.length);
    for (let i = 0; i < v1.length; i++) {
        assert.equal(v1[i], v2[i], `embedding diverged at dim ${i}`);
    }
});

test('cosineSim of L2-normalized embeddings stays within [-1, 1]', () => {
    const v1 = embed('hello world');
    const v2 = embed('chocolate cake recipe');
    const sim = cosineSim(v1, v2);
    assert.ok(sim >= -1 - 1e-4 && sim <= 1 + 1e-4, `cosine out of range: ${sim}`);
});

test('similar default topK is 5', () => {
    const corpus = Array.from({ length: 10 }, (_, i) => `corpus item number ${i}`);
    const matches = similar('any query', corpus);
    assert.equal(matches.length, 5, 'default topK should be 5');
});

test('similar with topK > corpus.length returns the full corpus', () => {
    const corpus = ['only one', 'just two', 'finally three'];
    const matches = similar('query', corpus, { topK: 100 });
    assert.equal(matches.length, 3, 'should return all corpus items, not pad or crash');
});

test('similar with empty corpus returns an empty array', () => {
    const matches = similar('any query', [], { topK: 5 });
    assert.deepEqual(matches, []);
});

test('TernError extends Error and exposes a stable code', () => {
    try {
        embed(null);
        assert.fail('should have thrown');
    } catch (err) {
        assert.ok(err instanceof Error, 'TernError extends Error');
        assert.ok(err instanceof TernError, 'instance check');
        assert.equal(err.name, 'TernError');
        assert.equal(err.code, 'INVALID_INPUT');
        assert.ok(typeof err.message === 'string' && err.message.length > 0);
    }
});

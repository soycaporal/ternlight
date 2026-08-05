// index.mjs — embed the entire Odyssey, live, on your CPU.  `node index.mjs`
import { embed } from '@ternlight/mini';
import { readFileSync, writeFileSync } from 'node:fs';

const chunks = JSON.parse(readFileSync('odyssey.json', 'utf8'));
console.log(`\n  Indexing the Odyssey — ${chunks.length.toLocaleString()} passages\n`);

const t0 = performance.now();
const vecs = chunks.map((text, i) => {
  const v = embed(text); // string -> 384-dim vector
  if (i % 100 === 0) bar(i + 1, chunks.length, t0);
  return Array.from(v);
});
bar(chunks.length, chunks.length, t0);

const secs = (performance.now() - t0) / 1000;
writeFileSync('index.json', JSON.stringify({ chunks, vecs }));
console.log(`\n\n  Done — ${chunks.length.toLocaleString()} embeddings in ${secs.toFixed(1)}s  (${Math.round(chunks.length / secs)} emb/sec)`);
console.log(`  No GPU. No API. No network. Just the CPU.\n`);

function bar(done, total, t0) {
  const frac = done / total;
  const fill = '█'.repeat(Math.round(frac * 32)).padEnd(32, '░');
  const eps = Math.round(done / ((performance.now() - t0) / 1000));
  process.stdout.write(`\r  [${fill}] ${String(Math.round(frac * 100)).padStart(3)}%   ${String(eps).padStart(4)} emb/sec`);
}

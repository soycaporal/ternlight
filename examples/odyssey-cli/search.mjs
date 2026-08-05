// search.mjs — semantic search over the indexed Odyssey.  `node search.mjs`
import { embed, cosineSim } from '@ternlight/mini';
import { readFileSync } from 'node:fs';

const { chunks, vecs } = JSON.parse(readFileSync('index.json', 'utf8'));

//  ↓↓↓  Swap which line is uncommented, re-run — instant, no re-indexing  ↓↓↓
// const query = 'weeping on the shore, longing for home';
// const query = 'a storm wrecks the ship at sea';
// const query = 'washed ashore, half-drowned and exhausted';
const query = 'praying to the gods for mercy';

const q = embed(query);
const hits = chunks
  .map((text, i) => ({ text, score: cosineSim(q, vecs[i]) }))   // cosine — no vector DB
  .sort((a, b) => b.score - a.score)
  .slice(0, 3);

console.log(`\n  "${query}"\n`);
for (const { text, score } of hits) console.log(`  ${score.toFixed(3)}  ${text}\n`);

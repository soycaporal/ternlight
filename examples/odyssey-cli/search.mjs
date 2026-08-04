// search.mjs — semantic search over the indexed Odyssey.  `node search.mjs`
import { embed, cosineSim } from '@ternlight/mini';
import { readFileSync } from 'node:fs';

const { chunks, vecs } = JSON.parse(readFileSync('index.json', 'utf8'));

//  ↓↓↓  Swap which line is uncommented, re-run — instant, no re-indexing  ↓↓↓
const query = 'tie me to the mast so I can hear the song and survive';
// const query = 'a witch turns the crew into pigs';                         
// const query = 'stringing the great bow that no other man can bend';       
// const query = 'a monster with one giant eye traps the men in a cave';

const q = embed(query);
const hits = chunks
  .map((text, i) => ({ text, score: cosineSim(q, vecs[i]) }))   // cosine — no vector DB
  .sort((a, b) => b.score - a.score)
  .slice(0, 3);

console.log(`\n  "${query}"\n`);
for (const { text, score } of hits) console.log(`  ${score.toFixed(3)}  ${text}\n`);

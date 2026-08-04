// probe.mjs — fire a battery of candidate demo queries, show top hit for each.
import { embed, cosineSim } from '@ternlight/mini';
import { readFileSync } from 'node:fs';
const { chunks, vecs } = JSON.parse(readFileSync('index.json', 'utf8'));

const QUERIES = [
  'the faithful wife who waits years for her husband to return',
  'a monster with one giant eye traps the men in a cave',
  'sailors lured toward the rocks by an irresistible song',
  'disguised as a beggar, unrecognized in his own home',
  'a witch turns the crew into pigs',
  'stringing the great bow that no other man can bend',
  'tie me to the mast so I can hear the song and survive',
  'the goddess who kept him on her island as her lover',
];

console.log(`\n  corpus: ${chunks.length} chunks\n`);
for (const query of QUERIES) {
  const q = embed(query);
  let best = { score: -1, text: '' };
  for (let i = 0; i < vecs.length; i++) {
    const s = cosineSim(q, vecs[i]);
    if (s > best.score) best = { score: s, text: chunks[i] };
  }
  console.log(`▸ ${query}`);
  console.log(`  ${best.score.toFixed(3)}  ${best.text.slice(0, 150)}${best.text.length > 150 ? '…' : ''}\n`);
}

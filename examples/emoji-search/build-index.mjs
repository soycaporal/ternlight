// build-index.mjs — precompute the emoji index with @ternlight/base.
// Each emoji gets a broad "label, tags, slang" vector; enriched emoji get a
// SECOND vector of the slang phrase alone, max-pooled at query time so idioms
// (GOAT, ship it, red flag) win without diluting the base vector.
import { embed, engineInfo } from '@ternlight/base';
import { writeFileSync } from 'node:fs';
import data from 'emojibase-data/en/data.json' with { type: 'json' };
import { ENRICH, GENZ } from './enrichment.mjs';

const norm = s => s.replace(/️/g, '');                     // ignore variation selectors when matching
const round = v => Array.from(v, x => Math.round(x * 1e5) / 1e5);

// Drop skin-tone components (group 2) and country/subdivision flags ("flag: X").
// Drop skin-tone components, country/subdivision flags, and the 24 clock-face
// dials (🕐–🕧 "one o'clock" / "one-thirty" — pure noise on any "time" query).
const items = data.filter(e =>
  e.group !== undefined && e.group !== 2 && e.label &&
  !/^flag: /.test(e.label) && !/o.clock|-thirty/i.test(e.label));

// Merge base enrichment + Gen Z slang (append where an emoji appears in both).
const merged = {};
for (const [k, v] of Object.entries(ENRICH)) merged[norm(k)] = v;
for (const [k, v] of Object.entries(GENZ)) merged[norm(k)] = merged[norm(k)] ? `${merged[norm(k)]}, ${v}` : v;
const enrichByNorm = new Map(Object.entries(merged));
const misses = [...enrichByNorm.keys()].filter(k => !items.some(e => norm(e.emoji) === k));
console.log(`emoji: ${items.length}  |  enrichment: ${enrichByNorm.size}  |  unmatched: ${misses.length}`);

console.log(`engine: ${engineInfo().split('|')[0].trim()} (base)\n`);
const t0 = performance.now();
const emojis = [], names = [], vecs = [];
let enriched = 0, nVecs = 0;
for (const e of items) {
  const extra = enrichByNorm.get(norm(e.emoji));
  const combined = [e.label, ...(e.tags || []), extra].filter(Boolean).join(', ');
  const list = [round(embed(combined))];                   // broad recall vector
  if (extra) { list.push(round(embed(extra))); enriched++; } // sharp slang vector
  emojis.push(e.emoji); names.push(e.label); vecs.push(list);
  nVecs += list.length;
}
const secs = (performance.now() - t0) / 1000;

const out = { model: 'ternlight-base', dim: vecs[0][0].length, count: emojis.length, emojis, names, vecs };
writeFileSync('emoji-index.json', JSON.stringify(out));
const mb = (Buffer.byteLength(JSON.stringify(out)) / 1024 / 1024).toFixed(1);
console.log(`built ${emojis.length} emoji / ${nVecs} vectors (${enriched} enriched) in ${secs.toFixed(1)}s  ->  emoji-index.json (${mb} MB)`);

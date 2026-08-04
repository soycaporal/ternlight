// prep.mjs — one-time offline: raw Gutenberg Odyssey -> clean ~40-word chunks.
// Run once, commit odyssey.json, never run on stage.
import { readFileSync, writeFileSync } from 'node:fs';

const raw = readFileSync('odyssey_raw.txt', 'utf8').split('\n');

// Keep only the narrative: BOOK I (line ~375) up to the endnotes block ([1] [ ...).
const start = raw.findIndex(l => /^BOOK I\b/.test(l));
const end   = raw.findIndex((l, i) => i > start && /^\[\d+\]\s/.test(l));
let text = raw.slice(start, end).join('\n');

// Scrub editorial cruft: {Greek} tokens, footnote superscripts glued to words
// (Menelaus36 -> Menelaus), bracketed editor insertions, and BOOK headers.
text = text
  .replace(/\{[^}]*\}/g, ' ')            // {Greek}
  .replace(/\d+/g, '')                   // footnote superscripts (prose spells numbers as words)
  .replace(/\[|\]/g, ' ')                // stray editorial brackets
  .replace(/^BOOK [IVXLC]+\.?\s*$/gm, ''); // book headers

const wc = s => s.split(/\s+/).filter(Boolean).length;
const TARGET = Number(process.env.TARGET || 50);

// Paragraphs (blank-line separated) -> pack sentences into ~40-word chunks.
const chunks = [];
for (const para of text.split(/\n\s*\n/)) {
  const p = para.replace(/\s+/g, ' ').trim();
  if (p.length < 40) continue;
  if (wc(p) <= TARGET) { chunks.push(p); continue; }
  let cur = '';
  for (const s of p.match(/[^.!?]+[.!?]+|\S.+$/g) || [p]) {
    if (cur && wc(cur) + wc(s) > TARGET) { chunks.push(cur.trim()); cur = ''; }
    cur += ' ' + s.trim();
  }
  if (cur.trim()) chunks.push(cur.trim());
}

writeFileSync('odyssey.json', JSON.stringify(chunks));
console.log(`Wrote odyssey.json — ${chunks.length} chunks, avg ${(chunks.reduce((a,c)=>a+wc(c),0)/chunks.length).toFixed(1)} words/chunk`);

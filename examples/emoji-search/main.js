// main.js — semantic emoji search, entirely in the browser on @ternlight/base.
//
// The emoji vectors are precomputed offline (build-index.mjs) with the SAME
// base model, so at runtime we only embed the QUERY per keystroke and cosine
// it against ~1,800 vectors. Emoji with a slang vector are max-pooled.

import init, { embed } from './wasm/tern_engine.js';

const $ = (id) => document.getElementById(id);
const grid = $('grid'), input = $('q'), latEl = $('lat'), chipsEl = $('chips');

const TOP_K = 8;
const CHIPS = [
  'so tired, put me to sleep',
  "that's hilarious, I'm dying",
  'the greatest of all time',
  'when your code finally works',
  'sending so much love',
  'I need coffee to survive',
  'my mind is blown',
  'party time, we won',
];

let EMOJIS = [], NAMES = [], VECS = [], DIM = 384;

// ── Boot ────────────────────────────────────────────────────────────────
(async function boot() {
  input.disabled = true;
  input.placeholder = 'loading the model…';

  await init();                                    // compile the base wasm
  const idx = await (await fetch('./emoji-index.json')).json();
  EMOJIS = idx.emojis; NAMES = idx.names; DIM = idx.dim;
  // Each emoji owns 1–2 vectors (broad + slang). Store as Float32Array for speed.
  VECS = idx.vecs.map((list) => list.map((v) => Float32Array.from(v)));

  buildChips();
  input.disabled = false;
  input.placeholder = 'this meeting could’ve been an email…';
  input.focus();
})().catch((err) => {
  console.error(err);
  input.placeholder = 'failed to load — serve over http, not file://';
});

// ── Search ─────────────────────────────────────────────────────────────
let debounce;
input.addEventListener('input', () => {
  clearTimeout(debounce);
  debounce = setTimeout(runSearch, 80);
});

function runSearch() {
  const q = input.value.trim();
  if (!q) { grid.innerHTML = ''; setLatency(null); return; }

  const t0 = performance.now();
  const qv = embed(q);                              // <-- the only per-keystroke embed
  const embedMs = performance.now() - t0;

  // max-pool cosine (vectors are L2-normalized, so dot == cosine)
  const scored = new Array(EMOJIS.length);
  for (let i = 0; i < EMOJIS.length; i++) {
    let best = -2;
    for (const v of VECS[i]) {
      let dot = 0;
      for (let j = 0; j < DIM; j++) dot += qv[j] * v[j];
      if (dot > best) best = dot;
    }
    scored[i] = { i, best };
  }
  scored.sort((a, b) => b.best - a.best);

  setLatency(embedMs);
  render(scored.slice(0, TOP_K));
}

function render(top) {
  grid.innerHTML = top.map(({ i }, rank) => `
    <div class="cell ${rank === 0 ? 'top' : ''}">
      <span class="emoji">${EMOJIS[i]}</span>
      <span class="name">${escapeHtml(NAMES[i])}</span>
    </div>`).join('');
}

function setLatency(ms) {
  const num = ms == null ? '&mdash;' : ms.toFixed(1);
  latEl.innerHTML = `<span class="lat-num">${num}</span><span class="lat-unit">ms</span>`;
  if (ms == null) return;
  latEl.classList.add('flash');
  setTimeout(() => latEl.classList.remove('flash'), 160);
}

// ── Chips ──────────────────────────────────────────────────────────────
function buildChips() {
  chipsEl.innerHTML = CHIPS.map((c) => `<button class="chip">${escapeHtml(c)}</button>`).join('');
  for (const btn of chipsEl.querySelectorAll('.chip')) {
    btn.addEventListener('click', () => {
      input.value = btn.textContent;
      input.focus();
      runSearch();
    });
  }
}

// ── QR overlay: click the QR or press "f" for a full-screen finale ──────
const overlay = $('qrOverlay');
const toggleQR = () => overlay.hidden = !overlay.hidden;
$('qrBtn').addEventListener('click', toggleQR);
overlay.addEventListener('click', toggleQR);
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') overlay.hidden = true;
  else if ((e.key === 'f' || e.key === 'F') && document.activeElement !== input) toggleQR();
});

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

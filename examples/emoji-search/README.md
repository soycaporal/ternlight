# Emoji search — semantic, in the browser

Type a feeling, get the emoji. A single page that embeds your query on your CPU
with [`@ternlight/base`](https://www.npmjs.com/package/@ternlight/base) and cosine-matches
it against ~1,650 emoji — no server, no API, nothing leaves the page.

**1,652 emoji · 1,794 precomputed vectors · ~2 ms per keystroke · fully offline.**

## Run it

WASM can't load from `file://`, so serve over HTTP:

```bash
cd examples/emoji-search
python3 -m http.server 8000      # or: npx serve -p 8000
```

Open <http://localhost:8000>. No `npm install` needed to run — the base wasm
and the emoji index are committed.

- Type, or click a suggested chip. Results appear per keystroke; the latency
  readout shows the query embed time.
- Press **F** (or click the QR) for a full-screen QR to `ternlight.dev`.

## How it works

The emoji vectors are **precomputed offline** with the same base model, so at
runtime the browser only embeds the *query* — one `embed()` per keystroke —
then does a plain cosine loop. Emoji never go through the model; each emoji is a
label on a text vector built from its name + CLDR keywords + a curated slang
phrase (see `enrichment.mjs`). Emoji with a slang phrase carry a second vector
that is max-pooled, so idioms ("the greatest of all time" → 🐐) win without
diluting the base vector.

Node-precomputed and browser-computed vectors are bit-identical (verified,
cosine = 1.000000), which is why precomputing is safe.

## Files

| File | Role |
|---|---|
| `index.html` · `style.css` · `main.js` | the page (no framework, no bundler) |
| `emoji-index.json` | precomputed vectors + emoji + names (shipped) |
| `wasm/` | base web-target build (model + tokenizer + engine) |
| `qr.svg` | offline QR to ternlight.dev |
| `build-index.mjs` · `enrichment.mjs` | offline pipeline to rebuild the index |

## Rebuild the index

Only needed if you change the enrichment or the model:

```bash
npm install                 # pulls @ternlight/base + emojibase-data (dev only)
npm run build:index         # -> emoji-index.json
```

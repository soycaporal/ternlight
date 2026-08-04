# Odyssey CLI — semantic search over Homer, on-device

A terminal demo: embed the **entire Odyssey** (~130k words) and search it by
meaning, running fully on your CPU via [`@ternlight/mini`](https://www.npmjs.com/package/@ternlight/mini).
No GPU, no API, no vector database — just `embed()` and a cosine loop.

**3,058 passages · ~17s one-time index · ~2ms per search · zero network calls.**

## Setup (once, before the demo)

```bash
cd examples/odyssey-cli
npm install          # pulls @ternlight/mini from npm (~5 MB wire)
node index.mjs       # embeds the whole Odyssey -> index.json (~17s)
```

## Files

| File | What it does |
|---|---|
| `index.mjs`  | Embeds every passage with a live progress bar → `index.json`. The speed flex. |
| `search.mjs` | Loads the index, embeds one editable `query`, prints the top 3. Instant. |
| `prep.mjs`   | One-time: raw Gutenberg text → clean `odyssey.json` chunks. Already run. |
| `probe.mjs`  | Fires a battery of candidate queries — use it to rehearse. |
| `odyssey.json` | 3,058 pre-chunked passages (committed). |
| `index.json` | Embeddings cache (git-ignored; created by `index.mjs`). |

## Run the demo

```bash
node index.mjs       # watch it embed the whole Odyssey live
node search.mjs      # semantic search — edit the `query` line and re-run
```

Editing the `query` in `search.mjs` and re-running is **instant** — the index
is already built, so only the query embeds.

## Queries that land (zero keyword overlap with the text)

| Query | Finds |
|---|---|
| `tie me to the mast so I can hear the song and survive` | the Sirens |
| `a witch turns the crew into pigs` | Circe |
| `stringing the great bow that no other man can bend` | the bow trial |
| `a monster with one giant eye traps the men in a cave` | the Cyclops |

None of these share words with the passage they find — that's the semantic win
a keyword search can't do.

## Regenerate the corpus (optional)

`odyssey.json` is committed, so you don't need this. To rebuild from source:

```bash
curl -sL https://www.gutenberg.org/cache/epub/1727/pg1727.txt -o odyssey_raw.txt
node prep.mjs                 # -> odyssey.json (default 50-word chunks)
TARGET=40 node prep.mjs       # smaller chunks: more, tighter, faster per embed
```

Corpus: Samuel Butler's public-domain prose translation
([Project Gutenberg #1727](https://www.gutenberg.org/ebooks/1727)).
`prep.mjs` strips the preface, endnotes, `{Greek}` tokens, and footnote numbers.

#!/usr/bin/env bash
# Pre-publish verification: prove every consumption path of every package
# works FROM THE TARBALL — the exact artifact `npm publish` uploads and the
# exact layout a user's node_modules will contain.
#
# Matrix per package (mini, base):
#   1. CJS require   → "node" + "require" conditions → src/index.js  → pkg-node
#   2. Node ESM      → "node" + "import"  conditions → src/index.mjs → pkg-node
#   3. Bundler/browser → "browser" condition          → src/index.browser.mjs
#      → pkg-bundler, executed via Node's ESM wasm-integration
#      (--experimental-wasm-modules) — the same module-graph mechanism
#      webpack's asyncWebAssembly / Vite's wasm plugin use.
#
# Each smoke asserts: 384-dim Float32Array, unit norm, the EXPECTED MODEL for
# the tier (d_model in engineInfo — catches "right package, wrong .bin"), and
# a semantic sanity pair. Failure exits non-zero; wire this before `npm publish`.
#
# Usage:  bash scripts/verify-release.sh            # both packages
#         PKG=base bash scripts/verify-release.sh   # one package

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d /tmp/ternlight-verify.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

declare -A EXPECT_DMODEL=([mini]=256 [base]=384)
PKGS="${PKG:-mini base}"

write_smokes() {
    local scope_name="$1" dmodel="$2" dir="$3"

    cat > "$dir/smoke.cjs" <<EOF
const { embed, cosineSim, engineInfo } = require('@ternlight/${scope_name}');
const v = embed('hello world');
if (!(v instanceof Float32Array) || v.length !== 384) throw new Error('bad embedding: ' + v.length);
const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
if (Math.abs(norm - 1) > 1e-3) throw new Error('not unit norm: ' + norm);
if (!engineInfo().includes('d_model=${dmodel}')) throw new Error('WRONG MODEL for ${scope_name}: ' + engineInfo());
const sim = cosineSim(embed('reset my password'), embed('I forgot my password'));
const neg = cosineSim(embed('reset my password'), embed('the weather is nice today'));
if (!(sim > 0.5 && neg < 0.3 && sim > neg)) throw new Error('semantic sanity failed: ' + sim + ' vs ' + neg);
console.log('  cjs      ✓  (' + engineInfo().match(/d_model=\\d+/)[0] + ', para=' + sim.toFixed(2) + ')');
EOF

    cat > "$dir/smoke.mjs" <<EOF
import { embed, cosineSim, engineInfo } from '@ternlight/${scope_name}';
const v = embed('hello world');
if (!(v instanceof Float32Array) || v.length !== 384) throw new Error('bad embedding: ' + v.length);
const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
if (Math.abs(norm - 1) > 1e-3) throw new Error('not unit norm: ' + norm);
if (!engineInfo().includes('d_model=${dmodel}')) throw new Error('WRONG MODEL for ${scope_name}: ' + engineInfo());
const sim = cosineSim(embed('reset my password'), embed('I forgot my password'));
if (sim <= 0.5) throw new Error('semantic sanity failed: ' + sim);
console.log('  esm      ✓  (' + engineInfo().match(/d_model=\\d+/)[0] + ')');
EOF
}

for name in $PKGS; do
    pkg_dir="$ROOT/packages/$name"
    dmodel="${EXPECT_DMODEL[$name]}"
    echo ""
    echo "── @ternlight/$name ─────────────────────────────────"

    # 1. Pack the tarball (what npm publish would upload)
    tarball=$(cd "$pkg_dir" && npm pack --pack-destination "$WORK" 2>/dev/null | tail -1)
    echo "  tarball  $tarball ($(du -h "$WORK/$tarball" | cut -f1 | xargs))"

    # 2. Fresh consumer project, install from the tarball
    consumer="$WORK/consume-$name"
    mkdir -p "$consumer"
    (cd "$consumer" && npm init -y >/dev/null 2>&1 && npm install --no-audit --no-fund "$WORK/$tarball" >/dev/null 2>&1)

    # 3. The three consumption smokes
    write_smokes "$name" "$dmodel" "$consumer"
    (cd "$consumer" && node smoke.cjs)
    (cd "$consumer" && node smoke.mjs)
    (cd "$consumer" && node --conditions=browser --experimental-wasm-modules smoke.mjs 2>/dev/null \
        | sed 's/esm      ✓/bundler  ✓/')
done

echo ""
echo "All consumption paths verified from tarballs. Safe to publish."

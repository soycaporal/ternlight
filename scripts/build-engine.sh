#!/usr/bin/env bash
# Build the Wasm engine and populate the @ternlight/* package(s).
#
# Stage 2 build (wasm-pack as the orchestrator). Tiers × targets:
#   tiers:   mini = d256, base = d384
#   targets: nodejs, bundler → published npm artifacts, per-package
#                              (packages/<tier>/pkg-node | pkg-bundler), routed to
#                              consumers via each package.json "exports" map.
#            web              → standalone browser/CDN build (explicit init(), no
#                              bundler). NOT part of the npm packages — landed in its
#                              own dimension-keyed tree: dist/web/d256/, dist/web/d384/.
# The default matrix is the published one (nodejs bundler); pass TARGET=web to build
# the standalone browser artifacts. Each tier's .bin is copied into
# engine/assets/model.bin (include_bytes!) before its builds.
#
# Usage:
#   bash scripts/build-engine.sh                        # published matrix (nodejs+bundler, both tiers)
#   PKG=packages/base bash scripts/build-engine.sh      # one tier, both published targets
#   PKG=packages/base TARGET=bundler bash scripts/build-engine.sh   # one cell
#   TARGET=web bash scripts/build-engine.sh             # web builds → dist/web/{d256,d384}
#   PKG=packages/base TARGET=web bash scripts/build-engine.sh       # base web only → dist/web/d384
#   TARGET="nodejs bundler web" bash scripts/build-engine.sh        # everything
#   WEB_OUT=path TARGET=web bash scripts/build-engine.sh            # override web output root
#   BIN=path/to/model.bin PKG=... bash scripts/build-engine.sh      # custom bin
#   PROFILE=debug ... / FEATURE=emb_int8 ...            # build variants
#
# See docs-local/tern-bundling.md → Stage 3 for the longer-term plan of
# dropping down to wasm-bindgen-cli directly (single shared wasm per package).

set -euo pipefail

PROFILE="${PROFILE:-release}"
FEATURE="${FEATURE:-emb_int4}"   # current ship target for both tiers
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENGINE_DIR="$ROOT/engine"

# Tier definitions: package dir → packed model .bin to embed.
default_bin_for() {
    case "$1" in
        */mini) echo "$ROOT/training/pack/out/model-int4.bin" ;;        # d256
        */base) echo "$ROOT/training/pack/out/d384/model-int4.bin" ;;   # d384
        *)      echo "" ;;
    esac
}

# Model dimension per tier — used to namespace the non-package (web) output tree.
dim_for() {
    case "$1" in
        */mini) echo "d256" ;;
        */base) echo "d384" ;;
        *)      echo "unknown" ;;
    esac
}

PKGS=${PKG:-"packages/mini packages/base"}
TARGETS=${TARGET:-"nodejs bundler"}

build_one() {
    local pkg_dir="$1" target="$2" bin="$3"

    # Output routing per target:
    #   nodejs / bundler → published npm artifacts, per-package (pkg-node / pkg-bundler)
    #   web             → standalone browser/CDN build, kept OUT of the npm packages in
    #                     its own dimension-keyed tree dist/web/<dim>/ (override root
    #                     with WEB_OUT=...).
    local target_dir
    case "$target" in
        nodejs)  target_dir="$ROOT/$pkg_dir/pkg-node" ;;
        bundler) target_dir="$ROOT/$pkg_dir/pkg-bundler" ;;
        web)     target_dir="$ROOT/${WEB_OUT:-dist/web}/$(dim_for "$pkg_dir")" ;;
        *) echo "ERROR: unknown target '$target' (want: nodejs | bundler | web)" >&2; return 1 ;;
    esac

    echo ""
    echo "── Building $pkg_dir ($target, $PROFILE, --features $FEATURE) ──"
    echo "   model: $bin"
    cp "$bin" "$ENGINE_DIR/assets/model.bin"

    cd "$ENGINE_DIR"
    if [[ "$PROFILE" == "release" ]]; then
        wasm-pack build --target "$target" --release --features "$FEATURE"
        if command -v wasm-opt >/dev/null 2>&1; then
            wasm-opt -Oz pkg/tern_engine_bg.wasm -o pkg/tern_engine_bg.wasm
        else
            echo "WARNING: wasm-opt not found — skipping size optimization"
        fi
    else
        wasm-pack build --target "$target" --features "$FEATURE"
    fi

    # Copy the glue + wasm + types — skip wasm-pack's auto package.json,
    # README, and .gitignore (the package ships its own metadata).
    rm -rf "$target_dir"
    mkdir -p "$target_dir"
    find pkg -maxdepth 1 -name 'tern_engine*' -exec cp {} "$target_dir/" \;

    local wasm_bytes wasm_mb
    wasm_bytes=$(wc -c <"$target_dir/tern_engine_bg.wasm")
    wasm_mb=$(awk "BEGIN {printf \"%.2f\", $wasm_bytes / 1024 / 1024}")
    echo "   → ${target_dir#"$ROOT"/}/tern_engine_bg.wasm = ${wasm_mb} MB"
}

for pkg_dir in $PKGS; do
    bin="${BIN:-$(default_bin_for "$pkg_dir")}"
    if [[ -z "$bin" || ! -f "$bin" ]]; then
        echo "ERROR: no model .bin for $pkg_dir (looked for: ${bin:-<none>})" >&2
        echo "       run training/pack/pack.py first, or pass BIN=..." >&2
        exit 1
    fi
    for target in $TARGETS; do
        build_one "$pkg_dir" "$target" "$bin"
    done
done

echo ""
echo "Done."

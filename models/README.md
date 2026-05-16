# models

> Pointers to model release artifacts. Binaries themselves are NOT committed to this repo — they're attached to GitHub Releases and pulled into packages at npm publish time.

## Why models live in releases, not git

- Binaries don't diff well in git
- Repo bloat compounds over time as variants/sizes ship
- npm packages bundle the model at publish time, so end users never see this layer
- GitHub Releases give us versioned URLs and download stats for free

## Where to find current models

Model binaries are attached to GitHub Releases of this repo:

```
https://github.com/soycaporal/ternlight/releases
```

Each release includes:
- `model-micro.bin` — d_model=256, ~3 MB (current default)
- `model-small.bin` — d_model=384, ~5 MB (planned future tier)

## Workflow

A maintainer runs `scripts/release-model.sh <tag> <path-to-bin>` to attach the binary to a release. The npm publish workflow then downloads the asset and bundles it into `packages/semantic/model.bin` before running `npm publish`.

```
training/distill/runs/<run-name>/checkpoint_ep<N>.pt    (NOT in git)
        ↓ training/pack/pack.py
training/pack/out/model.bin                              (NOT in git)
        ↓ scripts/release-model.sh
GitHub Release v0.1.0 (model-micro.bin asset)
        ↓ at npm publish time
packages/semantic/model.bin                              (bundled into the published package)
```

## Future tiers

A `models/registry.json` may live here later, mapping tier name → release asset URL → expected SHA256, so the JS publish step can validate downloads. Not needed at v0.1 (single tier).
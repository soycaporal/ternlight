"""Sidecar JSON manifest for a packed `.bin`.

The `.bin` alone is opaque. The sidecar `model.bin.json` records provenance —
who produced this artifact from what inputs, what its identity is — so a
downloaded `.bin` can be traced back to a training run.

Schema is forward-compatible: parsers tolerate extra keys, missing optional
keys default to None.
"""

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from format import EMB_NAMES, FORMAT_VERSION


def _sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    bin_path:           Path,
    manifest_path:      Path,
    embedding_format:   int,
    training_run_id:    str,
    code_commit:        str,
    source_ckpt_path:   str,
    eval_scorecard_path: str | None = None,
    source_data_manifest: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write the sidecar JSON.

    bin_path's sha256 is computed here and stored alongside — independently
    verifiable from the file contents at any later time.
    """
    bin_path = Path(bin_path)
    manifest_path = Path(manifest_path)
    sha256 = _sha256_hex(bin_path)
    payload = {
        "format_version":       FORMAT_VERSION,
        "embedding_format":     EMB_NAMES[embedding_format],
        "weights_format":       "ternary",
        "bin_path":             bin_path.name,
        "bin_bytes":            bin_path.stat().st_size,
        "bin_sha256":           sha256,
        "packed_at_iso":        datetime.datetime.now(datetime.UTC).isoformat(),
        "training_run_id":      training_run_id,
        "code_commit":          code_commit,
        "source_ckpt_path":     source_ckpt_path,
        "eval_scorecard_path":  eval_scorecard_path,
        "source_data_manifest": source_data_manifest,
    }
    if extra:
        payload["extra"] = extra
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")

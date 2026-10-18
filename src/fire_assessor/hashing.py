"""Streaming SHA-256 helpers (no whole-file reads)."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha256(obj) -> str:
    """Digest of a JSON-able object in canonical form (sorted keys, no whitespace variance)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def source_tree_hash(src_dir) -> str:
    """The run-config `srchash:` convention: sha256 over the bytes of every *.py directly in src_dir, sorted by name, first 16 hex."""
    h = hashlib.sha256()
    for f in sorted(Path(src_dir).glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def write_npz_deterministic(path, arrays: dict) -> None:
    """np.savez-compatible .npz whose bytes depend only on the arrays (np.savez stamps the current time into every zip entry, so two
    builds from identical inputs would differ and the artifact digest would not be reproducible). Entries are sorted, stored (no
    compression) and carry a fixed timestamp and permissions. np.load reads the result like any other .npz."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for k in sorted(arrays):
            zi = zipfile.ZipInfo(f"{k}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type, zi.external_attr = zipfile.ZIP_STORED, 0o600 << 16
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asarray(arrays[k]), allow_pickle=False)  # asarray keeps 0-d arrays 0-d
            z.writestr(zi, buf.getvalue())

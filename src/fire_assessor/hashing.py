"""Streaming SHA-256 helpers (no whole-file reads)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


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

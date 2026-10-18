"""Read-only artifact verification:  python -m fire_assessor.artifact_tools verify DIR [--original-checkpoint PATH]
                                     python -m fire_assessor.artifact_tools digest DIR   (digest an approval record must quote)"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .artifact import ArtifactError, ModelArtifact, binding_digest
from .hashing import sha256_file


def verify_artifact(directory, *, original_checkpoint=None, expected_version=None) -> dict:
    d = Path(directory)
    rep = {"directory": str(d), "integrity_ok": False, "status": None, "blockers": [], "operational_load": None, "checks": {}}
    orig_before = None
    if original_checkpoint is not None:
        st = os.stat(original_checkpoint)
        orig_before = (st.st_size, st.st_mtime_ns, sha256_file(original_checkpoint))
    try:
        art = ModelArtifact.load(d, expected_version, require_operational=False, check_checkpoint=True)
    except ArtifactError as e:
        rep["error"] = {"type": type(e).__name__, "message": str(e)}
        return rep
    rep.update(integrity_ok=True, status=art.status, blockers=art.blockers(), components=art.component_status())
    meta = json.loads((d / "artifact.json").read_text())
    rep["checks"] = {"files_sha256_verified": sorted(meta["files"]), "checkpoint_sha256": art.checkpoint_sha256,
                     "architecture_compatible_with_weights": True, "output_names": art.outputs["names"], "label_mapping": art.outputs["label_mapping"],
                     "binding_digest": binding_digest(meta)}
    try:
        ModelArtifact.load(d, expected_version, require_operational=True)
        rep["operational_load"] = "ALLOWED"
    except ArtifactError as e:
        rep["operational_load"] = f"REFUSED ({type(e).__name__})"
    if original_checkpoint is not None:
        st = os.stat(original_checkpoint)
        after = (st.st_size, st.st_mtime_ns, sha256_file(original_checkpoint))
        rep["checks"]["packaged_weights_equal_original"] = after[2] == art.checkpoint_sha256
        rep["checks"]["original_checkpoint_unchanged_by_verification"] = after == orig_before
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify"); v.add_argument("dir"); v.add_argument("--original-checkpoint"); v.add_argument("--expected-version")
    g = sub.add_parser("digest"); g.add_argument("dir")
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        r = verify_artifact(a.dir, original_checkpoint=a.original_checkpoint, expected_version=a.expected_version)
        print(json.dumps(r, indent=2)); raise SystemExit(0 if r["integrity_ok"] else 1)
    print(binding_digest(json.loads((Path(a.dir) / "artifact.json").read_text())))


if __name__ == "__main__":
    main()

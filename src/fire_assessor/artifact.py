"""Versioned, integrity-checked model artifact: thresholds, temperatures, OOD and authenticity state.

Layout:  <dir>/artifact.json  <dir>/arrays.npz  [<dir>/weights.pt]
artifact.json lists sha256 for every other file; loading fails closed on mismatch.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .authenticity import AuthenticityHead
from .decision import Thresholds
from .ood import KNNOOD, MahalanobisOOD


class ArtifactIntegrityError(Exception):
    pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ModelArtifact:
    model_version: str
    thresholds: Thresholds
    temperatures: dict  # {"flames": T, "smoke": T}
    uncertainty_margin: float = 0.05
    ood: Optional[object] = None
    ood_info: dict = field(default_factory=lambda: {"enabled": False})
    auth_head: Optional[AuthenticityHead] = None
    metadata: dict = field(default_factory=dict)  # data manifest hash, alpha/delta, metrics, git commit
    weights_path: Optional[str] = None

    def save(self, directory, weights_src=None) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        arrays = {}
        if self.ood is not None:
            for k, v in self.ood.state().items():
                arrays[f"ood_{k}"] = np.asarray(v)
        if self.auth_head is not None:
            for k, v in self.auth_head.state().items():
                arrays[f"auth_{k}"] = np.asarray(v)
        np.savez(d / "arrays.npz", **arrays)
        files = {"arrays.npz": sha256_file(d / "arrays.npz")}
        if weights_src is not None:
            (d / "weights.pt").write_bytes(Path(weights_src).read_bytes())
            files["weights.pt"] = sha256_file(d / "weights.pt")
        meta = {
            "model_version": self.model_version,
            "thresholds": {"flames": self.thresholds.flames, "smoke": self.thresholds.smoke},
            "temperatures": self.temperatures,
            "uncertainty_margin": self.uncertainty_margin,
            "ood": {"kind": getattr(self.ood, "kind", None), **self.ood_info},
            "has_auth_head": self.auth_head is not None,
            "metadata": self.metadata,
            "files": files,
        }
        (d / "artifact.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
        return d

    @classmethod
    def load(cls, directory, expected_version: Optional[str] = None) -> "ModelArtifact":
        d = Path(directory)
        meta = json.loads((d / "artifact.json").read_text())
        for name, digest in meta["files"].items():
            p = d / name
            if not p.exists() or sha256_file(p) != digest:
                raise ArtifactIntegrityError(f"checksum mismatch or missing file: {name}")
        if expected_version and meta["model_version"] != expected_version:
            raise ArtifactIntegrityError(
                f"version {meta['model_version']} != requested {expected_version}")
        arr = np.load(d / "arrays.npz", allow_pickle=False)
        sub = lambda pre: {k[len(pre):]: arr[k] for k in arr.files if k.startswith(pre)}
        kind = meta["ood"].get("kind")
        ood = None
        if kind == "knn":
            ood = KNNOOD.from_state(sub("ood_"))
        elif kind == "mahalanobis":
            ood = MahalanobisOOD.from_state(sub("ood_"))
        head = AuthenticityHead.from_state(sub("auth_")) if meta["has_auth_head"] else None
        th = meta["thresholds"]
        ood_info = {k: v for k, v in meta["ood"].items() if k != "kind"}
        return cls(meta["model_version"], Thresholds(th["flames"], th["smoke"]), meta["temperatures"],
                   meta["uncertainty_margin"], ood, ood_info, head, meta["metadata"],
                   str(d / "weights.pt") if "weights.pt" in meta["files"] else None)

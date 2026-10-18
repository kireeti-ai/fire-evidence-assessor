"""Label policy and split rules for the authenticity head (PROPOSED policy; see docs/authenticity/AUTH_HEAD_PLAN.md).

Turns the existing frozen-feature files + manifests into ONE table with a fixed schema so that fitting / threshold selection /
evaluation never depend on which file a row came from:

  block   (n, 768) float32   block-8 pre-norm CLS feature (key `block` of the feature files)
  y       (n, 2)   float64   [synthetic, edited_or_composited]; NaN = label undefined for that output (row is excluded from it)
  split   (n,)     str       train | val | cal | test | heldout_gen | heldout_source | control
  group   (n,)     str       unit of independence. Images that share a source photo / D-Fire near-duplicate cluster share a group
                             (namespace `c<cluster>`), generated images have one group each, control images one group each.
  source  (n,)     str       synth_set | edit_set | dfire | control | table
  tag     (n,)     str       generator family, edit type, benign op or real-source tag (for per-slice reporting)
  paths   (n,)     str

Label policy (PROPOSED, to be confirmed by a human owner):
  synthetic           = 1 for every generated image (any generator); 0 for every real image, benign-edited real image and edited image
  edited_or_composited = 1 for paste_flames / paste_smoke / cover_region edits; 0 for untouched originals and benign edits
                         (crop, brightness, contrast/colour, resize, JPEG); UNDEFINED (NaN) for generated images
  "real" negatives     = D-Fire originals (never labelled by anyone as edited/generated: an ASSUMPTION), edit-set originals and benign
                         edits, the real images of the synthetic set, and the control set (FIgLib + AoF frames).
numpy only; no network; reads the files it is given and nothing else.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .hashing import sha256_file

SPLITS = ("train", "val", "cal", "test", "heldout_gen", "heldout_source", "control")
_DFIRE_SPLIT = {"train": "train", "val": "val", "cal": "cal", "test": "test", "xsrc_aof": "heldout_source"}
OUTPUTS = ("synthetic", "edited_or_composited")
BLOCK_DIM = 768


class AuthDataError(Exception):
    """Inputs missing, malformed, misaligned or leaking; the build must stop (nothing is repaired silently)."""


def _read_csv(path) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _features(path, rows, key="path") -> np.ndarray:
    d = np.load(path, allow_pickle=False)
    if "block" not in d.files or "paths" not in d.files:
        raise AuthDataError(f"{path}: needs arrays `block` and `paths`")
    return d


def _align(d, wanted_paths, what):
    """Row i of the result = the feature row whose path equals wanted_paths[i]. Identity is by PATH, never by order."""
    idx = {str(p): i for i, p in enumerate(d["paths"].tolist())}
    if len(idx) != len(d["paths"]):
        raise AuthDataError(f"{what}: duplicate paths in the feature file")
    miss = [p for p in wanted_paths if p not in idx]
    if miss:
        raise AuthDataError(f"{what}: {len(miss)} manifest paths absent from the feature file (first: {miss[0]})")
    return np.array([idx[p] for p in wanted_paths])


def _src_entry(kind, feat, manifest, n, extra=None):
    e = {"kind": kind, "features": Path(feat).name, "features_sha256": sha256_file(feat), "n_rows": int(n)}
    if manifest is not None:
        e.update({"manifest": Path(manifest).name, "manifest_sha256": sha256_file(manifest)})
    return {**e, **(extra or {})}


def load_source(src: dict) -> tuple[dict, dict]:
    """src = {"kind": synth_set|edit_set|dfire|control|table, "features": path, "manifest": path[, "dfire_manifest"], "variant": str}.
    Returns (columns, provenance-entry)."""
    kind, feat = src["kind"], src["features"]
    if kind == "table":  # generic, already in the schema above (tests, other label sources)
        t = np.load(feat, allow_pickle=False)
        need = {"block", "y", "split", "group"}
        if not need <= set(t.files):
            raise AuthDataError(f"{feat}: table needs {sorted(need)}")
        n = len(t["y"])
        cols = {"block": t["block"], "y": np.asarray(t["y"], float), "split": t["split"].astype(str), "group": t["group"].astype(str),
                "source": np.array(["table"] * n),
                "tag": t["tag"].astype(str) if "tag" in t.files else np.array([""] * n),
                "paths": t["paths"].astype(str) if "paths" in t.files else np.array([f"row{i}" for i in range(n)])}
        return cols, _src_entry(kind, feat, None, n)
    man = src["manifest"]
    rows = _read_csv(man)
    d = _features(feat, rows)
    paths = [r["path"] for r in rows]
    ix = _align(d, paths, f"{kind}:{Path(feat).name}")
    X = np.asarray(d["block"])[ix]
    n = len(rows)
    y = np.full((n, 2), np.nan)
    if kind == "synth_set":
        syn = np.array([int(r["label_synthetic"]) for r in rows])
        y[:, 0] = syn
        y[:, 1] = np.where(syn == 1, np.nan, 0.0)  # a generated image is not "an edit of a photo": undefined, not 0
        split = np.array([r["partition"] for r in rows])
        group = np.array([r["cluster"] if r["label_synthetic"] == "1" else f"c{r['cluster']}" for r in rows])
        tag = np.array([r["family"] for r in rows])
    elif kind == "edit_set":
        y[:, 0] = np.array([int(r["synthetic"]) for r in rows])
        y[:, 1] = np.array([int(r["edited_or_composited"]) for r in rows])
        split = np.array([r["partition"] for r in rows])
        group = np.array([f"c{r['cluster']}" for r in rows])
        tag = np.array([r["edit_type"] if r["edited_or_composited"] == "1" else r["kind"] for r in rows])
    elif kind == "dfire":
        y[:] = 0.0
        bad = [r["partition_xsrc"] for r in rows if r["partition_xsrc"] not in _DFIRE_SPLIT]
        if bad:
            raise AuthDataError(f"unknown D-Fire partition_xsrc values: {sorted(set(bad))}")
        split = np.array([_DFIRE_SPLIT[r["partition_xsrc"]] for r in rows])
        group = np.array([f"c{r['cluster']}" for r in rows])
        tag = np.array(["dfire_raw_" + r["source"] for r in rows])
    elif kind == "control":
        y[:] = 0.0
        split = np.array(["control"] * n)
        group = np.array([f"ctl:{r['path']}" for r in rows])  # no sequence ids exist: each frame is its own group (optimistic CIs)
        tag = np.array(["control_" + r["source"] for r in rows])
    else:
        raise AuthDataError(f"unknown source kind {kind!r}")
    cols = {"block": X, "y": y, "split": split, "group": group, "source": np.array([kind] * n), "tag": tag, "paths": np.array(paths)}
    return cols, _src_entry(kind, feat, man, n)


def build_table(sources: list) -> tuple[dict, list]:
    """Concatenate sources; validate; returns (table, [provenance entries])."""
    parts, prov = [], []
    for s in sources:
        c, p = load_source(s)
        parts.append(c); prov.append(p)
    if not parts:
        raise AuthDataError("no data sources given")
    t = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    validate_table(t)
    return t, prov


def validate_table(t: dict) -> None:
    X, y = np.asarray(t["block"]), np.asarray(t["y"])
    n = len(y)
    if X.ndim != 2 or len(X) != n or y.shape != (n, 2) or any(len(t[k]) != n for k in ("split", "group", "source", "tag", "paths")):
        raise AuthDataError("table arrays have inconsistent shapes")
    if not np.isfinite(X).all():
        raise AuthDataError("block features contain NaN/Inf")
    if not np.isin(y[~np.isnan(y)], (0.0, 1.0)).all():
        raise AuthDataError("labels must be 0/1 (or NaN = undefined)")
    bad = sorted(set(map(str, t["split"])) - set(SPLITS))
    if bad:
        raise AuthDataError(f"unknown split names {bad}")
    if len(set(map(str, t["paths"]))) != n and "table" not in set(t["source"]):
        pass  # the same file name may legitimately occur in two sources (e.g. img/0001.jpg); identity is (source, path)


def group_leaks(t: dict, split_roles: dict) -> dict:
    """{group: [splits]} for every group that appears in more than one of the fit/threshold/selection/evaluation splits.
    Control images and held-out generators have their own groups; this is the leakage check the split rule promises."""
    used = {s for ss in split_roles.values() for s in ss}
    seen: dict = {}
    for g, s in zip(map(str, t["group"]), map(str, t["split"])):
        if s in used:
            seen.setdefault(g, set()).add(s)
    return {g: sorted(s) for g, s in seen.items() if len(s) > 1}

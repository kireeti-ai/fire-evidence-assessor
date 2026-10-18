"""Validation of feature exports used for fitting (train) and calibration (cal).

Principles (design sections 6 and 9):
  * calibration data = the `cal` partition ONLY; validation, test and held-out-source data never tune thresholds;
  * row identity must be established, not assumed: matching row ORDER and counts is supporting evidence, never identity.
    Identity is `PATH_PRESERVED` (export carries paths that match the manifest) or `CONTENT_VERIFIED` (every row's recomputed
    embedding matched its stored row, see scripts/verify_calibration_identity.py). Otherwise it is `NOT_ESTABLISHED`: a blocker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np

SCHEMA = {"logits": 2, "emb": 512, "block": 768, "y": 2}
ROLE_PARTITIONS = {"train": {"train"}, "cal": {"cal"}}
FORBIDDEN_FOR_CALIBRATION = {"val", "test", "xsrc_aof", "heldout_gen", "train"}


@dataclass
class Problem:
    code: str
    message: str
    blocker: bool = True


@dataclass
class ExportReport:
    role: str
    n_rows: int = 0
    identity_status: str = "NOT_ESTABLISHED"
    partition: Optional[str] = None
    problems: list = field(default_factory=list)

    @property
    def blockers(self) -> list:
        return [p for p in self.problems if p.blocker]

    @property
    def ok(self) -> bool:
        return not self.blockers


def _scalar(v):
    a = np.asarray(v)
    return str(a.item()) if a.shape == () else None


def validate_export(data: Mapping, *, role: str, identity_evidence: Optional[dict] = None, manifest_rows: Optional[list] = None) -> ExportReport:
    """role in {"train","cal"}. `identity_evidence` is the summary json written by verify_calibration_identity.py (or None)."""
    if role not in ROLE_PARTITIONS:
        raise ValueError(f"role must be one of {sorted(ROLE_PARTITIONS)}")
    rep = ExportReport(role)
    add = lambda code, msg, blocker=True: rep.problems.append(Problem(code, msg, blocker))
    keys = set(data.keys())
    for k, d in SCHEMA.items():
        if k not in keys:
            add("MISSING_KEY", f"export lacks required array {k!r}")
    if rep.problems:
        return rep
    n = len(data["y"])
    rep.n_rows = n
    for k, d in SCHEMA.items():
        a = np.asarray(data[k])
        if a.ndim != 2 or a.shape != (n, d):
            add("SHAPE", f"{k} has shape {a.shape}, expected ({n}, {d})")
        elif not np.isfinite(a).all():
            add("NONFINITE", f"{k} contains NaN/Inf")
    y = np.asarray(data["y"])
    if y.ndim == 2 and not np.isin(y, (0, 1)).all():
        add("LABELS_NOT_BINARY", "y contains values other than 0/1")
    for k in ("groups", "source", "paths", "partition"):
        if k in keys and len(np.asarray(data[k])) != n:
            add("ROW_COUNT", f"{k} has {len(np.asarray(data[k]))} rows, y has {n}")
    # ---- partition role: a recorded partition must be the role's partition; test/val/held-out data may never calibrate ----
    part = None
    if "partition" in keys:
        vals = {str(v) for v in np.asarray(data["partition"]).tolist()}
        part = vals.pop() if len(vals) == 1 else None
        if part is None:
            add("PARTITION_MIXED", "export mixes several partitions")
    elif identity_evidence and identity_evidence.get("partition"):
        part = identity_evidence["partition"]
    rep.partition = part
    if part is None:
        add("PARTITION_UNKNOWN", "partition of the export is not recorded in the file and not verified by identity evidence")
    elif part not in ROLE_PARTITIONS[role]:
        what = "must not tune thresholds" if role == "cal" else "is not the training partition"
        add("PARTITION_ROLE", f"export partition {part!r} {what} (role {role!r} requires {sorted(ROLE_PARTITIONS[role])})")
    # ---- identity ---------------------------------------------------------------------------------------------------------
    if "paths" in keys:
        paths = [str(p) for p in np.asarray(data["paths"]).tolist()]
        if len(set(paths)) != len(paths):
            add("DUPLICATE_PATHS", "export contains duplicate image paths")
        rep.identity_status = "PATH_PRESERVED"
        if manifest_rows is not None:
            by = {r["path"]: r for r in manifest_rows}
            miss = [p for p in paths if p not in by]
            if miss:
                add("PATH_NOT_IN_MANIFEST", f"{len(miss)} export paths absent from the manifest (first: {miss[0]})")
                rep.identity_status = "NOT_ESTABLISHED"
            else:
                lab = np.array([[int(by[p]["flames"]), int(by[p]["smoke"])] for p in paths], float)
                if not np.array_equal(lab, y):
                    add("LABEL_MISMATCH", "export labels differ from the manifest labels of the same paths")
                    rep.identity_status = "NOT_ESTABLISHED"
    elif (identity_evidence and identity_evidence.get("method") == "content_verification" and identity_evidence.get("rows_checked") == n
          and identity_evidence.get("rows_in_export") == n and identity_evidence.get("mismatch") == 0
          and identity_evidence.get("ambiguous_near_duplicate") == 0 and identity_evidence.get("content_verified") == n):
        rep.identity_status = "CONTENT_VERIFIED"
    else:
        extra = ""
        if identity_evidence and identity_evidence.get("method") == "content_verification":
            extra = (f" Content check: {identity_evidence.get('content_verified')} verified, {identity_evidence.get('ambiguous_near_duplicate')} "
                     f"ambiguous, {identity_evidence.get('mismatch')} mismatched of {identity_evidence.get('rows_checked')} rows.")
        add("IDENTITY_NOT_ESTABLISHED", "export has no per-row paths and no complete content verification; matching row order alone "
            "does not establish which image each row came from." + extra)
    if "paths" not in keys:
        add("NO_PATHS_IN_FILE", "export lacks `paths`; new exports from fire_assessor.train export preserve them", blocker=False)
    return rep


def _content_verified(ev: Optional[dict], partition: str) -> bool:
    return bool(ev and ev.get("method") == "content_verification" and ev.get("partition") == partition and ev.get("mismatch") == 0
                and ev.get("ambiguous_near_duplicate") == 0 and ev.get("content_verified") == ev.get("rows_checked") == ev.get("rows_in_export"))


def check_disjoint(train: Mapping, cal: Mapping, *, cal_identity_evidence: Optional[dict] = None) -> list:
    """Problems if train and calibration exports share samples.

    Checked by paths when both exports carry them; else, when the calibration export is content-verified against the manifest's `cal`
    rows and the train export records partition 'train', they are disjoint by the manifest's construction (clusters never span
    partitions; this property is re-checked by scripts/build_dfire_manifest.py and recorded in the provenance record)."""
    if "paths" in train and "paths" in cal:
        inter = set(map(str, np.asarray(train["paths"]).tolist())) & set(map(str, np.asarray(cal["paths"]).tolist()))
        return [Problem("TRAIN_CAL_OVERLAP", f"{len(inter)} images appear in both train and calibration exports")] if inter else []
    train_part = {str(v) for v in np.asarray(train.get("partition", [])).tolist()}
    if _content_verified(cal_identity_evidence, "cal") and train_part == {"train"}:
        return []
    g1 = {str(v) for v in np.asarray(train.get("groups", [])).tolist() if str(v)}
    g2 = {str(v) for v in np.asarray(cal.get("groups", [])).tolist() if str(v)}
    if g1 & g2:
        return [Problem("TRAIN_CAL_GROUP_OVERLAP", f"{len(g1 & g2)} sequence groups appear in both exports")]
    return [Problem("DISJOINTNESS_UNVERIFIED", "no per-row paths in both exports; train/cal disjointness cannot be verified from the files", blocker=True)]

"""Versioned, integrity-checked, FAIL-CLOSED model artifact.

Layout:  <dir>/artifact.json   <dir>/arrays.npz   <dir>/weights.pt
artifact.json lists a SHA-256 for every other file and describes the model (checkpoint hash, architecture, preprocessing, output
mapping), the calibration / OOD / authenticity components with their status, provenance references, known limitations, and an
approval block.

Status model (derived on load, never trusted from the file):
  INCOMPLETE    a required component or provenance field is missing (calibration, OOD decision, authenticity decision, ...)
  NOT_APPROVED  complete, but no valid human approval record bound to this artifact's content digest
  APPROVED      complete AND a valid approval record is bound to the digest

Nothing in this module can create an APPROVED artifact by itself: an approval record is human-authored evidence. The loader
refuses anything that is not APPROVED unless the caller passes require_operational=False (research / verification only).
"""
from __future__ import annotations

import copy
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .authenticity import AuthenticityHead
from .checkpoint import ArchitectureSpec, CheckpointFormatError, check_compatibility, inspect_checkpoint
from .config import OUTPUT_NAMES
from .decision import Thresholds
from .hashing import canonical_json_sha256, sha256_file, write_npz_deterministic  # sha256_file re-exported for callers
from .ood import KNNOOD, MahalanobisOOD

ARTIFACT_SCHEMA = "fire-assessor-artifact/1"
STATUS_INCOMPLETE, STATUS_NOT_APPROVED, STATUS_APPROVED = "INCOMPLETE", "NOT_APPROVED", "APPROVED"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_COMPONENT_OK = {
    "calibration": {"PROVIDED"},
    "ood": {"PROVIDED", "DISABLED_BY_POLICY"},
    "authenticity": {"PROVIDED", "EXCLUDED_BY_POLICY"},
}
_CAL_IDENTITY_OK = {"PATH_PRESERVED", "CONTENT_VERIFIED"}

DEFAULT_ARCHITECTURE = {
    "backbone": "ViT-B-16-quickgelu", "pretrained_init": "openai", "vision_width": 768, "layers": 12, "patch_size": 16,
    "image_size": 224, "embed_dim": 512,
    "lora": {"rank": 8, "alpha": 16.0, "targets": ["q", "v"], "blocks_zero_indexed": [8, 9, 10, 11]},
    "head": {"inputs": 512, "outputs": 2, "init": "normalised text-prompt embeddings x10, bias 0"},
    "block_feature_tap": {"block_1_indexed": 8, "dim": 768, "note": "pre-norm CLS token; consumed only by an authenticity head"},
}
DEFAULT_PREPROCESSING = {
    "color": "RGB", "resize": 224, "resize_mode": "shortest", "interpolation": "bicubic", "center_crop": 224,
    "mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711],
    "source": "open_clip pretrained cfg for ViT-B-16-quickgelu / openai",
}
DEFAULT_OUTPUTS = {
    "names": list(OUTPUT_NAMES), "label_mapping": {"0": OUTPUT_NAMES[0], "1": OUTPUT_NAMES[1]},
    "semantics": "two independent (multi-label) RAW logits; probability = sigmoid(logit / temperature); NOT calibrated until a calibration is provided",
    "embedding": {"dim": 512, "l2_normalised": True},
}
DEFAULT_LIMITATIONS = [
    "Advisory triage only: does not confirm a wildfire, a location or a time; every report still goes to a human.",
    "Fire/smoke visual-presence classifier only. It is not an authenticity or provenance detector.",
    "Trained on D-Fire (web/public images); no evaluation on citizen phone photos exists.",
    "Smoke generalises poorly to a new source (held-out fixed-camera ROC-AUC about 0.71 for the seed-0 checkpoint).",
    "Any false-negative bound is invalid under source shift (measured FNR 29% flames / 52% smoke on the held-out source).",
]


class ArtifactError(Exception):
    """Base class: every failure to produce a usable artifact."""


class ArtifactFormatError(ArtifactError):
    """artifact.json missing keys, wrong types, bad JSON or unknown schema."""


class ArtifactIntegrityError(ArtifactError):
    """Missing / modified / undeclared files, version or checkpoint-hash mismatch."""


class ArtifactIncompatibleError(ArtifactError):
    """weights.pt does not match the declared architecture."""


class ArtifactIncompleteError(ArtifactError):
    def __init__(self, blockers):
        super().__init__("artifact incomplete: " + "; ".join(blockers))
        self.blockers = list(blockers)


class ArtifactNotApprovedError(ArtifactError):
    pass


def binding_digest(meta: dict) -> str:
    """Digest an approval record must quote: canonical hash of artifact.json without its derived/approval fields."""
    return canonical_json_sha256({k: v for k, v in meta.items() if k not in ("approval", "artifact_status")})


def _approval_problems(record, digest: str) -> list:
    if not isinstance(record, dict):
        return ["no human approval record"]
    p = []
    for k in ("approver", "approved_at", "scope"):
        if not (isinstance(record.get(k), str) and record[k].strip()):
            p.append(f"approval record missing {k}")
    if not (isinstance(record.get("evidence"), list) and record["evidence"]):
        p.append("approval record has no evidence references")
    if record.get("artifact_binding_sha256") != digest:
        p.append("approval record is not bound to this artifact's content digest")
    return p


@dataclass
class ModelArtifact:
    model_version: str
    thresholds: Optional[Thresholds] = None
    temperatures: Optional[dict] = None  # {"flames": T, "smoke": T}
    uncertainty_margin: Optional[float] = None
    ood: Optional[object] = None
    ood_info: dict = field(default_factory=lambda: {"enabled": False})
    auth_head: Optional[AuthenticityHead] = None
    metadata: dict = field(default_factory=dict)
    weights_path: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    architecture: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_ARCHITECTURE))
    preprocessing: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_PREPROCESSING))
    outputs: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_OUTPUTS))
    calibration: dict = field(default_factory=lambda: {"status": "NOT_PROVIDED"})
    authenticity: dict = field(default_factory=lambda: {"status": "NOT_PROVIDED"})
    provenance: dict = field(default_factory=dict)
    limitations: list = field(default_factory=lambda: list(DEFAULT_LIMITATIONS))
    approval_record: Optional[dict] = None
    _binding: Optional[str] = field(default=None, repr=False, compare=False)

    # ---- derived status ------------------------------------------------------------------------------------------------
    def component_status(self) -> dict:
        ood = self.ood_info.get("status") or ("PROVIDED" if self.ood is not None else "NOT_PROVIDED")
        auth = self.authenticity.get("status", "NOT_PROVIDED")
        if self.auth_head is not None and auth == "NOT_PROVIDED":
            auth = "PROVIDED"
        return {"calibration": self.calibration.get("status", "NOT_PROVIDED"), "ood": ood, "authenticity": auth}

    def blockers(self) -> list:
        """Completeness problems (empty = complete). These are facts, not policy: nothing here is filled in automatically."""
        b, st = [], self.component_status()
        if self.checkpoint_sha256 is None:
            b.append("checkpoint SHA-256 not recorded")
        for comp, ok in _COMPONENT_OK.items():
            if st[comp] not in ok:
                b.append(f"{comp} status is {st[comp]} (needs one of {sorted(ok)})")
        cal = self.calibration
        if st["calibration"] == "PROVIDED":
            if self.thresholds is None or self.temperatures is None or self.uncertainty_margin is None:
                b.append("calibration marked PROVIDED but thresholds/temperatures/uncertainty margin are missing")
            pol = cal.get("policy") or {}
            for k in ("alpha", "delta"):
                if not isinstance(pol.get(k), (int, float)):
                    b.append(f"calibration policy.{k} not recorded (target false-negative rate / confidence are stakeholder decisions)")
            src = cal.get("source") or {}
            if src.get("partition") != "cal":
                b.append("calibration source partition is not 'cal' (validation/test/held-out data must not tune thresholds)")
            if src.get("identity_status") not in _CAL_IDENTITY_OK:
                b.append(f"calibration row identity not established (identity_status={src.get('identity_status')!r})")
            if not src.get("export_sha256"):
                b.append("calibration export SHA-256 not recorded")
        if st["ood"] == "DISABLED_BY_POLICY" and not self.ood_info.get("reason"):
            b.append("OOD disabled but no recorded reason")
        if st["ood"] == "PROVIDED" and (self.ood is None or not isinstance(self.ood_info.get("cutoff"), (int, float))):
            b.append("OOD marked PROVIDED but detector state/cutoff missing")
        if st["authenticity"] == "EXCLUDED_BY_POLICY" and not self.authenticity.get("reason"):
            b.append("authenticity excluded but no recorded reason")
        if st["authenticity"] == "EXCLUDED_BY_POLICY" and self.auth_head is not None:
            b.append("authenticity marked EXCLUDED_BY_POLICY but an authenticity head is packaged")
        if self.authenticity.get("status") == "PROVIDED":
            a = self.authenticity
            if self.auth_head is None:
                b.append("authenticity marked PROVIDED but no head state is packaged")
            else:
                try:
                    self.auth_head.validate((self.architecture.get("block_feature_tap") or {}).get("dim"))
                except ValueError as e:
                    b.append(f"authenticity head invalid: {e}")
                if self.auth_head.lo is None:
                    b.append("authenticity head has no recorded INCONCLUSIVE band")
            pol = a.get("policy") or {}
            for k in ("target_fpr", "delta_fpr", "inconclusive_fpr"):
                if not isinstance(pol.get(k), (int, float)):
                    b.append(f"authenticity policy.{k} not recorded (target false-flag rate / confidence / band are owner decisions)")
            for k in ("decided_by", "decided_at", "reference"):
                if not (isinstance(pol.get(k), str) and pol[k].strip()):
                    b.append(f"authenticity policy.{k} not recorded")
            src = (a.get("data") or {}).get("sources")
            if not (isinstance(src, list) and src and all(isinstance(x, dict) and re.match(_HEX64, str(x.get("features_sha256", ""))) for x in src)):
                b.append("authenticity data provenance (feature-file SHA-256s) not recorded")
            if not isinstance(a.get("label_policy"), dict) or not a["label_policy"]:
                b.append("authenticity label policy not recorded")
        for k in ("source", "initialization", "training_manifest"):
            if not (self.provenance or {}).get(k):
                b.append(f"provenance.{k} not recorded")
        return b

    def manifest(self, files: dict) -> dict:
        st = self.component_status()
        meta = {
            "schema_version": ARTIFACT_SCHEMA, "model_version": self.model_version,
            "checkpoint": {"file": "weights.pt", "sha256": self.checkpoint_sha256, "format": "torch zip: {'model': state_dict}"},
            "architecture": self.architecture, "preprocessing": self.preprocessing, "outputs": self.outputs,
            "calibration": self.calibration,
            "thresholds": None if self.thresholds is None else {"flames": self.thresholds.flames, "smoke": self.thresholds.smoke},
            "temperatures": self.temperatures, "uncertainty_margin": self.uncertainty_margin,
            "ood": {"kind": getattr(self.ood, "kind", None), **self.ood_info, "status": st["ood"]},
            "has_auth_head": self.auth_head is not None, "authenticity": {**self.authenticity, "status": st["authenticity"]},
            "provenance": self.provenance, "limitations": self.limitations, "metadata": self.metadata, "files": files,
        }
        digest = binding_digest(meta)
        blockers = self.blockers()
        if blockers:
            status = STATUS_INCOMPLETE
        else:
            status = STATUS_NOT_APPROVED if _approval_problems(self.approval_record, digest) else STATUS_APPROVED
        meta["artifact_status"] = status
        meta["approval"] = {"status": status if status == STATUS_APPROVED else STATUS_NOT_APPROVED, "reasons": blockers + (
            [] if blockers or status == STATUS_APPROVED else _approval_problems(self.approval_record, digest)),
            "record": self.approval_record, "binding_sha256": digest}
        return meta

    @property
    def status(self) -> str:
        """Status of this in-memory artifact (APPROVED only if complete and the record binds to the current content digest)."""
        if self.blockers():
            return STATUS_INCOMPLETE
        return STATUS_APPROVED if self._binding is not None and not _approval_problems(self.approval_record, self._binding) else STATUS_NOT_APPROVED

    # ---- persistence ---------------------------------------------------------------------------------------------------
    def save(self, directory, weights_src=None) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        arrays = {"_schema": np.array(ARTIFACT_SCHEMA)}
        if self.ood is not None:
            arrays.update({f"ood_{k}": np.asarray(v) for k, v in self.ood.state().items()})
        if self.auth_head is not None:
            arrays.update({f"auth_{k}": np.asarray(v) for k, v in self.auth_head.state().items()})
        write_npz_deterministic(d / "arrays.npz", arrays)
        files = {"arrays.npz": sha256_file(d / "arrays.npz")}
        if weights_src is not None:
            src = Path(weights_src)
            before = sha256_file(src)
            if self.checkpoint_sha256 is not None and self.checkpoint_sha256 != before:
                raise ArtifactIntegrityError("checkpoint SHA-256 differs from the declared value; refusing to package")
            self.checkpoint_sha256 = before
            shutil.copyfile(src, d / "weights.pt")  # the source is opened read-only and never modified
            files["weights.pt"] = sha256_file(d / "weights.pt")
            if files["weights.pt"] != before or sha256_file(src) != before:
                raise ArtifactIntegrityError("copy or source changed during packaging")
        meta = self.manifest(files)
        (d / "artifact.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
        return d

    @classmethod
    def load(cls, directory, expected_version: Optional[str] = None, *, require_operational: bool = True,
             check_checkpoint: Optional[bool] = None) -> "ModelArtifact":
        """Verify and load. With require_operational=True (default) anything that is not APPROVED raises; the consumer path must use it."""
        d = Path(directory)
        try:
            meta = json.loads((d / "artifact.json").read_text())
        except FileNotFoundError as e:
            raise ArtifactIntegrityError("artifact.json missing") from e
        except (ValueError, UnicodeDecodeError) as e:
            raise ArtifactFormatError(f"artifact.json is not valid JSON: {e}") from e
        if not isinstance(meta, dict):
            raise ArtifactFormatError("artifact.json is not an object")
        if meta.get("schema_version") != ARTIFACT_SCHEMA:
            raise ArtifactFormatError(f"unsupported schema_version {meta.get('schema_version')!r}")
        need = {"model_version": str, "files": dict, "checkpoint": dict, "architecture": dict, "preprocessing": dict, "outputs": dict,
                "calibration": dict, "ood": dict, "authenticity": dict, "provenance": dict, "limitations": list, "metadata": dict,
                "has_auth_head": bool, "approval": dict}
        for k, t in need.items():
            if not isinstance(meta.get(k), t):
                raise ArtifactFormatError(f"artifact.json field {k!r} missing or not {t.__name__}")
        files = meta["files"]
        for name, digest in files.items():
            if not isinstance(name, str) or "/" in name or "\\" in name or name.startswith(".") or not isinstance(digest, str) or not _HEX64.match(digest):
                raise ArtifactFormatError(f"bad files entry {name!r}")
            p = d / name
            if not p.is_file() or sha256_file(p) != digest:
                raise ArtifactIntegrityError(f"checksum mismatch or missing file: {name}")
        if expected_version and meta["model_version"] != expected_version:
            raise ArtifactIntegrityError(f"version {meta['model_version']} != requested {expected_version}")
        ck = meta["checkpoint"]
        if "weights.pt" in files and ck.get("sha256") != files["weights.pt"]:
            raise ArtifactIntegrityError("checkpoint.sha256 does not match the packaged weights.pt")
        out = meta["outputs"]
        if out.get("names") != list(OUTPUT_NAMES) or out.get("label_mapping") != {"0": OUTPUT_NAMES[0], "1": OUTPUT_NAMES[1]}:
            raise ArtifactIncompatibleError("output names / label mapping differ from this assessor's (flames_visible, smoke_visible)")
        if "arrays.npz" not in files:
            raise ArtifactIntegrityError("arrays.npz not declared")
        arr = np.load(d / "arrays.npz", allow_pickle=False)
        sub = lambda pre: {k[len(pre):]: arr[k] for k in arr.files if k.startswith(pre)}
        kind = meta["ood"].get("kind")
        ood = KNNOOD.from_state(sub("ood_")) if kind == "knn" else MahalanobisOOD.from_state(sub("ood_")) if kind == "mahalanobis" else None
        try:
            head = AuthenticityHead.from_state(sub("auth_")) if meta["has_auth_head"] else None
            if head is not None:
                head.validate()  # shapes / finiteness; the width is checked against the architecture tap by blockers() for PROVIDED heads
        except (KeyError, ValueError, TypeError) as e:
            raise ArtifactFormatError(f"authenticity head state unreadable or malformed: {e}") from e
        if head is None and any(k.startswith("auth_") for k in arr.files):
            raise ArtifactFormatError("arrays.npz holds authenticity state but artifact.json declares has_auth_head=false")
        th = meta.get("thresholds")
        if th is not None and not (isinstance(th, dict) and all(isinstance(th.get(k), (int, float)) for k in ("flames", "smoke"))):
            raise ArtifactFormatError("thresholds malformed")
        temps = meta.get("temperatures")
        if temps is not None and not (isinstance(temps, dict) and all(isinstance(temps.get(k), (int, float)) and temps[k] > 0 for k in ("flames", "smoke"))):
            raise ArtifactFormatError("temperatures malformed")
        art = cls(meta["model_version"], None if th is None else Thresholds(float(th["flames"]), float(th["smoke"])), temps,
                  meta.get("uncertainty_margin"), ood, {k: v for k, v in meta["ood"].items() if k != "kind"}, head, meta["metadata"],
                  str(d / "weights.pt") if "weights.pt" in files else None, ck.get("sha256"), meta["architecture"], meta["preprocessing"],
                  meta["outputs"], meta["calibration"], meta["authenticity"], meta["provenance"], meta["limitations"],
                  (meta["approval"] or {}).get("record"), binding_digest(meta))
        if check_checkpoint is None:
            check_checkpoint = require_operational
        if check_checkpoint:
            if art.weights_path is None:
                raise ArtifactIntegrityError("weights.pt not declared")
            try:
                problems = check_compatibility(inspect_checkpoint(art.weights_path), ArchitectureSpec.from_dict(meta["architecture"]))
            except (CheckpointFormatError, KeyError, TypeError, ValueError) as e:
                raise ArtifactIncompatibleError(f"checkpoint unreadable or architecture block malformed: {e}") from e
            if problems:
                raise ArtifactIncompatibleError(f"{len(problems)} architecture mismatch(es), first: {problems[0]}")
        if require_operational:
            undeclared = sorted(p.name for p in d.iterdir() if p.name not in files and p.name != "artifact.json")
            if undeclared:
                raise ArtifactIntegrityError(f"undeclared files in artifact directory: {undeclared}")
            blockers = art.blockers()
            if blockers:
                raise ArtifactIncompleteError(blockers)
            if art.status != STATUS_APPROVED:
                raise ArtifactNotApprovedError("; ".join(_approval_problems(art.approval_record, art._binding or "")))
        return art

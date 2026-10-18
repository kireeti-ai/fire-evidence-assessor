"""DEVELOPMENT-ONLY inference on a verified EXPERIMENTAL checkpoint.  EXPERIMENTAL / NOT FOR OPERATIONAL USE.

This module exists so the real trained network can be exercised in local development *without* weakening the production contract.
It is deliberately a separate code path from `artifact.ModelArtifact` / `pipeline.Assessor`:

* It never reads, writes, fabricates or relaxes an approval record, calibration, OOD state or artifact status, and it cannot produce
  one. Production loading (`ModelArtifact.load`, `GuardedAssessor`) is unchanged and still refuses an incomplete or unapproved bundle.
* It accepts a checkpoint only if ALL hold: the weights file's SHA-256 equals the hash the *caller* pins (the host application pins a
  specific audited checkpoint in code, never a value read from the file under test); a provenance record exists, names that same hash,
  and carries no approval; and the tensor names/shapes/dtypes match the documented architecture (checked without executing pickle).
* Its output is raw, uncalibrated logits. It has no probability, threshold, label, OOD verdict or authenticity verdict, because none of
  those exist for an unapproved checkpoint. `assessment_status` is always "EXPERIMENTAL" (never "COMPLETED") and `artifact_status` is
  always "EXPERIMENTAL_UNAPPROVED".
* It has no write path to incident status, alerts or assignments; it returns a dict and nothing else.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .checkpoint import ArchitectureSpec, CheckpointFormatError, check_compatibility, inspect_checkpoint
from .config import Limits, OUTPUT_NAMES
from .validation import ValidationError, validate_image

EXPERIMENTAL_NOTICE = "EXPERIMENTAL / NOT FOR OPERATIONAL USE"
RESULT_SCHEMA = "fire-assessor-experimental-result/1"
STATUS = "EXPERIMENTAL"
ARTIFACT_STATUS = "EXPERIMENTAL_UNAPPROVED"
SUPPORTED_PROVENANCE_RECORD_VERSION = 1

NOT_FOUND = "MODEL_EXPERIMENTAL_NOT_FOUND"
PROVENANCE_INVALID = "MODEL_EXPERIMENTAL_PROVENANCE_INVALID"
INTEGRITY_FAILED = "MODEL_EXPERIMENTAL_INTEGRITY_FAILED"
INCOMPATIBLE = "MODEL_EXPERIMENTAL_INCOMPATIBLE"
LOAD_FAILED = "MODEL_EXPERIMENTAL_LOAD_FAILED"
INFERENCE_FAILED = "MODEL_EXPERIMENTAL_INFERENCE_FAILED"
ERROR_CODES = (NOT_FOUND, PROVENANCE_INVALID, INTEGRITY_FAILED, INCOMPATIBLE, LOAD_FAILED, INFERENCE_FAILED)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_APPROVAL_KEYS = ("approval", "approved", "approved_by", "approved_at", "artifact_status", "artifact_binding_sha256")
MAX_ABS_LOGIT = 1.0e3  # a finite-but-absurd logit is a defect, not an output


class ExperimentalError(Exception):
    """Refusal to load or run the experimental path. `.code` is a stable machine-readable reason."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_provenance(provenance_path, expected_sha256: str) -> dict:
    """Read and validate the provenance record. It must document THIS hash and must not claim any approval."""
    if not _HEX64.match(expected_sha256 or ""):
        raise ExperimentalError(PROVENANCE_INVALID, "pinned checkpoint hash is not a SHA-256 hex digest")
    p = Path(provenance_path)
    if not p.is_file():
        raise ExperimentalError(PROVENANCE_INVALID, "provenance record not found")
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ExperimentalError(PROVENANCE_INVALID, "provenance record unreadable") from e
    if not isinstance(rec, dict) or rec.get("record_version") != SUPPORTED_PROVENANCE_RECORD_VERSION:
        raise ExperimentalError(PROVENANCE_INVALID, "unsupported provenance record version")
    ck = rec.get("checkpoint")
    if not isinstance(ck, dict) or ck.get("sha256") != expected_sha256 or not isinstance(ck.get("size_bytes"), int) or ck["size_bytes"] <= 0:
        raise ExperimentalError(PROVENANCE_INVALID, "provenance record does not document the pinned checkpoint")
    if any(k in rec for k in _APPROVAL_KEYS) or any(k in ck for k in _APPROVAL_KEYS):
        raise ExperimentalError(PROVENANCE_INVALID, "an experimental provenance record must not carry approval fields")
    return rec


def verify_experimental_checkpoint(weights_path, provenance_path, expected_sha256: str, spec: Optional[ArchitectureSpec] = None) -> dict:
    """Return the provenance record after verifying provenance, size, SHA-256 and architecture. Raises ExperimentalError."""
    rec = load_provenance(provenance_path, expected_sha256)
    w = Path(weights_path)
    if not w.is_file():
        raise ExperimentalError(NOT_FOUND, "checkpoint file not found")
    if w.stat().st_size != rec["checkpoint"]["size_bytes"]:
        raise ExperimentalError(INTEGRITY_FAILED, "checkpoint size differs from the provenance record")
    if sha256_file(w) != expected_sha256:
        raise ExperimentalError(INTEGRITY_FAILED, "checkpoint SHA-256 differs from the pinned value")
    try:
        problems = check_compatibility(inspect_checkpoint(w), spec or ArchitectureSpec())
    except CheckpointFormatError as e:
        raise ExperimentalError(INCOMPATIBLE, f"checkpoint format: {e}") from e
    if problems:
        raise ExperimentalError(INCOMPATIBLE, f"{len(problems)} architecture mismatches, first: {problems[0]}")
    return rec


class ExperimentalAssessor:
    """Scores one image to raw logits. Holds no approval, calibration or incident state."""

    def __init__(self, scorer: Any, checkpoint_sha256: str, limits: Limits = Limits()):
        self.scorer, self.checkpoint_sha256, self.limits = scorer, checkpoint_sha256, limits
        self.model_version = f"experimental:lora_s0@sha256:{checkpoint_sha256[:12]}"

    def _unavailable(self, code: str) -> dict:
        return {"schema": RESULT_SCHEMA, "assessment_status": "UNAVAILABLE", "experimental": True, "notice": EXPERIMENTAL_NOTICE,
                "model_version": self.model_version, "error_code": code, "artifact_status": None, "raw_logits": None,
                "decision": "REQUIRES_REVIEW"}

    def assess_bytes(self, data: bytes, image_id: str = "") -> dict:
        try:
            v = validate_image(data, self.limits)
        except ValidationError as e:  # bad upload: rejected, no scores
            return self._unavailable(e.code)
        t0 = time.perf_counter()
        try:
            out = self.scorer.score(v.image)
            logits = [float(out.flames_logit), float(out.smoke_logit)]
        except Exception:
            return self._unavailable(INFERENCE_FAILED)
        if not all(math.isfinite(x) and abs(x) <= MAX_ABS_LOGIT for x in logits):
            return self._unavailable(INFERENCE_FAILED)
        return {
            "schema": RESULT_SCHEMA, "assessment_status": STATUS, "experimental": True, "notice": EXPERIMENTAL_NOTICE,
            "model_version": self.model_version, "checkpoint_sha256": self.checkpoint_sha256, "error_code": None,
            "artifact_status": ARTIFACT_STATUS, "raw_logits": dict(zip(OUTPUT_NAMES, logits)), "scores_calibrated": False,
            "derived_label": None, "decision": "REQUIRES_REVIEW", "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "limitations": [
                "EXPERIMENTAL / NOT FOR OPERATIONAL USE: unapproved research checkpoint.",
                "Raw logits are uncalibrated. They are not probabilities and no threshold, OOD check or authenticity check was applied.",
                "The output does not confirm a fire, a location or a time, and must not change incident status or dispatch anyone.",
            ],
        }


def load_experimental(weights_path, provenance_path, *, expected_sha256: str, spec: Optional[ArchitectureSpec] = None,
                      scorer_factory: Optional[Callable[[str], Any]] = None, limits: Limits = Limits()) -> ExperimentalAssessor:
    """Verify the checkpoint, build the scorer and return an assessor. `scorer_factory(weights_path)` defaults to the real TorchScorer."""
    verify_experimental_checkpoint(weights_path, provenance_path, expected_sha256, spec)
    try:
        if scorer_factory is None:
            from .model import TorchScorer  # needs the [model] extra (torch, open_clip)

            scorer_factory = TorchScorer.from_artifact
        scorer = scorer_factory(str(weights_path))
    except ImportError as e:
        raise ExperimentalError(LOAD_FAILED, "model runtime (torch/open_clip) not installed") from e
    except Exception as e:
        raise ExperimentalError(LOAD_FAILED, f"model construction failed: {type(e).__name__}") from e
    return ExperimentalAssessor(scorer, expected_sha256, limits)

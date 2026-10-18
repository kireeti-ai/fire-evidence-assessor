"""Reference FAIL-CLOSED consumer adapter (mirrors what the website backend must do).

* Inference runs only if a complete, APPROVED, integrity-checked, architecture-compatible artifact loads and the model builds.
* Every other case returns an explicit UNAVAILABLE assessment with a specific error code: no scores, no label, no fabricated result.
* The output can only ever say REQUIRES_REVIEW; there is no field for dispatch, rejection or incident status. Humans decide.
"""
from __future__ import annotations

import importlib
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from .artifact import (ArtifactFormatError, ArtifactIncompatibleError, ArtifactIncompleteError, ArtifactIntegrityError, ArtifactNotApprovedError,
                       ModelArtifact)
from .config import LIMITATIONS, MODEL_VERSION
from .schema import Assessment

ERROR_CODES = {
    "not_configured": "MODEL_ARTIFACT_NOT_CONFIGURED", "not_found": "MODEL_ARTIFACT_NOT_FOUND", "integrity": "MODEL_ARTIFACT_INTEGRITY_FAILED",
    "format": "MODEL_ARTIFACT_MALFORMED", "incompatible": "MODEL_ARTIFACT_INCOMPATIBLE", "incomplete": "MODEL_ARTIFACT_INCOMPLETE",
    "not_approved": "MODEL_ARTIFACT_NOT_APPROVED", "load_failed": "MODEL_LOAD_FAILED",
}


def unavailable(code: str, model_version: str = "unconfigured") -> dict:
    return Assessment(assessment_status="UNAVAILABLE", model_version=model_version, error_code=code, decision="REQUIRES_REVIEW", review_priority="STANDARD",
                      limitations=["No approved Fire Evidence Assessor artifact is available in this environment.", *LIMITATIONS]).model_dump()


class GuardedAssessor:
    """Lazy, thread-safe loader. `scorer_factory(weights_path)` defaults to the OpenCLIP TorchScorer (needs the [model] extra)."""

    def __init__(self, artifact_dir: Optional[str], *, expected_version: str = MODEL_VERSION, scorer_factory: Optional[Callable[[str], Any]] = None):
        self.artifact_dir, self.expected_version, self.scorer_factory = (artifact_dir or "").strip(), expected_version, scorer_factory
        self._lock, self._assessor, self._error = threading.Lock(), None, None

    def _load(self):
        if self._assessor is not None or self._error is not None:
            return
        with self._lock:
            if self._assessor is not None or self._error is not None:
                return
            if not self.artifact_dir:
                self._error = ERROR_CODES["not_configured"]; return
            if not Path(self.artifact_dir).is_dir():
                self._error = ERROR_CODES["not_found"]; return
            try:
                art = ModelArtifact.load(self.artifact_dir, self.expected_version)          # operational: APPROVED or raise
                factory = self.scorer_factory or importlib.import_module("fire_assessor.model").TorchScorer.from_artifact
                from .pipeline import Assessor
                self._assessor = Assessor(factory(art.weights_path), art)                   # no allow_unapproved on this path, ever
            except ArtifactNotApprovedError:
                self._error = ERROR_CODES["not_approved"]
            except ArtifactIncompleteError:
                self._error = ERROR_CODES["incomplete"]
            except ArtifactIncompatibleError:
                self._error = ERROR_CODES["incompatible"]
            except ArtifactFormatError:
                self._error = ERROR_CODES["format"]
            except ArtifactIntegrityError:
                self._error = ERROR_CODES["integrity"]
            except Exception:  # missing dependency, model construction failure, ...
                self._error = ERROR_CODES["load_failed"]

    @property
    def error_code(self) -> Optional[str]:
        self._load()
        return self._error

    def assess(self, data: bytes, image_id: str, **kw) -> dict:
        self._load()
        if self._assessor is None:
            return unavailable(self._error or ERROR_CODES["load_failed"])
        return self._assessor.assess_bytes(data, image_id, **kw).model_dump()

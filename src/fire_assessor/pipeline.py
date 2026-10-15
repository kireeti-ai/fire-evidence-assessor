"""Assessment pipeline: validation -> quality -> model -> OOD -> authenticity -> decision policy.

Design properties (section 13):
  * The module has no write path to incident status. It only returns an `Assessment`; the host
    persists it through an `AssessmentSink` that can write the assessment field and nothing else.
  * Any failure (decode, timeout, model error) yields `assessment_status=UNAVAILABLE`; the report
    still goes to a ranger.
  * Logs carry image id, model version, scores, labels, latency, error codes; never image content.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime
from typing import Optional, Protocol

from .artifact import ModelArtifact
from .authenticity import ReuseIndex, assess_authenticity
from .calibration import sigmoid
from .config import LIMITATIONS, AppConfig
from .decision import decide
from .exif import read_exif
from .phash import phash_variants
from .quality import quality_flags
from .schema import Assessment, Authenticity, Scores
from .scorer import Scorer
from .validation import ValidationError, validate_image

log = logging.getLogger("fire_assessor")


class ImageStore(Protocol):
    def get_bytes(self, image_id: str) -> bytes: ...


class AssessmentSink(Protocol):
    """Write access is limited to the advisory assessment field."""

    def write_assessment(self, image_id: str, assessment: dict) -> None: ...


def _unavailable(version: str, code: str) -> Assessment:
    return Assessment(assessment_status="UNAVAILABLE", model_version=version, error_code=code,
                      limitations=list(LIMITATIONS))


class Assessor:
    def __init__(self, scorer: Scorer, artifact: ModelArtifact, config: AppConfig = AppConfig(),
                 reuse_index: Optional[ReuseIndex] = None):
        self.scorer, self.artifact, self.cfg, self.index = scorer, artifact, config, reuse_index

    def assess_bytes(self, data: bytes, image_id: str, *, upload_time: Optional[datetime] = None,
                     reported_location: Optional[tuple] = None) -> Assessment:
        t0, art = time.perf_counter(), self.artifact
        try:
            v = validate_image(data, self.cfg.limits)
            flags = quality_flags(v.image, self.cfg.quality)
            out = self.scorer.score(v.image)
            T = art.temperatures
            pf = float(sigmoid(out.flames_logit / T["flames"]))
            ps = float(sigmoid(out.smoke_logit / T["smoke"]))

            ood_flag = False
            if art.ood is not None and art.ood_info.get("enabled"):
                ood_flag = bool(art.ood.score(out.embedding)[0] > art.ood_info["cutoff"])

            pol_cfg = self.cfg.policy.__class__(
                uncertainty_margin=art.uncertainty_margin,
                smoke_high_priority=self.cfg.policy.smoke_high_priority,
                uncertain_high_priority=self.cfg.policy.uncertain_high_priority)
            pol = decide(pf, ps, art.thresholds, flags, ood_flag, pol_cfg)

            phs = phash_variants(v.image)
            status, signals = assess_authenticity(
                raw=v.raw, exif=read_exif(v.raw), ph=phs, image_id=image_id, embedding=out.embedding,
                block_feat=out.block_feature, head=art.auth_head, index=self.index,
                upload_time=upload_time, reported_location=reported_location, cfg=self.cfg.authenticity)
            if self.index is not None:
                self.index.add(image_id, phs[0], out.embedding)

            res = Assessment(
                assessment_status="COMPLETED", model_version=art.model_version,
                scores=Scores(flames_visible=pf, smoke_visible=ps), derived_label=pol.derived_label,
                review_priority=pol.review_priority, quality_flags=flags, ood_flag=ood_flag,
                authenticity=Authenticity(status=status, signals=signals), limitations=list(LIMITATIONS))
        except ValidationError as e:
            log.warning("assessment unavailable image_id=%s code=%s", image_id, e.code)
            return _unavailable(art.model_version, e.code)
        except Exception as e:  # model-load or inference error: fail soft, never block the report
            log.exception("assessment error image_id=%s", image_id)
            return _unavailable(art.model_version, type(e).__name__)
        log.info("assessed image_id=%s version=%s flames=%.3f smoke=%.3f label=%s priority=%s auth=%s ms=%.0f",
                 image_id, art.model_version, pf, ps, res.derived_label, res.review_priority,
                 status, (time.perf_counter() - t0) * 1000)
        return res


class AssessmentService:
    """Worker-side wrapper: read stored image -> assess with timeout and limited retries -> write field."""

    def __init__(self, assessor: Assessor, store: ImageStore, sink: AssessmentSink,
                 timeout_s: float = 30.0, max_retries: int = 1):
        self.assessor, self.store, self.sink = assessor, store, sink
        self.timeout_s, self.max_retries = timeout_s, max_retries
        self._pool = ThreadPoolExecutor(max_workers=1)

    def run(self, image_id: str, **kw) -> Assessment:
        version = self.assessor.artifact.model_version
        res = _unavailable(version, "NOT_RUN")
        for attempt in range(self.max_retries + 1):
            try:
                data = self.store.get_bytes(image_id)
                fut = self._pool.submit(self.assessor.assess_bytes, data, image_id, **kw)
                res = fut.result(timeout=self.timeout_s)
            except FutureTimeout:
                res = _unavailable(version, "TIMEOUT")
            except Exception as e:
                res = _unavailable(version, type(e).__name__)
            # deterministic validation failures are not retried
            if res.assessment_status == "COMPLETED" or res.error_code in {
                    "EMPTY_FILE", "FILE_TOO_LARGE", "UNSUPPORTED_TYPE", "DECODE_FAILED",
                    "TOO_MANY_PIXELS", "FORMAT_MISMATCH", "IMAGE_TOO_SMALL", "HEIC_UNSUPPORTED"}:
                break
        self.sink.write_assessment(image_id, res.model_dump())  # idempotent upsert keyed by image_id
        return res

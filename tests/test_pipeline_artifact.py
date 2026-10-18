import dataclasses
import inspect
import time

import numpy as np
import pytest

from fire_assessor import pipeline
from fire_assessor.artifact import ArtifactIntegrityError, ModelArtifact
from fire_assessor.authenticity import AuthenticityHead, ReuseIndex
from fire_assessor.decision import Thresholds
from fire_assessor.ood import KNNOOD, fit_ood
from fire_assessor.pipeline import Assessor, AssessmentService
from fire_assessor.schema import Assessment
from conftest import FakeScorer, make_assessor, textured, to_bytes


def test_completed_schema_and_advisory(artifact):
    res = make_assessor(FakeScorer(3, -3), artifact).assess_bytes(to_bytes(textured()), "img1")
    assert res.assessment_status == "COMPLETED" and res.decision == "REQUIRES_REVIEW"
    assert res.derived_label == "ACTIVE_FIRE" and res.review_priority == "HIGH"
    assert res.authenticity.status == "INCONCLUSIVE"  # no head, nothing found
    assert res.limitations[:2] == ["The image alone does not establish location or time.", "The assessment does not confirm a wildfire."]
    assert any(l.startswith("RESEARCH USE ONLY") for l in res.limitations) and res.artifact_status == "INCOMPLETE"  # unapproved artifact is labelled
    Assessment.model_validate(res.model_dump())


def test_temperature_applied(artifact):
    artifact.temperatures = {"flames": 100.0, "smoke": 1.0}  # flatten flames logit toward 0.5
    res = make_assessor(FakeScorer(3, -3), artifact).assess_bytes(to_bytes(textured()), "i")
    assert abs(res.scores.flames_visible - 0.5) < 0.02


@pytest.mark.parametrize("data", [b"", b"junk", b"\xff\xd8\xff" + b"0" * 50])
def test_bad_uploads_become_unavailable_not_exceptions(artifact, data):
    res = make_assessor(FakeScorer(), artifact).assess_bytes(data, "bad")
    assert res.assessment_status == "UNAVAILABLE" and res.error_code and res.scores is None


def test_scorer_failure_is_unavailable(artifact):
    class Boom:
        def score(self, im): raise RuntimeError("model load error")
    res = make_assessor(Boom(), artifact).assess_bytes(to_bytes(textured()), "x")
    assert res.assessment_status == "UNAVAILABLE" and res.error_code == "RuntimeError"


def test_ood_flag_routes_to_uncertain(artifact):
    rng = np.random.default_rng(0)
    e0, e1 = np.eye(8)[0] * 5, np.eye(8)[1] * 5
    ood = KNNOOD(); info = fit_ood(ood, rng.normal(0, 1, (200, 8)) + e0, rng.normal(0, 1, (100, 8)) + e0,
                                   rng.normal(0, 1, (100, 8)) + e1)
    artifact.ood, artifact.ood_info = ood, info
    class Far(FakeScorer):
        def score(self, im):
            o = super().score(im); o.embedding = np.eye(8)[1] * 5 + o.embedding * 0.1; return o
    res = make_assessor(Far(-3, -3), artifact).assess_bytes(to_bytes(textured()), "x")
    assert res.ood_flag and res.derived_label == "UNCERTAIN"


def test_quality_gate_in_pipeline(artifact):
    from PIL import Image
    res = make_assessor(FakeScorer(), artifact).assess_bytes(to_bytes(Image.new("RGB", (300, 300), (3, 3, 3))), "d")
    assert "TOO_DARK" in res.quality_flags and res.derived_label == "UNCERTAIN"


def test_reuse_detected_across_uploads(artifact):
    a = make_assessor(FakeScorer(), artifact, reuse_index=ReuseIndex())
    data = to_bytes(textured(7))
    assert a.assess_bytes(data, "first").authenticity.status == "INCONCLUSIVE"
    assert a.assess_bytes(data, "second").authenticity.status == "REUSED"


def test_artifact_roundtrip_and_integrity(artifact, tmp_path):
    rng = np.random.default_rng(0)
    artifact.ood = KNNOOD().fit(rng.normal(size=(50, 8)))
    artifact.ood_info = {"enabled": True, "cutoff": 0.9}
    artifact.auth_head = AuthenticityHead.fit(rng.normal(size=(40, 8)), rng.integers(0, 2, (40, 2)))
    artifact.metadata = {"manifest_hash": "abc"}
    artifact.save(tmp_path)
    loaded = ModelArtifact.load(tmp_path, "fire-assessor-v1.0.0", require_operational=False, check_checkpoint=False)
    assert loaded.thresholds == artifact.thresholds and loaded.ood_info["cutoff"] == 0.9
    assert loaded.auth_head is not None and loaded.ood.kind == "knn"
    with pytest.raises(ArtifactIntegrityError):
        ModelArtifact.load(tmp_path, "fire-assessor-v9")
    (tmp_path / "arrays.npz").write_bytes(b"tampered")
    with pytest.raises(ArtifactIntegrityError):
        ModelArtifact.load(tmp_path)


class Store:
    def __init__(self, data): self.data, self.n = data, 0
    def get_bytes(self, i):
        self.n += 1
        if self.n == 1 and self.data == "flaky": raise IOError("blip")
        return to_bytes(textured()) if self.data in ("ok", "flaky") else self.data


class Sink:
    def __init__(self): self.rows = {}
    def write_assessment(self, i, a): self.rows[i] = a


def test_service_retry_timeout_and_sink(artifact):
    sink = Sink()
    svc = AssessmentService(make_assessor(FakeScorer(), artifact), Store("flaky"), sink, max_retries=1)
    assert svc.run("a").assessment_status == "COMPLETED" and "a" in sink.rows
    svc = AssessmentService(make_assessor(FakeScorer(), artifact), Store(b""), sink, max_retries=3)
    st = svc.run("b"); assert st.error_code == "EMPTY_FILE" and svc.store.n == 1  # no retry on bad input
    class Slow(FakeScorer):
        def score(self, im): time.sleep(1); return super().score(im)
    svc = AssessmentService(make_assessor(Slow(), artifact), Store("ok"), sink, timeout_s=0.1, max_retries=0)
    assert svc.run("c").error_code == "TIMEOUT"
    assert sink.rows["c"]["assessment_status"] == "UNAVAILABLE"  # report still goes to a ranger


def test_no_incident_write_path():
    """The model path exposes only an assessment-field sink; nothing can set incident status."""
    methods = {n for n, _ in inspect.getmembers(pipeline.AssessmentSink, inspect.isfunction) if not n.startswith('__')}
    assert methods == {"write_assessment"}
    src = inspect.getsource(pipeline).lower()
    assert "incident" not in src.replace("incident status", "").replace("no write path to incident", "")
    assert set(Assessment.model_fields).isdisjoint({"incident_status", "status"})


def test_mirrored_repost_detected(artifact):
    from PIL import Image
    a = make_assessor(FakeScorer(), artifact, reuse_index=ReuseIndex())
    im = textured(11)
    a.assess_bytes(to_bytes(im), "orig")
    mirrored = to_bytes(im.transpose(Image.FLIP_LEFT_RIGHT))
    assert a.assess_bytes(mirrored, "mirror").authenticity.status == "REUSED"

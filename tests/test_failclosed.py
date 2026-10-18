import ast
import json
import pathlib

import pytest
from pydantic import ValidationError

import fire_assessor
from fire_assessor.adapter import ERROR_CODES, GuardedAssessor, unavailable
from fire_assessor.artifact import ArtifactIncompleteError, ArtifactNotApprovedError, ModelArtifact
from fire_assessor.build_artifact import calibrate_artifact
from fire_assessor.decision import Thresholds
from fire_assessor.pipeline import Assessor
from fire_assessor.schema import Assessment
from conftest import FakeScorer, approve_in_place, make_assessor, textured, to_bytes
from test_artifact_contract import POLICY, base, ckpt, exports, make_export, prov, save  # noqa: F401  (fixtures)

pytest.importorskip("torch")
IMG = to_bytes(textured())
stub = lambda logit_f=3.0, logit_s=-3.0: (lambda weights_path: FakeScorer(logit_f, logit_s))


@pytest.fixture
def approved(tmp_path, base, exports):
    d = tmp_path / "approved"
    calibrate_artifact(base, *exports, POLICY, d)
    approve_in_place(d)   # test stand-in for a human approval record
    return d


@pytest.fixture
def unapproved(tmp_path, base, exports):
    d = tmp_path / "unapproved"
    calibrate_artifact(base, *exports, POLICY, d)
    return d


def _edit(d, fn):
    f = d / "artifact.json"; m = json.loads(f.read_text()); fn(m); f.write_text(json.dumps(m))


def assert_explicit_unavailable(res, code):
    """Every refusal is an explicit state: no scores, no label, no derived priority beyond STANDARD, always human review."""
    assert res["assessment_status"] == "UNAVAILABLE" and res["error_code"] == code
    assert res["scores"] is None and res["derived_label"] is None and res["decision"] == "REQUIRES_REVIEW" and res["review_priority"] == "STANDARD"
    assert res["ood_flag"] is False and res["quality_flags"] == [] and res["authenticity"]["status"] == "INCONCLUSIVE"
    assert not any(s in json.dumps(res) for s in ("ACTIVE_FIRE", "SMOKE_ONLY", "NO_VISIBLE_FIRE"))   # nothing fabricated


# ------------------------------------------------------------------------ every refusal reason is explicit
@pytest.mark.parametrize("dir_arg,code", [(None, "not_configured"), ("", "not_configured"), ("/nonexistent/path", "not_found")])
def test_unconfigured_or_missing_directory(dir_arg, code):
    assert_explicit_unavailable(GuardedAssessor(dir_arg, scorer_factory=stub()).assess(IMG, "i"), ERROR_CODES[code])


def test_empty_directory_malformed_json_and_tampered_weights(tmp_path, approved):
    (tmp_path / "empty").mkdir()
    assert_explicit_unavailable(GuardedAssessor(str(tmp_path / "empty"), scorer_factory=stub()).assess(IMG, "i"), ERROR_CODES["integrity"])
    (approved / "artifact.json").write_text("{broken")
    assert_explicit_unavailable(GuardedAssessor(str(approved), scorer_factory=stub()).assess(IMG, "i"), ERROR_CODES["format"])


def test_tampered_weights(approved):
    with open(approved / "weights.pt", "ab") as f:
        f.write(b"0")
    assert_explicit_unavailable(GuardedAssessor(str(approved), scorer_factory=stub()).assess(IMG, "i"), ERROR_CODES["integrity"])


def test_incomplete_artifact(base):
    assert_explicit_unavailable(GuardedAssessor(str(base), scorer_factory=stub()).assess(IMG, "i"), ERROR_CODES["incomplete"])


def test_complete_but_unapproved_artifact(unapproved):
    assert_explicit_unavailable(GuardedAssessor(str(unapproved), scorer_factory=stub()).assess(IMG, "i"), ERROR_CODES["not_approved"])


def test_architecture_mismatch(approved):
    _edit(approved, lambda m: m["architecture"]["lora"].__setitem__("rank", 4))
    assert_explicit_unavailable(GuardedAssessor(str(approved), scorer_factory=stub()).assess(IMG, "i"), ERROR_CODES["incompatible"])


def test_model_construction_failure_is_unavailable_not_a_result(approved):
    def boom(path): raise RuntimeError("no open_clip / no cached base weights")
    assert_explicit_unavailable(GuardedAssessor(str(approved), scorer_factory=boom).assess(IMG, "i"), ERROR_CODES["load_failed"])


def test_failure_state_is_latched_and_calls_do_not_retry_loading(approved):
    calls = []
    def boom(path): calls.append(1); raise RuntimeError("x")
    g = GuardedAssessor(str(approved), scorer_factory=boom)
    g.assess(IMG, "a"); g.assess(IMG, "b")
    assert len(calls) == 1 and g.error_code == ERROR_CODES["load_failed"]


def test_all_error_codes_are_distinct_and_namespaced():
    assert len(set(ERROR_CODES.values())) == len(ERROR_CODES) and all(c.startswith("MODEL_") for c in ERROR_CODES.values())


def test_unavailable_dict_is_a_superset_of_the_existing_consumer_contract():
    r = unavailable("MODEL_ARTIFACT_NOT_CONFIGURED")
    assert {"assessment_status", "model_version", "error_code", "limitations"} <= set(r) and r["model_version"] == "unconfigured"


# ------------------------------------------------------------------------ the approved path (stub scorer; wiring only)
def test_only_an_approved_verified_artifact_runs_inference(approved):
    res = GuardedAssessor(str(approved), scorer_factory=stub(3.0, -3.0)).assess(IMG, "ok")
    assert res["assessment_status"] == "COMPLETED" and res["artifact_status"] == "APPROVED" and res["decision"] == "REQUIRES_REVIEW"
    assert not any(l.startswith("RESEARCH USE ONLY") for l in res["limitations"])


# ------------------------------------------------------------------------ assessor-level gate
def test_assessor_refuses_missing_calibration_even_in_research_mode():
    art = ModelArtifact("fire-assessor-v1.0.0")           # nothing calibrated
    for allow in (False, True):
        with pytest.raises(ArtifactIncompleteError, match="calibration value missing"):
            Assessor(FakeScorer(), art, allow_unapproved=allow)


def test_assessor_refuses_unapproved_by_default_and_labels_research_use(artifact):
    with pytest.raises(ArtifactNotApprovedError):
        Assessor(FakeScorer(), artifact)
    res = make_assessor(FakeScorer(), artifact).assess_bytes(IMG, "r")
    assert res.artifact_status in ("INCOMPLETE", "NOT_APPROVED") and any(l.startswith("RESEARCH USE ONLY") for l in res.limitations)


# ------------------------------------------------------------------------ no automatic operational decisions
@pytest.mark.parametrize("fl,sm", [(30, 30), (-30, -30), (30, -30), (0.0, 0.0), (0.4, 0.6)])
def test_every_outcome_requires_human_review(artifact, fl, sm):
    res = make_assessor(FakeScorer(fl, sm), artifact).assess_bytes(IMG, "x")
    assert res.decision == "REQUIRES_REVIEW" and res.review_priority in ("HIGH", "STANDARD") and res.derived_label in ("ACTIVE_FIRE", "SMOKE_ONLY", "NO_VISIBLE_FIRE", "UNCERTAIN")


@pytest.mark.parametrize("field,value", [("decision", "DISPATCH"), ("decision", "AUTO_REJECT"), ("decision", "VERIFIED"), ("review_priority", "CRITICAL"), ("assessment_status", "APPROVED")])
def test_schema_cannot_express_an_operational_action(field, value):
    base_kw = dict(assessment_status="COMPLETED", model_version="v")
    with pytest.raises(ValidationError):
        Assessment(**{**base_kw, field: value})


def test_assessment_has_no_incident_or_dispatch_fields():
    forbidden = {"incident_status", "dispatch", "dispatched", "verified", "rejected", "action", "incident_id", "status", "escalate"}
    assert forbidden.isdisjoint(Assessment.model_fields)


def test_decision_path_modules_import_nothing_that_can_act():
    src = pathlib.Path(fire_assessor.__file__).parent
    banned = ("incident", "dispatch", "requests", "httpx", "urllib", "sqlalchemy", "smtplib", "subprocess", "socket", "aiohttp")
    for name in ("pipeline.py", "adapter.py", "decision.py", "artifact.py", "schema.py"):
        for node in ast.walk(ast.parse((src / name).read_text())):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not [m for m in mods if any(b in m.lower() for b in banned)], (name, mods)


def test_service_wrapper_can_only_write_the_assessment_field():
    from fire_assessor.pipeline import AssessmentSink
    import inspect
    assert [n for n, _ in inspect.getmembers(AssessmentSink, inspect.isfunction) if not n.startswith("__")] == ["write_assessment"]

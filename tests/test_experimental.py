"""Development-only experimental path (src/fire_assessor/experimental.py). EXPERIMENTAL / NOT FOR OPERATIONAL USE.

The checkpoint here is a tiny TOY file (zeros; torch needed only to write it) with a matching toy ArchitectureSpec and a stub scorer.
`test_real_*` run the audited 345 MB checkpoint and skip unless FIRE_EVIDENCE_REAL_CHECKPOINT points at it.
"""
import hashlib
import json
import math
import os

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from fire_assessor import experimental as ex  # noqa: E402
from fire_assessor.checkpoint import ArchitectureSpec, expected_tensors  # noqa: E402
from fire_assessor.artifact import ModelArtifact  # noqa: E402
from conftest import FakeScorer, textured, to_bytes  # noqa: E402

TOY = ArchitectureSpec(width=8, layers=2, patch=4, image_size=8, embed_dim=6, lora_blocks=(1,), lora_rank=2, n_outputs=2)
IMG = to_bytes(textured())
REAL = os.getenv("FIRE_EVIDENCE_REAL_CHECKPOINT", "")
REAL_PROV = os.path.join(os.path.dirname(__file__), "..", "docs", "provenance", "lora_s0_provenance.json")
PINNED = "7bd5498f04838fe19240da2b7339d794cc5fc3948074a0d826c33303d5c42dfa"


@pytest.fixture
def toy(tmp_path):
    w, p = tmp_path / "weights.pt", tmp_path / "prov.json"
    torch.save({"model": {n: torch.zeros(s) for n, s in expected_tensors(TOY).items()}}, w)
    sha = hashlib.sha256(w.read_bytes()).hexdigest()
    p.write_text(json.dumps({"record_version": 1, "checkpoint": {"sha256": sha, "size_bytes": w.stat().st_size}}))
    return w, p, sha


def load(toy, **kw):
    w, p, sha = toy
    return ex.load_experimental(w, p, expected_sha256=kw.pop("sha", sha), spec=kw.pop("spec", TOY), scorer_factory=kw.pop("factory", lambda path: FakeScorer(2.0, -1.0)), **kw)


def test_valid_image_returns_labelled_finite_raw_logits_and_nothing_operational(toy):
    r = load(toy).assess_bytes(IMG, "i")
    assert r["assessment_status"] == "EXPERIMENTAL" and r["experimental"] is True and r["notice"] == "EXPERIMENTAL / NOT FOR OPERATIONAL USE"
    assert r["raw_logits"] == {"flames_visible": 2.0, "smoke_visible": -1.0} and r["scores_calibrated"] is False
    assert r["artifact_status"] == "EXPERIMENTAL_UNAPPROVED" and r["derived_label"] is None and r["decision"] == "REQUIRES_REVIEW"
    assert not ({"scores", "thresholds", "ood_flag", "authenticity", "review_priority", "incident_status", "dispatch"} & set(r))
    dumped = json.dumps(r)
    assert "COMPLETED" not in dumped and '"APPROVED"' not in dumped
    assert r["model_version"].startswith("experimental:") and r["checkpoint_sha256"] == toy[2]


@pytest.mark.parametrize("data,code", [(b"", "EMPTY_FILE"), (b"plain text", "UNSUPPORTED_TYPE"), (IMG[:150], "DECODE_FAILED")])
def test_invalid_images_are_rejected_without_calling_the_model(toy, data, code):
    scorer = FakeScorer()
    r = load(toy, factory=lambda p: scorer).assess_bytes(data, "i")
    assert r["assessment_status"] == "UNAVAILABLE" and r["error_code"] == code and r["raw_logits"] is None and scorer.calls == 0


@pytest.mark.parametrize("fl,sm", [(float("nan"), 0.0), (0.0, float("inf")), (5000.0, 0.0)])
def test_non_finite_or_absurd_logits_are_never_returned(toy, fl, sm):
    r = load(toy, factory=lambda p: FakeScorer(fl, sm)).assess_bytes(IMG, "i")
    assert r["assessment_status"] == "UNAVAILABLE" and r["error_code"] == ex.INFERENCE_FAILED and r["raw_logits"] is None


def test_scorer_exceptions_are_contained(toy):
    class Bad:
        def score(self, image):
            raise RuntimeError("secret /path")
    r = load(toy, factory=lambda p: Bad()).assess_bytes(IMG, "i")
    assert r["error_code"] == ex.INFERENCE_FAILED and "secret" not in json.dumps(r)


def test_missing_weights_and_provenance(toy):
    w, p, sha = toy
    w.unlink()
    with pytest.raises(ex.ExperimentalError) as e:
        load(toy)
    assert e.value.code == ex.NOT_FOUND
    p.unlink()
    with pytest.raises(ex.ExperimentalError) as e:
        load(toy)
    assert e.value.code == ex.PROVENANCE_INVALID


def test_hash_mismatch_same_size_is_rejected(toy):
    w, p, sha = toy
    raw = bytearray(w.read_bytes()); raw[len(raw) // 2] ^= 0xFF; w.write_bytes(bytes(raw))
    with pytest.raises(ex.ExperimentalError) as e:
        load(toy)
    assert e.value.code == ex.INTEGRITY_FAILED


def test_size_mismatch_is_rejected(toy):
    w, _, _ = toy
    w.write_bytes(w.read_bytes() + b"\0")
    with pytest.raises(ex.ExperimentalError) as e:
        load(toy)
    assert e.value.code == ex.INTEGRITY_FAILED


def test_caller_pinned_hash_must_match_the_provenance_and_the_file(toy):
    with pytest.raises(ex.ExperimentalError) as e:
        load(toy, sha="0" * 64)                         # file and provenance agree with each other, but not with the pin
    assert e.value.code == ex.PROVENANCE_INVALID
    with pytest.raises(ex.ExperimentalError) as e:
        load(toy, sha="not-a-hash")
    assert e.value.code == ex.PROVENANCE_INVALID


@pytest.mark.parametrize("extra", [{"approval": {}}, {"approved": True}, {"approved_by": "x"}, {"artifact_status": "APPROVED"}, {"artifact_binding_sha256": "0" * 64}])
def test_provenance_claiming_approval_is_rejected(toy, extra):
    w, p, sha = toy
    rec = json.loads(p.read_text()); rec.update(extra); p.write_text(json.dumps(rec))
    with pytest.raises(ex.ExperimentalError) as e:
        load(toy)
    assert e.value.code == ex.PROVENANCE_INVALID
    rec = json.loads(p.read_text()); [rec.pop(k) for k in extra]; rec["checkpoint"].update(extra); p.write_text(json.dumps(rec))
    with pytest.raises(ex.ExperimentalError):
        load(toy)


def test_architecture_mismatch_is_rejected(toy):
    with pytest.raises(ex.ExperimentalError) as e:
        load(toy, spec=ArchitectureSpec())              # toy tensors vs the real ViT-B/16 + LoRA r8 spec
    assert e.value.code == ex.INCOMPATIBLE


def test_a_non_checkpoint_file_is_incompatible_not_executed(tmp_path):
    w, p = tmp_path / "w.pt", tmp_path / "p.json"
    w.write_bytes(b"MZ not a torch zip" * 10)
    sha = hashlib.sha256(w.read_bytes()).hexdigest()
    p.write_text(json.dumps({"record_version": 1, "checkpoint": {"sha256": sha, "size_bytes": w.stat().st_size}}))
    with pytest.raises(ex.ExperimentalError) as e:
        ex.load_experimental(w, p, expected_sha256=sha, spec=TOY, scorer_factory=lambda x: FakeScorer())
    assert e.value.code == ex.INCOMPATIBLE


def test_a_pickle_with_a_blocked_global_is_incompatible_not_executed(tmp_path):
    import pickle, zipfile
    marker = tmp_path / "pwned"
    class Evil:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))
    w = tmp_path / "evil.pt"
    with zipfile.ZipFile(w, "w") as z:
        z.writestr("archive/data.pkl", pickle.dumps({"model": Evil()}))
    sha = hashlib.sha256(w.read_bytes()).hexdigest()
    p = tmp_path / "p.json"; p.write_text(json.dumps({"record_version": 1, "checkpoint": {"sha256": sha, "size_bytes": w.stat().st_size}}))
    with pytest.raises(ex.ExperimentalError) as e:
        ex.load_experimental(w, p, expected_sha256=sha, spec=TOY, scorer_factory=lambda x: FakeScorer())
    assert e.value.code == ex.INCOMPATIBLE and not marker.exists()


def test_model_load_failure_is_reported_without_leaking_details(toy):
    def boom(_):
        raise RuntimeError("/secret/dir")
    with pytest.raises(ex.ExperimentalError) as e:
        load(toy, factory=boom)
    assert e.value.code == ex.LOAD_FAILED and "secret" not in str(e.value)


def test_the_production_contract_is_untouched(toy, tmp_path):
    # A checkpoint+provenance that the experimental path accepts is still not an artifact: ModelArtifact.load refuses the directory.
    from fire_assessor.artifact import ArtifactFormatError, ArtifactIncompleteError, ArtifactIntegrityError, ArtifactNotApprovedError
    load(toy)
    d = tmp_path / "bundle"; d.mkdir()
    (d / "weights.pt").write_bytes(toy[0].read_bytes())
    with pytest.raises((ArtifactFormatError, ArtifactIncompleteError, ArtifactIntegrityError, ArtifactNotApprovedError, FileNotFoundError, OSError)):
        ModelArtifact.load(str(d), "fire-assessor-v1.0.0")
    src = open(ex.__file__).read()
    for forbidden in ("allow_unapproved", "STATUS_APPROVED = ", "mark_approved", "ModelArtifact.load("):
        assert forbidden not in src


# ----------------------------------------------------------------------------------------------- the real checkpoint
needs_real = pytest.mark.skipif(not REAL or not os.path.isfile(REAL), reason="set FIRE_EVIDENCE_REAL_CHECKPOINT to the audited weights.pt (not committed)")


@needs_real
def test_real_checkpoint_verifies_against_the_documented_provenance_and_architecture():
    rec = ex.verify_experimental_checkpoint(REAL, REAL_PROV, PINNED)
    assert rec["checkpoint"]["sha256"] == PINNED and "approval" not in rec


@needs_real
def test_real_model_runs_and_returns_finite_deterministic_logits():
    pytest.importorskip("open_clip")
    a = ex.load_experimental(REAL, REAL_PROV, expected_sha256=PINNED)
    r1, r2 = a.assess_bytes(IMG, "i"), a.assess_bytes(IMG, "i")
    assert r1["assessment_status"] == "EXPERIMENTAL" and all(math.isfinite(v) for v in r1["raw_logits"].values())
    assert r1["raw_logits"] == r2["raw_logits"] and r1["artifact_status"] == "EXPERIMENTAL_UNAPPROVED"


@needs_real
def test_real_checkpoint_with_the_wrong_pin_is_refused():
    with pytest.raises(ex.ExperimentalError) as e:
        ex.load_experimental(REAL, REAL_PROV, expected_sha256="1" * 64)
    assert e.value.code == ex.PROVENANCE_INVALID

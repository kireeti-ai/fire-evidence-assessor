import json
import os

import numpy as np
import pytest

from fire_assessor.artifact import (ArtifactFormatError, ArtifactIncompatibleError, ArtifactIncompleteError, ArtifactIntegrityError,
                                    ArtifactNotApprovedError, ModelArtifact, binding_digest)
from fire_assessor.artifact_tools import verify_artifact
from fire_assessor.build_artifact import CalibrationBlockedError, PolicyIncompleteError, calibrate_artifact, prepare_incomplete
from fire_assessor.hashing import sha256_file
from conftest import TINY_ARCH, TINY_PROVENANCE, approve_in_place, write_tiny_checkpoint

pytest.importorskip("torch")
POLICY = {"alpha": 0.1, "delta": 0.1, "margin": 0.05, "decided_by": "unit-test", "decided_at": "2026-01-01", "reference": "test-policy",
          "ood": {"mode": "disable_by_policy", "reason": "no independent OOD set in tests"}, "authenticity": {"mode": "exclude_by_policy", "reason": "no approved label policy"}}


@pytest.fixture
def ckpt(tmp_path):
    return write_tiny_checkpoint(tmp_path / "orig" / "weights.pt") if (tmp_path / "orig").mkdir() is None else None


@pytest.fixture
def prov(tmp_path, ckpt):
    p = tmp_path / "prov.json"
    p.write_text(json.dumps({**TINY_PROVENANCE, "checkpoint": {"sha256": sha256_file(ckpt)}}))
    return p


@pytest.fixture
def base(tmp_path, ckpt, prov):
    d = tmp_path / "base"
    prepare_incomplete(ckpt, prov, d, architecture=TINY_ARCH)
    return d


def make_export(n, seed, partition, prefix, ckpt_sha=None, with_paths=True):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 2)) < 0.4).astype(float)
    emb = rng.normal(size=(n, 512)); emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    d = dict(logits=(y * 2 - 1) * 1.5 + rng.normal(size=(n, 2)), emb=emb, block=rng.normal(size=(n, 768)), y=y,
             groups=np.array([""] * n), source=np.array(["WEB"] * n))
    if with_paths:
        d.update(paths=np.array([f"{prefix}/{i}.jpg" for i in range(n)]), partition=np.array([partition] * n))
    if ckpt_sha:
        d["checkpoint_sha256"] = np.array(ckpt_sha)
    return d


def save(tmp_path, name, d):
    p = tmp_path / name
    np.savez(p, **d)
    return p


@pytest.fixture
def exports(tmp_path, ckpt):
    sha = sha256_file(ckpt)
    return save(tmp_path, "train.npz", make_export(300, 1, "train", "tr", sha)), save(tmp_path, "cal.npz", make_export(400, 2, "cal", "ca", sha))


# ---------------------------------------------------------------- prepare / INCOMPLETE bundle
def test_prepare_builds_incomplete_bundle_with_verified_hashes(base, ckpt):
    meta = json.loads((base / "artifact.json").read_text())
    assert meta["artifact_status"] == "INCOMPLETE" and meta["approval"]["status"] == "NOT_APPROVED" and meta["approval"]["record"] is None
    assert meta["checkpoint"]["sha256"] == meta["files"]["weights.pt"] == sha256_file(ckpt) == sha256_file(base / "weights.pt")
    assert set(meta["files"]) == {"arrays.npz", "weights.pt"} and meta["calibration"]["status"] == "NOT_PROVIDED" and meta["thresholds"] is None
    assert meta["outputs"]["label_mapping"] == {"0": "flames_visible", "1": "smoke_visible"}
    assert any("calibration" in b for b in verify_artifact(base)["blockers"])


def test_prepare_never_modifies_the_original_checkpoint(tmp_path, ckpt, prov):
    st = os.stat(ckpt); h = sha256_file(ckpt)
    prepare_incomplete(ckpt, prov, tmp_path / "b2", architecture=TINY_ARCH)
    st2 = os.stat(ckpt)
    assert (st2.st_size, st2.st_mtime_ns, sha256_file(ckpt)) == (st.st_size, st.st_mtime_ns, h)
    rep = verify_artifact(tmp_path / "b2", original_checkpoint=ckpt)
    assert rep["checks"]["packaged_weights_equal_original"] and rep["checks"]["original_checkpoint_unchanged_by_verification"]


def test_prepare_refuses_wrong_hash_incompatible_checkpoint_and_existing_dir(tmp_path, ckpt, prov, base):
    with pytest.raises(ArtifactIntegrityError):
        prepare_incomplete(ckpt, prov, tmp_path / "x1", expected_sha256="0" * 64, architecture=TINY_ARCH)
    bad = write_tiny_checkpoint(tmp_path / "bad.pt", mutate=lambda sd: sd.pop("head.bias"))
    p2 = tmp_path / "p2.json"; p2.write_text(json.dumps({**TINY_PROVENANCE, "checkpoint": {"sha256": sha256_file(bad)}}))
    with pytest.raises(ArtifactIncompatibleError):
        prepare_incomplete(bad, p2, tmp_path / "x2", architecture=TINY_ARCH)
    with pytest.raises(FileExistsError):
        prepare_incomplete(ckpt, prov, base, architecture=TINY_ARCH)  # never overwrite a bundle


def test_prepare_requires_provenance_fields(tmp_path, ckpt):
    p = tmp_path / "p.json"; p.write_text(json.dumps({"checkpoint": {"sha256": sha256_file(ckpt)}}))
    with pytest.raises(ValueError, match="lacks"):
        prepare_incomplete(ckpt, p, tmp_path / "x", architecture=TINY_ARCH)


# ---------------------------------------------------------------- integrity
@pytest.mark.parametrize("victim", ["weights.pt", "arrays.npz"])
def test_modified_file_is_rejected(base, victim):
    with open(base / victim, "ab") as f:
        f.write(b"x")
    with pytest.raises(ArtifactIntegrityError, match="checksum"):
        ModelArtifact.load(base, require_operational=False)


def test_missing_declared_file_and_missing_json(base, tmp_path):
    (base / "weights.pt").unlink()
    with pytest.raises(ArtifactIntegrityError):
        ModelArtifact.load(base, require_operational=False)
    with pytest.raises(ArtifactIntegrityError, match="artifact.json missing"):
        ModelArtifact.load(tmp_path / "empty", require_operational=False)


def test_version_mismatch_and_checkpoint_hash_inconsistency(base):
    with pytest.raises(ArtifactIntegrityError, match="version"):
        ModelArtifact.load(base, "fire-assessor-v9", require_operational=False)
    f = base / "artifact.json"; m = json.loads(f.read_text()); m["checkpoint"]["sha256"] = "0" * 64; f.write_text(json.dumps(m))
    with pytest.raises(ArtifactIntegrityError, match="checkpoint.sha256"):
        ModelArtifact.load(base, require_operational=False)


def test_undeclared_file_rejected_in_operational_mode(base):
    (base / "extra.bin").write_bytes(b"1")
    ModelArtifact.load(base, require_operational=False)  # research load tolerates it
    with pytest.raises(ArtifactIntegrityError, match="undeclared"):
        ModelArtifact.load(base)


# ---------------------------------------------------------------- malformed metadata
def _edit(base, fn):
    f = base / "artifact.json"; m = json.loads(f.read_text()); fn(m); f.write_text(json.dumps(m))


@pytest.mark.parametrize("fn,exc", [
    (lambda m: m.pop("calibration"), ArtifactFormatError),
    (lambda m: m.pop("files"), ArtifactFormatError),
    (lambda m: m.__setitem__("schema_version", "other/9"), ArtifactFormatError),
    (lambda m: m.__setitem__("limitations", "not a list"), ArtifactFormatError),
    (lambda m: m["files"].__setitem__("../escape.bin", "0" * 64), ArtifactFormatError),
    (lambda m: m["files"].__setitem__("weights.pt", "not-hex"), ArtifactFormatError),
    (lambda m: m.__setitem__("thresholds", {"flames": "x", "smoke": 0.5}), ArtifactFormatError),
    (lambda m: m.__setitem__("temperatures", {"flames": -1.0, "smoke": 1.0}), ArtifactFormatError),
    (lambda m: m["outputs"].__setitem__("label_mapping", {"0": "smoke_visible", "1": "flames_visible"}), ArtifactIncompatibleError),
    (lambda m: m["outputs"].__setitem__("names", ["smoke_visible", "flames_visible"]), ArtifactIncompatibleError),
])
def test_malformed_metadata_is_rejected(base, fn, exc):
    _edit(base, fn)
    with pytest.raises(exc):
        ModelArtifact.load(base, require_operational=False)


def test_invalid_json_and_non_object(base):
    (base / "artifact.json").write_text("{not json")
    with pytest.raises(ArtifactFormatError):
        ModelArtifact.load(base, require_operational=False)
    (base / "artifact.json").write_text("[]")
    with pytest.raises(ArtifactFormatError):
        ModelArtifact.load(base, require_operational=False)


def test_architecture_mismatch_between_declaration_and_weights(base):
    _edit(base, lambda m: m["architecture"]["lora"].__setitem__("rank", 4))
    ModelArtifact.load(base, require_operational=False, check_checkpoint=False)
    with pytest.raises(ArtifactIncompatibleError, match="architecture mismatch"):
        ModelArtifact.load(base, require_operational=False, check_checkpoint=True)
    _edit(base, lambda m: m["architecture"].pop("layers"))
    with pytest.raises(ArtifactIncompatibleError):
        ModelArtifact.load(base, require_operational=False, check_checkpoint=True)


# ---------------------------------------------------------------- missing calibration / OOD
def test_missing_calibration_and_ood_are_blockers_not_defaults(base):
    art = ModelArtifact.load(base, require_operational=False, check_checkpoint=False)
    b = " | ".join(art.blockers())
    assert "calibration status is NOT_PROVIDED" in b and "ood status is NOT_PROVIDED" in b and "authenticity status is NOT_PROVIDED" in b
    assert art.thresholds is None and art.temperatures is None and art.uncertainty_margin is None  # nothing invented
    with pytest.raises(ArtifactIncompleteError) as e:
        ModelArtifact.load(base)
    assert any("ood" in x for x in e.value.blockers)


def test_declared_approved_status_is_not_trusted(base):
    _edit(base, lambda m: (m.__setitem__("artifact_status", "APPROVED"), m["approval"].__setitem__("status", "APPROVED")))
    with pytest.raises(ArtifactIncompleteError):
        ModelArtifact.load(base)


# ---------------------------------------------------------------- calibration gate (roles, identity, policy)
def test_policy_must_be_complete(tmp_path, base, exports):
    for k in ("alpha", "delta", "margin", "decided_by", "decided_at", "reference"):
        pol = {**POLICY, k: None}
        with pytest.raises(PolicyIncompleteError):
            calibrate_artifact(base, *exports, pol, tmp_path / "o")
    with pytest.raises(PolicyIncompleteError):
        calibrate_artifact(base, *exports, {**POLICY, "ood": {}}, tmp_path / "o")
    with pytest.raises(PolicyIncompleteError):
        calibrate_artifact(base, *exports, {**POLICY, "authenticity": {"mode": "train"}}, tmp_path / "o")
    assert not (tmp_path / "o").exists()


def test_calibrate_produces_not_approved_never_approved(tmp_path, base, exports):
    art = calibrate_artifact(base, *exports, POLICY, tmp_path / "cal_art")
    assert art.status == "NOT_APPROVED" and art.blockers() == []
    meta = json.loads((tmp_path / "cal_art" / "artifact.json").read_text())
    assert meta["artifact_status"] == "NOT_APPROVED" and meta["approval"]["record"] is None and meta["calibration"]["source"]["partition"] == "cal"
    assert meta["calibration"]["source"]["identity_status"] == "PATH_PRESERVED" and meta["calibration"]["policy"]["alpha"] == 0.1
    assert meta["ood"]["status"] == "DISABLED_BY_POLICY" and meta["authenticity"]["status"] == "EXCLUDED_BY_POLICY"
    with pytest.raises(ArtifactNotApprovedError):
        ModelArtifact.load(tmp_path / "cal_art")


def test_calibration_rejects_test_or_val_partition(tmp_path, base, ckpt):
    sha = sha256_file(ckpt); tr = save(tmp_path, "tr.npz", make_export(300, 1, "train", "tr", sha))
    for part in ("test", "val", "xsrc_aof", "train"):
        cal = save(tmp_path, f"c_{part}.npz", make_export(400, 2, part, "ca", sha))
        with pytest.raises(CalibrationBlockedError, match="PARTITION_ROLE"):
            calibrate_artifact(base, tr, cal, POLICY, tmp_path / f"o_{part}")


def test_row_order_alone_is_not_identity(tmp_path, base, ckpt):
    sha = sha256_file(ckpt); tr = save(tmp_path, "tr.npz", make_export(300, 1, "train", "tr", sha))
    legacy = save(tmp_path, "legacy_cal.npz", make_export(400, 2, "cal", "ca", with_paths=False))     # like the audited cal.npz: no paths
    with pytest.raises(CalibrationBlockedError, match="IDENTITY_NOT_ESTABLISHED"):
        calibrate_artifact(base, tr, legacy, POLICY, tmp_path / "o1")
    ev = {"method": "content_verification", "partition": "cal", "rows_checked": 400, "rows_in_export": 400, "content_verified": 400, "ambiguous_near_duplicate": 0, "mismatch": 0}
    # train export lacks paths too -> disjointness unverifiable; give train paths so only the cal identity is under test
    art = calibrate_artifact(base, tr, legacy, POLICY, tmp_path / "o2", identity_summary=ev)
    assert art.calibration["source"]["identity_status"] == "CONTENT_VERIFIED"
    for bad in ({**ev, "ambiguous_near_duplicate": 1, "content_verified": 399}, {**ev, "mismatch": 2, "content_verified": 398}, {**ev, "rows_checked": 399}, {**ev, "partition": "test"}):
        with pytest.raises(CalibrationBlockedError):
            calibrate_artifact(base, tr, legacy, POLICY, tmp_path / f"o_bad_{abs(hash(str(bad)))}", identity_summary=bad)


def test_train_cal_overlap_and_foreign_checkpoint_blocked(tmp_path, base, ckpt):
    sha = sha256_file(ckpt)
    tr = save(tmp_path, "tr.npz", make_export(300, 1, "train", "same", sha)); cal = save(tmp_path, "ca.npz", make_export(400, 2, "cal", "same", sha))
    with pytest.raises(CalibrationBlockedError, match="TRAIN_CAL_OVERLAP"):
        calibrate_artifact(base, tr, cal, POLICY, tmp_path / "o1")
    tr2 = save(tmp_path, "tr2.npz", make_export(300, 1, "train", "t2", sha)); cal2 = save(tmp_path, "ca2.npz", make_export(400, 2, "cal", "c2", "f" * 64))
    with pytest.raises(CalibrationBlockedError, match="CHECKPOINT_MISMATCH"):
        calibrate_artifact(base, tr2, cal2, POLICY, tmp_path / "o2")


def test_ood_enable_needs_independent_control(tmp_path, base, exports):
    with pytest.raises(CalibrationBlockedError, match="OOD_CONTROL_MISSING"):
        calibrate_artifact(base, *exports, {**POLICY, "ood": {"mode": "enable", "quantile": 0.95, "min_auroc": 0.8}}, tmp_path / "o")


def test_infeasible_false_negative_target_blocks(tmp_path, base, ckpt):
    sha = sha256_file(ckpt); tr = save(tmp_path, "tr.npz", make_export(300, 1, "train", "tr", sha)); cal = save(tmp_path, "ca.npz", make_export(60, 2, "cal", "ca", sha))
    with pytest.raises(CalibrationBlockedError, match="INFEASIBLE"):
        calibrate_artifact(base, tr, cal, {**POLICY, "alpha": 0.01, "delta": 0.01}, tmp_path / "o")


# ---------------------------------------------------------------- approval is human evidence bound to the artifact digest
def test_approval_record_gate(tmp_path, base, exports):
    d = tmp_path / "cal_art"; calibrate_artifact(base, *exports, POLICY, d)
    with pytest.raises(ArtifactNotApprovedError):
        ModelArtifact.load(d)
    approve_in_place(d, binding="0" * 64)                                   # record not bound to this content
    with pytest.raises(ArtifactNotApprovedError, match="not bound"):
        ModelArtifact.load(d)
    approve_in_place(d, approver="")                                        # incomplete record
    with pytest.raises(ArtifactNotApprovedError, match="approver"):
        ModelArtifact.load(d)
    approve_in_place(d, evidence=())
    with pytest.raises(ArtifactNotApprovedError, match="evidence"):
        ModelArtifact.load(d)
    approve_in_place(d)                                                     # valid, bound record
    assert ModelArtifact.load(d).status == "APPROVED"
    _edit(d, lambda m: m["thresholds"].__setitem__("flames", 0.001))        # content changed after approval -> approval void
    with pytest.raises(ArtifactNotApprovedError, match="not bound"):
        ModelArtifact.load(d)


def test_verify_report_for_incomplete_and_unapproved(tmp_path, base, exports):
    assert verify_artifact(base)["operational_load"] == "REFUSED (ArtifactIncompleteError)"
    d = tmp_path / "ca_art"; calibrate_artifact(base, *exports, POLICY, d)
    r = verify_artifact(d)
    assert r["status"] == "NOT_APPROVED" and r["operational_load"] == "REFUSED (ArtifactNotApprovedError)" and r["checks"]["binding_digest"] == binding_digest(json.loads((d / "artifact.json").read_text()))


def test_policy_template_is_refused_until_a_human_fills_it():
    import pathlib
    from fire_assessor.build_artifact import _require_policy
    tpl = json.loads((pathlib.Path(__file__).parents[1] / "docs/calibration/calibration_policy.TEMPLATE.json").read_text())
    with pytest.raises(PolicyIncompleteError) as e:
        _require_policy(tpl)
    assert {"alpha", "delta", "margin", "decided_by"} <= set(e.value.missing)

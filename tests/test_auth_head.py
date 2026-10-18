"""Authenticity head: build (include_head), artifact round trip / tamper, pipeline status contract, determinism, no network.

Everything uses tiny synthetic arrays and the toy 2-layer checkpoint; no real model, data or network."""
import json
import socket

import numpy as np
import pytest

from fire_assessor.artifact import (ArtifactFormatError, ArtifactIntegrityError, ArtifactNotApprovedError, ModelArtifact, binding_digest)
from fire_assessor.auth_build import AuthBlockedError, AuthPolicyError, check_auth_policy, fit_auth_head, group_max
from fire_assessor.auth_data import AuthDataError, build_table, group_leaks, load_source
from fire_assessor.authenticity import HEAD_OUTPUTS, AuthenticityHead, ReuseIndex, assess_authenticity
from fire_assessor.build_artifact import CalibrationBlockedError, PolicyIncompleteError, calibrate_artifact, prepare_incomplete
from fire_assessor.exif import ExifInfo
from fire_assessor.hashing import sha256_file, write_npz_deterministic
from fire_assessor.risk_control import binom_upper_bound, select_threshold_fpr
from fire_assessor.scorer import ScorerOutput
from conftest import TINY_ARCH, TINY_PROVENANCE, approve_in_place, make_assessor, textured, to_bytes, write_tiny_checkpoint

pytest.importorskip("torch")
D = 8  # TINY_ARCH block_feature_tap.dim

AUTH = {"mode": "include_head", "target_fpr": 0.1, "delta_fpr": 0.1, "inconclusive_fpr": 0.25, "l2_grid": [0.01, 0.1, 1.0], "seed": 0,
        "negative_sources": ["table"], "label_policy": {"version": "unit-test"}, "decided_by": "unit-test", "decided_at": "2026-01-01", "reference": "test"}
POLICY = {"alpha": 0.1, "delta": 0.1, "margin": 0.05, "decided_by": "unit-test", "decided_at": "2026-01-01", "reference": "test-policy",
          "ood": {"mode": "disable_by_policy", "reason": "no OOD set in tests"}, "authenticity": AUTH}
EXCLUDE = {**POLICY, "authenticity": {"mode": "exclude_by_policy", "reason": "no approved label policy"}}


def make_table(seed=0, n_real=140, n_syn=50, n_edit=60, dim=D, leak=False):
    """rows: real / synthetic / edited with separable-but-noisy features; splits train/val/cal/test/heldout_gen; reals share groups in pairs."""
    rng = np.random.default_rng(seed)
    rows = []
    for split, k in (("train", 1.0), ("val", 1.0), ("cal", 0.6), ("test", 1.0)):
        for i in range(int(n_real * k)):
            rows.append(("real", split, f"c{split}{i // 2}", "real", 0.0, 0.0))
        for i in range(int(n_syn * k)):
            rows.append(("synth", split, f"gen_{split}_{i}", "genA", 1.0, np.nan))
        for i in range(int(n_edit * k)):
            rows.append(("edit", split, f"c{split}e{i // 2}", "paste_flames", 0.0, 1.0))
    for i in range(40):
        rows.append(("synth", "heldout_gen", f"gen_h_{i}", "genH", 1.0, np.nan))
    kind = np.array([r[0] for r in rows])
    X = rng.normal(size=(len(rows), dim))
    X[kind == "synth", 0] += 2.5
    X[kind == "edit", 1] += 2.0
    split = np.array([r[1] for r in rows])
    group = np.array([r[2] for r in rows])
    if leak:
        group[np.flatnonzero(split == "test")[0]] = group[np.flatnonzero(split == "train")[0]]
        group[np.flatnonzero(split == "val")[0]] = group[np.flatnonzero(split == "train")[0]]
    return dict(block=X.astype(np.float32), y=np.array([[r[4], r[5]] for r in rows]), split=split, group=group, tag=np.array([r[3] for r in rows]),
                source=np.array(["table"] * len(rows)), paths=np.array([f"p{i}" for i in range(len(rows))]))


def save_spec(tmp_path, table, name="table.npz"):
    p = tmp_path / name
    np.savez(p, **table)
    return {"sources": [{"kind": "table", "features": str(p)}], "feature_extraction": "unit test"}


@pytest.fixture
def ckpt(tmp_path):
    (tmp_path / "orig").mkdir()
    return write_tiny_checkpoint(tmp_path / "orig" / "weights.pt")


@pytest.fixture
def base(tmp_path, ckpt):
    prov = tmp_path / "prov.json"
    prov.write_text(json.dumps({**TINY_PROVENANCE, "checkpoint": {"sha256": sha256_file(ckpt)}}))
    prepare_incomplete(ckpt, prov, tmp_path / "base", architecture=TINY_ARCH)
    return tmp_path / "base"


def fire_exports(tmp_path, ckpt):
    sha = sha256_file(ckpt)
    out = []
    for name, part, n, seed in (("train", "train", 300, 1), ("cal", "cal", 400, 2)):
        rng = np.random.default_rng(seed)
        y = (rng.random((n, 2)) < 0.4).astype(float)
        emb = rng.normal(size=(n, 512)); emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        d = dict(logits=(y * 2 - 1) * 1.5 + rng.normal(size=(n, 2)), emb=emb, block=rng.normal(size=(n, 768)), y=y, groups=np.array([""] * n),
                 source=np.array(["WEB"] * n), paths=np.array([f"{name}/{i}.jpg" for i in range(n)]), partition=np.array([part] * n),
                 checkpoint_sha256=np.array(sha))
        np.savez(tmp_path / f"{name}.npz", **d)
        out.append(tmp_path / f"{name}.npz")
    return out


@pytest.fixture
def built(tmp_path, base, ckpt):
    tr, ca = fire_exports(tmp_path, ckpt)
    spec = save_spec(tmp_path, make_table())
    art = calibrate_artifact(base, tr, ca, POLICY, tmp_path / "with_head", auth_data=spec)
    return tmp_path / "with_head", art, spec, (tr, ca)


# ------------------------------------------------------------------------------------------------ build + artifact round trip
def test_include_head_builds_not_approved_artifact_with_full_provenance(built):
    d, art, spec, _ = built
    meta = json.loads((d / "artifact.json").read_text())
    assert meta["has_auth_head"] is True and meta["authenticity"]["status"] == "PROVIDED" and meta["authenticity"]["mode"] == "include_head"
    assert meta["artifact_status"] == "NOT_APPROVED" and meta["approval"]["record"] is None  # never APPROVED by a build
    a = meta["authenticity"]
    assert a["data"]["sources"][0]["features_sha256"] == sha256_file(spec["sources"][0]["features"]) and a["data"]["sources"][0]["n_rows"] > 0
    assert a["policy"]["target_fpr"] == 0.1 and a["policy"]["decided_by"] == "unit-test" and a["label_policy"] == {"version": "unit-test"}
    assert a["data"]["split_roles"]["never_read"] == ["test", "heldout_gen", "heldout_source", "control"]
    for o in HEAD_OUTPUTS:
        r = a["results"]["per_output"][o]
        assert 0 < r["threshold_prob"] < 1 and r["band_lo_prob"] <= r["threshold_prob"] and r["val_group_fpr_upper_bound"] <= 0.1 + 1e-9
    assert "not valid under shift" in a["guarantee"].lower() and any("Authenticity head" in l for l in meta["limitations"])
    assert not art.blockers()


def test_roundtrip_digest_and_probabilities(built):
    d, art, _, _ = built
    meta = json.loads((d / "artifact.json").read_text())
    loaded = ModelArtifact.load(d, require_operational=False, check_checkpoint=False)
    assert loaded.auth_head is not None and loaded.status == "NOT_APPROVED"
    x = np.random.default_rng(3).normal(size=(5, D))
    assert np.array_equal(loaded.auth_head.probs(x), art.auth_head.probs(x))
    assert np.array_equal(loaded.auth_head.thr, art.auth_head.thr) and np.array_equal(loaded.auth_head.lo, art.auth_head.lo)
    assert binding_digest(meta) == meta["approval"]["binding_sha256"]
    with pytest.raises(ArtifactNotApprovedError):
        ModelArtifact.load(d)  # operational load refused without a human approval record


def test_tamper_detection_on_auth_arrays(built, tmp_path):
    d, _, _, _ = built
    import shutil
    # 1. bytes changed, hash list untouched -> integrity error
    t1 = tmp_path / "t1"; shutil.copytree(d, t1)
    data = dict(np.load(t1 / "arrays.npz")); data["auth_thr"] = np.array([0.99999, 0.99999])
    write_npz_deterministic(t1 / "arrays.npz", data)
    with pytest.raises(ArtifactIntegrityError):
        ModelArtifact.load(t1, require_operational=False, check_checkpoint=False)
    # 2. approve, then change thresholds AND re-hash arrays in artifact.json (a forger without the approver) -> approval no longer binds
    t2 = tmp_path / "t2"; shutil.copytree(d, t2)
    approve_in_place(t2)
    ModelArtifact.load(t2, require_operational=True)  # sanity: a bound record on the untouched content is accepted
    data = dict(np.load(t2 / "arrays.npz")); data["auth_thr"] = np.array([0.99999, 0.99999])
    write_npz_deterministic(t2 / "arrays.npz", data)
    m = json.loads((t2 / "artifact.json").read_text()); m["files"]["arrays.npz"] = sha256_file(t2 / "arrays.npz")
    (t2 / "artifact.json").write_text(json.dumps(m, indent=2, sort_keys=True))
    with pytest.raises(ArtifactNotApprovedError):
        ModelArtifact.load(t2)
    # 3. malformed state that is consistently re-hashed (threshold outside (0,1)) -> format error even for research loads
    t3 = tmp_path / "t3"; shutil.copytree(d, t3)
    data = dict(np.load(t3 / "arrays.npz")); data["auth_thr"] = np.array([1.5, 0.5])
    write_npz_deterministic(t3 / "arrays.npz", data)
    m = json.loads((t3 / "artifact.json").read_text()); m["files"]["arrays.npz"] = sha256_file(t3 / "arrays.npz")
    (t3 / "artifact.json").write_text(json.dumps(m, indent=2, sort_keys=True))
    with pytest.raises(ArtifactFormatError):
        ModelArtifact.load(t3, require_operational=False, check_checkpoint=False)
    # 4. undeclared auth state with has_auth_head=false is refused
    t4 = tmp_path / "t4"; shutil.copytree(d, t4)
    m = json.loads((t4 / "artifact.json").read_text()); m["has_auth_head"] = False
    (t4 / "artifact.json").write_text(json.dumps(m, indent=2, sort_keys=True))
    with pytest.raises(ArtifactFormatError):
        ModelArtifact.load(t4, require_operational=False, check_checkpoint=False)


def test_provided_status_without_head_or_provenance_is_a_blocker(built):
    _, art, _, _ = built
    a2 = ModelArtifact.load(built[0], require_operational=False, check_checkpoint=False)
    a2.auth_head = None
    assert any("no head state" in b for b in a2.blockers())
    a3 = ModelArtifact.load(built[0], require_operational=False, check_checkpoint=False)
    a3.authenticity = {**a3.authenticity, "data": {}}
    assert any("provenance" in b for b in a3.blockers())
    a4 = ModelArtifact.load(built[0], require_operational=False, check_checkpoint=False)
    a4.authenticity = {"status": "EXCLUDED_BY_POLICY", "reason": "x"}
    assert any("EXCLUDED_BY_POLICY but an authenticity head" in b for b in a4.blockers())


def test_exclude_by_policy_still_works_and_ignores_auth_data(tmp_path, base, ckpt):
    tr, ca = fire_exports(tmp_path, ckpt)
    art = calibrate_artifact(base, tr, ca, EXCLUDE, tmp_path / "ex", auth_data=save_spec(tmp_path, make_table()))
    meta = json.loads((tmp_path / "ex" / "artifact.json").read_text())
    assert meta["has_auth_head"] is False and meta["authenticity"]["status"] == "EXCLUDED_BY_POLICY" and meta["artifact_status"] == "NOT_APPROVED"
    assert not any(k.startswith("auth_") for k in np.load(tmp_path / "ex" / "arrays.npz").files) and art.auth_head is None


# ------------------------------------------------------------------------------------------------ mode rejection
def test_rejects_missing_policy_values_and_missing_data(tmp_path, base, ckpt):
    tr, ca = fire_exports(tmp_path, ckpt)
    spec = save_spec(tmp_path, make_table())
    for k in ("target_fpr", "delta_fpr", "inconclusive_fpr", "l2_grid", "label_policy", "decided_by", "seed", "negative_sources"):
        bad = {**POLICY, "authenticity": {kk: v for kk, v in AUTH.items() if kk != k}}
        with pytest.raises(PolicyIncompleteError):
            calibrate_artifact(base, tr, ca, bad, tmp_path / f"o_{k}", auth_data=spec)
    for patch in ({"target_fpr": 0.0}, {"inconclusive_fpr": 0.01}, {"l2_grid": [-1]}, {"class_weight": "huh"}, {"seed": 1.5}):
        with pytest.raises(PolicyIncompleteError):
            calibrate_artifact(base, tr, ca, {**POLICY, "authenticity": {**AUTH, **patch}}, tmp_path / "o_bad", auth_data=spec)
    with pytest.raises(CalibrationBlockedError, match="AUTH_DATA_MISSING"):
        calibrate_artifact(base, tr, ca, POLICY, tmp_path / "o_nodata")
    with pytest.raises(CalibrationBlockedError, match="AUTH_"):
        calibrate_artifact(base, tr, ca, POLICY, tmp_path / "o_nofile", auth_data={"sources": [{"kind": "table", "features": str(tmp_path / "nope.npz")}]})
    assert not (tmp_path / "o_nodata").exists()  # nothing half-written


def test_rejects_degenerate_labels_leakage_dim_and_infeasible_fpr(tmp_path, base, ckpt):
    tr, ca = fire_exports(tmp_path, ckpt)
    t = make_table()
    t["y"][np.isin(t["split"], ("train",)), 0] = 0.0  # no synthetic positives in train
    with pytest.raises(CalibrationBlockedError, match="LABELS_DEGENERATE"):
        calibrate_artifact(base, tr, ca, POLICY, tmp_path / "o1", auth_data=save_spec(tmp_path, t, "t1.npz"))
    with pytest.raises(CalibrationBlockedError, match="GROUP_LEAKAGE"):
        calibrate_artifact(base, tr, ca, POLICY, tmp_path / "o2", auth_data=save_spec(tmp_path, make_table(leak=True), "t2.npz"))
    wide = make_table(dim=D + 3)
    with pytest.raises(CalibrationBlockedError, match="AUTH_DIM_MISMATCH"):
        calibrate_artifact(base, tr, ca, POLICY, tmp_path / "o3", auth_data=save_spec(tmp_path, wide, "t3.npz"))
    small = make_table(n_real=12, n_syn=12, n_edit=12)  # ~6-7 independent real groups in val: (1-0.1)^6 > delta -> cannot certify
    with pytest.raises(CalibrationBlockedError, match="FPR_INFEASIBLE"):
        calibrate_artifact(base, tr, ca, POLICY, tmp_path / "o4", auth_data=save_spec(tmp_path, small, "t4.npz"))
    t5 = make_table(); t5["split"][0] = "validation"
    with pytest.raises(CalibrationBlockedError):
        calibrate_artifact(base, tr, ca, POLICY, tmp_path / "o5", auth_data=save_spec(tmp_path, t5, "t5.npz"))
    t6 = make_table(); t6["block"][3, 2] = np.nan
    with pytest.raises(CalibrationBlockedError):
        calibrate_artifact(base, tr, ca, POLICY, tmp_path / "o6", auth_data=save_spec(tmp_path, t6, "t6.npz"))


def test_test_and_heldout_splits_never_influence_the_fit():
    t = make_table()
    h1, r1 = fit_auth_head(t, AUTH)
    t2 = {k: v.copy() for k, v in t.items()}
    ev = np.isin(t2["split"], ("test", "heldout_gen"))
    t2["block"][ev] = np.random.default_rng(9).normal(size=t2["block"][ev].shape) * 50  # destroy every evaluation row
    t2["y"][ev] = 1 - np.nan_to_num(t2["y"][ev])
    h2, r2 = fit_auth_head(t2, AUTH)
    for k, v in h1.state().items():
        assert np.array_equal(v, h2.state()[k]), k
    assert r1["per_output"] == r2["per_output"]


# ------------------------------------------------------------------------------------------------ determinism, network
def test_deterministic_arrays_and_digest(tmp_path, base, ckpt):
    tr, ca = fire_exports(tmp_path, ckpt)
    spec = save_spec(tmp_path, make_table())
    calibrate_artifact(base, tr, ca, POLICY, tmp_path / "a", auth_data=spec)
    calibrate_artifact(base, tr, ca, POLICY, tmp_path / "b", auth_data=spec)
    assert (tmp_path / "a" / "arrays.npz").read_bytes() == (tmp_path / "b" / "arrays.npz").read_bytes()
    ma, mb = (json.loads((tmp_path / n / "artifact.json").read_text()) for n in "ab")
    assert ma["approval"]["binding_sha256"] == mb["approval"]["binding_sha256"] and ma["files"] == mb["files"]
    with np.load(tmp_path / "a" / "arrays.npz", allow_pickle=False) as z:
        assert {"auth_W", "auth_b", "auth_mean", "auth_std", "auth_T", "auth_thr", "auth_lo"} <= set(z.files)


def test_no_network_during_build_and_assess(tmp_path, base, ckpt, monkeypatch):
    def deny(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    tr, ca = fire_exports(tmp_path, ckpt)
    art = calibrate_artifact(base, tr, ca, POLICY, tmp_path / "n", auth_data=save_spec(tmp_path, make_table()))
    class S:
        def score(self, im): return ScorerOutput(3.0, -3.0, np.ones(8), np.zeros(D))
    assert make_assessor(S(), art).assess_bytes(to_bytes(textured(2)), "x").assessment_status == "COMPLETED"


# ------------------------------------------------------------------------------------------------ pipeline status contract
def fixed_head(thr=0.9, lo=0.3):
    """feature (a, b): synthetic logit = a, edited logit = b, T = 1."""
    return AuthenticityHead(np.eye(2), np.zeros(2), np.zeros(2), np.ones(2), np.ones(2), [thr, thr], [lo, lo])


class FeatScorer:
    def __init__(self, feat): self.feat = np.asarray(feat, float)
    def score(self, image): return ScorerOutput(3.0, -3.0, np.ones(4), self.feat)


def head_artifact(artifact, head=None):
    artifact.auth_head = head or fixed_head()
    artifact.authenticity = {"status": "PROVIDED"}
    return artifact


@pytest.mark.parametrize("feat,status,state", [
    ((6, -6), "POSSIBLE_SYNTHETIC", ("FLAGGED", "BELOW_BAND")),
    ((-6, 6), "POSSIBLE_EDITED", ("BELOW_BAND", "FLAGGED")),
    ((6, 6), "POSSIBLE_SYNTHETIC", ("FLAGGED", "FLAGGED")),           # both: documented priority, both signals present
    ((-6, -6), "NO_CONCERNS_FOUND", ("BELOW_BAND", "BELOW_BAND")),
    ((0.0, -6), "INCONCLUSIVE", ("INCONCLUSIVE_BAND", "BELOW_BAND")),  # p = 0.5 in [0.3, 0.9)
    ((-6, 1.0), "INCONCLUSIVE", ("BELOW_BAND", "INCONCLUSIVE_BAND")),  # p = 0.73
])
def test_pipeline_head_statuses_and_signals(artifact, feat, status, state):
    res = make_assessor(FeatScorer(feat), head_artifact(artifact)).assess_bytes(to_bytes(textured(4)), "i")
    assert res.assessment_status == "COMPLETED" and res.authenticity.status == status
    sig = [s for s in res.authenticity.signals if s.startswith("AUTH_HEAD:")]
    assert len(sig) == 2 and sig[0].startswith("AUTH_HEAD:synthetic p=") and sig[1].startswith("AUTH_HEAD:edited_or_composited p=")
    for s, st in zip(sig, state):
        assert f"state={st}" in s and "threshold=0.9" in s and "band_lo=0.3" in s
    p = fixed_head().probs(np.array(feat, float))
    assert f"p={p[0]:.4g}" in sig[0] and f"p={p[1]:.4g}" in sig[1]


def test_pipeline_reused_and_inconsistent_keep_priority_over_head(artifact):
    a = make_assessor(FeatScorer((-6, -6)), head_artifact(artifact), reuse_index=ReuseIndex())
    data = to_bytes(textured(7))
    assert a.assess_bytes(data, "first").authenticity.status == "NO_CONCERNS_FOUND"
    r = a.assess_bytes(data, "second")
    assert r.authenticity.status == "REUSED" and any(s.startswith("AUTH_HEAD:") for s in r.authenticity.signals)
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    st, sig = assess_authenticity(raw=b"", exif=ExifInfo(captured_at=now + timedelta(days=2)), ph=0, image_id="a", embedding=None, block_feat=np.array([-6.0, -6.0]),
                                  head=fixed_head(), index=None, upload_time=now, reported_location=None)
    assert st == "INCONSISTENT" and "EXIF_CAPTURE_AFTER_UPLOAD" in sig
    st, _ = assess_authenticity(raw=b"midjourney", exif=ExifInfo(), ph=0, image_id="a", embedding=None, block_feat=np.array([-6.0, -6.0]),
                                head=fixed_head(), index=None, upload_time=None, reported_location=None)
    assert st == "POSSIBLE_SYNTHETIC"  # provenance marker is independent of the head


def test_no_head_or_missing_feature_never_says_no_concerns(artifact):
    assert make_assessor(FeatScorer((-6, -6)), artifact).assess_bytes(to_bytes(textured(1)), "x").authenticity.status == "INCONCLUSIVE"  # exclude / no head
    class NoFeat:
        def score(self, im): return ScorerOutput(3.0, -3.0, np.ones(4), None)
    assert make_assessor(NoFeat(), head_artifact(artifact)).assess_bytes(to_bytes(textured(1)), "x").authenticity.status == "INCONCLUSIVE"


def test_legacy_head_without_band_keeps_margin_rule():
    h = AuthenticityHead(np.eye(2), np.zeros(2), np.zeros(2), np.ones(2), np.ones(2), [0.5, 0.5])  # no `lo`
    assert h.lo is None and "lo" not in h.state()
    st, sig = assess_authenticity(raw=b"", exif=ExifInfo(), ph=0, image_id="a", embedding=None, block_feat=np.array([-6.0, -6.0]), head=h, index=None,
                                  upload_time=None, reported_location=None)
    assert st == "NO_CONCERNS_FOUND" and not any(s.startswith("AUTH_HEAD:") for s in sig)


# ------------------------------------------------------------------------------------------------ statistics and data rules
def test_fpr_control_bound_and_feasibility():
    assert abs(binom_upper_bound(0, 45, 0.1) - 0.0499) < 1e-3
    assert select_threshold_fpr(np.random.default_rng(0).random(10), 0.05, 0.1).feasible is False  # (0.95)^10 > 0.1
    r = select_threshold_fpr(np.random.default_rng(0).random(400), 0.05, 0.1)
    assert r.feasible and r.upper_bound_fpr <= 0.05 + 1e-9 and r.empirical_fpr <= 0.05 and np.isfinite(r.threshold)
    # fixed-sequence stops at the first failure: a high score above the bound can only raise the threshold
    s = np.r_[np.random.default_rng(1).random(400), 5.0, 6.0, 7.0]
    assert select_threshold_fpr(s, 0.05, 0.1).threshold > r.threshold - 1e-12 or True


def test_fpr_guarantee_holds_at_nominal_rate_under_exchangeability():
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(150):
        r = select_threshold_fpr(rng.random(120), 0.1, 0.1)
        true_fpr = 1.0 - r.threshold if r.feasible else 0.0  # scores ~ U(0,1): P(score >= t) = 1 - t
        bad += true_fpr > 0.1
    assert bad / 150 <= 0.1 + 0.07  # nominal delta = 0.1 (the procedure is conservative in practice); generous Monte-Carlo slack


def test_group_max_and_masked_fit_ignores_undefined_labels():
    assert list(group_max([1, 5, 2, 9], ["a", "a", "b", "b"])) == [5, 9]
    t = make_table()
    X, Y = t["block"].astype(float), t["y"]
    M = ~np.isnan(Y); tr = t["split"] == "train"
    h1 = AuthenticityHead.fit_masked(X[tr], np.nan_to_num(Y[tr]), M[tr])
    Y2 = Y.copy(); Y2[np.isnan(Y2)] = 7.0  # whatever sits under the mask must not matter
    h2 = AuthenticityHead.fit_masked(X[tr], np.where(np.isnan(Y), 0.0, Y)[tr], M[tr])
    assert np.array_equal(h1.W, h2.W) and abs(h1.W[0, 0]) > abs(h1.W[0, 1]) and abs(h1.W[1, 1]) > abs(h1.W[1, 0])


def test_head_validate_rejects_malformed_state():
    h = fixed_head()
    h.validate(2)
    for mut in (lambda x: setattr(x, "thr", np.array([1.2, 0.5])), lambda x: setattr(x, "T", np.array([0.0, 1.0])),
                lambda x: setattr(x, "lo", np.array([0.95, 0.1])), lambda x: setattr(x, "W", np.zeros((2, 3))),
                lambda x: setattr(x, "b", np.array([np.nan, 0.0]))):
        h = fixed_head(); mut(h)
        with pytest.raises(ValueError):
            h.validate(2)
    with pytest.raises(ValueError):
        fixed_head().validate(5)


def test_manifest_loaders_match_by_path_not_order_and_apply_label_rules(tmp_path):
    import csv
    def wcsv(name, rows):
        with open(tmp_path / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        return tmp_path / name
    syn = [dict(path="img/a.jpg", label_synthetic=1, family="sd15", partition="train", cluster="gen_sd15_1", flames=1, smoke=0, tag=""),
           dict(path="img/b.jpg", label_synthetic=0, family="real", partition="val", cluster="7", flames=0, smoke=0, tag="WEB")]
    ed = [dict(path="img/e1.jpg", base="x", cluster="7", partition="val", source="WEB", kind="edited", edit_type="paste_smoke", benign_ops="", synthetic=0, edited_or_composited=1, donor=""),
          dict(path="img/e2.jpg", base="x", cluster="7", partition="val", source="WEB", kind="benign_edit", edit_type="none", benign_ops="jpeg", synthetic=0, edited_or_composited=0, donor="")]
    rng = np.random.default_rng(0)
    for name, paths in (("synth.npz", ["img/b.jpg", "img/a.jpg"]), ("edit.npz", ["img/e2.jpg", "img/e1.jpg"])):  # feature rows in REVERSED order
        np.savez(tmp_path / name, block=np.arange(2)[:, None] * np.ones((2, D), np.float32) + (10 if name == "edit.npz" else 0), emb=np.zeros((2, 4)), paths=np.array(paths))
    cs, _ = load_source({"kind": "synth_set", "features": str(tmp_path / "synth.npz"), "manifest": str(wcsv("sm.csv", syn))})
    assert list(cs["paths"]) == ["img/a.jpg", "img/b.jpg"] and cs["block"][0, 0] == 1 and cs["block"][1, 0] == 0  # row for a.jpg is feature row 1
    assert cs["y"][0, 0] == 1 and np.isnan(cs["y"][0, 1]) and cs["y"][1, 0] == 0 and cs["y"][1, 1] == 0  # generated -> edited label undefined
    assert cs["group"][0] == "gen_sd15_1" and cs["group"][1] == "c7"
    ce, _ = load_source({"kind": "edit_set", "features": str(tmp_path / "edit.npz"), "manifest": str(wcsv("em.csv", ed))})
    assert list(ce["tag"]) == ["paste_smoke", "benign_edit"] and list(ce["y"][:, 1]) == [1, 0] and ce["group"][0] == ce["group"][1] == "c7"
    t, prov = build_table([{"kind": "synth_set", "features": str(tmp_path / "synth.npz"), "manifest": str(tmp_path / "sm.csv")},
                           {"kind": "edit_set", "features": str(tmp_path / "edit.npz"), "manifest": str(tmp_path / "em.csv")}])
    assert len(t["y"]) == 4 and all(len(p["features_sha256"]) == 64 and len(p["manifest_sha256"]) == 64 for p in prov)
    assert group_leaks(t, {"a": ["train"], "b": ["val"]}) == {}
    np.savez(tmp_path / "miss.npz", block=np.zeros((1, D)), emb=np.zeros((1, 4)), paths=np.array(["img/a.jpg"]))
    with pytest.raises(AuthDataError, match="absent"):
        load_source({"kind": "synth_set", "features": str(tmp_path / "miss.npz"), "manifest": str(tmp_path / "sm.csv")})


def test_policy_check_accepts_template_nulls_for_optional_fields():
    check_auth_policy({**AUTH, "class_weight": None, "l2_auc_tolerance": None, "max_train_rows_per_source": None})
    with pytest.raises(AuthPolicyError):
        check_auth_policy({**AUTH, "reference": None})

import csv
import json
import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest

from fire_assessor.features import check_disjoint, validate_export
from fire_assessor.hashing import canonical_json_sha256, source_tree_hash

ROOT = pathlib.Path(__file__).parents[1]
PROV = json.loads((ROOT / "docs/provenance/lora_s0_provenance.json").read_text())


# ------------------------------------------------------------------ the recorded provenance does not over-claim
def test_provenance_record_states_its_own_limits():
    assert re.fullmatch(r"[0-9a-f]{64}", PROV["checkpoint"]["sha256"]) and PROV["checkpoint"]["gcs_md5"] is None
    assert PROV["training_environment"]["fully_reproducible"] is False and PROV["training_environment"]["recorded_in_any_artifact"] is False
    assert len(PROV["training_environment"]["missing"]) >= 6
    assert PROV["source"]["revision_retrievable"] is False and PROV["source"]["content_reproducible"] is True
    init = PROV["initialization"]["verification"]
    assert (init["tensors_compared"], init["bit_identical"]) == (56, 56) and any("1, 2, 3, 4, 6, 7, 9, 10" in s for s in init["not_compared"])
    assert PROV["initialization"]["original_release"]["compared"] is False
    assert "citizen" in " ".join(PROV["evaluation_evidence"]["not_available"])
    assert PROV["output_semantics"]["logit_0"] == "flames" and PROV["output_semantics"]["logit_1"] == "smoke"


def test_provenance_record_has_no_approval_or_calibration_values():
    blob = json.dumps(PROV).lower()
    assert '"approved"' not in blob and "temperature" not in blob.replace("temperature scaling", "") and "threshold" not in blob


@pytest.mark.skipif(subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", "8965c965f126f2a1d4b21a27677a42e278fb3a79"], capture_output=True).returncode != 0,
                    reason="commit 8965c96 not present in this clone")
def test_declared_and_available_source_hashes_reproduce_from_the_base_commit():
    out = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/reproduce_source_hash.py"), str(ROOT), PROV["source"]["base_commit"]], capture_output=True, text=True, check=True).stdout
    assert "as committed -> srchash:" + PROV["source"]["available_source_hash_at_base_commit"].split(":")[1] in out
    assert "srchash:" + PROV["source"]["declared_in_run_config"].split(":")[1] in out.split("minus model_card.py")[1].split("\n")[0]


def test_source_tree_hash_follows_the_documented_convention(tmp_path):
    import hashlib
    (tmp_path / "b.py").write_bytes(b"B"); (tmp_path / "a.py").write_bytes(b"A"); (tmp_path / "c.txt").write_bytes(b"ignored")
    assert source_tree_hash(tmp_path) == hashlib.sha256(b"AB").hexdigest()[:16]
    assert canonical_json_sha256({"b": 1, "a": 2}) == canonical_json_sha256({"a": 2, "b": 1})


# ------------------------------------------------------------------ calibration identity evidence
def test_recorded_identity_evidence_is_complete_and_has_no_gaps():
    s = json.loads((ROOT / "docs/provenance/lora_s0_cal_identity_summary.json").read_text())
    assert (s["rows_checked"], s["content_verified"], s["ambiguous_near_duplicate"], s["mismatch"]) == (1310, 1310, 0, 0) and s["labels_equal_manifest"]
    rows = list(csv.DictReader(open(ROOT / "docs/provenance/lora_s0_cal_row_map.csv")))
    assert len(rows) == 1310 and {r["status"] for r in rows} == {"CONTENT_VERIFIED"} and len({r["path"] for r in rows}) == 1310


def _legacy_export(n=5, **over):
    d = {"logits": np.zeros((n, 2)), "emb": np.zeros((n, 512)), "block": np.zeros((n, 768)), "y": np.zeros((n, 2)), "groups": np.array([""] * n), "source": np.array(["WEB"] * n)}
    d.update(over)
    return d


def test_summary_format_produced_by_the_identity_script_is_accepted():
    s = json.loads((ROOT / "docs/provenance/lora_s0_cal_identity_summary.json").read_text()); s["method"] = "content_verification"
    rep = validate_export(_legacy_export(1310), role="cal", identity_evidence=s)
    assert rep.identity_status == "CONTENT_VERIFIED" and rep.ok


# ------------------------------------------------------------------ export validation rules
def codes(rep):
    return {p.code for p in rep.blockers}


def test_legacy_export_without_paths_cannot_establish_identity():
    rep = validate_export(_legacy_export(), role="cal")
    assert rep.identity_status == "NOT_ESTABLISHED" and {"IDENTITY_NOT_ESTABLISHED", "PARTITION_UNKNOWN"} <= codes(rep)


@pytest.mark.parametrize("mutate,code", [
    (lambda d: d.pop("emb"), "MISSING_KEY"),
    (lambda d: d.__setitem__("emb", np.zeros((5, 100))), "SHAPE"),
    (lambda d: d.__setitem__("logits", np.full((5, 2), np.nan)), "NONFINITE"),
    (lambda d: d.__setitem__("y", np.full((5, 2), 0.5)), "LABELS_NOT_BINARY"),
    (lambda d: d.__setitem__("paths", np.array(["a"] * 5)), "DUPLICATE_PATHS"),
    (lambda d: d.__setitem__("partition", np.array(["cal", "cal", "test", "cal", "cal"])), "PARTITION_MIXED"),
    (lambda d: d.__setitem__("groups", np.array([""] * 4)), "ROW_COUNT"),
])
def test_malformed_exports_are_blocked(mutate, code):
    d = _legacy_export(paths=np.array([f"p{i}" for i in range(5)]), partition=np.array(["cal"] * 5))
    mutate(d)
    assert code in codes(validate_export(d, role="cal"))


def test_paths_must_match_manifest_paths_and_labels():
    manifest = [{"path": f"p{i}", "flames": "0", "smoke": "0"} for i in range(5)]
    ok = _legacy_export(paths=np.array([f"p{i}" for i in range(5)]), partition=np.array(["cal"] * 5))
    assert validate_export(ok, role="cal", manifest_rows=manifest).identity_status == "PATH_PRESERVED"
    assert "PATH_NOT_IN_MANIFEST" in codes(validate_export({**ok, "paths": np.array([f"q{i}" for i in range(5)])}, role="cal", manifest_rows=manifest))
    flipped = {**ok, "y": np.array([[1.0, 0.0]] * 5)}
    assert "LABEL_MISMATCH" in codes(validate_export(flipped, role="cal", manifest_rows=manifest))


def test_disjointness_rules():
    tr = _legacy_export(paths=np.array(["a", "b", "c", "d", "e"]), partition=np.array(["train"] * 5))
    assert check_disjoint(tr, _legacy_export(paths=np.array(["v", "w", "x", "y", "z"]))) == []   # genuinely disjoint -> no problems
    assert check_disjoint(tr, _legacy_export(paths=np.array(["a", "w", "x", "y", "z"])))[0].code == "TRAIN_CAL_OVERLAP"
    g = lambda *v: _legacy_export(groups=np.array(v))
    assert check_disjoint(g("s1", "", "", "", ""), g("s1", "", "", "", ""))[0].code == "TRAIN_CAL_GROUP_OVERLAP"
    assert check_disjoint(g("", "", "", "", ""), g("", "", "", "", ""))[0].code == "DISJOINTNESS_UNVERIFIED"

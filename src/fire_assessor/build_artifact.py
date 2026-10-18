"""Build artifact bundles. Two stages; neither can produce an APPROVED artifact (approval is human evidence, see artifact.py).

  prepare    checkpoint + provenance record  ->  INCOMPLETE bundle (no calibration / OOD / authenticity values are invented)
  calibrate  base bundle + train/cal exports + explicit POLICY file  ->  NOT_APPROVED bundle (complete, awaiting a human approval record)

The policy file carries the stakeholder decisions (target false-negative rate alpha, confidence delta, uncertainty margin, OOD and
authenticity decisions, who decided and when). A missing or null value stops the build; nothing is defaulted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .artifact import (DEFAULT_ARCHITECTURE, ArtifactIncompatibleError, ArtifactIntegrityError, ModelArtifact)
from .auth_build import AuthBlockedError, AuthPolicyError, check_auth_policy, fit_auth_head
from .auth_data import AuthDataError, build_table
from .authenticity import AuthenticityHead  # noqa: F401  (kept for callers)
from .calibration import calibration_report, fit_temperature, sigmoid
from .checkpoint import ArchitectureSpec, check_compatibility, inspect_checkpoint
from .config import MODEL_VERSION
from .decision import Thresholds
from .features import check_disjoint, validate_export
from .hashing import sha256_file
from .ood import KNNOOD, fit_ood
from .risk_control import select_threshold


class PolicyIncompleteError(Exception):
    def __init__(self, missing):
        super().__init__("policy incomplete, refusing to build: " + "; ".join(missing))
        self.missing = list(missing)


class CalibrationBlockedError(Exception):
    def __init__(self, problems):
        super().__init__("calibration blocked: " + "; ".join(f"{p.code}: {p.message}" for p in problems))
        self.problems = list(problems)


def _fresh_dir(out) -> Path:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty; artifacts are immutable, stage a new directory")
    return out


def prepare_incomplete(weights, provenance_path, out, *, expected_sha256=None, architecture=None, model_version=MODEL_VERSION) -> ModelArtifact:
    prov = json.loads(Path(provenance_path).read_text())
    for k in ("checkpoint", "source", "initialization", "training_manifest"):
        if not prov.get(k):
            raise ValueError(f"provenance record lacks {k!r}")
    want = expected_sha256 or prov["checkpoint"]["sha256"]
    got = sha256_file(weights)
    if got != want:
        raise ArtifactIntegrityError(f"checkpoint SHA-256 {got} != expected {want}")
    arch = architecture or DEFAULT_ARCHITECTURE
    problems = check_compatibility(inspect_checkpoint(weights), ArchitectureSpec.from_dict(arch))
    if problems:
        raise ArtifactIncompatibleError(f"{len(problems)} architecture mismatch(es), first: {problems[0]}")
    why = "no approved calibration policy or independent calibration export is available"
    art = ModelArtifact(
        model_version, checkpoint_sha256=got, architecture=arch,
        calibration={"status": "NOT_PROVIDED", "reason": why},
        ood_info={"enabled": False, "status": "NOT_PROVIDED", "reason": "LoRA-feature OOD bank not generated and no independent OOD evaluation approved"},
        authenticity={"status": "NOT_PROVIDED", "reason": "authenticity representation and label policy are model-owner decisions; the checkpoint has no authenticity head"},
        provenance=prov, metadata={"built_by": "fire_assessor.build_artifact prepare", "note": "INCOMPLETE bundle: weights + contract only"})
    art.save(_fresh_dir(out), weights_src=weights)
    return art


def _require_policy(policy: dict) -> None:
    miss = [k for k in ("alpha", "delta", "margin", "decided_by", "decided_at", "reference") if policy.get(k) in (None, "")]
    ood, auth = policy.get("ood") or {}, policy.get("authenticity") or {}
    if ood.get("mode") not in ("enable", "disable_by_policy"):
        miss.append("ood.mode ('enable' or 'disable_by_policy')")
    elif ood["mode"] == "enable" and any(ood.get(k) is None for k in ("quantile", "min_auroc")):
        miss.append("ood.quantile / ood.min_auroc")
    elif ood["mode"] == "disable_by_policy" and not ood.get("reason"):
        miss.append("ood.reason")
    if auth.get("mode") == "include_head":
        try:
            check_auth_policy(auth)
        except AuthPolicyError as e:
            miss += [f"authenticity.{m}" for m in e.missing]
    elif auth.get("mode") != "exclude_by_policy" or not auth.get("reason"):
        miss.append("authenticity.mode='exclude_by_policy' with a reason, or 'include_head' with the head policy (target_fpr, delta_fpr, inconclusive_fpr, "
                    "l2_grid, seed, negative_sources, label_policy, decided_by, decided_at, reference)")
    if miss:
        raise PolicyIncompleteError(miss)
    if not (0 < policy["alpha"] < 1 and 0 < policy["delta"] < 1 and 0 <= policy["margin"] < 0.5):
        raise PolicyIncompleteError(["alpha/delta must be in (0,1) and margin in [0,0.5)"])


AUTH_LIMITATIONS = [
    "Authenticity head (synthetic / edited indications) is advisory and PROPOSED-policy only: it was fitted on frozen-feature research data (D-Fire-derived "
    "edits, four open-weight generators), never evaluated on citizen phone photos, and detects unseen generators and edit types unreliably.",
    "A false-flag-rate bound for the authenticity head holds only for images exchangeable with the validation images; it is invalid under shift "
    "(measured: real FIgLib control frames are flagged synthetic far above the target rate).",
    "NO_CONCERNS_FOUND from the authenticity layer never means verified real.",
]


def _build_auth_head(base, apol: dict, auth_data):
    """Fit the authenticity head from the feature files named in `auth_data` ({"sources": [...]}) on the TRAIN split only; choose temperature
    and threshold on the validation split. Returns (head, artifact.json `authenticity` block, extra limitation strings)."""
    if not (isinstance(auth_data, dict) and auth_data.get("sources")):
        raise CalibrationBlockedError([type("P", (), {"code": "AUTH_DATA_MISSING", "message": "authenticity.mode 'include_head' needs --auth-data (a JSON spec listing feature files and manifests)"})()])
    try:
        table, prov = build_table(auth_data["sources"])
        dim = (base.architecture.get("block_feature_tap") or {}).get("dim")
        if table["block"].shape[1] != dim:
            raise AuthBlockedError("AUTH_DIM_MISMATCH", f"feature width {table['block'].shape[1]} != architecture block_feature_tap.dim {dim}")
        head, rep = fit_auth_head(table, apol)
    except (AuthDataError, AuthBlockedError, FileNotFoundError, KeyError) as e:
        code = getattr(e, "code", type(e).__name__)
        raise CalibrationBlockedError([type("P", (), {"code": f"AUTH_{code}", "message": getattr(e, "message", str(e))})()]) from e
    except AuthPolicyError as e:
        raise PolicyIncompleteError(e.missing) from e
    keep = ("target_fpr", "delta_fpr", "inconclusive_fpr", "l2_grid", "l2_auc_tolerance", "class_weight", "seed", "negative_sources",
            "max_train_rows_per_source", "decided_by", "decided_at", "reference")
    meta = {
        "status": "PROVIDED", "mode": "include_head", "outputs": ["synthetic", "edited_or_composited"],
        "method": "L2 logistic regression (damped Newton, class-balanced) on the block-8 pre-norm CLS feature (key `block`); per-output temperature (NLL) "
                  "and a threshold chosen on the validation split by an exact one-sided binomial bound on the false-flag rate over independent real groups; "
                  "an INCONCLUSIVE band [band_lo, threshold) from the validation real-group score distribution",
        "policy": {k: apol[k] for k in keep if k in apol},
        "label_policy": apol["label_policy"],
        "feature": {"key": "block", "tap_block_1_indexed": 8, "dim": int(table["block"].shape[1]),
                    "extracted_with": auth_data.get("feature_extraction", "NOT RECORDED"),
                    "note": "head fitted on features of the frozen base model; the pipeline feeds it the LoRA checkpoint's block-8 tap (identical weights in blocks 1-8, "
                            "equal up to numerical noise; see docs/authenticity/AUTH_HEAD_PLAN.md)"},
        "data": {"sources": prov, "split_roles": {"train": "fit", "cal": "L2 selection", "val": "temperature + threshold + band",
                                                    "never_read": ["test", "heldout_gen", "heldout_source", "control"]},
                 "split_rule": "rows are split by source photo / D-Fire cluster; a group never spans train, cal and val (checked, build stops otherwise)"},
        "results": rep,
        "guarantee": "With probability >= 1 - delta_fpr the false-flag rate on REAL images exchangeable with the validation real groups is <= target_fpr at the "
                     "recorded threshold. Not valid under shift (new generator, camera, source, post-processing, phone photos).",
    }
    return head, json.loads(json.dumps(meta, default=_jsonable)), AUTH_LIMITATIONS


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.str_):
        return str(o)
    raise TypeError(type(o))


def calibrate_artifact(base_dir, train_path, cal_path, policy: dict, out, *, identity_summary=None, manifest_rows=None, control_path=None,
                       auth_data=None) -> ModelArtifact:
    _require_policy(policy)
    base = ModelArtifact.load(base_dir, require_operational=False, check_checkpoint=True)
    tr, cal = (dict(np.load(p, allow_pickle=False)) for p in (train_path, cal_path))
    problems = []
    for data, role, ev in ((tr, "train", None), (cal, "cal", identity_summary)):
        rep = validate_export(data, role=role, identity_evidence=ev, manifest_rows=manifest_rows)
        problems += rep.blockers
        if role == "cal":
            cal_rep = rep
        if "checkpoint_sha256" in data and str(data["checkpoint_sha256"]) != base.checkpoint_sha256:
            problems.append(type("P", (), {"code": "CHECKPOINT_MISMATCH", "message": f"{role} export was produced by a different checkpoint"})())
    problems += check_disjoint(tr, cal, cal_identity_evidence=identity_summary)
    problems = [p for p in problems if getattr(p, "blocker", True)]
    if problems:
        raise CalibrationBlockedError(problems)
    T, th, report = {}, {}, {}
    for k, name in enumerate(("flames", "smoke")):
        T[name] = fit_temperature(cal["logits"][:, k], cal["y"][:, k])
        rc = select_threshold(sigmoid(cal["logits"][:, k] / T[name]), cal["y"][:, k], policy["alpha"], policy["delta"])
        if not rc.feasible:
            raise CalibrationBlockedError([type("P", (), {"code": "INFEASIBLE", "message": f"{name}: cannot certify FNR<={policy['alpha']} at delta={policy['delta']} with {rc.n_pos} calibration positives"})()])
        th[name] = rc.threshold
        report[name] = {**calibration_report(cal["logits"][:, k], cal["y"][:, k], T[name]), "risk_control": rc.__dict__}
    ood_pol, ood, ood_info = policy["ood"], None, {"enabled": False}
    if ood_pol["mode"] == "disable_by_policy":
        ood_info = {"enabled": False, "status": "DISABLED_BY_POLICY", "reason": ood_pol["reason"]}
    else:
        if control_path is None:
            raise CalibrationBlockedError([type("P", (), {"code": "OOD_CONTROL_MISSING", "message": "OOD mode 'enable' needs an independent control export (unrelated real images) to measure AUROC"})()])
        ctl = np.load(control_path, allow_pickle=False)
        ood = KNNOOD()
        info = fit_ood(ood, tr["emb"], cal["emb"], ctl["emb"], quantile=ood_pol["quantile"], min_auroc=ood_pol["min_auroc"])
        if info["enabled"]:
            ood_info = {**info, "status": "PROVIDED", "control_export_sha256": sha256_file(control_path)}
        else:
            ood, ood_info = None, {"enabled": False, "status": "DISABLED_BY_POLICY", "reason": f"policy: disable if AUROC < {ood_pol['min_auroc']}; {info['reason']}", "auroc": info["auroc"]}
    auth_head, auth_meta, extra_limits = None, None, []
    if policy["authenticity"]["mode"] == "include_head":
        auth_head, auth_meta, extra_limits = _build_auth_head(base, policy["authenticity"], auth_data)
    new = ModelArtifact(
        base.model_version, Thresholds(th["flames"], th["smoke"]), T, policy["margin"], ood, ood_info, auth_head, {**base.metadata, "calibrated_by": "fire_assessor.build_artifact calibrate"},
        None, base.checkpoint_sha256, base.architecture, base.preprocessing, base.outputs,
        {"status": "PROVIDED", "method": "per-head temperature scaling (NLL, golden-section) + Learn-then-Test fixed-sequence binomial threshold on the cal partition",
         "policy": {k: policy[k] for k in ("alpha", "delta", "margin", "decided_by", "decided_at", "reference")},
         "source": {"partition": "cal", "export_sha256": sha256_file(cal_path), "n_rows": cal_rep.n_rows, "identity_status": cal_rep.identity_status,
                    "train_export_sha256": sha256_file(train_path)},
         "results": report, "guarantee": "holds only if deployment data is exchangeable with the calibration partition; invalid under source shift"},
        auth_meta or {"status": "EXCLUDED_BY_POLICY", "reason": policy["authenticity"]["reason"]}, base.provenance, list(base.limitations) + extra_limits, None)
    new.save(_fresh_dir(out), weights_src=base.weights_path)
    return new


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--weights", required=True); p.add_argument("--provenance", required=True); p.add_argument("--out", required=True)
    p.add_argument("--expected-sha256")
    c = sub.add_parser("calibrate"); c.add_argument("--base", required=True); c.add_argument("--train", required=True); c.add_argument("--cal", required=True)
    c.add_argument("--policy", required=True); c.add_argument("--identity-summary"); c.add_argument("--control"); c.add_argument("--auth-data"); c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        art = prepare_incomplete(a.weights, a.provenance, a.out, expected_sha256=a.expected_sha256)
    else:
        ev = json.loads(Path(a.identity_summary).read_text()) if a.identity_summary else None
        if ev is not None:
            ev["method"] = ev.get("method", "content_verification")
        art = calibrate_artifact(a.base, a.train, a.cal, json.loads(Path(a.policy).read_text()), a.out, identity_summary=ev, control_path=a.control,
                              auth_data=json.loads(Path(a.auth_data).read_text()) if a.auth_data else None)
    print(json.dumps({"out": str(a.out), "status": art.status, "blockers": art.blockers()}, indent=2))


if __name__ == "__main__":
    main()

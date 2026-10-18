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
    if auth.get("mode") != "exclude_by_policy" or not auth.get("reason"):
        miss.append("authenticity.mode='exclude_by_policy' with a reason (training an authenticity head needs an approved label policy and is not done here)")
    if miss:
        raise PolicyIncompleteError(miss)
    if not (0 < policy["alpha"] < 1 and 0 < policy["delta"] < 1 and 0 <= policy["margin"] < 0.5):
        raise PolicyIncompleteError(["alpha/delta must be in (0,1) and margin in [0,0.5)"])


def calibrate_artifact(base_dir, train_path, cal_path, policy: dict, out, *, identity_summary=None, manifest_rows=None, control_path=None) -> ModelArtifact:
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
    new = ModelArtifact(
        base.model_version, Thresholds(th["flames"], th["smoke"]), T, policy["margin"], ood, ood_info, None, {**base.metadata, "calibrated_by": "fire_assessor.build_artifact calibrate"},
        None, base.checkpoint_sha256, base.architecture, base.preprocessing, base.outputs,
        {"status": "PROVIDED", "method": "per-head temperature scaling (NLL, golden-section) + Learn-then-Test fixed-sequence binomial threshold on the cal partition",
         "policy": {k: policy[k] for k in ("alpha", "delta", "margin", "decided_by", "decided_at", "reference")},
         "source": {"partition": "cal", "export_sha256": sha256_file(cal_path), "n_rows": cal_rep.n_rows, "identity_status": cal_rep.identity_status,
                    "train_export_sha256": sha256_file(train_path)},
         "results": report, "guarantee": "holds only if deployment data is exchangeable with the calibration partition; invalid under source shift"},
        {"status": "EXCLUDED_BY_POLICY", "reason": policy["authenticity"]["reason"]}, base.provenance, base.limitations, None)
    new.save(_fresh_dir(out), weights_src=base.weights_path)
    return new


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--weights", required=True); p.add_argument("--provenance", required=True); p.add_argument("--out", required=True)
    p.add_argument("--expected-sha256")
    c = sub.add_parser("calibrate"); c.add_argument("--base", required=True); c.add_argument("--train", required=True); c.add_argument("--cal", required=True)
    c.add_argument("--policy", required=True); c.add_argument("--identity-summary"); c.add_argument("--control"); c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        art = prepare_incomplete(a.weights, a.provenance, a.out, expected_sha256=a.expected_sha256)
    else:
        ev = json.loads(Path(a.identity_summary).read_text()) if a.identity_summary else None
        if ev is not None:
            ev["method"] = ev.get("method", "content_verification")
        art = calibrate_artifact(a.base, a.train, a.cal, json.loads(Path(a.policy).read_text()), a.out, identity_summary=ev, control_path=a.control)
    print(json.dumps({"out": str(a.out), "status": art.status, "blockers": art.blockers()}, indent=2))


if __name__ == "__main__":
    main()

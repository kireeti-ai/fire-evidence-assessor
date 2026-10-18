"""Read-only validation of a calibrated (NOT_APPROVED) artifact against path-preserving exports.

Applies the artifact's frozen temperatures and thresholds to held-out partitions and reports discrimination, calibration and
operating-point behaviour with grouped (cluster) bootstrap intervals. It tunes nothing. The evidence goes to the human approver.

Partition rules (docs/calibration/CALIBRATION_AND_EVALUATION_READINESS.md):
  cal       calibration source (reported for reference; in-sample for the thresholds)
  val       used for checkpoint selection (reported, not an independent estimate)
  test      one-time in-distribution report: every run is appended to the access log in --out-dir
  xsrc_aof  held-out source: shift evaluation only

Usage: python scripts/validate_artifact.py --artifact DIR --exports EXPORT_DIR --out-dir docs/validation
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fire_assessor.artifact import ModelArtifact  # noqa: E402
from fire_assessor.calibration import brier, ece, sigmoid  # noqa: E402
from fire_assessor.decision import decide  # noqa: E402
from fire_assessor.config import PolicyConfig  # noqa: E402
from fire_assessor.hashing import sha256_file  # noqa: E402
from fire_assessor.metrics import grouped_bootstrap_ci, pr_auc, prf, roc_auc  # noqa: E402

HEADS = ("flames", "smoke")
PARTITIONS = ("cal", "val", "test", "xsrc_aof")


def _groups(d):
    g = np.asarray(d["groups"]).astype(str) if "groups" in d else np.array([""] * len(d["y"]))
    return np.array([x if x else f"_row{i}" for i, x in enumerate(g)])


def _ci(fn, y, s, g, n_boot):
    lo, hi = grouped_bootstrap_ci(fn, y, s, g, n_boot=n_boot)
    return [round(lo, 4), round(hi, 4)]


def evaluate(art: ModelArtifact, d: dict, n_boot: int) -> dict:
    out = {"n": int(len(d["y"])), "n_groups": int(len(set(_groups(d)))), "heads": {}}
    g = _groups(d)
    for k, name in enumerate(HEADS):
        y = d["y"][:, k].astype(int)
        logit = d["logits"][:, k].astype(float)
        T = art.temperatures[name]
        p = sigmoid(logit / T)
        th = getattr(art.thresholds, name)
        flagged = (p >= th).astype(int)
        fnr_fn = lambda yy, ss, th=th: prf(yy, (ss >= th).astype(int))["fnr"]
        prec_fn = lambda yy, ss, th=th: prf(yy, (ss >= th).astype(int))["precision"]
        m = prf(y, flagged)
        out["heads"][name] = {
            "prevalence": round(float(y.mean()), 4),
            "roc_auc": round(roc_auc(y, p), 4), "roc_auc_ci95": _ci(roc_auc, y, p, g, n_boot),
            "pr_auc": round(pr_auc(y, p), 4), "pr_auc_ci95": _ci(pr_auc, y, p, g, n_boot),
            "temperature": T, "threshold": round(float(th), 6),
            "ece_raw": round(ece(sigmoid(logit), y), 4), "ece_calibrated": round(ece(p, y), 4),
            "brier_raw": round(brier(sigmoid(logit), y), 4), "brier_calibrated": round(brier(p, y), 4),
            "fnr": round(m["fnr"], 4), "fnr_ci95": _ci(fnr_fn, y, p, g, n_boot),
            "precision": round(m["precision"], 4), "precision_ci95": _ci(prec_fn, y, p, g, n_boot),
            "flagged_share": round(float(flagged.mean()), 4), "confusion": {k2: m[k2] for k2 in ("tp", "fp", "tn", "fn")},
        }
    labels = {}
    for i in range(len(d["y"])):
        pf = sigmoid(d["logits"][i, 0] / art.temperatures["flames"])
        ps = sigmoid(d["logits"][i, 1] / art.temperatures["smoke"])
        r = decide(float(pf), float(ps), art.thresholds, [], False, PolicyConfig(uncertainty_margin=art.uncertainty_margin))
        labels[r.derived_label] = labels.get(r.derived_label, 0) + 1
    out["derived_label_counts"] = labels
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", required=True, type=Path)
    ap.add_argument("--exports", required=True, type=Path, help="directory with export_<partition>.npz")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--partitions", default=",".join(PARTITIONS))
    a = ap.parse_args(argv)

    art = ModelArtifact.load(a.artifact, require_operational=False, check_checkpoint=True)
    if art.thresholds is None:
        raise SystemExit("artifact has no calibration; run build_artifact calibrate first")
    digest = sha256_file(Path(a.artifact) / "artifact.json")
    report = {"artifact": str(a.artifact), "artifact_json_sha256": digest, "model_version": art.model_version,
              "artifact_status": art.status, "checkpoint_sha256": art.checkpoint_sha256,
              "uncertainty_margin": art.uncertainty_margin,
              "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "partitions": {}}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for part in [p for p in a.partitions.split(",") if p]:
        f = a.exports / f"export_{part}.npz"
        if not f.exists():
            report["partitions"][part] = {"error": "export missing"}
            continue
        d = dict(np.load(f, allow_pickle=False))
        problems = []
        if "checkpoint_sha256" in d and str(d["checkpoint_sha256"]) != art.checkpoint_sha256:
            problems.append("export was produced by a different checkpoint")
        if "partition" in d and {str(v) for v in d["partition"].tolist()} != {part}:
            problems.append("export partition field does not match its name")
        if problems:
            report["partitions"][part] = {"error": "; ".join(problems)}
            continue
        res = evaluate(art, d, a.n_boot)
        res["export_sha256"] = sha256_file(f)
        res["role"] = {"cal": "calibration source (in-sample for thresholds)", "val": "checkpoint-selection partition (not independent)",
                       "test": "one-time in-distribution report", "xsrc_aof": "held-out source: shift evaluation"}[part]
        report["partitions"][part] = res
        if part == "test":
            with open(a.out_dir / "test_access_log.md", "a") as log:
                log.write(f"- {report['generated_at']}  test export sha256 {res['export_sha256']}  artifact.json sha256 {digest}  purpose: one-time in-distribution report\n")
    (a.out_dir / "validation_report.json").write_text(json.dumps(report, indent=2))

    lines = ["# Artifact validation report", "",
             f"Artifact `{art.model_version}` status **{art.status}** (checkpoint `{art.checkpoint_sha256[:12]}...`, artifact.json `{digest[:12]}...`).",
             "Thresholds and temperatures were frozen before this run; nothing here tunes them. Intervals are 95% grouped (cluster) bootstrap.", ""]
    for part, res in report["partitions"].items():
        lines += [f"## {part}", ""]
        if "error" in res:
            lines += [f"Not evaluated: {res['error']}", ""]
            continue
        lines += [f"{res['role']}. n={res['n']}, groups={res['n_groups']}.", "",
                  "| Head | prevalence | ROC-AUC (CI) | PR-AUC (CI) | ECE raw to cal | FNR (CI) | precision (CI) | flagged |", "|---|---|---|---|---|---|---|---|"]
        for name, h in res["heads"].items():
            lines.append(f"| {name} | {h['prevalence']} | {h['roc_auc']} {h['roc_auc_ci95']} | {h['pr_auc']} {h['pr_auc_ci95']} | "
                         f"{h['ece_raw']} to {h['ece_calibrated']} | {h['fnr']} {h['fnr_ci95']} | {h['precision']} {h['precision_ci95']} | {h['flagged_share']} |")
        lines += ["", f"Derived labels (no quality or OOD gating): {res['derived_label_counts']}", ""]
    (a.out_dir / "validation_report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

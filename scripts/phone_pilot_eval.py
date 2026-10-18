"""Phone-photo pilot: annotation agreement, consensus manifest, and per-policy slice report.

  manifest : two annotator CSVs + candidates.csv -> agreement report and phone_manifest.csv (input to `train export`)
  report   : export_phone.npz + phone_manifest.csv + policy artifacts -> miss / false-alarm tables by slice

Annotator CSV columns: file, flames, smoke, context, quality, unsure, reason. Candidates CSV (from collect_phone_pilot.py) carries
file, make, model, license, author, title, group. Primary evaluation uses only images where both annotators agree on flames and
smoke, neither marked unsure, and neither marked quality=poor; every exclusion is counted.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def kappa(a, b):
    a, b = np.asarray(a).astype(int), np.asarray(b).astype(int)
    po = float((a == b).mean())
    pe = float(sum((a == v).mean() * (b == v).mean() for v in (0, 1)))
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def read(path):
    with open(path, newline="") as f:
        return {r["file"]: r for r in csv.DictReader(f)}


def cmd_manifest(a):
    ann1, ann2 = read(a.annotator1), read(a.annotator2)
    cands = read(a.candidates)
    files = sorted(set(ann1) & set(ann2) & set(cands))
    agree_stats, rows, excluded = {}, [], {"disagree_flames": 0, "disagree_smoke": 0, "unsure": 0, "poor_quality": 0}
    for fld in ("flames", "smoke"):
        agree_stats[fld] = {"kappa": round(kappa([int(ann1[f][fld]) for f in files], [int(ann2[f][fld]) for f in files]), 3),
                            "percent_agreement": round(float(np.mean([ann1[f][fld] == ann2[f][fld] for f in files])), 3)}
    agree_stats["context"] = {"percent_agreement": round(float(np.mean([ann1[f]["context"] == ann2[f]["context"] for f in files])), 3)}
    for f in files:
        x, y, c = ann1[f], ann2[f], cands[f]
        if x["flames"] != y["flames"]:
            excluded["disagree_flames"] += 1
            continue
        if x["smoke"] != y["smoke"]:
            excluded["disagree_smoke"] += 1
            continue
        if x["unsure"] == "1" or y["unsure"] == "1":
            excluded["unsure"] += 1
            continue
        if "poor" in (x["quality"], y["quality"]):
            excluded["poor_quality"] += 1
            continue
        ctx = x["context"] if x["context"] == y["context"] else "disputed"
        rows.append({"path": f"img/{f}", "flames": x["flames"], "smoke": x["smoke"], "partition": "phone", "group": c.get("group", ""),
                     "source": f"{c.get('make', '')}|{c.get('model', '')}", "context": ctx, "license": c.get("license", ""), "author": c.get("author", ""), "title": c.get("title", "")})
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "phone_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    rep = {"n_annotated_by_both": len(files), "n_in_primary_evaluation": len(rows), "excluded": excluded, "agreement": agree_stats,
           "label_counts": {"flames": int(sum(int(r["flames"]) for r in rows)), "smoke": int(sum(int(r["smoke"]) for r in rows)),
                            "neither": int(sum(1 for r in rows if r["flames"] == "0" and r["smoke"] == "0"))},
           "context_counts": {k: sum(1 for r in rows if r["context"] == k) for k in sorted({r["context"] for r in rows})},
           "annotators": "two independent AI reviewers following ANNOTATION_GUIDE.md; NOT human annotators"}
    (out / "annotation_agreement.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


def cmd_report(a):
    from fire_assessor.artifact import ModelArtifact
    from fire_assessor.calibration import sigmoid
    from fire_assessor.metrics import grouped_bootstrap_ci, pr_auc, roc_auc
    from compare_policies import decisions  # noqa: E402

    d = dict(np.load(a.export, allow_pickle=False))
    man = list(csv.DictReader(open(a.manifest, newline="")))
    paths = [str(p) for p in d["paths"].tolist()]
    by = {r["path"]: r for r in man}
    meta = [by[p] for p in paths]
    y = d["y"].astype(int)
    pos = y.sum(axis=1) > 0
    ctx = np.array([m["context"] for m in meta])
    make = np.array([m["source"].split("|")[0].lower() or "unknown" for m in meta])
    groups = np.array([m["group"] or f"_r{i}" for i, m in enumerate(meta)])
    report = {"n": int(len(y)), "n_positive": int(pos.sum()), "n_negative": int((~pos).sum()), "options": {}}
    for item in a.option:
        name, dirp = item.split("=", 1)
        art = ModelArtifact.load(dirp, require_operational=False, check_checkpoint=False)
        lab, pri = decisions(art, d)
        miss = lab == "NO_VISIBLE_FIRE"
        alarm = np.isin(lab, ("ACTIVE_FIRE", "SMOKE_ONLY"))
        high = pri == "HIGH"
        unc = lab == "UNCERTAIN"

        def rate(num, den):
            return None if not den.any() else round(float(num[den].mean()), 4), int(den.sum())

        def boot(num, den):
            def fn(yy, ss, num=num, den=den):  # grouped bootstrap over indices
                idx = ss.astype(int)
                return float(num[idx][den[idx]].mean()) if den[idx].any() else float("nan")
            lo, hi = grouped_bootstrap_ci(fn, np.zeros(len(num)), np.arange(len(num)), groups, n_boot=300)
            return [round(lo, 3), round(hi, 3)]

        o = {"thresholds": {"flames": round(float(art.thresholds.flames), 4), "smoke": round(float(art.thresholds.smoke), 4)}, "margin": art.uncertainty_margin,
             "miss_rate_positive": [*rate(miss, pos), boot(miss, pos)], "false_alarm_rate_negative": [*rate(alarm, ~pos), boot(alarm, ~pos)],
             "high_priority_rate_negative": [*rate(high, ~pos), boot(high, ~pos)], "uncertain_share_all": [*rate(unc, np.ones_like(pos)), boot(unc, np.ones_like(pos))]}
        pf = sigmoid(d["logits"][:, 0] / art.temperatures["flames"])
        ps = sigmoid(d["logits"][:, 1] / art.temperatures["smoke"])
        o["flames_auc"] = round(roc_auc(y[:, 0], pf), 4)
        o["smoke_auc"] = round(roc_auc(y[:, 1], ps), 4)
        o["flames_auc_ci95"] = [round(v, 3) for v in grouped_bootstrap_ci(roc_auc, y[:, 0], pf, groups, n_boot=300)]
        o["smoke_auc_ci95"] = [round(v, 3) for v in grouped_bootstrap_ci(roc_auc, y[:, 1], ps, groups, n_boot=300)]
        o["by_context"] = {c: {"n": int((ctx == c).sum()), "labelled_fire_or_smoke_alarm": rate(alarm, ctx == c)[0], "labelled_NO_VISIBLE_FIRE": rate(miss, ctx == c)[0], "high_priority": rate(high, ctx == c)[0]} for c in sorted(set(ctx))}
        o["by_phone_make"] = {m: {"n": int((make == m).sum()), "miss_rate_positive": rate(miss, pos & (make == m))[0], "false_alarm_negative": rate(alarm, ~pos & (make == m))[0]} for m in sorted(set(make)) if (make == m).sum() >= 8}
        report["options"][name] = o
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phone_pilot_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)[:6000])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--annotator1", required=True); m.add_argument("--annotator2", required=True); m.add_argument("--candidates", required=True); m.add_argument("--out-dir", required=True)
    r = sub.add_parser("report")
    r.add_argument("--export", required=True); r.add_argument("--manifest", required=True); r.add_argument("--option", action="append", required=True); r.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    return cmd_manifest(a) if a.cmd == "manifest" else cmd_report(a)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())

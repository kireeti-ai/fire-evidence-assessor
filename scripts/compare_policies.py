"""Side-by-side trade-offs of calibrated policy options on held-out exports. Read-only; tunes nothing.

For every artifact (policy option) and every partition this applies the artifact's frozen temperatures, thresholds and
uncertainty margin through the real decision function and reports, per fire/smoke ground truth:

  miss        a truly fire/smoke-positive image labelled NO_VISIBLE_FIRE (it would be queued at STANDARD priority)
  false alarm a truly negative image labelled ACTIVE_FIRE or SMOKE_ONLY (HIGH priority)
  uncertain   labelled UNCERTAIN (reviewed with the uncertainty reason; HIGH when above threshold or inside the margin)
  HIGH        review_priority HIGH
Every outcome is REQUIRES_REVIEW: the model never rejects anything; it only orders the queue.

Per-1000-report workload projections combine the class-conditional rates with assumed field prevalences (assumptions, not data).

Usage: python scripts/compare_policies.py --option A=DIR --option B=DIR --exports EXPORT_DIR --out-dir OUT [--extra NAME=export.npz]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fire_assessor.artifact import ModelArtifact  # noqa: E402
from fire_assessor.calibration import sigmoid  # noqa: E402
from fire_assessor.config import PolicyConfig  # noqa: E402
from fire_assessor.decision import decide  # noqa: E402

PARTS = ("cal", "val", "test", "xsrc_aof")
PREVALENCES = (0.05, 0.20, 0.50)


def decisions(art, d):
    cfg = PolicyConfig(uncertainty_margin=art.uncertainty_margin)
    pf = sigmoid(d["logits"][:, 0] / art.temperatures["flames"])
    ps = sigmoid(d["logits"][:, 1] / art.temperatures["smoke"])
    labels, prio = [], []
    for a, b in zip(pf, ps):
        r = decide(float(a), float(b), art.thresholds, [], False, cfg)
        labels.append(r.derived_label)
        prio.append(r.review_priority)
    return np.array(labels), np.array(prio)


def _groups(d):
    g = np.asarray(d["groups"]).astype(str) if "groups" in d else np.array([""] * len(d["y"]))
    return np.array([x if x else f"_row{i}" for i, x in enumerate(g)])


def _rate(mask_num, mask_den):
    return float(mask_num[mask_den].mean()) if mask_den.any() else float("nan")


def _boot(fn, groups, n_boot=500, seed=0):
    ug = np.unique(groups)
    members = {k: np.flatnonzero(groups == k) for k in ug}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.choice(ug, size=len(ug), replace=True)])
        v = fn(idx)
        if v == v:
            vals.append(v)
    return [round(float(np.quantile(vals, 0.025)), 4), round(float(np.quantile(vals, 0.975)), 4)] if vals else [None, None]


def summarise(art, d, n_boot):
    lab, pri = decisions(art, d)
    y = d["y"].astype(int)
    pos = (y.sum(axis=1) > 0)                 # fire or smoke visible
    neg = ~pos
    miss = lab == "NO_VISIBLE_FIRE"
    alarm = np.isin(lab, ("ACTIVE_FIRE", "SMOKE_ONLY"))
    unc = lab == "UNCERTAIN"
    high = pri == "HIGH"
    g = _groups(d)
    out = {"n": int(len(y)), "n_pos": int(pos.sum()), "n_neg": int(neg.sum()), "prevalence": round(float(pos.mean()), 4),
           "label_counts": {k: int((lab == k).sum()) for k in ("ACTIVE_FIRE", "SMOKE_ONLY", "NO_VISIBLE_FIRE", "UNCERTAIN")}}
    rates = {
        "miss_rate_any_positive": (miss, pos), "false_alarm_rate_negative": (alarm, neg),
        "uncertain_share_all": (unc, np.ones_like(pos)), "uncertain_share_positive": (unc, pos), "uncertain_share_negative": (unc, neg),
        "high_priority_share_all": (high, np.ones_like(pos)), "high_priority_rate_positive": (high, pos), "high_priority_rate_negative": (high, neg),
    }
    for name, (num, den) in rates.items():
        out[name] = round(_rate(num, den), 4)
        out[name + "_ci95"] = _boot(lambda idx, num=num, den=den: _rate(num[idx], den[idx]), g, n_boot)
    for k, name in enumerate(("flames", "smoke")):
        flagged = sigmoid(d["logits"][:, k] / art.temperatures[name]) >= getattr(art.thresholds, name)
        yk = y[:, k].astype(bool)
        out[f"{name}_head_fnr"] = round(_rate(~flagged, yk), 4)
        out[f"{name}_head_fnr_ci95"] = _boot(lambda idx, f=flagged, yk=yk: _rate(~f[idx], yk[idx]), g, n_boot)
        out[f"{name}_head_precision"] = round(float(yk[flagged].mean()), 4) if flagged.any() else None
    tpr_high, fpr_high = out["high_priority_rate_positive"], out["high_priority_rate_negative"]
    out["per_1000_reports"] = {
        f"prevalence_{int(p * 100)}pct": {
            "HIGH_priority_items": round(1000 * (p * tpr_high + (1 - p) * fpr_high), 1),
            "of_which_real_fire_or_smoke": round(1000 * p * tpr_high, 1),
            "real_fire_or_smoke_labelled_NO_VISIBLE_FIRE": round(1000 * p * out["miss_rate_any_positive"], 1),
            "UNCERTAIN_items": round(1000 * (p * out["uncertain_share_positive"] + (1 - p) * out["uncertain_share_negative"]), 1),
        } for p in PREVALENCES}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--option", action="append", required=True, help="NAME=ARTIFACT_DIR (repeat)")
    ap.add_argument("--exports", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--extra", action="append", default=[], help="NAME=export.npz (e.g. phone_pilot=...); needs logits, y")
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args(argv)
    sets = {p: a.exports / f"export_{p}.npz" for p in PARTS}
    for item in a.extra:
        n, p = item.split("=", 1)
        sets[n] = Path(p)
    report = {"options": {}, "assumptions": {"prevalences": PREVALENCES,
              "note": "Projections apply class-conditional rates to assumed prevalences; field prevalence is unknown."}}
    for item in a.option:
        name, d = item.split("=", 1)
        art = ModelArtifact.load(d, require_operational=False, check_checkpoint=False)
        res = {"artifact_dir": d, "status": art.status, "thresholds": {"flames": round(float(art.thresholds.flames), 5), "smoke": round(float(art.thresholds.smoke), 5)},
               "temperatures": art.temperatures, "margin": art.uncertainty_margin,
               "policy": art.metadata.get("policy") or (art.calibration_info or {}).get("policy") if hasattr(art, "calibration_info") else None, "partitions": {}}
        for part, f in sets.items():
            if not Path(f).exists():
                res["partitions"][part] = {"error": "missing"}
                continue
            res["partitions"][part] = summarise(art, dict(np.load(f, allow_pickle=False)), a.n_boot)
        report["options"][name] = res
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "policy_comparison.json").write_text(json.dumps(report, indent=2, default=str))
    names = list(report["options"])
    md = ["# Policy options: measured trade-offs", "",
          "Frozen thresholds applied to held-out exports; nothing is tuned here. Rates are class-conditional. 95% grouped bootstrap intervals in brackets.", ""]
    for part in sets:
        md += [f"## {part}", "", "| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        first = report["options"][names[0]]["partitions"].get(part, {})
        if "error" in first:
            md += ["| not available | " + " | ".join("-" for _ in names) + " |", ""]
            continue
        md.append(f"| n (positives / negatives) | " + " | ".join(f"{report['options'][n]['partitions'][part]['n']} ({report['options'][n]['partitions'][part]['n_pos']} / {report['options'][n]['partitions'][part]['n_neg']})" for n in names) + " |")
        for key, label in (("miss_rate_any_positive", "Real fire/smoke labelled NO_VISIBLE_FIRE (miss)"), ("flames_head_fnr", "Flames head FNR"), ("smoke_head_fnr", "Smoke head FNR"),
                           ("false_alarm_rate_negative", "Negatives labelled ACTIVE_FIRE/SMOKE_ONLY (false alarm)"), ("high_priority_rate_negative", "Negatives at HIGH priority"),
                           ("uncertain_share_all", "UNCERTAIN share (all)"), ("high_priority_share_all", "HIGH-priority share (all)"), ("flames_head_precision", "Flames flag precision"), ("smoke_head_precision", "Smoke flag precision")):
            cells = []
            for n in names:
                r = report["options"][n]["partitions"][part]
                ci = r.get(key + "_ci95") or r.get(key.replace("rate_", "rate_") + "_ci95")
                cells.append(f"{r[key]} {ci}" if ci and ci[0] is not None else f"{r[key]}")
            md.append(f"| {label} | " + " | ".join(cells) + " |")
        for p in PREVALENCES:
            k = f"prevalence_{int(p * 100)}pct"
            md.append(f"| Per 1000 reports at {int(p * 100)}% real: HIGH items / real among them / real fire labelled NO_VISIBLE_FIRE / UNCERTAIN | " + " | ".join(
                "{} / {} / {} / {}".format(*[report['options'][n]['partitions'][part]['per_1000_reports'][k][m] for m in ("HIGH_priority_items", "of_which_real_fire_or_smoke", "real_fire_or_smoke_labelled_NO_VISIBLE_FIRE", "UNCERTAIN_items")]) for n in names) + " |")
        md.append("")
    (a.out_dir / "policy_comparison.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

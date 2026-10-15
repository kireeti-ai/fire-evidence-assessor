"""Stage 2/3: authenticity head on frozen CLIP features. Block-8 CLS vs final-layer embedding.

Protocol: train on partition=train, pick L2 on val, pick the operating threshold on val NEGATIVES
(false-flag rate <= 5% on real + benign-edited photos), report on test only. Edit type `cover_region`
is held out from training for the held-out-type result. Bootstrap resamples whole clusters.
"""
import csv, json, sys
import numpy as np
from fire_assessor.authenticity import AuthenticityHead
from fire_assessor.metrics import grouped_bootstrap_ci, pr_auc, roc_auc

F = "/home/user/data/feat"
rows = list(csv.DictReader(open("/home/user/data/authenticity/edits_v1/edit_manifest.csv")))
clean = np.load(f"{F}/edits_clean.npz")
assert list(clean["paths"]) == [r["path"] for r in rows]
y = np.array([int(r["edited_or_composited"]) for r in rows]); part = np.array([r["partition"] for r in rows])
et = np.array([r["edit_type"] for r in rows]); kind = np.array([r["kind"] for r in rows]); cl = np.array([r["cluster"] for r in rows])
HELD = "cover_region"

def fit_eval(key, train_mask, tag):
    X = clean[key]; tr, va = train_mask, part == "val"
    best = None
    for l2 in (1e-3, 1e-2, 1e-1, 1.0):
        h = AuthenticityHead.fit(X[tr], y[tr, None].astype(float), l2=l2)
        a = roc_auc(y[va], h.logits(X[va])[:, 0])
        if best is None or a > best[0]: best = (a, l2, h)
    _, l2, h = best
    sv = h.logits(X[va])[:, 0]; thr = np.quantile(sv[y[va] == 0], 0.95)  # FPR<=5% on val negatives
    out = {"l2": l2, "val_auc": best[0], "threshold_logit": float(thr)}
    te = part == "test"
    def rep(mask, X_, name):
        s = h.logits(X_)[:, 0]
        pos, neg = mask & (y == 1), mask & (y == 0)
        auc = roc_auc(y[mask], s[mask]); lo, hi = grouped_bootstrap_ci(roc_auc, y[mask], s[mask], cl[mask], n_boot=300)
        return {"auc": auc, "auc_ci95": [lo, hi], "pr_auc": pr_auc(y[mask], s[mask]),
                "tpr_at_val_thr": float((s[pos] >= thr).mean()) if pos.any() else None,
                "false_flag_rate_all_neg": float((s[neg] >= thr).mean()),
                "false_flag_orig": float((s[neg & (kind == "original")] >= thr).mean()),
                "false_flag_benign": float((s[neg & (kind == "benign_edit")] >= thr).mean()), "n_pos": int(pos.sum()), "n_neg": int(neg.sum())}
    out["test_all_types"] = rep(te, clean[key], "all")
    out["test_seen_types"] = rep(te & (et != HELD), clean[key], "seen")
    out["test_heldout_type"] = rep(te & ((et == HELD) | (y == 0)), clean[key], "held")
    for d in ("jpeg40", "resize50", "blur2"):
        dg = np.load(f"{F}/edits_test_{d}.npz"); tp = [r["path"] for r in rows if r["partition"] == "test"]
        assert list(dg["paths"]) == tp
        full = np.zeros_like(clean[key]); full[te] = dg[key]
        out[f"degraded_{d}"] = rep(te, full, d)
    return out

res = {}
for key, name in (("block", "block8_cls"), ("emb", "final_embedding")):
    res[f"{name}__train_all_types"] = fit_eval(key, part == "train", name)
    res[f"{name}__train_without_{HELD}"] = fit_eval(key, (part == "train") & (et != HELD), name)
json.dump(res, open("/home/user/fire-evidence-assessor/docs/data/authenticity_edit_baseline.json", "w"), indent=1)
for k, v in res.items():
    print(k)
    for s in ("test_all_types", "test_seen_types", "test_heldout_type", "degraded_jpeg40", "degraded_resize50", "degraded_blur2"):
        r = v[s]; print(f"  {s:20s} auc {r['auc']:.3f} [{r['auc_ci95'][0]:.3f},{r['auc_ci95'][1]:.3f}]  TPR@thr {r['tpr_at_val_thr']:.3f}  false-flag orig {r['false_flag_orig']:.3f} benign {r['false_flag_benign']:.3f}")

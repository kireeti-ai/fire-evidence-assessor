"""Track B baselines on frozen CLIP features: zero-shot prompt contrast and linear probe.
Evaluated on the cluster-split test partition and on the held-out cross-source set (AoF fixed cameras).
CIs resample whole clusters (AoF: sequence blocks). No tuning on test."""
import csv, json
import numpy as np
from fire_assessor.authenticity import AuthenticityHead
from fire_assessor.metrics import grouped_bootstrap_ci, pr_auc, roc_auc

F = "/home/user/data/feat"
rows = list(csv.DictReader(open("/home/user/data/manifests/dfire_manifest.csv")))
d = np.load(f"{F}/dfire_clean.npz"); assert list(d["paths"]) == [r["path"] for r in rows]
emb = d["emb"] / np.linalg.norm(d["emb"], axis=1, keepdims=True)
t = np.load(f"{F}/text_emb.npz")
Y = np.array([[int(r["flames"]), int(r["smoke"])] for r in rows], float)
part = np.array([r["partition_xsrc"] for r in rows]); cl = np.array([r["group"] or r["cluster"] for r in rows]);  # AoF: 25-frame sequence blocks (its cluster is one giant component)
src = np.array([r["source"] for r in rows])
zs = np.stack([emb @ t[f"{k}_pos"] - emb @ t[f"{k}_neg"] for k in ("flames", "smoke")], 1)
res = {"n": {p: int((part == p).sum()) for p in np.unique(part)}}
def rep(S, mask, k):
    y, s = Y[mask, k], S[mask, k]
    lo, hi = grouped_bootstrap_ci(roc_auc, y, s, cl[mask], n_boot=300); plo, phi = grouped_bootstrap_ci(pr_auc, y, s, cl[mask], n_boot=300)
    return {"roc_auc": roc_auc(y, s), "roc_ci95": [lo, hi], "pr_auc": pr_auc(y, s), "pr_ci95": [plo, phi], "prevalence": float(y.mean())}
tr, va = part == "train", part == "val"
best = max(((np.mean([roc_auc(Y[va, k], (h := AuthenticityHead.fit(emb[tr], Y[tr], l2=l2, epochs=400)).logits(emb[va])[:, k]) for k in (0, 1)]), l2, h) for l2 in (1e-3, 1e-2, 1e-1, 1.0)), key=lambda x: x[0])
h = best[2]; lp = h.logits(emb); res["linear_probe_l2"] = best[1]
for name, S in (("zero_shot", zs), ("linear_probe", lp)):
    res[name] = {}
    for evalset, mask in (("test", part == "test"), ("xsrc_aof", part == "xsrc_aof")):
        res[name][evalset] = {("flames", "smoke")[k]: rep(S, mask, k) for k in (0, 1)}
    # per original source slice on test + aof
    res[name]["by_source_test"] = {s: {("flames", "smoke")[k]: roc_auc(Y[(part == "test") & (src == s), k], S[(part == "test") & (src == s), k]) for k in (0, 1)} for s in np.unique(src[part == "test"])}
json.dump(res, open("/home/user/fire-evidence-assessor/docs/data/fire_frozen_baselines.json", "w"), indent=1)
for name in ("zero_shot", "linear_probe"):
    for ev in ("test", "xsrc_aof"):
        for k in ("flames", "smoke"):
            r = res[name][ev][k]; print(f"{name:13s} {ev:9s} {k:6s} ROC {r['roc_auc']:.3f} [{r['roc_ci95'][0]:.3f},{r['roc_ci95'][1]:.3f}]  PR {r['pr_auc']:.3f} (prev {r['prevalence']:.2f})")

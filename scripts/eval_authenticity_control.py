"""Control for the fire-presence shortcut: negatives and positives restricted to images whose BASE photo
contains fire or smoke, so 'has fire/smoke' carries no label information. Positives = paste edits."""
import csv, json
import numpy as np
from fire_assessor.authenticity import AuthenticityHead
from fire_assessor.metrics import grouped_bootstrap_ci, roc_auc

F = "/home/user/data/feat"
rows = list(csv.DictReader(open("/home/user/data/authenticity/edits_v1/edit_manifest.csv")))
dm = {r["path"]: r for r in csv.DictReader(open("/home/user/data/manifests/dfire_manifest.csv"))}
clean = np.load(f"{F}/edits_clean.npz")
y = np.array([int(r["edited_or_composited"]) for r in rows]); part = np.array([r["partition"] for r in rows])
et = np.array([r["edit_type"] for r in rows]); cl = np.array([r["cluster"] for r in rows])
basefire = np.array([int(dm[r["base"]]["flames"]) or int(dm[r["base"]]["smoke"]) for r in rows])
out = {"base_fire_rate": {"negatives": float(basefire[y == 0].mean()), "paste_positives": float(basefire[(y == 1) & (et != "cover_region")].mean())}}
use = (basefire == 1) & ((y == 0) | (et != "cover_region"))
for key, name in (("block", "block8_cls"), ("emb", "final_embedding")):
    X = clean[key]; tr, va, te = use & (part == "train"), use & (part == "val"), use & (part == "test")
    best = max(((roc_auc(y[va], (h := AuthenticityHead.fit(X[tr], y[tr, None].astype(float), l2=l2)).logits(X[va])[:, 0]), l2, h) for l2 in (1e-3, 1e-2, 1e-1, 1.0)), key=lambda t: t[0])
    h = best[2]; s = h.logits(X[te])[:, 0]
    lo, hi = grouped_bootstrap_ci(roc_auc, y[te], s, cl[te], n_boot=300)
    out[name] = {"auc": roc_auc(y[te], s), "ci95": [lo, hi], "n_test": int(te.sum()), "n_pos": int(y[te].sum()), "l2": best[1]}
    print(name, out[name])
print(out["base_fire_rate"])
json.dump(out, open("/home/user/fire-evidence-assessor/docs/data/authenticity_edit_control.json", "w"), indent=1)

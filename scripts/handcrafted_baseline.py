"""Stage 2 non-learned forensic baseline: noise residual, tile inconsistency, blockiness, error level analysis.
Features -> standardised logistic regression, same protocol as eval_authenticity.py."""
import csv, io, json, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageFilter
from fire_assessor.authenticity import AuthenticityHead
from fire_assessor.metrics import grouped_bootstrap_ci, roc_auc

def feats(path):
    im = Image.open(path).convert("RGB"); a = np.asarray(im, dtype=np.float64)
    g = np.asarray(im.convert("L"), dtype=np.float64)
    res = g - np.asarray(im.convert("L").filter(ImageFilter.GaussianBlur(1.5)), dtype=np.float64)
    h, w = g.shape; th, tw = h // 4, w // 4
    tiles = np.array([res[i*th:(i+1)*th, j*tw:(j+1)*tw].std() for i in range(4) for j in range(4)])
    k = ((res - res.mean()) ** 4).mean() / (res.var() ** 2 + 1e-9)
    col_b = np.abs(np.diff(g, axis=1))[:, 7::8].mean() / (np.abs(np.diff(g, axis=1)).mean() + 1e-9)
    row_b = np.abs(np.diff(g, axis=0))[7::8, :].mean() / (np.abs(np.diff(g, axis=0)).mean() + 1e-9)
    b = io.BytesIO(); im.save(b, "JPEG", quality=90); b.seek(0)
    ela = np.abs(a - np.asarray(Image.open(b).convert("RGB"), dtype=np.float64)).mean(axis=2)
    et = np.array([ela[i*th:(i+1)*th, j*tw:(j+1)*tw].mean() for i in range(4) for j in range(4)])
    sat = a.max(2) - a.min(2)
    return [res.std(), np.log(res.var() + 1e-6), k, tiles.std() / (tiles.mean() + 1e-9), tiles.max() / (tiles.min() + 1e-9),
            col_b, row_b, ela.mean(), ela.std(), et.std() / (et.mean() + 1e-9), et.max() / (et.min() + 1e-9),
            sat.mean(), sat.std(), g.mean(), g.std(), np.log(h * w)]

if __name__ == "__main__":
    base = "/home/user/data/authenticity/edits_v1"
    rows = list(csv.DictReader(open(f"{base}/edit_manifest.csv")))
    with Pool(4) as p: X = np.array(p.map(feats, [f"{base}/{r['path']}" for r in rows], chunksize=32))
    np.save("/home/user/data/feat/edits_handcrafted.npy", X)
    y = np.array([int(r["edited_or_composited"]) for r in rows]); part = np.array([r["partition"] for r in rows])
    et = np.array([r["edit_type"] for r in rows]); cl = np.array([r["cluster"] for r in rows]); kind = np.array([r["kind"] for r in rows])
    tr, va, te = part == "train", part == "val", part == "test"
    best = max(((roc_auc(y[va], (h := AuthenticityHead.fit(X[tr], y[tr, None].astype(float), l2=l2)).logits(X[va])[:, 0]), l2, h) for l2 in (1e-3, 1e-2, 1e-1, 1.0)), key=lambda t: t[0])
    h = best[2]; s = h.logits(X)[:, 0]; out = {"l2": best[1], "val_auc": best[0]}
    for name, m in (("test_all", te), ("test_seen_types", te & (et != "cover_region")), ("test_cover_region_vs_neg", te & ((et == "cover_region") | (y == 0)))):
        lo, hi = grouped_bootstrap_ci(roc_auc, y[m], s[m], cl[m], n_boot=300); out[name] = {"auc": roc_auc(y[m], s[m]), "ci95": [lo, hi]}
    print(json.dumps(out, indent=1)); json.dump(out, open("/home/user/fire-evidence-assessor/docs/data/authenticity_handcrafted_baseline.json", "w"), indent=1)

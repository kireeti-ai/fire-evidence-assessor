"""Stage 3: real vs AI-generated images, per generator family, held-out generators, degradations; plus how easily the
fire/smoke linear probe is fooled by generated fire images. Train: real vs sd15+sdxl (train partition). Held out entirely:
pixart, kolors. Threshold: 5% false flags on real val+cal images (test untouched). CIs resample whole clusters."""
import csv, json
import numpy as np
from fire_assessor.authenticity import AuthenticityHead
from fire_assessor.metrics import grouped_bootstrap_ci, roc_auc

F = "/home/user/data/feat"
rows = list(csv.DictReader(open(f"{F}/synth_manifest.csv")))
y = np.array([int(r["label_synthetic"]) for r in rows]); part = np.array([r["partition"] for r in rows]); fam = np.array([r["family"] for r in rows])
cl = np.array([r["cluster"] for r in rows]); flames = np.array([int(r["flames"]) for r in rows]); smoke = np.array([int(r["smoke"]) for r in rows])
feats = {d: np.load(f"{F}/synth_{d}.npz") for d in ("clean", "jpeg40", "resize50", "blur2")}
assert list(feats["clean"]["paths"]) == [r["path"] for r in rows]
real_te = (part == "test") & (y == 0)
sets = {"seen generators (sd15+sdxl)": real_te | ((part == "test") & np.isin(fam, ["sd15", "sdxl"])),
        "held-out generators (pixart+kolors)": real_te | (part == "heldout_gen")}
res = {}
for key, nm in (("block", "block8_cls"), ("emb", "final_embedding")):
    X = feats["clean"][key]; tr, va = part == "train", part == "val"
    best = max(((roc_auc(y[va], (h := AuthenticityHead.fit(X[tr], y[tr, None].astype(float), l2=l2)).logits(X[va])[:, 0]), l2, h) for l2 in (1e-3, 1e-2, 1e-1, 1.0)), key=lambda t: t[0])
    h = best[2]; thrmask = np.isin(part, ["val", "cal"]) & (y == 0)
    thr = float(np.quantile(h.logits(X[thrmask])[:, 0], 0.95)); out = {"l2": best[1], "n_real_for_threshold": int(thrmask.sum())}
    for d, f in feats.items():
        s = h.logits(f[key])[:, 0]; o = {}
        for sn, m in sets.items():
            lo, hi = grouped_bootstrap_ci(roc_auc, y[m], s[m], cl[m], n_boot=300)
            o[sn] = {"auc": roc_auc(y[m], s[m]), "auc_ci95": [lo, hi], "false_flag_real": float((s[m & (y == 0)] >= thr).mean())}
        o["catch_rate_by_family"] = {g: float((s[(fam == g) & ((part == "test") | (part == "heldout_gen"))] >= thr).mean()) for g in ("sd15", "sdxl", "pixart", "kolors")}
        out[d] = o
    # control: real photos from sources never seen in training (FIgLib, AoF). A good artifact detector leaves these unflagged.
    ctl = np.load(f"{F}/control_clean.npz"); cs = np.array([r["source"] for r in csv.DictReader(open("/home/user/data/authenticity/control_set/control_manifest.csv"))])
    sc = h.logits(ctl[key])[:, 0]; out["control_real_other_sources_false_flag"] = {s_: float((sc[cs == s_] >= thr).mean()) for s_ in np.unique(cs)}
    out["threshold_fpr_on_real_dfire_val_cal"] = 0.05
    res[nm] = out
# fire-head exploitability: D-Fire linear probe applied to generated images
dm = list(csv.DictReader(open("/home/user/data/manifests/dfire_manifest.csv"))); dd = np.load(f"{F}/dfire_clean.npz")
Y = np.array([[int(r["flames"]), int(r["smoke"])] for r in dm], float); pt = np.array([r["partition_xsrc"] for r in dm])
n = lambda e: e / np.linalg.norm(e, axis=1, keepdims=True)
E = n(dd["emb"]); probe = AuthenticityHead.fit(E[pt == "train"], Y[pt == "train"], l2=1e-2, epochs=400)
S = probe.logits(E); Sg = probe.logits(n(feats["clean"]["emb"]))
ex = {}
for k, h_ in ((0, "flames"), (1, "smoke")):
    thr = np.quantile(S[(pt == "val") & (Y[:, k] == 0), k], 0.95)   # 5% false positives on real D-Fire negatives
    lab = (flames if k == 0 else smoke); m_syn = (y == 1)
    ex[h_] = {"auc_on_generated": roc_auc(lab[m_syn], Sg[m_syn, k]), "generated_positive_flagged": float((Sg[m_syn & (lab == 1), k] >= thr).mean()),
              "generated_negative_flagged": float((Sg[m_syn & (lab == 0), k] >= thr).mean()), "real_dfire_test_positive_flagged": float((S[(pt == "test") & (Y[:, k] == 1), k] >= thr).mean()),
              "n_generated_pos": int((m_syn & (lab == 1)).sum()), "n_generated_neg": int((m_syn & (lab == 0)).sum())}
    tags = np.array([r["tag"] for r in rows]); ex[h_]["flagged_by_hard_negative_tag"] = {t: float((Sg[m_syn & (tags == t) & (lab == 0), k] >= thr).mean()) for t in np.unique(tags[m_syn & (lab == 0)]) if t and (m_syn & (tags == t) & (lab == 0)).sum() >= 10}
res["fire_head_on_generated_images"] = ex
json.dump(res, open("/home/user/fire-evidence-assessor/docs/data/authenticity_synthetic.json", "w"), indent=1)
for nm in ("block8_cls", "final_embedding"):
    print(nm, "control (real, unseen sources) false-flag rate:", res[nm]["control_real_other_sources_false_flag"])
    for d in ("clean", "jpeg40", "resize50", "blur2"):
        o = res[nm][d]; print(f"{nm:16s} {d:9s} " + " | ".join(f"{k.split(' ')[0]} {k.split('(')[0].split()[1]}: AUC {v['auc']:.3f} [{v['auc_ci95'][0]:.2f},{v['auc_ci95'][1]:.2f}] ff {v['false_flag_real']:.2f}" for k, v in o.items() if isinstance(v, dict) and "auc" in v), {g: round(c, 2) for g, c in o["catch_rate_by_family"].items()})
print(json.dumps(ex, indent=1))

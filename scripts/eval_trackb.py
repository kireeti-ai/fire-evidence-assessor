"""Track B evaluation from exported logits: LoRA (3 seeds) vs ResNet-50 / EfficientNet-B0 (3 seeds), WiSE-FT sweep,
calibration + risk-controlled thresholds, and a check of the guarantee under source shift (AoF).
Sets: test (cluster split, same sources) and xsrc_aof (held-out fixed-camera source). Seeds: mean +- sd."""
import csv, glob, json, os, sys
import numpy as np
from fire_assessor.calibration import ece, fit_temperature, sigmoid, brier
from fire_assessor.metrics import grouped_bootstrap_ci, pr_auc, prf, roc_auc
from fire_assessor.risk_control import select_threshold

RUNS = sys.argv[1] if len(sys.argv) > 1 else "/home/user/data/runs"
rows = list(csv.DictReader(open("/home/user/data/manifests/dfire_manifest.csv")))
byp = {p: [r for r in rows if r["partition_xsrc"] == p] for p in ("val", "cal", "test", "xsrc_aof")}
grp = {p: np.array([r["group"] or r["cluster"] for r in byp[p]]) for p in byp}
H = ("flames", "smoke")

def load_lora(d, sub=""):
    return {p: np.load(f"{d}/{sub}{p}.npz") for p in ("val", "cal", "test", "xsrc_aof") if os.path.exists(f"{d}/{sub}{p}.npz")}
def metrics(S, Y, p):
    return {H[k]: {"roc": roc_auc(Y[:, k], S[:, k]), "pr": pr_auc(Y[:, k], S[:, k])} for k in (0, 1)}

res = {"methods": {}, "wise": {}, "risk_control": {}}
dirs_ = lambda pat: sorted(d for d in glob.glob(f"{RUNS}/{pat}") if os.path.isdir(d))  # directories only (skip *.log)
methods = {"lora": [d for d in dirs_("lora_s*") if os.path.exists(f"{d}/test.npz")], "resnet50": [d for d in dirs_("resnet50_s*") if os.path.exists(f"{d}/preds.npz")], "efficientnet_b0": [d for d in dirs_("efficientnet_b0_s*") if os.path.exists(f"{d}/preds.npz")]}
for name, dirs in methods.items():
    per = {p: [] for p in ("val", "test", "xsrc_aof")}
    for d in dirs:
        if name == "lora": L = load_lora(d); get = lambda p: (L[p]["logits"], L[p]["y"]) if p in L else None
        else:
            f = f"{d}/preds.npz"; get = (lambda f: (lambda p: (np.load(f)[f"{p}_logits"], np.load(f)[f"{p}_y"]) if os.path.exists(f) else None))(f)
        for p in per:
            g = get(p)
            if g is not None: per[p].append(metrics(g[0], g[1], p))
    res["methods"][name] = {"n_seeds": len(dirs)}
    for p, ms in per.items():
        if not ms: continue
        res["methods"][name][p] = {h: {m: [float(np.mean([x[h][m] for x in ms])), float(np.std([x[h][m] for x in ms]))] for m in ("roc", "pr")} for h in H}

# WiSE-FT sweep (seed 0): pick alpha on VAL; report val / test / xsrc_aof for each alpha
for d in sorted(glob.glob(f"{RUNS}/lora_s0/wise_*")):
    if os.path.isdir(d):
        a = os.path.basename(d).split("_")[1]; L = load_lora(d)
        res["wise"][a] = {p: metrics(L[p]["logits"], L[p]["y"], p) for p in L}

# calibration + risk control on LoRA seed 0 (alpha=1): fit on cal, evaluate on test and on shifted xsrc_aof
FNR_TARGET, DELTA = 0.05, 0.1
L = load_lora(f"{RUNS}/lora_s0")
if "cal" in L:
    for k, h in enumerate(H):
        T = fit_temperature(L["cal"]["logits"][:, k], L["cal"]["y"][:, k])
        pc = sigmoid(L["cal"]["logits"][:, k] / T); rc = select_threshold(pc, L["cal"]["y"][:, k], FNR_TARGET, DELTA)
        out = {"temperature": T, "threshold": rc.threshold, "target_fnr": FNR_TARGET, "delta": DELTA, "feasible": rc.feasible, "cal_empirical_fnr": rc.empirical_fnr}
        for p in ("test", "xsrc_aof"):
            if p in L:
                s, y = sigmoid(L[p]["logits"][:, k] / T), L[p]["y"][:, k]; m = prf(y, (s >= rc.threshold).astype(int))
                out[p] = {"fnr": m["fnr"], "recall": m["recall"], "precision": m["precision"], "review_fraction": float((s >= rc.threshold).mean()),
                          "ece_after": ece(s, y), "ece_before": ece(sigmoid(L[p]["logits"][:, k]), y), "brier_after": brier(s, y)}
        res["risk_control"][h] = out
json.dump(res, open("/home/user/fire-evidence-assessor/docs/data/trackb_results.json", "w"), indent=1, default=float)
for n, v in res["methods"].items():
    for p in ("test", "xsrc_aof"):
        if p in v: print(f"{n:16s} {p:9s} " + "  ".join(f"{h}: ROC {v[p][h]['roc'][0]:.3f}+-{v[p][h]['roc'][1]:.3f} PR {v[p][h]['pr'][0]:.3f}" for h in H))
for a, v in sorted(res["wise"].items()): print("wise", a, {p: {h: round(v[p][h]["roc"], 3) for h in H} for p in v})
for h, o in res["risk_control"].items(): print("risk", h, json.dumps(o, default=float)[:600])

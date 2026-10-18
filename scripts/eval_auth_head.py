"""Honest held-out evaluation of the authenticity head (docs/authenticity/AUTH_HEAD_RESULTS.md).

usage: python scripts/eval_auth_head.py --policy docs/calibration/auth_policy.PROPOSED.json --data-spec docs/authenticity/auth_data_spec.EXAMPLE.json \
           --feat-dir /home/user/data/feat --out docs/authenticity/AUTH_HEAD_RESULTS.json [--n-boot 500]

Fits the head EXACTLY as `build_artifact calibrate` does (fire_assessor.auth_build.fit_auth_head: train split fits, cal picks L2, val picks temperature /
threshold / band), then reports on splits that fitting never read: test, held-out generators (pixart, kolors), held-out source (AoF), control
(FIgLib + AoF frames), and the jpeg40 / blur2 / resize50 variants. Bootstrap CIs resample whole groups (source photo / D-Fire cluster).
A second fit without the `cover_region` edit type estimates generalisation to an unseen edit type; a third (sensitivity only) adds raw D-Fire negatives.
CPU, numpy only.
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fire_assessor.auth_build import fit_auth_head
from fire_assessor.auth_data import build_table, load_source
from fire_assessor.authenticity import assess_authenticity
from fire_assessor.exif import ExifInfo
from fire_assessor.metrics import grouped_bootstrap_ci, pr_auc, roc_auc
from fire_assessor.risk_control import binom_upper_bound

NB = 500
REAL_TAGS_TEST = {"synth_real": ("synth_set", "real"), "edit_original": ("edit_set", "original"), "edit_benign": ("edit_set", "benign_edit")}


def ci(fn, y, s, g, seed=0):
    lo, hi = grouped_bootstrap_ci(fn, y, s, g, n_boot=NB, seed=seed)
    return [lo, hi]


def rate(mask_flag, g, seed=0):
    f = np.asarray(mask_flag, float)
    if len(f) == 0:
        return {"n": 0, "rate": None, "ci95": None}
    lo, hi = grouped_bootstrap_ci(lambda y, s: float(np.mean(s)), np.zeros(len(f)), f, g, n_boot=NB, seed=seed)
    k = int(f.sum())
    return {"n": int(len(f)), "k": k, "rate": k / len(f), "ci95_grouped_bootstrap": [lo, hi],
            "clopper_pearson_upper95_iid": binom_upper_bound(k, len(f), 0.05)}


def status_from(P, head):
    """Vectorised twin of assess_authenticity's head rule (no other signals): returns array of status strings."""
    flag = P >= head.thr
    st = np.full(len(P), "NO_CONCERNS_FOUND", dtype=object)
    inc = ((P >= head.lo) & ~flag).any(1)
    st[inc] = "INCONCLUSIVE"
    st[flag[:, 1]] = "POSSIBLE_EDITED"
    st[flag[:, 0]] = "POSSIBLE_SYNTHETIC"
    return st


def evaluate(head, T, X, label=""):
    """T: table columns for the rows X (features may be degraded variants)."""
    P = head.probs(X)
    Y, sp, src, tag, grp = T["y"], T["split"], T["source"], T["tag"], T["group"]
    out = {"split_rows": {}}
    flag = P >= head.thr
    band = (P >= head.lo) & ~flag
    res = {}
    # ---- discrimination --------------------------------------------------------------------------------------------------------
    def disc(name, j, pos_mask, neg_mask):
        m = pos_mask | neg_mask
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return None
        y = pos_mask[m].astype(int); s = P[m, j]; g = grp[m]
        return {"n_pos": int(pos_mask.sum()), "n_neg": int(neg_mask.sum()), "roc_auc": roc_auc(y, s), "roc_auc_ci95": ci(roc_auc, y, s, g),
                "pr_auc": pr_auc(y, s), "pr_auc_ci95": ci(pr_auc, y, s, g), "prevalence_in_eval_set": float(y.mean())}
    syn_real_test = (sp == "test") & (src == "synth_set") & (Y[:, 0] == 0)
    res["synthetic__test_seen_generators"] = disc("syn", 0, (sp == "test") & (Y[:, 0] == 1), syn_real_test)
    res["synthetic__heldout_generators"] = disc("syn", 0, (sp == "heldout_gen") & (Y[:, 0] == 1), syn_real_test)
    for gname in ("pixart", "kolors", "sd15", "sdxl"):
        sel = (tag == gname) & np.isin(sp, ("test", "heldout_gen"))
        res[f"synthetic__{gname}"] = disc("syn", 0, sel, syn_real_test)
    edit_neg = (sp == "test") & (src == "edit_set") & (Y[:, 1] == 0)
    edit_pos = (sp == "test") & (src == "edit_set") & (Y[:, 1] == 1)
    res["edited__test_all_types"] = disc("ed", 1, edit_pos, edit_neg)
    for et in ("paste_flames", "paste_smoke", "cover_region"):
        res[f"edited__{et}"] = disc("ed", 1, edit_pos & (tag == et), edit_neg)
    out["discrimination"] = res
    # ---- operating point: detection rates on positives -------------------------------------------------------------------------
    det = {}
    for gname in ("sd15", "sdxl", "pixart", "kolors"):
        m = (tag == gname) & np.isin(sp, ("test", "heldout_gen"))
        det[f"synthetic_output__{gname}"] = {**rate(flag[m, 0], grp[m]), "inconclusive_band_share": float(band[m, 0].mean()) if m.any() else None,
                                              "heldout_generator": gname in ("pixart", "kolors")}
    for et in ("paste_flames", "paste_smoke", "cover_region"):
        m = edit_pos & (tag == et)
        det[f"edited_output__{et}"] = {**rate(flag[m, 1], grp[m]), "inconclusive_band_share": float(band[m, 1].mean()) if m.any() else None}
    det["edited_output__all_edit_types"] = {**rate(flag[edit_pos, 1], grp[edit_pos])}
    det["edited_output__on_generated_images (undefined label; informational)"] = {**rate(flag[(Y[:, 0] == 1) & np.isin(sp, ("test", "heldout_gen")), 1], grp[(Y[:, 0] == 1) & np.isin(sp, ("test", "heldout_gen"))])}
    out["detection_at_threshold"] = det
    # ---- false-flag rate on REAL images (per real source; the policy target applies to the validation mix, not to these) -------
    ff = {}
    real_sets = {
        "synth_set_real (D-Fire, normalised, test)": syn_real_test,
        "edit_set_original (test)": (sp == "test") & (src == "edit_set") & (tag == "original"),
        "edit_set_benign_edit (test)": (sp == "test") & (src == "edit_set") & (tag == "benign_edit"),
        "dfire_raw_WEB (test)": (sp == "test") & (tag == "dfire_raw_WEB"),
        "dfire_raw_all (test)": (sp == "test") & (src == "dfire"),
        "dfire_raw_AoF held-out source": (sp == "heldout_source"),
        "control_figlib": tag == "control_figlib",
        "control_aof": tag == "control_aof",
        "control_all": sp == "control",
    }
    for name, m in real_sets.items():
        if m.sum():
            ff[name] = {"synthetic_output": {**rate(flag[m, 0], grp[m]), "inconclusive_band_share": float(band[m, 0].mean())},
                        "edited_output": {**rate(flag[m, 1], grp[m]), "inconclusive_band_share": float(band[m, 1].mean())},
                        "any_flag": rate(flag[m].any(1), grp[m]), "any_inconclusive_or_flag": float((flag[m].any(1) | band[m].any(1)).mean())}
    out["false_flag_on_real"] = ff
    # ---- status mix (what the pipeline would report from the head alone) --------------------------------------------------
    st = status_from(P, head)
    mix = {}
    groups_for_mix = {"real_test_mix(synth_real+edit_orig+edit_benign)": syn_real_test | edit_neg, "control_all": sp == "control",
                      "generated_test+heldout": (Y[:, 0] == 1) & np.isin(sp, ("test", "heldout_gen")), "edited_test": edit_pos}
    for name, m in groups_for_mix.items():
        vals, cnt = np.unique(st[m], return_counts=True)
        mix[name] = {"n": int(m.sum()), **{str(v): float(c / m.sum()) for v, c in zip(vals, cnt)}}
    out["status_mix_head_only"] = mix
    return out, P, st


def load_variant(feat_dir, name, rows_expected_paths):
    d = np.load(Path(feat_dir) / name, allow_pickle=False)
    assert list(map(str, d["paths"])) == list(rows_expected_paths), f"{name}: row identity differs from clean features"
    return d["block"].astype(float)


def main():
    global NB
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True); ap.add_argument("--data-spec", required=True); ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args(); NB = a.n_boot
    pol = json.load(open(a.policy))["authenticity"]
    spec = json.load(open(a.data_spec))
    table, prov = build_table(spec["sources"])
    X = table["block"].astype(float)
    head, rep = fit_auth_head(table, pol)
    res = {"policy_used": {k: pol[k] for k in pol if k != "label_policy"}, "label_policy": pol["label_policy"], "data_sources": prov, "fit_report": rep,
           "bootstrap": {"n_boot": NB, "resampling_unit": "group (D-Fire cluster / source photo; generated and control images are singletons)"}}
    res["main_head"], _, _ = evaluate(head, table, X)
    # ---- robustness: same images, degraded before feature extraction, SAME head/threshold ---------------------------------------
    rob = {}
    n_idx = {kind: np.flatnonzero(table["source"] == kind) for kind in ("synth_set", "edit_set")}
    for d in ("jpeg40", "blur2", "resize50"):
        Xd = X.copy()
        # synthetic set: all rows degraded; edit set: only test-partition rows degraded (others stay clean and are not evaluated)
        si = n_idx["synth_set"]
        Xd[si] = load_variant(a.feat_dir, f"synth_{d}.npz", table["paths"][si])
        ei = n_idx["edit_set"]; et = ei[table["split"][ei] == "test"]
        Xd[et] = load_variant(a.feat_dir, f"edits_test_{d}.npz", table["paths"][et])
        sub = np.zeros(len(X), bool); sub[si] = True; sub[et] = True
        T_ = {k: v[sub] for k, v in table.items()}
        r, _, _ = evaluate(head, T_, Xd[sub])
        r = {k: r[k] for k in ("discrimination", "detection_at_threshold", "status_mix_head_only")}
        r["false_flag_on_real"] = {k: v for k, v in evaluate(head, T_, Xd[sub])[0]["false_flag_on_real"].items() if k.startswith(("synth_set", "edit_set"))}
        rob[d] = r
    res["robustness_same_head_same_threshold"] = rob
    # ---- sensitivity: feature noise at the size of the measured LoRA-vs-frozen block-8 difference --------------------------------
    rng = np.random.default_rng(pol["seed"])
    sens = {}
    for rel in (0.0027, 0.0064):  # median / max relative L2 difference measured on 32 train images (docs/authenticity/BLOCK8_EQUIVALENCE.json)
        N = rng.normal(size=X.shape); N *= (rel * np.linalg.norm(X, axis=1) / np.linalg.norm(N, axis=1))[:, None]
        P0, P1 = head.probs(X), head.probs(X + N)
        ev = np.isin(table["split"], ("test", "heldout_gen", "control"))
        z0, z1 = head.logits(X), head.logits(X + N)
        sens[str(rel)] = {"max_abs_logit_change": [float(np.abs(z1 - z0)[ev, j].max()) for j in (0, 1)],
                          "median_abs_logit_change": [float(np.median(np.abs(z1 - z0)[ev, j])) for j in (0, 1)],
                          "flag_flip_rate_on_eval_rows": [float(((P0[ev, j] >= head.thr[j]) != (P1[ev, j] >= head.thr[j])).mean()) for j in (0, 1)]}
    res["noise_sensitivity_isotropic"] = {"note": "isotropic Gaussian noise; the real LoRA-vs-frozen difference is structured, so this is only an order-of-magnitude check", **sens}
    # ---- held-out EDIT TYPE: refit without cover_region (evaluation-only ablation; this head is not what ships) ----------------
    h2, rep2 = fit_auth_head(table, pol, exclude_train_tags=("cover_region",))
    r2, _, _ = evaluate(h2, table, X)
    res["ablation_heldout_edit_type_cover_region"] = {"fit_report_edited": rep2["per_output"]["edited_or_composited"], "l2_chosen": rep2["l2_chosen"],
        "edited_discrimination": {k: v for k, v in r2["discrimination"].items() if k.startswith("edited__")},
        "edited_detection": {k: v for k, v in r2["detection_at_threshold"].items() if k.startswith("edited_output__")},
        "false_flag_edit_set_test": {k: v["edited_output"] for k, v in r2["false_flag_on_real"].items() if k.startswith("edit_set")}}
    # ---- sensitivity: also train/validate on raw D-Fire negatives -------------------------------------------------------------
    pol3 = {**pol, "negative_sources": ["synth_set", "edit_set", "dfire"]}
    h3, rep3 = fit_auth_head(table, pol3)
    r3, _, _ = evaluate(h3, table, X)
    res["sensitivity_with_raw_dfire_negatives"] = {"fit_report": rep3["per_output"], "l2_chosen": rep3["l2_chosen"], "rows": rep3["rows"],
        "discrimination": r3["discrimination"], "detection_at_threshold": r3["detection_at_threshold"], "false_flag_on_real": r3["false_flag_on_real"]}
    # ---- cross-check: vectorised status rule == assess_authenticity on a sample ------------------------------------------------
    P = head.probs(X); st = status_from(P, head)
    idx = np.random.default_rng(1).choice(len(X), 400, replace=False)
    bad = 0
    for i in idx:
        s_, _sig = assess_authenticity(raw=b"", exif=ExifInfo(), ph=0, image_id="x", embedding=None, block_feat=X[i], head=head, index=None, upload_time=None, reported_location=None)
        bad += int(s_ != st[i])
    res["status_rule_crosscheck"] = {"rows_checked": 400, "mismatches": bad}
    json.dump(res, open(a.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", a.out)


if __name__ == "__main__":
    main()

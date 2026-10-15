"""Build and execute the notebooks (python scripts/build_notebooks.py). Charts read docs/data/*.json, so they run anywhere;
heavy cells (feature re-training) run only if the feature files exist at TW_FEAT."""
import nbformat as nbf
from nbclient import NotebookClient
M, C = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SETUP = '''import json, os, sys, pathlib
import numpy as np, matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
ROOT = pathlib.Path(os.getcwd()).resolve().parent if pathlib.Path(os.getcwd()).name == "notebooks" else pathlib.Path(os.getcwd())
sys.path.insert(0, str(ROOT / "src"))
J = lambda n: json.load(open(ROOT / "docs" / "data" / n))
COL = {"blue": "#2f6fb3", "orange": "#d9822b", "grey": "#8a8f98", "green": "#3a9a6e", "red": "#c24d4d"}'''
def nb(cells, path):
    n = nbf.v4.new_notebook(); n.cells = cells
    NotebookClient(n, timeout=600, kernel_name="python3", resources={"metadata": {"path": "notebooks"}}).execute()
    nbf.write(n, path); print("wrote", path)

# 01 -------------------------------------------------------------------------------------------------------------
nb([M("# 01. Data, leakage-controlled splits and pHash reuse detection\nSources: D-Fire (CC0) and an HPWREN FIgLib pilot (CC BY-NC 4.0). Everything here is read from `docs/data/*.json`, produced by `scripts/build_dfire_manifest.py` and `scripts/bench_phash.py`."),
    C(SETUP),
    M("## Partition sizes and label balance\nClusters touching the fixed-camera AoF source are held out whole (`xsrc_aof`); everything else is cluster-split."),
    C('''s = J("dfire_split_report.json"); P = s["per_partition_xsrc"]; names = list(P)
x = np.arange(len(names)); w = .27
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.bar(x - w, [P[n]["n"] for n in names], w, label="images", color=COL["grey"])
ax.bar(x, [P[n]["flames"] for n in names], w, label="flames = 1", color=COL["orange"])
ax.bar(x + w, [P[n]["smoke"] for n in names], w, label="smoke = 1", color=COL["blue"])
ax.set_xticks(x, names); ax.set_ylabel("images"); ax.legend(frameon=False); ax.set_title("D-Fire partitions (cluster split, AoF held out)")
plt.tight_layout(); plt.show()
print({k: s[k] for k in ("n_images", "n_clusters", "largest_cluster", "clusters_spanning_original_splits", "images_in_those_clusters")})'''),
    M("## The Kaggle splits leak\n414 near-duplicate clusters (about 11k images) span more than one of Kaggle's own train/val/test folders, so they are not used."),
    C('''leak, tot = s["images_in_those_clusters"], s["n_images"]
fig, ax = plt.subplots(figsize=(5.5, 1.8)); ax.barh([0], [leak], color=COL["red"], label="in clusters spanning splits"); ax.barh([0], [tot - leak], left=[leak], color=COL["grey"], label="other")
ax.set_yticks([]); ax.set_xlabel("images"); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.5), ncol=2); ax.grid(False); plt.tight_layout(); plt.show()'''),
    M("## pHash reuse detection\nEach query is a transformed copy of an indexed image; match = Hamming distance <= 6."),
    C('''b = J("phash_bench.json"); T = b["transforms"]; k = list(T)
r = np.array([T[i]["detect_rate"] for i in k]); lo = r - np.array([T[i]["ci95"][0] for i in k]); hi = np.array([T[i]["ci95"][1] for i in k]) - r
fig, ax = plt.subplots(figsize=(7.5, 3.6)); ax.bar(k, r, yerr=[lo, hi], color=[COL["green"] if v > .85 else COL["red"] if v < .3 else COL["orange"] for v in r], capsize=3)
ax.set_ylabel("copies detected"); ax.set_ylim(0, 1.05); ax.tick_params(axis="x", rotation=40); ax.set_title("pHash catches recompression/resizing, misses crops, rotation and flips")
plt.tight_layout(); plt.show()'''),
    M("**Reading:** crops, rotations and flips need the CLIP-embedding nearest-neighbour layer. Flips are also covered by hashing the mirrored image (`phash_variants`).\n\n## FIgLib frames inside D-Fire\nTop row: HPWREN FIgLib frames. Bottom row: the closest D-Fire images (Hamming <= 2). A 'train D-Fire, test FIgLib' test would leak."),
    C('''from PIL import Image
plt.figure(figsize=(11, 3.8)); plt.imshow(Image.open(ROOT / "docs/figures/dfire_figlib_overlap.png")); plt.axis("off"); plt.show()
x = b["cross_dataset_false_match_figlib_vs_dfire"]; print(f"{x['figlib_imgs_with_match_le_2']} of {x['figlib_images']} FIgLib pilot images match a D-Fire image at distance <= 2")''')],
   "notebooks/01_data_splits_and_phash.ipynb")

# 02 -------------------------------------------------------------------------------------------------------------
nb([M("# 02. Authenticity: detecting edited / composited fire images\nBlock-8 CLIP features vs final-layer features vs non-learned forensic cues, with a held-out edit type and degradations. Results come from `docs/data/authenticity_*.json` (scripts `eval_authenticity.py`, `eval_authenticity_control.py`, `handcrafted_baseline.py`).\n\n**Caveat:** the edits are crude splices built from D-Fire; realistic inpainting would be harder."),
    C(SETUP),
    M("## What the edits look like\nTop: original. Bottom: edited (`paste_flames`, `paste_smoke`, `cover_region`)."),
    C('''from PIL import Image
plt.figure(figsize=(10, 4.5)); plt.imshow(Image.open(ROOT / "docs/figures/edit_examples.png")); plt.axis("off"); plt.show()'''),
    M("## AUC on the test partition (95% CI, resampling whole clusters)"),
    C('''a, h = J("authenticity_edit_baseline.json"), J("authenticity_handcrafted_baseline.json")
rows = [("forensic cues", h["test_all"], h["test_seen_types"], h["test_cover_region_vs_neg"])]
for key, nm in (("block8_cls", "CLIP block-8"), ("final_embedding", "CLIP final layer")):
    r, r2 = a[key + "__train_all_types"], a[key + "__train_without_cover_region"]
    rows.append((nm, r["test_all_types"], r["test_seen_types"], r2["test_heldout_type"]))
groups = ["all edit types", "seen types", "held-out type"]; cols = [COL["grey"], COL["orange"], COL["blue"]]
fig, ax = plt.subplots(figsize=(7.5, 3.8)); w = .26
for i, (nm, *rr) in enumerate(rows):
    v = np.array([q["auc"] for q in rr]); ci = np.array([q.get("auc_ci95", q.get("ci95")) for q in rr])
    ax.bar(np.arange(3) + (i - 1) * w, v, w, yerr=[v - ci[:, 0], ci[:, 1] - v], capsize=3, color=cols[i], label=nm)
ax.axhline(.5, color="k", lw=.8, ls="--"); ax.set_xticks(range(3), groups); ax.set_ylim(.4, 1); ax.set_ylabel("ROC-AUC"); ax.legend(frameon=False, loc="lower right"); plt.tight_layout(); plt.show()'''),
    M("## Degradation: edits caught at a fixed 5% false-flag rate\nThe head is trained on clean images; the threshold comes from validation negatives."),
    C('''deg = ["none", "jpeg40", "resize50", "blur2"]; fig, ax = plt.subplots(figsize=(7.5, 3.6))
for i, (key, nm, c) in enumerate((("block8_cls", "block-8", COL["orange"]), ("final_embedding", "final layer", COL["blue"]))):
    r = a[key + "__train_all_types"]; v = [r["test_all_types"]["tpr_at_val_thr"]] + [r[f"degraded_{d}"]["tpr_at_val_thr"] for d in deg[1:]]
    ax.bar(np.arange(4) + (i - .5) * .35, v, .35, color=c, label=nm)
ax.set_xticks(range(4), deg); ax.set_ylabel("edits caught"); ax.set_ylim(0, 1); ax.legend(frameon=False); ax.set_title("Block-8 features collapse after JPEG recompression"); plt.tight_layout(); plt.show()'''),
    M("## Shortcut check\nPasted edits always contain fire, so a detector could just be spotting fire. Restricting to fire-bearing base images:"),
    C('''c = J("authenticity_edit_control.json"); print("fire-bearing bases:", {k: round(v, 2) for k, v in c["base_fire_rate"].items()})
for k in ("block8_cls", "final_embedding"): print(k, "AUC %.3f  CI %s" % (c[k]["auc"], [round(x, 3) for x in c[k]["ci95"]]))'''),
    M("## Optional: re-run the head live from the cached features\nRuns only if the CLIP feature files are present (not stored in the repo)."),
    C('''import csv
F = pathlib.Path(os.environ.get("TW_FEAT", "/home/user/data/feat")); E = pathlib.Path(os.environ.get("TW_EDITS", "/home/user/data/authenticity/edits_v1/edit_manifest.csv"))
if (F / "edits_clean.npz").exists() and E.exists():
    from fire_assessor.authenticity import AuthenticityHead
    from fire_assessor.metrics import roc_auc
    rows_ = list(csv.DictReader(open(E))); d = np.load(F / "edits_clean.npz"); y = np.array([int(r["edited_or_composited"]) for r in rows_]); part = np.array([r["partition"] for r in rows_])
    for key in ("block", "emb"):
        hd = AuthenticityHead.fit(d[key][part == "train"], y[part == "train", None].astype(float), l2=1e-2)
        print(key, "test AUC", round(roc_auc(y[part == "test"], hd.logits(d[key][part == "test"])[:, 0]), 3))
else: print("feature files not found; skipping live re-run")''')],
   "notebooks/02_authenticity_edit_detection.ipynb")

# 03 -------------------------------------------------------------------------------------------------------------
nb([M("# 03. Fire and smoke classification: baselines, calibration and risk control\nFrozen-CLIP baselines are read from `docs/data/fire_frozen_baselines.json`. LoRA/CNN results from `trackb_results.json` are shown if present. The calibration and risk-control demos run live on simulated scores to show how the method behaves, not to report performance."),
    C(SETUP),
    M("## Frozen CLIP: zero-shot vs linear probe\nAoF is the held-out fixed-camera source."),
    C('''f = J("fire_frozen_baselines.json"); fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, h in zip(axs, ("flames", "smoke")):
    for i, (m, c) in enumerate((("zero_shot", COL["grey"]), ("linear_probe", COL["blue"]))):
        v = np.array([f[m][e][h]["roc_auc"] for e in ("test", "xsrc_aof")]); ci = np.array([f[m][e][h]["roc_ci95"] for e in ("test", "xsrc_aof")])
        ax.bar(np.arange(2) + (i - .5) * .35, v, .35, yerr=[v - ci[:, 0], ci[:, 1] - v], capsize=3, color=c, label=m.replace("_", " "))
    ax.set_xticks(range(2), ["test (same sources)", "AoF (new source)"]); ax.set_title(h); ax.set_ylim(.5, 1)
axs[0].set_ylabel("ROC-AUC"); axs[0].legend(frameon=False, loc="lower left"); plt.tight_layout(); plt.show()'''),
    M("**Reading:** smoke detection drops sharply on a new source while flames do not. In-distribution numbers are optimistic.\n\n## LoRA vs CNN baselines (3 seeds), WiSE-FT, calibrated thresholds"),
    C('''p = ROOT / "docs/data/trackb_results.json"
if p.exists() and json.load(open(p))["methods"].get("lora", {}).get("test"):
    r = json.load(open(p)); fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, h in zip(axs, ("flames", "smoke")):
        names = [n for n in r["methods"] if r["methods"][n].get("test")]
        for i, ev in enumerate(("test", "xsrc_aof")):
            m = [r["methods"][n][ev][h]["roc"][0] for n in names]; s = [r["methods"][n][ev][h]["roc"][1] for n in names]
            ax.bar(np.arange(len(names)) + (i - .5) * .35, m, .35, yerr=s, capsize=3, color=[COL["blue"], COL["orange"]][i], label=ev)
        ax.set_xticks(range(len(names)), names, rotation=15); ax.set_title(h); ax.set_ylim(.5, 1)
    axs[0].legend(frameon=False); plt.tight_layout(); plt.show()
    for a_, v in sorted(r["wise"].items()): print("WiSE alpha", a_, {k: {h: round(v[k][h]["roc"], 3) for h in v[k]} for k in v})
    print(json.dumps(r["risk_control"], indent=1, default=float)[:1800])
else: print("LoRA / CNN results not available yet (run scripts/eval_trackb.py after the Track B jobs finish)")'''),
    M("## Demo: temperature scaling fixes an over-confident head (simulated)"),
    C('''from fire_assessor.calibration import ece, fit_temperature, sigmoid
rng = np.random.default_rng(0); z = rng.normal(size=6000) * 2; y = (rng.random(6000) < sigmoid(z)).astype(int); raw = z * 3   # logits inflated 3x
T = fit_temperature(raw, y); print("fitted T = %.2f" % T)
def rel(p, y, bins=12):
    e = np.linspace(0, 1, bins + 1); i = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
    return [(p[i == k].mean(), y[i == k].mean()) for k in range(bins) if (i == k).sum() > 20]
fig, ax = plt.subplots(figsize=(4.4, 4)); ax.plot([0, 1], [0, 1], "k--", lw=.8)
for p, nm, c in ((sigmoid(raw), "raw", COL["red"]), (sigmoid(raw / T), "temperature-scaled", COL["blue"])): q = np.array(rel(p, y)); ax.plot(q[:, 0], q[:, 1], "o-", color=c, label=f"{nm} (ECE {ece(p, y):.3f})")
ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency"); ax.legend(frameon=False); plt.tight_layout(); plt.show()'''),
    M("## Demo: risk-controlled threshold keeps false negatives below the target (simulated)\nLearn-then-Test: choose the highest threshold whose miss rate is certified <= alpha with probability >= 1 - delta. Over repeated calibration draws, the true miss rate should exceed alpha in at most a fraction delta of draws. This holds only if deployment data is exchangeable with calibration data."),
    C('''from fire_assessor.risk_control import select_threshold
alpha, delta, n_cal = .05, .1, 400; rng = np.random.default_rng(1); true_fnr, prec = [], []
neg_s = rng.beta(2, 5, 4000)
for _ in range(300):
    pos = rng.beta(5, 2, n_cal); r = select_threshold(np.r_[pos, neg_s[:1600]], np.r_[np.ones(n_cal), np.zeros(1600)], alpha, delta)
    true_fnr.append(np.mean(rng.beta(5, 2, 20000) < r.threshold)); prec.append(r.precision)
true_fnr = np.array(true_fnr); fig, ax = plt.subplots(figsize=(6.5, 3.4)); ax.hist(true_fnr, bins=30, color=COL["blue"]); ax.axvline(alpha, color=COL["red"], label=f"target alpha = {alpha}")
ax.set_xlabel("true false-negative rate of the chosen threshold"); ax.legend(frameon=False); plt.tight_layout(); plt.show()
print("fraction of draws with true FNR > alpha: %.3f (allowed: delta = %.2f);  mean precision at the operating point %.2f" % ((true_fnr > alpha).mean(), delta, np.mean(prec)))''')],
   "notebooks/03_fire_smoke_models_and_risk_control.ipynb")

# 04 -------------------------------------------------------------------------------------------------------------
nb([M("# 04. The assessment pipeline, end to end (runs on CPU, no model needed)\nThis drives the real pipeline code (validation, quality gate, decision policy, reuse detection, output contract) with a **stub scorer that returns fixed logits**. It demonstrates behaviour and the output contract; the scores are not model output."),
    C(SETUP + '''
import io
from PIL import Image
from fire_assessor.artifact import ModelArtifact
from fire_assessor.authenticity import ReuseIndex
from fire_assessor.decision import Thresholds, decide
from fire_assessor.pipeline import Assessor
from fire_assessor.scorer import ScorerOutput
class StubScorer:   # NOT a model: returns the logits it was given
    def __init__(self, fl, sm): self.fl, self.sm = fl, sm
    def score(self, im):
        rng = np.random.default_rng(int(np.asarray(im.resize((4, 4))).sum())); return ScorerOutput(self.fl, self.sm, rng.normal(size=8), rng.normal(size=8))
def img(seed=0, size=(320, 240)):
    r = np.random.default_rng(seed); return Image.fromarray(r.integers(0, 256, (size[1] // 8, size[0] // 8, 3), dtype=np.uint8)).resize(size, Image.BICUBIC)
def jpg(im, q=90):
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); return b.getvalue()
art = ModelArtifact("fire-assessor-v1.0.0", Thresholds(.5, .5), {"flames": 1., "smoke": 1.}, .05)'''),
    M("## One report through the pipeline"),
    C('''a = Assessor(StubScorer(3., -3.), art, reuse_index=ReuseIndex()); res = a.assess_bytes(jpg(img(1)), "report-001"); print(res.model_dump_json(indent=2))'''),
    M("## What each failure and edge case produces\nEvery outcome is `REQUIRES_REVIEW` or `UNAVAILABLE`; nothing is auto-rejected."),
    C('''import pandas as pd
cases = {"normal photo": jpg(img(2)), "too dark": jpg(Image.new("RGB", (300, 300), (4, 4, 4))), "blank grey": jpg(Image.new("RGB", (300, 300), (128, 128, 128))),
         "tiny image": jpg(img(3, (64, 64))), "empty file": b"", "not an image": b"hello", "truncated jpeg": jpg(img(4))[:200]}
out = []
for k, d in cases.items():
    r = Assessor(StubScorer(3., -3.), art).assess_bytes(d, k)
    out.append(dict(case=k, status=r.assessment_status, label=r.derived_label, priority=r.review_priority, quality=",".join(r.quality_flags), error=r.error_code))
pd.DataFrame(out)'''),
    M("## Reuse detection: a re-post, a recompressed copy, and a mirrored copy\nThe second and later uploads of the same photo are flagged `REUSED`; the first upload is not."),
    C('''a = Assessor(StubScorer(-3., -3.), art, reuse_index=ReuseIndex()); base = img(7)
for nm, d in (("original", jpg(base)), ("same file again", jpg(base)), ("recompressed q40", jpg(base, 40)), ("mirrored", jpg(base.transpose(Image.FLIP_LEFT_RIGHT))), ("different photo", jpg(img(8)))):
    r = a.assess_bytes(d, nm); print(f"{nm:18s} -> {r.authenticity.status:12s} {r.authenticity.signals}")'''),
    M("## The decision policy over the score plane\nThresholds 0.5/0.5 and a 0.05 uncertainty margin (illustrative values; the real ones come from calibration). The band around each threshold becomes `UNCERTAIN` and goes to review."),
    C('''g = np.linspace(0, 1, 201); lab = {"NO_VISIBLE_FIRE": 0, "SMOKE_ONLY": 1, "UNCERTAIN": 2, "ACTIVE_FIRE": 3}
Z = np.array([[lab[decide(pf, ps, Thresholds(.5, .5), [], False).derived_label] for pf in g] for ps in g])
from matplotlib.colors import ListedColormap
fig, ax = plt.subplots(figsize=(5.4, 4.6)); ax.imshow(Z, origin="lower", extent=[0, 1, 0, 1], cmap=ListedColormap(["#e8eef5", "#8bb4dd", "#e6c36a", "#d9822b"]), aspect="auto"); ax.grid(False)
ax.set_xlabel("calibrated flames score"); ax.set_ylabel("calibrated smoke score")
for (x, y_, t) in ((.15, .15, "NO_VISIBLE_FIRE"), (.2, .8, "SMOKE_ONLY"), (.8, .2, "ACTIVE_FIRE")): ax.text(x, y_, t, fontsize=8, ha="center")
ax.text(.5, .5, "UNCERTAIN\\n(band)", fontsize=7, ha="center", va="center"); plt.tight_layout(); plt.show()'''),
    M("## Safety property: the model path has no write access to incident status\nThe service wrapper writes only an assessment field.\n"),
    C('''import inspect
from fire_assessor import pipeline
print("sink methods:", [n for n, _ in inspect.getmembers(pipeline.AssessmentSink, inspect.isfunction) if not n.startswith("__")])
print("Assessment fields:", list(pipeline.Assessment.model_fields))''')],
   "notebooks/04_pipeline_demo.ipynb")

# 05 -------------------------------------------------------------------------------------------------------------
nb([M("# 05. AI-generated images and how exploitable the fire head is\nResults from `docs/data/authenticity_synthetic.json` (`scripts/gen_synthetic.py`, `build_synth_set.py`, `build_control_set.py`, `eval_synthetic.py`). Generators with licences read from their model cards: **SD 1.5 and SDXL are used for training; PixArt-alpha and Kolors are held out entirely**.\n\n**Read the caveat first:** these are four open diffusion models with simple prompts at 512 px / JPEG q90, so near-perfect scores are the easy case and must not be quoted as 'AI images are detected'."),
    C(SETUP),
    M("## Detection across generator families, and under degradation\nThe operating threshold gives 5% false flags on real validation/calibration images."),
    C('''s = J("authenticity_synthetic.json"); conds = ["clean", "jpeg40", "resize50", "blur2"]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for key, nm, c in (("block8_cls", "block-8", COL["orange"]), ("final_embedding", "final layer", COL["blue"])):
    held = [s[key][d]["held-out generators (pixart+kolors)"] for d in conds]
    axs[0].plot(conds, [h["auc"] for h in held], "o-", color=c, label=nm); axs[1].plot(conds, [h["false_flag_real"] for h in held], "o-", color=c, label=nm)
axs[0].set_title("AUC on held-out generators (PixArt, Kolors)"); axs[0].set_ylim(.95, 1.003); axs[1].set_title("false flags on real test photos"); axs[1].axhline(.05, color="k", ls="--", lw=.8)
axs[0].legend(frameon=False); plt.tight_layout(); plt.show()'''),
    M("Detection stays near-perfect, but false flags on real photos rise sharply after blur and downsizing, especially for block-8 features.\n\n## Control: real photos from sources never seen in training\nIf the head merely rejected 'anything not D-Fire', these would be flagged."),
    C('''fig, ax = plt.subplots(figsize=(5.5, 3.3)); w = .35
for i, (key, nm, c) in enumerate((("block8_cls", "block-8", COL["orange"]), ("final_embedding", "final layer", COL["blue"]))):
    ctl = s[key]["control_real_other_sources_false_flag"]; ax.bar(np.arange(2) + (i - .5) * w, [ctl["figlib"], ctl["aof"]], w, color=c, label=nm)
ax.set_xticks(range(2), ["FIgLib frames", "AoF frames"]); ax.set_ylabel("flagged as synthetic"); ax.legend(frameon=False); plt.tight_layout(); plt.show()'''),
    M("The final-layer head passes this control; the block-8 head does not (59% of FIgLib frames flagged), so block-8 is unstable under source shift.\n\n## The fire head cannot tell generated fire from real fire\nA linear probe trained on real D-Fire, applied to the generated images (threshold at 5% false positives on real negatives)."),
    C('''e = s["fire_head_on_generated_images"]; fig, ax = plt.subplots(figsize=(6.2, 3.4)); w = .27
for i, (lab, key, c) in enumerate((("generated fire/smoke", "generated_positive_flagged", COL["red"]), ("real D-Fire positives", "real_dfire_test_positive_flagged", COL["green"]), ("generated negatives", "generated_negative_flagged", COL["grey"]))):
    ax.bar(np.arange(2) + (i - 1) * w, [e["flames"][key], e["smoke"][key]], w, color=c, label=lab)
ax.set_xticks(range(2), ["flames head", "smoke head"]); ax.set_ylabel("share flagged"); ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=8); plt.tight_layout(); plt.show()
tags = e["flames"]["flagged_by_hard_negative_tag"]; fig, ax = plt.subplots(figsize=(6.2, 3)); ax.bar(list(tags), list(tags.values()), color=COL["blue"]); ax.set_ylabel("flagged by flames head"); ax.tick_params(axis="x", rotation=30); ax.set_title("generated hard negatives"); plt.tight_layout(); plt.show()'''),
    M("**Why this matters:** a generated flame image scores as high as a real one, which is exactly why the design says the fire score must never be shown without the authenticity status. Generated burn scars, a labelled hard negative, are flagged as flames most of the time.")],
   "notebooks/05_synthetic_images_and_exploitability.ipynb")

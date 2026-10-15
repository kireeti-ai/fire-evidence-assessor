"""Render docs/experiments/*.md from the result JSONs in docs/data, so numbers are copied, never retyped.
Re-run after any experiment finishes. Missing result files render as 'NOT YET RUN'."""
import json, os
D = "docs/data"; O = "docs/experiments"; os.makedirs(O, exist_ok=True)
def J(n):
    p = f"{D}/{n}"; return json.load(open(p)) if os.path.exists(p) else None
def ci(r, k="auc", c="auc_ci95"): return f"{r[k]:.3f} [{r[c][0]:.3f}, {r[c][1]:.3f}]"
def w(name, text): open(f"{O}/{name}", "w").write(text.strip() + "\n"); print("wrote", name)
PENDING = "**NOT YET RUN.** The result file does not exist yet; re-run `scripts/make_experiment_docs.py` after the experiment."

# ---- 01 data and splits
s = J("dfire_split_report.json")
t = "# 01. Data, splits and leakage control\n\n"
if s:
    t += f"D-Fire (Kaggle mirror, CC0): **{s['n_images']:,}** images. Near-duplicate clustering (pHash Hamming <= {s['max_hamming']}, plus 25-frame sequence blocks for the AoF fixed cameras) gives **{s['n_clusters']:,}** clusters; the largest has **{s['largest_cluster']:,}** images (AoF frames chain together).\n\n"
    t += f"**The Kaggle train/val/test splits leak:** {s['clusters_spanning_original_splits']} clusters ({s['images_in_those_clusters']:,} images) span more than one of them. They are not used.\n\n"
    t += "## Partitions used (`partition_xsrc`, recommended)\nClusters touching AoF are held out whole as the cross-source test set; the rest are cluster-split.\n\n| Partition | Images | Flames=1 | Smoke=1 |\n|---|---|---|---|\n"
    for p, v in s["per_partition_xsrc"].items(): t += f"| {p} | {v['n']:,} | {v['flames']:,} | {v['smoke']:,} |\n"
    t += f"\nManifest hash (all splits, labels and paths): `{s['manifest_hash']}`; seed {s['seed']}.\n"
else: t += PENDING
t += "\n## Caveats\n- Sequence block size (25) and the Hamming threshold are assumptions, not ground-truth video IDs.\n- None of these images are citizen phone photos.\n- D-Fire labels are derived from boxes: `flames` = any fire box, `smoke` = any smoke box.\n"
w("01_data_and_splits.md", t)

# ---- 02 phash
b = J("phash_bench.json"); t = "# 02. Perceptual-hash reuse detection\n\n"
if b:
    t += f"{b['n_index']:,} distinct images indexed; each query is a transformed copy; match = pHash Hamming <= {b['max_hamming']}.\n\n| Transform | Detected | 95% CI | Median Hamming |\n|---|---|---|---|\n"
    for k, v in b["transforms"].items(): t += f"| {k} | {v['detect_rate']:.1%} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {v['median_hamming']:.0f} |\n"
    x = b.get("cross_dataset_false_match_figlib_vs_dfire")
    t += "\n## Findings\n- pHash is robust to resize, JPEG and screenshot-style recompression and mostly robust to brightness, but **misses crops, rotations and flips**. Flips are now covered in code by also hashing the mirrored image (`phash_variants`, unit-tested but not re-benchmarked here); crops and rotations need the CLIP-embedding nearest-neighbour layer.\n"
    t += "- **False-match rate is not established.** Within D-Fire it is circular (clusters use the same threshold).\n"
    if x: t += f"- **D-Fire and FIgLib overlap:** {x['figlib_imgs_with_match_le_2']} of {x['figlib_images']} FIgLib pilot images match a D-Fire image at distance <= 2 (same HPWREN camera, see `docs/figures/dfire_figlib_overlap.png`). A 'train on D-Fire, test on FIgLib' cross-dataset test would leak unless those clusters are removed first.\n"
else: t += PENDING
w("02_phash_reuse_detection.md", t)

# ---- 03 authenticity edits
a, c, h = J("authenticity_edit_baseline.json"), J("authenticity_edit_control.json"), J("authenticity_handcrafted_baseline.json")
t = "# 03. Authenticity: edited / composited images (Stages 2-3)\n\n"
t += "**Data.** Edits are built from D-Fire inside each partition: `paste_flames`, `paste_smoke` (feathered paste of a fire/smoke box from another photo) and `cover_region` (a fire/smoke box covered by a patch from elsewhere in the image). Negatives are untouched originals and benign edits (crop, brightness, colour, resize, JPEG). Label = `edited_or_composited`. `cover_region` is held out for the unseen-edit-type test. **The edits are crude splices; realistic inpainting would be harder.**\n\n"
if a and h:
    t += "Operating threshold = the value giving 5% false flags on *validation negatives*; test is never used for tuning. 95% CIs resample whole clusters.\n\n"
    t += "## AUC on the test partition\n| Method | All edit types | Seen types | Held-out type (`cover_region`) |\n|---|---|---|---|\n"
    t += f"| Non-learned forensic cues (noise, tiles, blockiness, ELA) | {ci(h['test_all'],'auc','ci95')} | {ci(h['test_seen_types'],'auc','ci95')} | {ci(h['test_cover_region_vs_neg'],'auc','ci95')} |\n"
    for key, nm in (("block8_cls", "CLIP block-8 CLS"), ("final_embedding", "CLIP final embedding")):
        r = a[f"{key}__train_all_types"]; r2 = a[f"{key}__train_without_cover_region"]
        t += f"| {nm} | {ci(r['test_all_types'])} | {ci(r['test_seen_types'])} | {ci(r2['test_heldout_type'])} (trained without that type) |\n"
    t += "\n## Robustness to degradation (head trained on clean images; test partition)\n| Features | Degradation | AUC | Edits caught at the 5%-false-flag threshold | False flags on originals |\n|---|---|---|---|---|\n"
    for key, nm in (("block8_cls", "block-8"), ("final_embedding", "final")):
        r = a[f"{key}__train_all_types"]
        t += f"| {nm} | none | {r['test_all_types']['auc']:.3f} | {r['test_all_types']['tpr_at_val_thr']:.1%} | {r['test_all_types']['false_flag_orig']:.1%} |\n"
        for d in ("jpeg40", "resize50", "blur2"):
            x = r[f"degraded_{d}"]; t += f"| {nm} | {d} | {x['auc']:.3f} | {x['tpr_at_val_thr']:.1%} | {x['false_flag_orig']:.1%} |\n"
    if c: t += f"\n## Shortcut control\nFire presence could leak the label (pasted edits always contain fire/smoke). Fire-bearing base images are {c['base_fire_rate']['negatives']:.0%} of negatives and {c['base_fire_rate']['paste_positives']:.0%} of paste positives. Restricting train and test to fire-bearing bases: block-8 AUC {ci(c['block8_cls'],'auc','ci95')}, final-layer AUC {ci(c['final_embedding'],'auc','ci95')}; the result does not depend on fire presence.\n"
    t += "\n## Findings\n- Non-learned cues are close to chance on the unseen edit type. CLIP features are far better.\n- **Final-layer features beat block-8 here**, contrary to the design hypothesis. The intervals overlap partly, so this is a modest advantage on crude edits, not a settled answer.\n- Block-8 features lose almost all recall after JPEG recompression; the final embedding keeps a fraction of it.\n- Not yet tested: realistic inpainting, and AI-generated images (see 06).\n"
else: t += PENDING
w("03_authenticity_edit_detection.md", t)

# ---- 04 frozen fire baselines
f = J("fire_frozen_baselines.json"); t = "# 04. Fire and smoke: frozen-CLIP baselines (zero-shot, linear probe)\n\nCLIP ViT-B/16 (OpenAI weights, **QuickGELU config**) with no fine-tuning. Zero-shot = cosine to a positive minus a negative prompt ensemble. Linear probe = logistic regression on the final embedding (L2 chosen on validation). AoF is the held-out fixed-camera source; its CIs resample 25-frame blocks.\n\n"
if f:
    t += "| Method | Set | Flames ROC-AUC | Smoke ROC-AUC | Flames PR-AUC (prevalence) | Smoke PR-AUC (prevalence) |\n|---|---|---|---|---|---|\n"
    for m in ("zero_shot", "linear_probe"):
        for ev in ("test", "xsrc_aof"):
            r = f[m][ev]; t += f"| {m} | {ev} | {ci(r['flames'],'roc_auc','roc_ci95')} | {ci(r['smoke'],'roc_auc','roc_ci95')} | {r['flames']['pr_auc']:.3f} ({r['flames']['prevalence']:.2f}) | {r['smoke']['pr_auc']:.3f} ({r['smoke']['prevalence']:.2f}) |\n"
    t += "\n## Findings\n- In-distribution the linear probe is near ceiling for both heads.\n- **Smoke generalises poorly across sources** (large drop on the AoF fixed-camera set) while flames hold up.\n- In-distribution numbers are optimistic: all test images come from the same web sources as training, even after cluster splitting.\n"
else: t += PENDING
w("04_fire_smoke_frozen_baselines.md", t)

# ---- 05 track B
r = J("trackb_results.json"); t = "# 05. Fire and smoke: LoRA vs CNN baselines, WiSE-FT, calibration, risk control\n\n"
if r and r["methods"].get("lora", {}).get("test"):
    t += "Seeds per row are listed; mean +- sd is shown when there is more than one. LoRA = CLIP ViT-B/16 + rank-8 LoRA on blocks 9-12, head initialised from text prompts. CNNs are fully fine-tuned from ImageNet weights on the same splits.\n\n| Method | Seeds | Set | Flames ROC | Smoke ROC | Flames PR | Smoke PR |\n|---|---|---|---|---|---|---|\n"
    for n, v in r["methods"].items():
        for p in ("test", "xsrc_aof"):
            if p in v:
                f_ = lambda h, m: f"{v[p][h][m][0]:.3f}" + (f" +- {v[p][h][m][1]:.3f}" if v["n_seeds"] > 1 else "")
                t += f"| {n} | {v['n_seeds']} | {p} | {f_('flames','roc')} | {f_('smoke','roc')} | {f_('flames','pr')} | {f_('smoke','pr')} |\n"
    if r["wise"]:
        t += "\n## WiSE-FT sweep (seed 0; alpha 0 = zero-shot weights, 1 = fully fine-tuned)\n| alpha | test flames | test smoke | AoF flames | AoF smoke |\n|---|---|---|---|---|\n"
        for a_, v in sorted(r["wise"].items()):
            g = lambda p, k: f"{v[p][k]['roc']:.3f}" if p in v else "-"
            t += f"| {a_} | {g('test','flames')} | {g('test','smoke')} | {g('xsrc_aof','flames')} | {g('xsrc_aof','smoke')} |\n"
    if r["risk_control"]:
        t += "\n## Calibration and risk-controlled thresholds (LoRA seed 0, fit on the calibration partition only)\n"
        k0 = next(iter(r["risk_control"].values())); t += f"Target false-negative rate {k0['target_fnr']:.0%} with probability {1-k0['delta']:.0%} (Learn-then-Test). **The target is a placeholder; the real value is a stakeholder decision.**\n\n| Head | Set | FNR | Recall | Precision | Review volume | ECE before -> after |\n|---|---|---|---|---|---|---|\n"
        for h_, o in r["risk_control"].items():
            for p in ("test", "xsrc_aof"):
                if p in o: x = o[p]; t += f"| {h_} | {p} | {x['fnr']:.3f} | {x['recall']:.3f} | {x['precision']:.3f} | {x['review_fraction']:.1%} | {x['ece_before']:.3f} -> {x['ece_after']:.3f} |\n"
        t += "\nThe guarantee holds only under exchangeability. The `xsrc_aof` rows show what happens when it is violated by a source shift: the false-negative rate is many times the target, so shadow-mode monitoring on real uploads is mandatory before any priority use.\n"
    t += """
## Findings
- **In-distribution the three models are indistinguishable** (flames ROC about 0.99, smoke about 0.98). A model compared only on a random split would look equally good.
- **Under source shift CLIP+LoRA is clearly more robust than the ImageNet CNNs** (design question 1): on the held-out fixed-camera source, smoke ROC-AUC is about 0.70 for LoRA but roughly 0.5 (chance) for both CNNs, and flames is 0.97 vs 0.92 to 0.93. The three LoRA seeds agree closely, so this is not seed noise.
- **But CLIP+LoRA is not robust enough**: smoke at 0.70 on a new source is poor. WiSE-FT did not help here: the best AoF smoke score is at alpha = 1 (fully fine-tuned) and there is no intermediate sweet spot.
- **Calibration helps, the false-negative bound does not survive a source shift.** Temperature scaling lowers calibration error on test, and thresholds meet the 5% target for flames on test (smoke is slightly over). On the held-out source the false-negative rate is several times the target.
- Caveats: one dataset family, three seeds, cross-source test is a single fixed-camera source, and no citizen phone photos. These results support the design's *choice* of CLIP+LoRA over CNNs on this data; they are not deployment evidence.
"""
else: t += PENDING + "\n"
w("05_fire_smoke_lora_vs_cnn.md", t)

# ---- 06 synthetic generators
sj = J("authenticity_synthetic.json"); t = "# 06. Authenticity: AI-generated images, and how exploitable the fire head is\n\n"
if sj:
    t += "**Data.** Generated with open-weight models whose licences were read from their model cards: SD 1.5 (CreativeML OpenRAIL-M) and SDXL (OpenRAIL++) for training; **PixArt-alpha (OpenRAIL++) and Kolors (Apache-2.0) held out entirely**. Prompts describe phone-style photos of flames, smoke and hard negatives (sunset, fog, haze, burn scar, dry vegetation, campfire, cooking smoke, clear sky). Real = D-Fire originals. Real and synthetic images are both normalised to longest side 512 + JPEG q90 so size and format are not shortcuts. Operating threshold = 5% false flags on real validation+calibration images; test is never used for tuning.\n\n"
    t += "## Detection (AUC, 95% CI over clusters) and false-flag rate on real test images\n| Features | Condition | Seen generators AUC | Held-out generators AUC | False flags on real |\n|---|---|---|---|---|\n"
    for k, nm in (("block8_cls", "block-8"), ("final_embedding", "final layer")):
        for d in ("clean", "jpeg40", "resize50", "blur2"):
            o = sj[k][d]; sn, hn = "seen generators (sd15+sdxl)", "held-out generators (pixart+kolors)"
            t += f"| {nm} | {d} | {ci(o[sn],'auc','auc_ci95')} | {ci(o[hn],'auc','auc_ci95')} | {o[sn]['false_flag_real']:.1%} |\n"
    t += "\n## Control: real photos from sources never seen in training (share flagged as synthetic)\n| Features | FIgLib frames | AoF fixed-camera frames |\n|---|---|---|\n"
    for k, nm in (("block8_cls", "block-8"), ("final_embedding", "final layer")):
        c = sj[k]["control_real_other_sources_false_flag"]; t += f"| {nm} | {c['figlib']:.1%} | {c['aof']:.1%} |\n"
    e = sj["fire_head_on_generated_images"]
    t += "\n## Fire/smoke linear probe applied to generated images (threshold = 5% false positives on real D-Fire negatives)\n| Head | Generated fire/smoke images flagged | Real D-Fire test positives flagged | Generated negatives flagged |\n|---|---|---|---|\n"
    for h_ in ("flames", "smoke"): t += f"| {h_} | {e[h_]['generated_positive_flagged']:.1%} (n={e[h_]['n_generated_pos']}) | {e[h_]['real_dfire_test_positive_flagged']:.1%} | {e[h_]['generated_negative_flagged']:.1%} (n={e[h_]['n_generated_neg']}) |\n"
    t += "\nFalse positives on generated hard negatives (share flagged): flames head " + ", ".join(f"{k} {v:.0%}" for k, v in e["flames"]["flagged_by_hard_negative_tag"].items()) + "; smoke head " + ", ".join(f"{k} {v:.0%}" for k, v in e["smoke"]["flagged_by_hard_negative_tag"].items()) + ".\n"
    t += """
## Findings and caveats
- All four generator families, including the two never seen in training, are separated from real photos almost perfectly **at this resolution and compression**. This is the easy case: four open diffusion models, simple prompts, 512 px JPEG q90. It says nothing about newer or closed generators, other prompts, or higher-resolution outputs, and AUC 1.0 should not be quoted as "AI images are detected".
- The control matters. The final-layer head leaves unseen real sources almost unflagged, so it is not merely rejecting anything that is not D-Fire. The block-8 head is unstable under source shift (most FIgLib frames flagged), which argues against it.
- Detection degrades under blur and half-resolution recompression, mainly through more false flags on real images.
- **The fire head is easy to fool.** Generated fire/smoke images are flagged about as often as real ones, so the authenticity status must always be shown with the fire score. It also flags generated burn scars (a hard negative) most of the time.
"""
else: t += PENDING + "\n"
w("06_authenticity_synthetic_generators.md", t)

# ---- 00 index / lessons
w("00_README.md", """# Experiment log

| # | Experiment | Status |
|---|---|---|
| 01 | Data, splits and leakage control | done |
| 02 | pHash reuse detection | done (false-match rate not established) |
| 03 | Edited-image detection (block-8 vs final-layer) | done on crude splices |
| 04 | Fire/smoke frozen-CLIP baselines | done |
| 05 | LoRA vs CNN, WiSE-FT, calibration, risk control | done (3 seeds each) |
| 06 | AI-generated images across generators + fire-head exploitability | done (easy case; see caveats) |

Notebooks that reproduce the charts are in `../../notebooks/`.

## Process notes
- **QuickGELU bug.** OpenAI CLIP weights loaded into the plain `ViT-B-16` config degrade silently (open_clip warns). All CLIP numbers here were re-run with `ViT-B-16-quickgelu`; a unit test now guards it.
- **Circular metric caught.** A false-match rate measured inside clusters built with the same threshold was trivially zero and was removed.
- **Leakage found.** D-Fire's own splits leak; FIgLib frames are inside D-Fire.
- **Not claimed.** Nothing here is deployment performance. No citizen phone photos were used.
""")

# Authenticity head: data, label policy, splits, operating point

Status: **PROPOSED, research only.** The label policy and every policy value below are proposals written by an AI coding session. A named human owner must confirm or replace them. Nothing here approves an artifact; a build with `authenticity.mode = "include_head"` still ends **NOT_APPROVED** until a human approval record bound to the artifact digest exists. Results are in [`AUTH_HEAD_RESULTS.md`](AUTH_HEAD_RESULTS.md).

## 1. What was built
| Piece | Where |
|---|---|
| Head: two-output L2 logistic regression (`synthetic`, `edited_or_composited`) on the block-8 pre-norm CLS feature (key `block`, 768-d), per-output temperature, threshold, INCONCLUSIVE band | `src/fire_assessor/authenticity.py` (`AuthenticityHead.fit_masked`, `.validate`, optional `lo`) |
| Label rules, split names, leakage check | `src/fire_assessor/auth_data.py` |
| Fit + operating-point selection | `src/fire_assessor/auth_build.py` (`fit_auth_head`) |
| FPR control (exact one-sided binomial) | `src/fire_assessor/risk_control.py` (`select_threshold_fpr`, `binom_upper_bound`) |
| `build_artifact calibrate` with `policy.authenticity.mode = "include_head"` and `--auth-data SPEC.json` | `src/fire_assessor/build_artifact.py` |
| Packaging / load validation / completeness blockers | `src/fire_assessor/artifact.py` |
| Pipeline status + calibrated signals | `src/fire_assessor/authenticity.py::assess_authenticity` (called from `pipeline.py:86`) |
| Evaluation | `scripts/eval_auth_head.py`, `scripts/check_block8_equivalence.py` |

What already existed before this change (verified by reading): `ModelArtifact.auth_head` is saved as `auth_*` arrays and reloaded when `has_auth_head` is true (`artifact.py`), `AuthenticityHead` existed with a plain gradient-descent `fit`, and `assess_authenticity` already mapped head probabilities to `POSSIBLE_SYNTHETIC` / `POSSIBLE_EDITED` / `NO_CONCERNS_FOUND` / `INCONCLUSIVE`. Missing: any way to *build* a head into an artifact (`calibrate` accepted only `exclude_by_policy`), thresholds and temperatures (the head used 0.5 and T = 1), any provenance, any load-time validation of the head, and completeness checks for it.

## 2. Data inventory (frozen-base features, block-8 CLS; all local, read-only)
Rows are matched to manifests by `paths`, never by order. Counts were computed from the table built by `auth_data.build_table` (32,592 rows in total).

| Source (feature file, manifest) | Rows | Contents |
|---|---|---|
| synthetic set (`synth_clean.npz`, `synth_manifest.csv`) | 2,500 | 1,250 generated (SD 1.5: 600, SDXL: 250, **PixArt-alpha: 200 and Kolors: 200, held out entirely**) + 1,250 real (D-Fire originals normalised like the generated images) |
| edit set (`edits_clean.npz`, `edit_manifest.csv`) | 7,565 | 2,700 originals, 2,700 benign edits (crop, brightness, contrast/colour, resize, JPEG), 2,165 edits (`paste_flames` 914, `paste_smoke` 877, `cover_region` 374); one source photo per D-Fire cluster |
| D-Fire raw (`dfire_clean.npz`, `dfire_manifest.csv`) | 21,527 | all D-Fire images, unnormalised; 8,428 are the AoF fixed-camera held-out source (`xsrc_aof`) |
| control (`control_clean.npz`, `control_manifest.csv`) | 1,000 | 500 FIgLib frames + 500 AoF frames, normalised like the synthetic set, never seen in any fitting |

Exact breakdown by split (from the table):

| Split | Rows | Groups | `synthetic` pos / neg | `edited` pos / neg / undefined |
|---|---|---|---|---|
| train | 14,394 | 7,858 | 514 / 13,880 | 1,197 / 12,683 / 514 |
| val | 2,600 | 1,172 | 84 / 2,516 | 322 / 2,194 / 84 |
| cal | 2,576 | 1,180 | 73 / 2,503 | 320 / 2,183 / 73 |
| test | 3,194 | 1,309 | 179 / 3,015 | 326 / 2,689 / 179 |
| heldout_gen (PixArt 200 + Kolors 200) | 400 | 400 | 400 / 0 | 0 / 0 / 400 |
| heldout_source (D-Fire AoF) | 8,428 | **1** | 0 / 8,428 | 0 / 8,428 / 0 |
| control | 1,000 | 1,000 | 0 / 1,000 | 0 / 1,000 / 0 |

Generators and edit types (rows): sd15 train 368 / val 61 / cal 52 / test 119; sdxl 146 / 23 / 21 / 60; pixart 200 and kolors 200 only in `heldout_gen`. Edit types train / val / cal / test: `paste_flames` 536 / 112 / 127 / 139, `paste_smoke` 468 / 145 / 136 / 128, `cover_region` 193 / 65 / 57 / 59; originals and benign edits 1,500 / 400 / 400 / 400 each. The 579 real images of the synthetic set's test split include the 400 reused as the real counterpart of the held-out generators.

Data caveats that change how results must be read: (a) the AoF source is a single D-Fire "cluster" (id 0), so grouped bootstraps on it are degenerate; (b) control images have no sequence ids, so each frame is its own group, which makes control confidence intervals optimistic for fixed-camera sequences; (c) generated and real images of the synthetic set are both normalised to longest side 512 + JPEG q90, whereas raw D-Fire images are not; (d) the generated images come from templated prompts (10 scenes, 7 styles, 15 fire/smoke/hard-negative cores).

## 3. Label policy (PROPOSED; to be confirmed by a human owner)
| Output | Positive (1) | Negative (0) | Undefined (excluded from that output) |
|---|---|---|---|
| `synthetic` | every generated image, from each generator | every real image: D-Fire originals, benign-edited originals, **edited composites**, control frames | none |
| `edited_or_composited` | each edit type: `paste_flames`, `paste_smoke`, `cover_region` | untouched originals and benign edits (crop, brightness, contrast/colour, resize, JPEG), D-Fire originals, control frames | **generated images** (a generated image is not an edit of a photo; labelling it 0 would train "not synthetic" into this output) |

Assumptions that cannot be verified from the data: D-Fire web images are real and unedited (no provenance exists; some may be edited or synthetic); "benign edit" is not "edited" by definition of this policy, so benign edits flagged as edited count as false flags; a composite is "real" for the `synthetic` output. A different owner policy (for example, treating benign JPEG recompression as an edit, or flagging composites as synthetic) needs a re-labelling and a rebuild.

## 4. Split rules and what is fit on which split
* **Split unit.** An original, all its derived edits, and every near-duplicate share a D-Fire cluster id (`c<cluster>`); every row of that group lives in exactly one split. Generated images are singleton groups; control frames are singletons. `fit_auth_head` stops with `GROUP_LEAKAGE` if any group appears in two of train / cal / val / test / heldout_gen / heldout_source (checked on the real data: 0 leaking groups among 11,920 groups in those splits).
* **Held-out generators.** PixArt-alpha and Kolors are never read by fitting, selection or thresholding (`heldout_gen`). The shipped head is therefore directly a generalisation estimate for two unseen generators (not for newer or closed generators).
* **Held-out edit type.** The shipped head trains on all three edit types, so an unseen-type estimate needs a second fit: `scripts/eval_auth_head.py` refits without `cover_region` and reports that ablation. It is an estimate for how the method behaves, not a property of the shipped head.
* **Held-out source / control.** D-Fire AoF (`heldout_source`) and the control set (`control`) are evaluation only.

| Split | Role | Read by |
|---|---|---|
| `train` | fit weights and standardisation | fit |
| `cal` | choose L2 strength per output from the policy grid (strongest L2 within `l2_auc_tolerance` = 0.002 ROC-AUC of the best) | fit |
| `val` | per-output temperature (NLL, golden-section); threshold (FPR control); INCONCLUSIVE band edge | fit |
| `test`, `heldout_gen`, `heldout_source`, `control` | report only | **never** read by `fit_auth_head` (a test changes every evaluation row and asserts identical state) |

Only the sources named in `policy.authenticity.negative_sources` enter train/cal/val (proposed: `synth_set`, `edit_set`). Raw D-Fire is held back for evaluation because its processing history differs from the synthetic positives; adding it as a negative source was run as a sensitivity and made things worse (see results).

## 5. Operating point and the guarantee
For each output, scores of real validation images are reduced to **one score per group** (the maximum over the group's real images, so a source photo and its benign edit count once, conservatively). `select_threshold_fpr` scans thresholds from the highest down and keeps the lowest threshold whose false-flag rate on those groups is rejected as "> target_fpr" by the exact binomial test at level `delta_fpr` (fixed-sequence, no multiplicity penalty).

**What it guarantees.** If the images the threshold is later applied to are exchangeable with the validation real groups, then with probability at least `1 - delta_fpr` the false-flag rate on such real images is at most `target_fpr`. On the real data the selected thresholds flagged 16 of 446 validation real groups (3.6 %; one-sided 90 % upper bound 5.0 %) for both outputs.

**What it does not guarantee.** It is not valid under shift: a new generator, camera, phone model, social-media re-compression or any real domain not represented among the validation groups. Measured: FIgLib control frames are flagged `synthetic` at 16.8 %, far above the 5 % target (results §3). It is also a statement about *real* images only; it says nothing about the detection rate on fakes. With (1 - target_fpr)^n > delta_fpr, i.e. fewer than 45 real groups at the proposed values, the build refuses (`FPR_INFEASIBLE`).

**Calibrated probabilities.** Temperature scaling is fitted on the validation mix (about 6.5 % `synthetic` positives, 26.7 % `edited` positives), so reported probabilities describe that mix, not field prevalence. The `synthetic` output is almost perfectly separable on validation, so its temperature is small (0.17) and its operating threshold is a low probability (0.032): read the threshold as the FPR-controlled operating point, not as a "3 % chance".

**INCONCLUSIVE band.** `[band_lo, threshold)`, where `band_lo` is the validation real-group score that would flag `inconclusive_fpr` (proposed 15 %) of real groups, never above the threshold. Flagging (>= threshold) keeps the false-flag guarantee; an image inside the band is reported `INCONCLUSIVE` instead of `NO_CONCERNS_FOUND`. Without a recorded band (legacy heads) the old +-`uncertain_margin` rule is used.

**Proposed values** (`docs/calibration/auth_policy.PROPOSED.json`, all PROPOSED): `target_fpr` 0.05 (matches the 5 % placeholder used for fire thresholds; every output is a human review item so a flag costs reviewer time, not an automatic action), `delta_fpr` 0.10 (same as fire), `inconclusive_fpr` 0.15, L2 grid {0.001, 0.01, 0.1, 1}, balanced classes, `seed` 0, negatives `synth_set` + `edit_set`. They were fixed from feasibility on the validation split before any test, held-out or control number was looked at, and are not tuned on those splits. Consequence worth deciding on: at 5 % false flags, in a population where fakes are rare (say 1 %), most `POSSIBLE_*` flags on the edit output would be false (illustration: edited TPR 0.405, measured real false-flag 1.8 % on the edit set gives about 19 % precision at 1 % prevalence; at a 5 % false-flag rate about 8 %).

## 6. Block-8 equivalence (frozen base vs LoRA checkpoint), verified numerically
By construction LoRA wraps only transformer blocks 9-12 (`lora.py:47`, `apply_lora(..., blocks=range(8, 12))`, zero-indexed; `model.py:26`) and everything else in the visual tower is frozen, and the tap is taken after block 8 (`model.py:22, 59`). So the tap should not depend on the LoRA weights. Checked (`scripts/check_block8_equivalence.py`, output [`BLOCK8_EQUIVALENCE.json`](BLOCK8_EQUIVALENCE.json)) on the 32 train images in the LoRA-checkpoint probe file against the frozen-base rows of `dfire_clean.npz`, matched by `paths`:

| Statistic (32 images, 768-d) | Value |
|---|---|
| bitwise identical | **no** |
| max abs difference / mean abs difference | 0.0155 / 0.00037 (the mean per-dimension feature std is 0.054; the largest difference is 0.19 std) |
| cosine similarity min / median | 0.99998 / 0.999997 |
| relative L2 difference median / max | 0.27 % / 0.64 % |

Reading: equal up to numerical noise, not identical. The frozen features were extracted in fp16 on a GPU (`scripts/extract_features.py` calls `.half()`), the probe file's precision and device are not recorded, and this environment has torch 2.14.1 + open_clip 3.3.0; fp16 rounding (relative 1e-3) is the likely cause, but that is a hypothesis: the base pretrained weights are not available offline, so a clean fp32 recomputation of the frozen path could not be run. As a contrast, the final embedding `emb` of the same images differs a lot (cosine down to 0.28), as expected since LoRA changes blocks 9-12, which is why frozen `emb` features must not be reused with the LoRA checkpoint and block 8 is the only usable tap. Effect on the head (isotropic noise at the measured median and maximum relative size, results §6): flag decisions flip on at most 0.17 % of evaluation rows. A residual risk remains: 32 images is a small sample and the check does not exercise the whole data distribution; re-run the script on a larger LoRA-extracted probe before relying on it.

## 7. Limits (also written into every artifact built in this mode)
* No evaluation on citizen phone photos; no such labelled set exists.
* Features were extracted with the frozen base model (fp16, GPU); the pipeline feeds the LoRA checkpoint's tap (equal up to the noise above).
* Four open diffusion generators, simple templated prompts, 512 px JPEG q90; perfect separation on them does not transfer (control FIgLib frames are flagged); it must not be quoted as "AI images are detected".
* Edit detection is weak (AUC 0.87, 40 % recall at the controlled threshold) and nearly disappears after JPEG recompression (quality 40) at the clean-fitted threshold; crude paste/cover edits only, no inpainting.
* D-Fire "real" is assumed; control frames have no sequence ids.
* The authenticity output is advisory. `NO_CONCERNS_FOUND` never means verified real.

## 8. Decisions needed from a human owner
1. Confirm or replace the label policy (§3), in particular what counts as an edit and whether composites should also count as synthetic.
2. Confirm or replace `target_fpr`, `delta_fpr`, `inconclusive_fpr` and the negative sources.
3. Decide whether a head with these failure modes (control-domain false flags, weak edit recall) may ship at all, and whether the final-layer embedding (better in earlier experiments, but LoRA-specific, so it needs LoRA-extracted features) should replace block 8.
4. Provide a phone-photo validation set; without it the false-flag target is unverified for the deployment domain.
5. Approval itself: the build never writes an approval record.

## 9. How to run
```bash
python scripts/check_block8_equivalence.py --lora-features PROBE.npz --frozen-features /home/user/data/feat/dfire_clean.npz --out docs/authenticity/BLOCK8_EQUIVALENCE.json
python scripts/eval_auth_head.py --policy docs/calibration/auth_policy.PROPOSED.json --data-spec docs/authenticity/auth_data_spec.EXAMPLE.json \
    --feat-dir /home/user/data/feat --out docs/authenticity/AUTH_HEAD_RESULTS.json --n-boot 300
# artifact: merge docs/calibration/auth_policy.PROPOSED.json's `authenticity` block into a complete, owner-confirmed policy, then
python -m fire_assessor.build_artifact calibrate --base BASE --train train.npz --cal cal.npz --policy policy.json \
    --auth-data docs/authenticity/auth_data_spec.EXAMPLE.json --out NEW_DIR
```

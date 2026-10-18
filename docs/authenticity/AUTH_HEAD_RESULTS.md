# Authenticity head: held-out results

Machine-readable twin: [`AUTH_HEAD_RESULTS.json`](AUTH_HEAD_RESULTS.json) (produced by `scripts/eval_auth_head.py`, 300 bootstrap resamples, seed 0). Policy: the **PROPOSED** values in `docs/calibration/auth_policy.PROPOSED.json` (not an owner decision). Head: block-8 CLS of the **frozen base model**, fitted on train, L2 picked on cal, temperature + threshold + band picked on val; test, held-out generators, held-out source and control were never read by the fit (enforced and tested). Confidence intervals resample whole groups (D-Fire cluster / source photo; generated and control images are singletons). **This is not evidence that the head is fit for operational use** (see section 9).

## 1. Operating point chosen on the validation split
| | `synthetic` | `edited_or_composited` |
|---|---|---|
| L2 (chosen from {0.001, 0.01, 0.1, 1}) | 1.0 | 0.01 |
| temperature | 0.167 | 1.33 |
| threshold (calibrated probability) | 0.032 | 0.860 |
| INCONCLUSIVE band [lo, threshold) | [0.00007, 0.032) | [0.648, 0.860) |
| validation real groups / flagged | 446 / 16 (3.6 %, one-sided 90 % bound 5.0 %) | 446 / 16 (3.6 %, bound 5.0 %) |
| validation positives caught at threshold | 100 % (84) | 41.6 % (322) |
| validation ROC-AUC | 1.000 | 0.869 |
| real groups in the band (validation) | 11.4 % | 11.4 % |

The synthetic output separates validation almost perfectly, so its temperature and threshold are extreme; the threshold is an FPR-controlled operating point, not a probability of being fake.

## 2. Held-out discrimination (test split; 95 % grouped-bootstrap CI)
| Task | n pos / n neg | ROC-AUC | PR-AUC |
|---|---|---|---|
| synthetic, seen generators (SD 1.5, SDXL) | 179 / 579 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| synthetic, **held-out generators** (PixArt-alpha, Kolors) | 400 / 579 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| edited, all types | 326 / 800 | 0.866 [0.849, 0.888] | 0.786 [0.760, 0.820] |
| edited, `paste_flames` | 139 / 800 | 0.850 [0.818, 0.886] | 0.608 [0.528, 0.684] |
| edited, `paste_smoke` | 128 / 800 | 0.902 [0.873, 0.935] | 0.778 [0.720, 0.842] |
| edited, `cover_region` | 59 / 800 | 0.827 [0.776, 0.871] | 0.373 [0.249, 0.502] |

Negatives for `synthetic` are the 579 real (normalised) test images of the synthetic set; for `edited` the 800 untouched originals and benign edits of the edit set's test split. PR-AUC depends on the evaluation prevalence (24 % positives in the seen-generator row, 41 % in the held-out-generator row, 29 % for edited), not on field prevalence.

Held-out edit type (evaluation-only refit **without** `cover_region`): all-type AUC 0.852; `cover_region` AUC 0.790 (0.827 when seen); detection of `cover_region` at that fit's own threshold 11.9 % (20.3 % when seen).

Caution on the perfect `synthetic` numbers: they are an easy case (four open diffusion models, templated prompts, all images 512 px JPEG q90, real images from D-Fire) and are confounded by scene content and style; section 3 (control rows) shows they do not transfer to other real photos. They must not be quoted as "AI images are detected".

## 3. Operating-point behaviour: false flags on real images and detection of fakes
Share flagged at the validation-selected thresholds (95 % grouped-bootstrap CI). The policy target (5 %) refers to the validation mix of real groups; per-source rates differ.

| Real images | n | flagged `synthetic` | flagged `edited` | any flag or INCONCLUSIVE |
|---|---|---|---|---|
| synthetic-set reals (D-Fire, normalised, test) | 579 | 5.2 % [3.6, 6.7] | 0.3 % [0.0, 0.9] | 24.9 % |
| edit-set originals (test) | 400 | 2.5 % [1.0, 4.1] | 2.8 % [1.2, 4.2] | 25.8 % |
| edit-set benign edits (test) | 400 | 2.5 % [1.0, 4.2] | 0.8 % [0.0, 1.5] | 17.8 % |
| raw D-Fire WEB (test) | 1,208 | 1.7 % [0.9, 2.2] | 3.7 % [1.7, 6.8] | 24.1 % |
| raw D-Fire, all (test) | 1,310 | 1.8 % [1.1, 2.5] | 3.6 % [1.7, 6.2] | 24.7 % |
| raw D-Fire AoF held-out source | 8,428 | 0.0 % | 1.1 % | 8.8 % |
| **control: FIgLib frames** | 500 | **16.8 % [13.5, 20.3]** | 0.0 % | **74.6 %** |
| control: AoF frames | 500 | 0.4 % [0.0, 1.0] | 0.0 % | 8.2 % |
| control, all | 1,000 | 8.6 % [6.7, 10.2] | 0.0 % | 41.4 % |

AoF is a single group, so its bootstrap interval is degenerate; control frames are treated as independent although fixed-camera frames are not, so those intervals are optimistic. Status mix the pipeline would report from the head alone (no other signals): real test images 76.9 % `NO_CONCERNS_FOUND`, 18.3 % `INCONCLUSIVE`, 3.6 % `POSSIBLE_SYNTHETIC`, 1.2 % `POSSIBLE_EDITED`; control 58.6 % / 32.8 % / 8.6 % / 0 %.

Detection of fakes at the threshold (test + held-out generators):
| Positive | n | flagged | in INCONCLUSIVE band |
|---|---|---|---|
| SD 1.5 (seen) | 119 | 100 % | 0 % |
| SDXL (seen) | 60 | 100 % (one-sided 95 % lower bound 95 %) | 0 % |
| **PixArt-alpha (held out)** | 200 | 100 % (lower bound 98.5 %) | 0 % |
| **Kolors (held out)** | 200 | 100 % (lower bound 98.5 %) | 0 % |
| edit `paste_flames` | 139 | 30.2 % [23.4, 37.4] | 33.8 % |
| edit `paste_smoke` | 128 | 60.9 % [52.3, 68.8] | 11.7 % |
| edit `cover_region` | 59 | 20.3 % [11.9, 29.7] | 35.6 % |
| all edit types | 326 | 40.5 % [35.6, 45.3] | n/a |

Of the edited images, 39.3 % are reported `POSSIBLE_EDITED`, 2.1 % `POSSIBLE_SYNTHETIC`, 31.0 % `INCONCLUSIVE` and 27.6 % `NO_CONCERNS_FOUND`. Informational: 17.1 % of generated images also trip the `edited` output (its label is undefined for them).

## 4. INCONCLUSIVE share
On the validation real groups the band holds 11.4 % (by construction about `inconclusive_fpr` minus the flag rate). On held-out real test images 17.8 %-25.8 % of images are flagged or inconclusive, 74.6 % of FIgLib frames. A deployment where most real images are `INCONCLUSIVE` or flagged gives reviewers little: this is the practical cost of the 15 % band at the proposed values.

## 5. Robustness (same images degraded before feature extraction; same head, same clean-fitted threshold)
| Variant | `synthetic` AUC seen / held-out gen | `synthetic` detection sd15 / sdxl / pixart / kolors | `synthetic` false flags on synthetic-set reals | `edited` AUC | `edited` detection (all types) | `edited` false flags on edit-set originals |
|---|---|---|---|---|---|---|
| clean | 1.000 / 1.000 | 100 / 100 / 100 / 100 % | 5.2 % | 0.866 | 40.5 % | 2.8 % |
| jpeg40 | 1.000 / 1.000 | 96.6 / 95.0 / 99.0 / 100 % | 0.0 % | 0.806 [0.786, 0.828] | **0.6 %** | 0.0 % |
| blur2 | 0.990 / 0.993 | 98.3 / 100 / 100 / 99.5 % | **11.1 %** | 0.790 [0.767, 0.815] | 22.1 % | 2.0 % |
| resize50 | 0.998 / 0.999 | 98.3 / 100 / 100 / 99.5 % | 4.7 % | 0.828 [0.806, 0.852] | 38.7 % | 2.8 % |

Reading: ranking quality degrades moderately; the clean-fitted threshold does not. The `edited` output flags almost nothing after JPEG quality-40 recompression (any realistic upload pipeline recompresses), and blur doubles the `synthetic` false-flag rate. Threshold and temperature are not robust to post-processing; they would need re-selection on data that reflects the deployment pipeline.

## 6. Sensitivity to the LoRA-vs-frozen block-8 difference
Equivalence check (`BLOCK8_EQUIVALENCE.json`): cosine >= 0.99998, relative L2 0.27 % median / 0.64 % max on 32 images, not bitwise identical. Injecting isotropic Gaussian noise of those sizes into all features changes the `synthetic` logit by at most 0.013-0.028 and the `edited` logit by at most 0.087-0.207 (medians 0.002-0.005 and 0.016-0.038); decisions flip on 0.00-0.02 % (`synthetic`) and 0.09-0.17 % (`edited`) of test/held-out/control rows. Isotropic noise is only an order-of-magnitude proxy for the real structured difference. Not verified: the head applied to LoRA-extracted features on the full evaluation sets (no such features exist).

## 7. Sensitivity: raw D-Fire as additional negatives (not proposed)
Training and validating also on raw D-Fire (`negative_sources` + `dfire`) makes things worse: false flags on synthetic-set reals 15.5 % (vs 5.2 %), on FIgLib control 34.6 % (vs 16.8 %), edited AUC 0.833 (vs 0.866). The threshold, chosen on a different negative mix, no longer transfers to the processing-matched images. This is why raw D-Fire is evaluation-only in the proposal.

## 8. Reproducibility
Same inputs and seed give identical head arrays and byte-identical `arrays.npz` (tested; also checked on the real data through the build helper `_build_auth_head`, twice). The evaluation script's status rule matches `assess_authenticity` on 400 random rows (0 mismatches).

## 9. Limitations (plain)
* **No phone-photo evaluation.** Every number is on D-Fire-derived images, FIgLib/AoF camera frames and four open diffusion generators.
* **Generated-image detection does not generalise to unseen real domains.** Perfect AUC on held-out PixArt/Kolors coexists with 16.8 % of real FIgLib frames flagged as synthetic and 74.6 % flagged or inconclusive. Held-out generators here are two more open diffusion models with the same prompts; newer or closed generators, other prompts and resolutions are untested.
* **Edit detection is weak** (40 % recall at about 2-3 % false flags on its own distribution; 20 % for `cover_region`), only crude paste/cover edits were tested (no inpainting, no generative fill), and recall collapses after JPEG recompression.
* **The false-flag bound is conditional on exchangeability** with the validation real groups and is violated here by FIgLib. Per-source false-flag rates range from 0 % to 16.8 % against a 5 % target.
* **Features come from the frozen base model**; the production tap is the LoRA checkpoint's. They are equal up to measured numerical noise (0.27 % median relative L2) on 32 images.
* Calibrated probabilities describe the validation mix, not field prevalence; the `synthetic` output is near-separable so its temperature is extreme.
* D-Fire reals are assumed real and unedited; control sequence structure is unknown; AoF is one group.
* Earlier experiments (`docs/experiments/03`, `06`) found final-layer embeddings better than block 8 for both tasks and block-8 unstable under source shift; block 8 is used here because it is the only tap the LoRA checkpoint leaves unchanged.
* The artifact built in this mode ends **NOT_APPROVED**; nothing in this work is an approval, and the head is **not production-ready**.

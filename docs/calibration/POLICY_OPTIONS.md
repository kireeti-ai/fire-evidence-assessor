# Policy options for the fire-report assessment

Status: **for the project owner to choose from. Nothing here is approved and no approval record exists.** All numbers are measured on held-out exports with each option's frozen thresholds (`scripts/compare_policies.py`; raw results `docs/validation/policy_comparison.json`). Nothing was tuned on them.

## How to read the labels
The model never accepts or rejects a report. Every outcome is REQUIRES_REVIEW. The policy only decides the **label and queue priority**:

| Label | Priority | Meaning |
|---|---|---|
| ACTIVE_FIRE / SMOKE_ONLY | HIGH | a score is above its threshold |
| UNCERTAIN | HIGH if above threshold or inside the margin, else STANDARD | score near the threshold, or a quality or unfamiliar-photo warning |
| NO_VISIBLE_FIRE | STANDARD | both scores below their thresholds. **Not proof of absence** |

"Miss" below means a photo that truly shows fire or smoke but is labelled NO_VISIBLE_FIRE (so it would sit at STANDARD priority). "False alarm" means a photo with neither, labelled ACTIVE_FIRE or SMOKE_ONLY.

## The two options (and one diagnostic)
| | Option A (current) | Option B (stricter) | B2 (diagnostic only) |
|---|---|---|---|
| Target false-negative rate alpha | 0.05 | **0.02** | 0.02 |
| Confidence delta | 0.10 | 0.10 | 0.10 |
| Uncertainty margin | 0.05 | 0.05 | **0.01** |
| Flames threshold (calibrated probability) | 0.168 | 0.020 | 0.020 |
| Smoke threshold | 0.658 | 0.271 | 0.271 |
| Policy file | `calibration_policy.PROPOSED.json` | `calibration_policy.PROPOSED_ALPHA002.json` | `calibration_policy.DIAGNOSTIC_ALPHA002_MARGIN001.json` |

Binding digests (not approved): A `1fb451b3...11fa`, B `1ef5fc46...ef8f`, B2 `2cd8eeb0...beb6`. B2 is not one of the two requested options. It is included only because it shows what the margin does to B (below).

## Measured trade-offs

**In distribution (test partition, 1,310 images, 829 with fire or smoke)**
| | A | B | B2 |
|---|---|---|---|
| Miss (real fire/smoke labelled NO_VISIBLE_FIRE) | 1.2% [0.5, 2.0] | 0.0% | 0.1% [0, 0.5] |
| Flames head false-negative rate | 2.5% | 0.4% | 0.4% |
| Smoke head false-negative rate | 6.0% | 2.3% | 2.3% |
| False alarm (negatives labelled fire/smoke) | 1.7% [0.6, 2.9] | 4.4% [2.8, 6.3] | 11.6% [8.7, 14.4] |
| UNCERTAIN share of all images | 3.1% | **55.2%** | 22.0% |
| HIGH-priority share of all images | 63.6% | **100%** | 80.1% |
| Flames flag precision | 0.90 | 0.59 | 0.59 |

**Held-out fixed-camera source (8,428 images; the closest stand-in we have for an unseen camera)**
| | A | B | B2 |
|---|---|---|---|
| Miss | **37.3% [32.5, 42.7]** | 0.0% | 19.4% [15.6, 23.5] |
| Flames / smoke head false-negative rate | 29.5% / 51.5% | 13.8% / 34.6% | 13.8% / 34.6% |
| False alarm | 20.8% [16.7, 24.0] | 5.2% | 35.6% [30.8, 40.1] |
| UNCERTAIN share | 4.6% | **86.1%** | 19.7% |
| HIGH-priority share | 40.0% | **100%** | 66.9% |

## Control Room workload, per 1,000 uploaded images
Assumes the share of uploads that really show fire or smoke. **That share is unknown in the field**; these are projections from the rates above, not forecasts. Columns: HIGH-priority items / real fire among them / real fire labelled NO_VISIBLE_FIRE / UNCERTAIN items.

*In-distribution rates*
| Real share | A | B | B2 |
|---|---|---|---|
| 5% | 77 / 49 / 0.6 / 14 | 1000 / 50 / 0 / 924 | 487 / 50 / 0.1 / 333 |
| 20% | 221 / 198 / 2.4 / 18 | 1000 / 200 / 0 / 829 | 567 / 200 / 0.2 / 304 |

*Held-out-camera rates*
| Real share | A | B | B2 |
|---|---|---|---|
| 5% | 266 / 31 / 19 / 40 | 1000 / 50 / 0 / 937 | 588 / 40 / 10 / 218 |
| 20% | 323 / 125 / 75 / 43 | 1000 / 200 / 0 / 905 | 623 / 161 / 39 / 209 |

## What this shows
1. **B as specified is unusable for triage.** Alpha 0.02 pushes the flames threshold down to 0.020. The 0.05 margin is wider than the threshold itself, so most scores fall inside the uncertainty band and every item becomes HIGH priority. The queue is no longer ordered. Making alpha stricter only helps if the margin is narrowed with it. That is the point of B2.
2. **A stricter alpha buys fewer misses at a large workload cost.** B2 vs A in distribution: misses fall from 1.2% to 0.1%, but false alarms rise from 1.7% to 11.6% and the HIGH-priority queue grows from 64% to 80% of items. On the unseen source, misses fall from 37% to 19% while false alarms rise from 21% to 36%.
3. **Neither option is safe if NO_VISIBLE_FIRE lowers anyone's attention.** On the unseen source, option A labels about 37% of real fire or smoke photos NO_VISIBLE_FIRE. A false-negative guarantee set on one dataset does not hold on another camera source. **Recommended rule under either option: the assessment may raise attention but must never lower it. Every report is reviewed by a person regardless of its label.** Then "miss" costs a slower queue position, not a missed report.
4. **The in-distribution numbers are optimistic.** They come from the same kind of images the model was trained and calibrated on.

## Recommendation (the owner decides)
- For shadow mode, run **A** as the displayed-to-nobody baseline and compute **B2** offline from the same stored scores, so both can be compared against Control Room decisions without a second deployment. Do not use B as specified.
- Choose the final policy only after reviewing the phone-photo pilot (`docs/validation/phone_pilot/`) and the shadow-mode report.
- Changing alpha or margin means rebuilding the bundle, which changes its digest and voids any approval.

## Not decided here
Out-of-distribution (OOD) detection stays disabled: no independent control set has been passed through this checkpoint, so it cannot be validated. Authenticity detection is excluded from fire-report decisions (see `docs/integration/`).

## Reproduce
```
python scripts/compare_policies.py --option A=<bundle A> --option B=<bundle B> --option B2=<bundle B2> --exports <export dir> --out-dir <out>
```
Bundles are built with `build_artifact calibrate --policy <file above>` from the same base and exports (`docs/artifact/PROMOTION_RUNBOOK.md`).

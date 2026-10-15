# 03. Authenticity: edited / composited images (Stages 2-3)

**Data.** Edits are built from D-Fire inside each partition: `paste_flames`, `paste_smoke` (feathered paste of a fire/smoke box from another photo) and `cover_region` (a fire/smoke box covered by a patch from elsewhere in the image). Negatives are untouched originals and benign edits (crop, brightness, colour, resize, JPEG). Label = `edited_or_composited`. `cover_region` is held out for the unseen-edit-type test. **The edits are crude splices; realistic inpainting would be harder.**

Operating threshold = the value giving 5% false flags on *validation negatives*; test is never used for tuning. 95% CIs resample whole clusters.

## AUC on the test partition
| Method | All edit types | Seen types | Held-out type (`cover_region`) |
|---|---|---|---|
| Non-learned forensic cues (noise, tiles, blockiness, ELA) | 0.613 [0.598, 0.632] | 0.632 [0.610, 0.657] | 0.528 [0.468, 0.603] |
| CLIP block-8 CLS | 0.865 [0.848, 0.887] | 0.873 [0.852, 0.896] | 0.798 [0.741, 0.850] (trained without that type) |
| CLIP final embedding | 0.896 [0.876, 0.916] | 0.905 [0.885, 0.928] | 0.826 [0.768, 0.874] (trained without that type) |

## Robustness to degradation (head trained on clean images; test partition)
| Features | Degradation | AUC | Edits caught at the 5%-false-flag threshold | False flags on originals |
|---|---|---|---|---|
| block-8 | none | 0.865 | 52.1% | 5.0% |
| block-8 | jpeg40 | 0.822 | 1.8% | 0.0% |
| block-8 | resize50 | 0.832 | 46.0% | 6.0% |
| block-8 | blur2 | 0.800 | 29.4% | 2.8% |
| final | none | 0.896 | 69.0% | 7.2% |
| final | jpeg40 | 0.842 | 43.6% | 1.8% |
| final | resize50 | 0.884 | 70.9% | 8.5% |
| final | blur2 | 0.857 | 72.1% | 14.2% |

## Shortcut control
Fire presence could leak the label (pasted edits always contain fire/smoke). Fire-bearing base images are 66% of negatives and 67% of paste positives. Restricting train and test to fire-bearing bases: block-8 AUC 0.868 [0.841, 0.896], final-layer AUC 0.898 [0.872, 0.925]; the result does not depend on fire presence.

## Findings
- Non-learned cues are close to chance on the unseen edit type. CLIP features are far better.
- **Final-layer features beat block-8 here**, contrary to the design hypothesis. The intervals overlap partly, so this is a modest advantage on crude edits, not a settled answer.
- Block-8 features lose almost all recall after JPEG recompression; the final embedding keeps a fraction of it.
- Not yet tested: realistic inpainting, and AI-generated images (see 06).

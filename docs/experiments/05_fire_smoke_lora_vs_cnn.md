# 05. Fire and smoke: LoRA vs CNN baselines, WiSE-FT, calibration, risk control

Seeds per row are listed; mean +- sd is shown when there is more than one. LoRA = CLIP ViT-B/16 + rank-8 LoRA on blocks 9-12, head initialised from text prompts. CNNs are fully fine-tuned from ImageNet weights on the same splits.

| Method | Seeds | Set | Flames ROC | Smoke ROC | Flames PR | Smoke PR |
|---|---|---|---|---|---|---|
| lora | 3 | test | 0.991 +- 0.000 | 0.986 +- 0.000 | 0.987 +- 0.001 | 0.987 +- 0.001 |
| lora | 3 | xsrc_aof | 0.966 +- 0.002 | 0.698 +- 0.006 | 0.876 +- 0.003 | 0.548 +- 0.005 |
| resnet50 | 3 | test | 0.987 +- 0.002 | 0.975 +- 0.002 | 0.984 +- 0.001 | 0.979 +- 0.002 |
| resnet50 | 3 | xsrc_aof | 0.915 +- 0.031 | 0.463 +- 0.023 | 0.743 +- 0.050 | 0.359 +- 0.024 |
| efficientnet_b0 | 3 | test | 0.988 +- 0.003 | 0.977 +- 0.002 | 0.986 +- 0.003 | 0.979 +- 0.003 |
| efficientnet_b0 | 3 | xsrc_aof | 0.927 +- 0.006 | 0.511 +- 0.006 | 0.726 +- 0.035 | 0.401 +- 0.011 |

## WiSE-FT sweep (seed 0; alpha 0 = zero-shot weights, 1 = fully fine-tuned)
| alpha | test flames | test smoke | AoF flames | AoF smoke |
|---|---|---|---|---|
| 0.0 | 0.907 | 0.868 | 0.892 | 0.680 |
| 0.25 | 0.971 | 0.952 | 0.915 | 0.678 |
| 0.5 | 0.987 | 0.980 | 0.934 | 0.666 |
| 0.75 | 0.991 | 0.985 | 0.952 | 0.689 |
| 1.0 | 0.991 | 0.986 | 0.965 | 0.707 |

## Calibration and risk-controlled thresholds (LoRA seed 0, fit on the calibration partition only)
Target false-negative rate 5% with probability 90% (Learn-then-Test). **The target is a placeholder; the real value is a stakeholder decision.**

| Head | Set | FNR | Recall | Precision | Review volume | ECE before -> after |
|---|---|---|---|---|---|---|
| flames | test | 0.025 | 0.975 | 0.897 | 40.6% | 0.052 -> 0.020 |
| flames | xsrc_aof | 0.295 | 0.705 | 0.842 | 13.3% | 0.020 -> 0.040 |
| smoke | test | 0.060 | 0.940 | 0.959 | 57.7% | 0.041 -> 0.020 |
| smoke | xsrc_aof | 0.515 | 0.485 | 0.560 | 29.4% | 0.151 -> 0.195 |

The guarantee holds only under exchangeability. The `xsrc_aof` rows show what happens when it is violated by a source shift: the false-negative rate is many times the target, so shadow-mode monitoring on real uploads is mandatory before any priority use.

## Findings
- **In-distribution the three models are indistinguishable** (flames ROC about 0.99, smoke about 0.98). A model compared only on a random split would look equally good.
- **Under source shift CLIP+LoRA is clearly more robust than the ImageNet CNNs** (design question 1): on the held-out fixed-camera source, smoke ROC-AUC is about 0.70 for LoRA but roughly 0.5 (chance) for both CNNs, and flames is 0.97 vs 0.92 to 0.93. The three LoRA seeds agree closely, so this is not seed noise.
- **But CLIP+LoRA is not robust enough**: smoke at 0.70 on a new source is poor. WiSE-FT did not help here: the best AoF smoke score is at alpha = 1 (fully fine-tuned) and there is no intermediate sweet spot.
- **Calibration helps, the false-negative bound does not survive a source shift.** Temperature scaling lowers calibration error on test, and thresholds meet the 5% target for flames on test (smoke is slightly over). On the held-out source the false-negative rate is several times the target.
- Caveats: one dataset family, three seeds, cross-source test is a single fixed-camera source, and no citizen phone photos. These results support the design's *choice* of CLIP+LoRA over CNNs on this data; they are not deployment evidence.

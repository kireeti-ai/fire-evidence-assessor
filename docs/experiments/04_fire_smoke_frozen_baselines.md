# 04. Fire and smoke: frozen-CLIP baselines (zero-shot, linear probe)

CLIP ViT-B/16 (OpenAI weights, **QuickGELU config**) with no fine-tuning. Zero-shot = cosine to a positive minus a negative prompt ensemble. Linear probe = logistic regression on the final embedding (L2 chosen on validation). AoF is the held-out fixed-camera source; its CIs resample 25-frame blocks.

| Method | Set | Flames ROC-AUC | Smoke ROC-AUC | Flames PR-AUC (prevalence) | Smoke PR-AUC (prevalence) |
|---|---|---|---|---|---|
| zero_shot | test | 0.894 [0.871, 0.914] | 0.933 [0.911, 0.952] | 0.810 (0.37) | 0.937 (0.59) |
| zero_shot | xsrc_aof | 0.906 [0.871, 0.938] | 0.646 [0.600, 0.686] | 0.776 (0.16) | 0.567 (0.34) |
| linear_probe | test | 0.991 [0.985, 0.995] | 0.982 [0.971, 0.990] | 0.984 (0.37) | 0.985 (0.59) |
| linear_probe | xsrc_aof | 0.963 [0.949, 0.976] | 0.734 [0.695, 0.769] | 0.869 (0.16) | 0.585 (0.34) |

## Findings
- In-distribution the linear probe is near ceiling for both heads.
- **Smoke generalises poorly across sources** (large drop on the AoF fixed-camera set) while flames hold up.
- In-distribution numbers are optimistic: all test images come from the same web sources as training, even after cluster splitting.

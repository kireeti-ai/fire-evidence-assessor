# Artifact validation report

Artifact `fire-assessor-v1.0.0` status **NOT_APPROVED** (checkpoint `7bd5498f0483...`, artifact.json `d8773fd5a181...`).
Thresholds and temperatures were frozen before this run; nothing here tunes them. Intervals are 95% grouped (cluster) bootstrap.

## cal

calibration source (in-sample for thresholds). n=1310, groups=1310.

| Head | prevalence | ROC-AUC (CI) | PR-AUC (CI) | ECE raw to cal | FNR (CI) | precision (CI) | flagged |
|---|---|---|---|---|---|---|---|
| flames | 0.3603 | 0.9881 [0.9821, 0.993] | 0.9855 [0.9792, 0.9908] | 0.0453 to 0.0159 | 0.036 [0.0207, 0.054] | 0.8922 [0.8628, 0.9186] | 0.3893 |
| smoke | 0.6023 | 0.9824 [0.9741, 0.9893] | 0.983 [0.9723, 0.991] | 0.0395 to 0.0176 | 0.0393 [0.026, 0.0532] | 0.9607 [0.9479, 0.9734] | 0.6023 |

Derived labels (no quality or OOD gating): {'ACTIVE_FIRE': 481, 'UNCERTAIN': 55, 'SMOKE_ONLY': 325, 'NO_VISIBLE_FIRE': 449}

## val

checkpoint-selection partition (not independent). n=1310, groups=1310.

| Head | prevalence | ROC-AUC (CI) | PR-AUC (CI) | ECE raw to cal | FNR (CI) | precision (CI) | flagged |
|---|---|---|---|---|---|---|---|
| flames | 0.3473 | 0.9924 [0.9887, 0.9954] | 0.9869 [0.9809, 0.9917] | 0.0454 to 0.0191 | 0.022 [0.009, 0.037] | 0.89 [0.8621, 0.9132] | 0.3817 |
| smoke | 0.6275 | 0.9824 [0.9762, 0.9876] | 0.9876 [0.9813, 0.9921] | 0.0369 to 0.0178 | 0.0462 [0.0319, 0.0592] | 0.9561 [0.9428, 0.9676] | 0.626 |

Derived labels (no quality or OOD gating): {'ACTIVE_FIRE': 487, 'UNCERTAIN': 31, 'SMOKE_ONLY': 356, 'NO_VISIBLE_FIRE': 436}

## test

one-time in-distribution report. n=1310, groups=1310.

| Head | prevalence | ROC-AUC (CI) | PR-AUC (CI) | ECE raw to cal | FNR (CI) | precision (CI) | flagged |
|---|---|---|---|---|---|---|---|
| flames | 0.3733 | 0.9913 [0.9861, 0.9955] | 0.9885 [0.9819, 0.9936] | 0.0516 to 0.0198 | 0.0245 [0.0118, 0.0394] | 0.8966 [0.8681, 0.9214] | 0.4061 |
| smoke | 0.5885 | 0.9857 [0.9798, 0.9911] | 0.9867 [0.9782, 0.9935] | 0.0414 to 0.0197 | 0.0597 [0.044, 0.0765] | 0.959 [0.9447, 0.9728] | 0.5771 |

Derived labels (no quality or OOD gating): {'ACTIVE_FIRE': 511, 'SMOKE_ONLY': 281, 'UNCERTAIN': 41, 'NO_VISIBLE_FIRE': 477}

## xsrc_aof

held-out source: shift evaluation. n=8428, groups=380.

| Head | prevalence | ROC-AUC (CI) | PR-AUC (CI) | ECE raw to cal | FNR (CI) | precision (CI) | flagged |
|---|---|---|---|---|---|---|---|
| flames | 0.1583 | 0.9648 [0.949, 0.9768] | 0.8743 [0.807, 0.9167] | 0.0197 to 0.0403 | 0.2946 [0.2081, 0.4067] | 0.8417 [0.7666, 0.8903] | 0.1327 |
| smoke | 0.3396 | 0.7072 [0.6743, 0.7419] | 0.5531 [0.4876, 0.6237] | 0.1513 to 0.1946 | 0.5154 [0.4661, 0.5643] | 0.5597 [0.485, 0.6341] | 0.294 |

Derived labels (no quality or OOD gating): {'NO_VISIBLE_FIRE': 5061, 'ACTIVE_FIRE': 974, 'UNCERTAIN': 391, 'SMOKE_ONLY': 2002}

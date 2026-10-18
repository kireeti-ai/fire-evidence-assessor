# Calibration and evaluation readiness

Status: **BLOCKED on policy decisions and on a deployment-domain evaluation.** Nothing here was tuned on the test set.

## 1. Partition roles (D-Fire manifest, `partition_xsrc`)
| Partition | Rows | Role | May it tune thresholds? |
|---|---|---|---|
| `train` | 9,169 | fit LoRA + head; fit the OOD bank | no |
| `val` | 1,310 | checkpoint selection (early stopping) and the WiSE-FT / hyper-parameter choices | no (it already selected the checkpoint) |
| `cal` | 1,310 | temperature scaling, risk-controlled thresholds, OOD in-domain cutoff | **yes, the only partition that may** |
| `test` | 1,310 | one-time in-distribution report | **never** (no access log exists) |
| `xsrc_aof` | 8,428 | held-out fixed-camera source: shift evaluation only | **never** |
Enforced in code: `validate_export(role="cal")` blocks any export whose partition is `train`, `val`, `test`, `xsrc_aof` or unknown (`PARTITION_ROLE` / `PARTITION_UNKNOWN`).

## 2. Can calibration rows be mapped to manifest identities and image paths?
- **The stored `cal.npz` has no paths or IDs.** Matching order and counts is *not* treated as identity (`IDENTITY_NOT_ESTABLISHED` blocker).
- **Identity was established by content** (`scripts/verify_calibration_identity.py`): every manifest `cal` image was re-run through the checkpoint on CPU; each stored row's embedding matched its own image (cosine min 0.9996, median 0.9999993) and no other stored row matched within tolerance. **1,310 / 1,310 CONTENT_VERIFIED, 0 ambiguous, 0 mismatched.** Evidence: `docs/provenance/lora_s0_cal_identity_summary.json` and the per-row path map `lora_s0_cal_row_map.csv`. This also shows `cal.npz` came from this checkpoint.
- Caveats: verified in another environment (CPU, open_clip 3.3.0; logit differences up to 0.026); the file itself still carries no IDs, so the row map is the record. New exports (`fire_assessor.train export`) now store `paths`, `partition`, manifest hash, checkpoint SHA-256 and source hash.
- Real-file validation results: `docs/calibration/real_export_validation.json`. `cal.npz` alone: blocked; with the content-verification evidence: accepted. `val.npz`, `test.npz`, `xsrc_aof.npz` are rejected as calibration data.
- **Still blocking:** the **LoRA-feature train export does not exist** (`features/dfire_clean.npz` is frozen-base and must not be used), so train/cal disjointness cannot be recorded from files (it holds by the manifest: 0 of 10,670 clusters span partitions) and no OOD bank can be fitted.

## 3. Threshold-selection procedure (implemented, not yet run on approved inputs)
1. Per head, fit a temperature on `cal` logits (NLL, golden-section); report ECE and Brier before/after.
2. On temperature-scaled probabilities, choose the highest threshold whose false-negative rate is certified <= alpha with probability >= 1 - delta (Learn-then-Test, fixed-sequence exact binomial test over positives). If no threshold qualifies, **stop** (`INFEASIBLE`) instead of loosening alpha.
3. Record precision and review volume at the operating point; the uncertainty margin comes from the policy file.
4. The guarantee holds only if deployment data is exchangeable with `cal`. **It is not**: on the held-out source the false-negative rate was 29% (flames) and 52% (smoke) against a 5% placeholder target.

**Policy decisions required (no defaults):** alpha (target false-negative rate), delta (confidence), uncertainty margin, OOD mode, authenticity mode, who decided, when, reference. Template: `docs/calibration/calibration_policy.TEMPLATE.json`; `calibrate` refuses while any value is null.

## 4. OOD evaluation requirements
Three disjoint roles: (a) LoRA **train** features form the kNN bank; (b) `cal` in-domain features set the cutoff (95th percentile); (c) an **independent control** set of unrelated real images measures AUROC; the detector is enabled only if AUROC meets the owners' minimum, else it ships `DISABLED_BY_POLICY` with the reason. All three must be extracted through the LoRA checkpoint. Existing `control_clean.npz` is frozen-base: unusable. None of this has been run.

## 5. Known degradation under source shift (seed-0 checkpoint, from stored exports)
| | In-distribution test | Held-out fixed-camera source (AoF) |
|---|---|---|
| Flames ROC-AUC | 0.9913 | 0.9648 |
| Smoke ROC-AUC | 0.9857 | **0.7072** (PR-AUC 0.553) |
Three-seed results and the CNN comparison are in `docs/experiments/05_fire_smoke_lora_vs_cnn.md`. Smoke detection is the weak head on a new source.

## 6. Required before any operational claim
1. A **representative citizen-phone-photo evaluation**: consented or licensed phone-style photos with hard negatives (sunsets, fog, haze, burn scars, dry vegetation, cooking smoke), annotated by a documented labeling guide with measured inter-annotator agreement. No such set exists; D-Fire and generated images are not substitutes.
2. A calibration and threshold study repeated on that domain, then **shadow mode** on real uploads before any prioritization.
3. The test partition (and the phone-photo test set) used once per finalist with an access log.
4. The owners' approval of alpha, delta, margin, and the OOD and authenticity decisions.

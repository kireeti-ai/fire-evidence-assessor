# Promotion runbook: from research checkpoint to an artifact a production backend can load

Applies to `fire-assessor-v1.0.0` built from `runs/lora_s0/weights.pt`. This file records what was done and what only a human can do. **Nothing here approves the model.**

## Where things stand
| Item | State |
|---|---|
| Checkpoint | `runs/lora_s0/weights.pt`, SHA-256 `7bd5498f04838fe19240da2b7339d794cc5fc3948074a0d826c33303d5c42dfa`. Packaged byte-identical; never modified, never copied over |
| Retraining | Not needed and not done |
| Calibration | PROVIDED (temperature scaling + Learn-then-Test thresholds on the `cal` partition) from a **proposed** policy (`docs/calibration/calibration_policy.PROPOSED.json`), not an owner decision |
| OOD | DISABLED_BY_POLICY (no independent LoRA-feature control set) |
| Authenticity | v1.0.0: EXCLUDED_BY_POLICY (synthetic/edited/reused reported as not assessed). Optional research variant v1.1.0-authhead-research carries a block-8 head; see below |
| Approval | **none**. Status is NOT_APPROVED; the website adapter refuses to run it operationally |

## Reproducible build (inputs and hashes)
Inputs live in `gs://terrawatch-fire-ace-478304`:
| Object | Use |
|---|---|
| `runs/lora_s0/weights.pt` | the checkpoint (SHA-256 above) |
| `manifests/dfire_manifest.csv` | D-Fire manifest; `partition := partition_xsrc` gives manifest hash `c04b4b2f551957d759468cfa699cad827c394189bd65a65164d475b346dd2789` |
| `raw/dfire_kaggle/zip/smoke-fire-detection-yolo.zip` | images, SHA-256 `65d81b602991261b47ddc71281e99108362a30336a448e1f8d21352c825c6d83` |
| `work/artifact_build_20261018/export_{train,cal,val,test,xsrc_aof}.npz` | path-preserving LoRA feature exports (made on a GCP L4 VM; cal/val/test logits are bit-identical to the original research exports) |

Export hashes (SHA-256):
```
87f2144fa0e56f535f4734eb803993de9068cacf52566c6b4103b45116188e6c  export_cal.npz
2f15c81b9f64d5521a3ad4c76de74ab5418ea57afedb6241f34def181c048c4c  export_train.npz
463005d03a2da4b0d788a29f475eaf559f67023093cbd72019b2a66dc0cc3dbd  export_val.npz
6935933a696cf1a6f5a4e18df8361af9d25403adf794693223b1225c5ecf9562  export_test.npz
1f7f62045d3b8e72ac76d1325c8d22c8923b597fb8b3fb29216a258e8f3d0035  export_xsrc_aof.npz
```
`scripts/build_production_bundle.sh` runs: prepare, calibrate, verify, validate.

## What a human must do before enabling production
1. **Read** `docs/validation/validation_report.md`. In distribution the model is strong (test ROC-AUC flames 0.991, smoke 0.986). On the held-out fixed-camera source it is not (smoke ROC-AUC 0.707; false-negative rate 29% flames, 52% smoke). No citizen-phone-photo evaluation exists. The statistical guarantee behind the thresholds does not hold under that shift.
2. **Decide** whether to accept the proposed policy (alpha 0.05, delta 0.10, margin 0.05) or replace it. Replacing it means rebuilding the bundle (a new digest).
3. **Decide** the approved scope, for example: "advisory triage of citizen/ranger report images with mandatory Control Room review; not for dispatch decisions; not for community posts' fire logic".
4. **Write the approval record** (template: `approval_record.TEMPLATE.json`) into a **copy** of the bundle's `artifact.json` under `approval.record`, with `artifact_binding_sha256` from `python -m fire_assessor.artifact_tools digest <bundle>`.
5. **Verify**: `python -m fire_assessor.artifact_tools verify <bundle> --original-checkpoint <weights.pt>` must report `APPROVED` and `operational_load` accepted.
6. **Shadow mode first**: enable on the website with real uploads and compare assessments to Control Room decisions before relying on any label.

## Website configuration once step 5 passes
```
TERRAWATCH_ENV=production
FIRE_EVIDENCE_EXPERIMENTAL_ENABLED=false
FIRE_EVIDENCE_INFERENCE_ENABLED=true
FIRE_EVIDENCE_REPO=/absolute/path/to/fire-evidence-assessor
FIRE_EVIDENCE_ARTIFACT_DIR=/absolute/path/to/approved-artifact
FIRE_EVIDENCE_MODEL_VERSION=fire-assessor-v1.0.0
```
Human Control Room review remains authoritative; an assessment never changes incident status or dispatches anyone.


## Bundles built (both NOT_APPROVED; the website refuses both for operational use)
| Bundle | GCS prefix (new objects; the research run is untouched) | Binding digest | artifact.json | arrays.npz | weights.pt |
|---|---|---|---|---|---|
| `fire-assessor-v1.0.0` (authenticity excluded) | `artifacts/fire-assessor-v1.0.0-NOT_APPROVED/bundle/` | `1fb451b3f85c9c4f2463d1f323c3a0c51b5269f28c49e0d600931a5afb7511fa` | `71cbc70dcf76e2083a8b67bfea8aa943b7f4c73571f7c26fca6f6f3f215c5a77` | `e62b2f6de092c7615fa4a3c7cd2302a111acd1ba025027145e3e16056a70411e` | `7bd5498f...2dfa` |
| `fire-assessor-v1.1.0-authhead-research` (block-8 authenticity head) | `artifacts/fire-assessor-v1.1.0-authhead-research-NOT_APPROVED/bundle/` | `a42960d9a6e1c774e21c9a9092d21855da006de359b9c57afe42c5df019fc54b` | `18bead2cae977785f18aa85be4462e4c2c14a12d3486628317d362caaac0a911` | `e40cba3ee3cc3ed3565a5ee972e84247ef516e172b9e4b240149888d0494b740` | `7bd5498f...2dfa` |

Rebuilds are byte-identical (checked for v1.0.0). Validation evidence for v1.0.0: `docs/validation/`. Authenticity head evidence: `docs/authenticity/AUTH_HEAD_RESULTS.md`.

Build the head variant: `build_artifact prepare ... --model-version fire-assessor-v1.1.0-authhead-research`, then `calibrate ... --policy docs/calibration/policy_with_auth_head.PROPOSED.json --auth-data docs/authenticity/auth_data_spec.EXAMPLE.json` (paths in the data spec are local to the research machine).

### Do not ship the authenticity head without reading its results
On held-out data it flags about 16.8% of real FIgLib camera frames as synthetic (target 5%), detects about 40% of crude edits on clean images, and edit detection collapses after JPEG recompression. Its false-flag guarantee does not hold under source shift. Use it, if at all, for moderator triage where a false flag costs a human look, never as evidence. Sanity run through the real LoRA model on three images (an original, a flames-pasted edit and an SD 1.5 image) gave NO_CONCERNS_FOUND, POSSIBLE_EDITED and POSSIBLE_SYNTHETIC; three images are an anecdote, not a validation.

### End-to-end check on the website adapter
With `FIRE_EVIDENCE_INFERENCE_ENABLED=true` and the v1.0.0 bundle (no approval record), `assess_image_async` returns UNAVAILABLE with `MODEL_ARTIFACT_NOT_APPROVED`: the production path fails closed.

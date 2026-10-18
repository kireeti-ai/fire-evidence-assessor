# Promotion runbook: from research checkpoint to an artifact a production backend can load

Applies to `fire-assessor-v1.0.0` built from `runs/lora_s0/weights.pt`. This file records what was done and what only a human can do. **Nothing here approves the model.**

## Where things stand
| Item | State |
|---|---|
| Checkpoint | `runs/lora_s0/weights.pt`, SHA-256 `7bd5498f04838fe19240da2b7339d794cc5fc3948074a0d826c33303d5c42dfa`. Packaged byte-identical; never modified, never copied over |
| Retraining | Not needed and not done |
| Calibration | PROVIDED (temperature scaling + Learn-then-Test thresholds on the `cal` partition) from a **proposed** policy (`docs/calibration/calibration_policy.PROPOSED.json`), not an owner decision |
| OOD | DISABLED_BY_POLICY (no independent LoRA-feature control set) |
| Authenticity | see the bundle's `artifact.json` (`authenticity.status`) |
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

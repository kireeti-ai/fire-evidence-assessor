# TerraWatch website integration checklist (assessor side)

**Scope and honesty.** The actual TerraWatch `website` repository was **not accessible** in this session, and nothing was integrated into it. The only consumer code inspected is the public clone `RESQ-LoRa/dashboard` at `48789eeb` (a shallow, read-only checkout; it was **not modified**). Whether that repository is the `website` repository is **unverified**. Treat the observations below as evidence from the dashboard, and run the checklist against the real website repo. **Inference is not enabled anywhere.**

## A. What the dashboard already does correctly (verified by reading it)
| Property | Evidence (dashboard clone) |
|---|---|
| Fail-closed when unconfigured | `backend/app/services/image_assessment.py`: no `FIRE_EVIDENCE_REPO` / `FIRE_EVIDENCE_ARTIFACT_DIR` -> `UNAVAILABLE`, `MODEL_ARTIFACT_NOT_CONFIGURED`; test `backend/tests/test_image_assessment_contract.py` |
| Any load failure -> `UNAVAILABLE` | broad `except` in `_load_service`; uploads stay persisted |
| Advisory only | `citizen.py` returns "Automated assessment remains advisory."; no backend code reads `derived_label`, `review_priority` or `REQUIRES_REVIEW` to change incident state (grep found none) |
| Human review is a separate field | `resq_incident_images.human_review_status` (default `NOT_REVIEWED`), reviewer and time columns |
| Result persisted for audit | `assessment` JSONB + `assessment_status`, `model_version`, `authenticity_status` |

## B. Gaps and required changes (checklist for the website repo)
1. **Use this repo's gate, not a bypass.** The adapter must call `ModelArtifact.load(dir, expected_version)` with the **default** `require_operational=True` (it does today) and must **never** construct `Assessor(..., allow_unapproved=True)`. That flag is for offline research only; add a unit test that greps the backend for it.
2. **Distinct error codes.** Today every failure other than "not configured" is the exception class name (`ArtifactIntegrityError`, ...). Adopt `fire_assessor.adapter.ERROR_CODES` (`MODEL_ARTIFACT_NOT_FOUND`, `_INTEGRITY_FAILED`, `_MALFORMED`, `_INCOMPATIBLE`, `_INCOMPLETE`, `_NOT_APPROVED`, `MODEL_LOAD_FAILED`) or call `fire_assessor.adapter.GuardedAssessor` directly, and persist the code in `assessment_error`.
3. **Complete the UNAVAILABLE payload** so it always satisfies the contract: `decision: "REQUIRES_REVIEW"`, `review_priority`, `quality_flags`, `ood_flag`, `authenticity {status: INCONCLUSIVE}`, `limitations`, with **no `scores` and no `derived_label`**. `fire_assessor.adapter.unavailable(code)` produces exactly that.
4. **Never render or store an UNAVAILABLE result as "no fire".** UI copy: "Automated check unavailable"; the report proceeds to a ranger. Add a frontend test that `NO_VISIBLE_FIRE` is never shown when `assessment_status != COMPLETED`.
5. **Show the artifact status.** A `COMPLETED` result carries `artifact_status`; the UI must show a research banner for anything but `APPROVED` (the backend should in any case never receive one from the operational adapter).
6. **Blocking inference in an async route.** `upload_report_evidence` is `async def` and calls the synchronous `assess_image` in the event loop (`citizen.py`, `workflow.py`, `operations.py`). Before any real model is enabled, run inference in a worker/thread pool with a timeout and bounded retries (see `pipeline.AssessmentService`), so a slow model cannot stall the server. Timeouts must produce `UNAVAILABLE`, not an error to the citizen.
7. **Failure latch.** `_load_error` is cached for the process lifetime. Acceptable (fail-closed), but surface it in a health endpoint and document that replacing an artifact requires a restart.
8. **Supply chain.** The adapter prepends `FIRE_EVIDENCE_REPO/src` to `sys.path` and imports from it. Pin that repository to a commit hash, verify it at deploy time, and run it in a pinned environment (see D). Never point it at a mutable checkout.
9. **Configuration pin.** Set `FIRE_EVIDENCE_ARTIFACT_DIR` only to a verified, immutable, **APPROVED** bundle; change it together with `FIRE_EVIDENCE_MODEL_VERSION`; retain the previous directory for rollback; never overwrite a live bundle.
10. **No operational side effects from model output.** Keep the invariant that no code path changes incident status, dispatches, escalates, or rejects a report from an assessment. Add a backend test that inserts every possible assessment (including extreme scores) and asserts the incident row is unchanged. Citizens must not be penalised on a flag; wording stays neutral ("we may ask for more information").
11. **Audit.** Log `image_id`, `model_version`, `artifact_status`, `error_code`, latency; never image content. Record who reviewed (`human_reviewed_by`) when a ranger overrides.
12. **Privacy.** EXIF GPS is read by the assessor for consistency checks only; confirm retention and consent policy before any uploaded image is used for training.

## C. Acceptance tests to add in the website repo
- Artifact missing / empty / malformed / tampered weights / incomplete / unapproved / incompatible / model-build failure each -> `UNAVAILABLE` with the matching code and **no scores, no label** (mirror `tests/test_failclosed.py` here).
- An `APPROVED` bundle with a stub scorer -> `COMPLETED`, `decision == REQUIRES_REVIEW`, incident unchanged.
- `assess_image` never raises to the request handler.

## D. Runtime to pin before enabling anything (unresolved)
Python >= 3.10, `torch` >= 2.1, `open_clip_torch` >= 2.24, NumPy, Pillow, Pydantic; **exact versions must be locked and recorded** (the training environment was not). OpenCLIP `ViT-B-16-quickgelu` base weights are **not needed at inference** (the checkpoint contains the full vision tower; the loader builds the architecture with `pretrained=None`), but the OpenCLIP package is needed for architecture and preprocessing. Provide it from controlled storage; no runtime downloads.

## E. Reconciliation with the user's local assessor working copy
The dashboard docs describe assessor features (`fire_assessor.features`, `train --init-spec`, `build_artifact --initialization/--manifest-hash/--source-revision`) from a **local working copy not available here**. This repository now provides equivalents with a different shape (`features.py` validation, `build_artifact prepare/calibrate`, a provenance record embedded in the bundle). Reconcile the two before the website points at either; do not run both layouts against one bundle.

## F. Status
`ARTIFACT_STATUS: INCOMPLETE`; `OPERATIONAL_INFERENCE: DISABLED`; the adapter must stay fail-closed. Approval requires the evidence list in `docs/calibration/CALIBRATION_AND_EVALUATION_READINESS.md` section 6 and the unresolved items in `docs/provenance/SOURCE_AND_INITIALIZATION_SPEC.md`.

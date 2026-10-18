# Experimental (development-only) inference — EXPERIMENTAL / NOT FOR OPERATIONAL USE

Running the real network is **not** the same as having an approved fire-detection system. This path exists so developers can exercise the real
checkpoint locally. It does not approve, calibrate or validate anything, and the production artifact contract is unchanged.

## What it is
`src/fire_assessor/experimental.py`, a code path separate from `ModelArtifact` / `Assessor` / `GuardedAssessor`:

| Property | Behaviour |
|---|---|
| Accepts | one checkpoint whose SHA-256 equals the hash **the host pins in code**, documented by a provenance record (`docs/provenance/lora_s0_provenance.json`) naming that same hash and carrying no approval field; size and tensor names/shapes/dtypes must match the documented architecture (checked without executing pickle) |
| Returns | `assessment_status: "EXPERIMENTAL"`, `artifact_status: "EXPERIMENTAL_UNAPPROVED"`, raw uncalibrated `raw_logits` (flames_visible, smoke_visible), the notice `EXPERIMENTAL / NOT FOR OPERATIONAL USE`, `decision: REQUIRES_REVIEW` |
| Never returns | `COMPLETED`, `APPROVED`, probabilities, a derived label, thresholds, an OOD verdict, an authenticity verdict, an incident status, a dispatch hint |
| Refuses (explicit codes) | `MODEL_EXPERIMENTAL_NOT_FOUND`, `_PROVENANCE_INVALID`, `_INTEGRITY_FAILED`, `_INCOMPATIBLE`, `_LOAD_FAILED`, `_INFERENCE_FAILED`; bad images return the validation code (`EMPTY_FILE`, `UNSUPPORTED_TYPE`, `DECODE_FAILED`, `IMAGE_TOO_SMALL`, ...) with no logits |

It cannot create approval records, calibration, OOD state or artifact statuses, and it does not import `ModelArtifact.load`.

## Verified on the audited checkpoint (2026-10-16, CPU only)
- Checkpoint re-downloaded from `gs://terrawatch-fire-ace-478304/runs/lora_s0/weights.pt` (generation 1792100300386495, crc32c `dlSr1Q==`, 345,237,991 bytes):
  SHA-256 `7bd5498f04838fe19240da2b7339d794cc5fc3948074a0d826c33303d5c42dfa` — equal to the audited value.
- `python scripts/experimental_smoke.py <weights.pt>` — see the table below. Inputs are synthetic images only.

| Measurement | Value |
|---|---|
| Device | CPU (4 cores, 15 GB RAM); no GPU present; GPU **not** tested |
| Checkpoint SHA-256 + architecture verification | 0.4 s |
| Model load (build ViT-B/16 + LoRA, strict `load_state_dict`) | 6.6 s |
| Single-image inference, 640x480 JPEG, 15 runs | median 130 ms, p95 151 ms, min 109 ms |
| Peak RSS | about 1.03 GB (0.36 GB before load) |
| Determinism | two identical runs gave identical logits |
| Python / torch / torchvision / open_clip_torch / timm | 3.13.16 / 2.14.1+cpu / 0.29.1+cpu / 3.3.0 / 1.0.30 |
| numpy / pillow | 2.5.3 / 12.3.0 |
| transformers | not installed, not used |

Sample output (synthetic image of orange flame-like shapes; logits are raw, uncalibrated, and say nothing about real fires):
`{"assessment_status": "EXPERIMENTAL", "raw_logits": {"flames_visible": 1.8648, "smoke_visible": -2.1549}, "notice": "EXPERIMENTAL / NOT FOR OPERATIONAL USE"}`

The OpenCLIP warning `No pretrained weights loaded ... initialized randomly` is expected: the model is built uninitialised and every tensor is then
overwritten by the verified checkpoint using a strict `load_state_dict` (a missing or extra tensor would raise).

## Reproduce
```
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/experimental-inference.txt
pip install -e . --no-deps
python scripts/experimental_smoke.py /path/to/weights.pt --json smoke.json
FIRE_EVIDENCE_REAL_CHECKPOINT=/path/to/weights.pt python -m pytest tests/test_experimental.py
```
The checkpoint itself is never committed; obtain it from the bucket above and verify its SHA-256 first.

## Still blocking operational approval (unchanged)
No approved calibration or OOD configuration; no evaluation on deployment-like phone photographs (held-out-source flames AUC 0.965, smoke AUC 0.707;
observed false-negative rates under shift 29% flames / 52% smoke); training environment not fully recorded; training-time source revision not
retrievable; licensing of the data and the pretrained weights not resolved by a person; no human approval record. See `docs/provenance/lora_s0_provenance.json`.

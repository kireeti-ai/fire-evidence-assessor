# Fire Evidence Assessor artifact contract (schema `fire-assessor-artifact/1`)

An artifact is an **immutable directory** of exactly three files. It is consumed fail-closed: anything that is not a complete, integrity-checked, architecture-compatible and **human-approved** bundle produces `UNAVAILABLE`, never a result.

```
artifact.json   description, component status, provenance, limitations, approval block, SHA-256 of the other files
weights.pt      the checkpoint, byte-identical to the audited original (SHA-256 recorded and checked)
arrays.npz      OOD detector state / authenticity-head state when provided; otherwise only a schema marker
```

## artifact.json
| Field | Content |
|---|---|
| `schema_version`, `model_version` | `fire-assessor-artifact/1`; version string the consumer pins (`fire-assessor-v1.0.0`) |
| `checkpoint` | file name, **SHA-256** (must equal `files["weights.pt"]`), format |
| `architecture` | backbone, widths, layers, patch/image size, embedding dim, LoRA (rank, alpha, targets, zero-indexed blocks), head, block-8 tap |
| `preprocessing` | RGB, resize 224 bicubic, centre crop 224, OpenAI mean/std |
| `outputs` | names `[flames_visible, smoke_visible]`, **label mapping `0 -> flames_visible, 1 -> smoke_visible`**, semantics (raw logits; sigmoid after temperature) |
| `calibration` | `status` (`NOT_PROVIDED` / `PROVIDED`), method, **policy** (alpha, delta, margin, decided_by, decided_at, reference), **source** (partition `cal`, export SHA-256, row identity status), results; legacy keys `thresholds`, `temperatures`, `uncertainty_margin` are `null` unless provided |
| `ood` | kind, `status` (`NOT_PROVIDED` / `PROVIDED` / `DISABLED_BY_POLICY`), config, cutoff, reason |
| `authenticity` / `has_auth_head` | `status` (`NOT_PROVIDED` / `PROVIDED` / `EXCLUDED_BY_POLICY`), reason |
| `provenance` | the provenance record (`docs/provenance/lora_s0_provenance.json`): source identity, initialization, training manifest, config, environment gaps, evidence |
| `limitations` | known limitations, shown with every use |
| `approval` | `status`, `reasons`, `record` (human), `binding_sha256` |
| `files` | `{name: sha256}` for `weights.pt` and `arrays.npz` |

## Status model (derived at load, never trusted from the file)
- **INCOMPLETE**: a required component or provenance field is missing (calibration, an OOD decision, an authenticity decision, checkpoint hash, provenance).
- **NOT_APPROVED**: complete, but no valid approval record.
- **APPROVED**: complete **and** a record with approver, date, scope, evidence references and `artifact_binding_sha256` equal to this artifact's content digest (`python -m fire_assessor.artifact_tools digest DIR`). Any later edit to the content voids the approval.

No tool in this repository writes an approval record or sets `APPROVED`; that is a human act. `calibrate` yields at most `NOT_APPROVED`.

## Load rules (`ModelArtifact.load(dir, expected_version)`; `require_operational=True` by default)
1. `artifact.json` present, valid JSON, known schema, all required fields with correct types, safe file names, hex digests, well-formed thresholds/temperatures.
2. Every declared file exists and its SHA-256 matches; `checkpoint.sha256` equals the packaged `weights.pt`; version matches the pin; **no undeclared files** in the directory.
3. Output names and label mapping equal the assessor's (`flames_visible`, `smoke_visible`).
4. **Architecture check without PyTorch**: the checkpoint's tensor names, shapes and dtypes must equal the declared architecture exactly (restricted unpickler; pickle code is never executed).
5. Completeness (no blockers) and approval-binding checks. Failures raise `ArtifactFormatError`, `ArtifactIntegrityError`, `ArtifactIncompatibleError`, `ArtifactIncompleteError` or `ArtifactNotApprovedError`.
`Assessor(...)` independently refuses an artifact without calibration values, and one that is not APPROVED unless the caller passes `allow_unapproved=True` (offline research only; every output then carries `artifact_status` and a `RESEARCH USE ONLY` limitation).

## Tooling
```bash
# 1. INCOMPLETE bundle from the audited checkpoint (hash-checked, architecture-checked, original opened read-only)
python -m fire_assessor.build_artifact prepare --weights /path/weights.pt \
    --provenance docs/provenance/lora_s0_provenance.json --out artifacts/fire-assessor-v1.0.0-INCOMPLETE
# 2. read-only verification (exit 1 on integrity failure; reports blockers and whether an operational load would be refused)
python -m fire_assessor.artifact_tools verify DIR --original-checkpoint /path/weights.pt
# 3. only after the policy file is filled in by the owners and a path-preserving LoRA train export exists:
python -m fire_assessor.build_artifact calibrate --base DIR --train train.npz --cal cal.npz --policy policy.json \
    [--identity-summary docs/provenance/lora_s0_cal_identity_summary.json] [--control control.npz] --out NEW_DIR
```

## Verification results for the existing checkpoint (2026-10-16)
`artifacts/fire-assessor-v1.0.0-INCOMPLETE/` (git-ignored; 345 MB) was built from the audited checkpoint and verified:

| Check | Result |
|---|---|
| `weights.pt` SHA-256 | `7bd5498f04838fe19240da2b7339d794cc5fc3948074a0d826c33303d5c42dfa`, equals the declared and the original value |
| Packaged weights equal original | yes; original unchanged by packaging and by verification (size, mtime and hash identical before/after) |
| Architecture vs. tensors | compatible: 172 tensors, 86,292,996 parameters, exact name/shape/dtype set |
| Output mapping | `0 -> flames_visible`, `1 -> smoke_visible` |
| Status | **INCOMPLETE**, 3 blockers: calibration, OOD, authenticity all `NOT_PROVIDED` |
| Operational load | **REFUSED** (`ArtifactIncompleteError`) |

### What is deliberately not in the artifact, and why
- **Calibration** (thresholds, temperatures, margin): needs the owners' target false-negative rate and confidence (alpha, delta), the uncertainty margin, a path-preserving calibration export, and a LoRA-feature train export. None is approved or available. Values computed during analysis with placeholder targets are not packaged.
- **OOD**: needs an OOD bank from the LoRA checkpoint's training features (not generated), an in-domain cutoff from calibration data, and an independent control set (existing control features came from the frozen base model and are unusable).
- **Authenticity head**: the checkpoint has none; representation and label policy are owner decisions.
- **Approval**: none exists.

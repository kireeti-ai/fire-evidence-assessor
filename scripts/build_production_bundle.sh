#!/usr/bin/env bash
# Rebuild the NOT_APPROVED production bundle from the audited checkpoint and path-preserving exports.
# Never writes an approval record. Refuses to overwrite an existing output directory.
# Usage: scripts/build_production_bundle.sh WEIGHTS_PT EXPORT_DIR OUT_DIR [POLICY_JSON]
set -euo pipefail
WEIGHTS=${1:?weights.pt}; EXPORTS=${2:?dir containing the five export_PARTITION.npz files}; OUT=${3:?output dir}
POLICY=${4:-docs/calibration/calibration_policy.PROPOSED.json}
EXPECTED=7bd5498f04838fe19240da2b7339d794cc5fc3948074a0d826c33303d5c42dfa
[ ! -e "$OUT" ] || { echo "refusing: $OUT exists"; exit 2; }
mkdir -p "$OUT"
export PYTHONPATH=src
python3 -m fire_assessor.build_artifact prepare --weights "$WEIGHTS" --provenance docs/provenance/lora_s0_provenance.json \
  --out "$OUT/incomplete" --expected-sha256 "$EXPECTED"
python3 -m fire_assessor.build_artifact calibrate --base "$OUT/incomplete" --train "$EXPORTS/export_train.npz" \
  --cal "$EXPORTS/export_cal.npz" --policy "$POLICY" --out "$OUT/fire-assessor-v1.0.0"
python3 -m fire_assessor.artifact_tools verify "$OUT/fire-assessor-v1.0.0" --original-checkpoint "$WEIGHTS"
python3 scripts/validate_artifact.py --artifact "$OUT/fire-assessor-v1.0.0" --exports "$EXPORTS" --out-dir "$OUT/validation"
python3 -m fire_assessor.artifact_tools digest "$OUT/fire-assessor-v1.0.0"

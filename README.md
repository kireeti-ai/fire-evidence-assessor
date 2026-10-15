# TerraWatch Fire Evidence Assessor

Advisory triage for one uploaded photograph: fire/smoke evidence, uncertainty, and authenticity signals.
Implements `TerraWatch_Fire_Evidence_Assessor_Final_Design_v2.md`. It never confirms a wildfire and never auto-rejects.

**No deployment performance is claimed.** Measured results so far are on public benchmarks and generated images; see `docs/data/`.

## Layout
- `src/fire_assessor/` validation, quality gate, EXIF/provenance, pHash reuse index, calibration, risk-controlled thresholds,
  OOD, authenticity head, decision policy, pipeline + service wrapper, versioned artifact, CLIP+LoRA model, training.
- `scripts/` data manifest + splits, benchmarks, edit/synthetic set builders, feature extraction, evaluations.
- `docs/data/` license table, data card, result files.

## Experiments and notebooks
- `docs/experiments/` experiment write-ups, generated from `docs/data/*.json` by `scripts/make_experiment_docs.py` (numbers are copied, never retyped).
- `notebooks/` executed notebooks with charts: `01` data/splits/pHash, `02` edited-image detection, `03` fire/smoke models + calibration + risk control,
  `04` end-to-end pipeline demo (CPU, stub scorer). Rebuild with `python scripts/build_notebooks.py`.

## Run
```
pip install -e '.[dev]'          # CPU: everything but the model
pip install -e '.[dev,model]'    # torch + open_clip for training / inference
pytest
```
Backbone note: OpenAI CLIP weights must be loaded as `ViT-B-16-quickgelu` (`model.CLIP_NAME`); plain `ViT-B-16` silently degrades them.

## Data and splits
See `docs/data/DATA_CARD.md`. Splits are by near-duplicate cluster (pHash + sequence groups), never by image; D-Fire's own
Kaggle splits leak and must not be used. The fixed-camera AoF source is held out whole as a cross-source test.

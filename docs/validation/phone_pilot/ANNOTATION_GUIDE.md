# Phone-photo pilot: annotation guide

Purpose: label openly licensed phone photos so the assessor's fire and smoke outputs can be checked on phone-style images. This is a **pilot**. Its annotators are AI reviewers, not people, and it does not replace the human-annotated study required before approval (see `docs/calibration/CALIBRATION_AND_EVALUATION_READINESS.md`, section 6).

## What to label (look at the image only; ignore the file name, category and description)
| Field | Values | Rule |
|---|---|---|
| `flames` | 0 / 1 | 1 if an open flame is clearly visible anywhere in the frame, whatever the cause or size (wildfire, grass fire, bonfire, cooking fire, candle-sized flames do NOT count unless clearly a fire). 0 if only glow, sunset colours, lights, embers without flame, or burn scars |
| `smoke` | 0 / 1 | 1 if smoke from burning is clearly visible: a plume or haze that plausibly comes from fire (wildfire, grass or crop fire, structure fire, bonfire, cooking fire). 0 for fog, mist, clouds, steam, dust, vehicle exhaust, industrial chimney or cooling-tower plumes, and ordinary urban haze unless it clearly shows fire smoke |
| `context` | one of `wildfire`, `structure_or_urban`, `controlled_or_cooking_or_ceremony`, `industrial`, `no_fire_scene` | what the scene is. `wildfire` = vegetation or forest fire (including grass and crop fires) |
| `quality` | `ok` / `poor` | `poor` if too dark, blurred, tiny subject or heavily obstructed to judge |
| `unsure` | 0 / 1 | 1 if you cannot decide `flames` or `smoke` with reasonable confidence |

## Procedure
1. Label each image independently. Do not discuss images with the other annotator.
2. Give a one-line reason for every `flames=1`, `smoke=1` or `unsure=1` label.
3. Do not infer from the file name. Judge the pixels.
4. Agreement is measured (Cohen's kappa per field). Disagreements and `unsure` images are excluded from the primary evaluation and reported as a count; they are not resolved by the model's output.

## Limits to state wherever results are shown
AI annotators can share blind spots with the model being evaluated; images come from Wikimedia Commons (skewed toward photographers who upload to Commons); phone make and model come from EXIF and are not verified; sample size is small. None of this approves anything.

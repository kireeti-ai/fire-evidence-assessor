# Experiment log

| # | Experiment | Status |
|---|---|---|
| 01 | Data, splits and leakage control | done |
| 02 | pHash reuse detection | done (false-match rate not established) |
| 03 | Edited-image detection (block-8 vs final-layer) | done on crude splices |
| 04 | Fire/smoke frozen-CLIP baselines | done |
| 05 | LoRA vs CNN, WiSE-FT, calibration, risk control | done (3 seeds each) |
| 06 | AI-generated images across generators + fire-head exploitability | done (easy case; see caveats) |

Notebooks that reproduce the charts are in `../../notebooks/`.

## Process notes
- **QuickGELU bug.** OpenAI CLIP weights loaded into the plain `ViT-B-16` config degrade silently (open_clip warns). All CLIP numbers here were re-run with `ViT-B-16-quickgelu`; a unit test now guards it.
- **Circular metric caught.** A false-match rate measured inside clusters built with the same threshold was trivially zero and was removed.
- **Leakage found.** D-Fire's own splits leak; FIgLib frames are inside D-Fire.
- **Not claimed.** Nothing here is deployment performance. No citizen phone photos were used.

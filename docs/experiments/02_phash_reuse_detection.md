# 02. Perceptual-hash reuse detection

2,000 distinct images indexed; each query is a transformed copy; match = pHash Hamming <= 6.

| Transform | Detected | 95% CI | Median Hamming |
|---|---|---|---|
| identical | 100.0% | [1.000, 1.000] | 0 |
| resize_50 | 100.0% | [1.000, 1.000] | 0 |
| jpeg_q60 | 100.0% | [1.000, 1.000] | 0 |
| jpeg_q30 | 100.0% | [1.000, 1.000] | 0 |
| crop_90 | 25.9% | [0.240, 0.279] | 10 |
| crop_75 | 0.4% | [0.001, 0.006] | 22 |
| brightness_1.3 | 89.6% | [0.883, 0.910] | 2 |
| rotate_5 | 5.9% | [0.049, 0.069] | 14 |
| flip_h | 0.0% | [0.000, 0.000] | 32 |
| screenshot_like | 100.0% | [1.000, 1.000] | 0 |

## Findings
- pHash is robust to resize, JPEG and screenshot-style recompression and mostly robust to brightness, but **misses crops, rotations and flips**. Flips are now covered in code by also hashing the mirrored image (`phash_variants`, unit-tested but not re-benchmarked here); crops and rotations need the CLIP-embedding nearest-neighbour layer.
- **False-match rate is not established.** Within D-Fire it is circular (clusters use the same threshold).
- **D-Fire and FIgLib overlap:** 81 of 531 FIgLib pilot images match a D-Fire image at distance <= 2 (same HPWREN camera, see `docs/figures/dfire_figlib_overlap.png`). A 'train on D-Fire, test on FIgLib' cross-dataset test would leak unless those clusters are removed first.

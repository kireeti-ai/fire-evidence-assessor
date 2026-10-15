# 06. Authenticity: AI-generated images, and how exploitable the fire head is

**Data.** Generated with open-weight models whose licences were read from their model cards: SD 1.5 (CreativeML OpenRAIL-M) and SDXL (OpenRAIL++) for training; **PixArt-alpha (OpenRAIL++) and Kolors (Apache-2.0) held out entirely**. Prompts describe phone-style photos of flames, smoke and hard negatives (sunset, fog, haze, burn scar, dry vegetation, campfire, cooking smoke, clear sky). Real = D-Fire originals. Real and synthetic images are both normalised to longest side 512 + JPEG q90 so size and format are not shortcuts. Operating threshold = 5% false flags on real validation+calibration images; test is never used for tuning.

## Detection (AUC, 95% CI over clusters) and false-flag rate on real test images
| Features | Condition | Seen generators AUC | Held-out generators AUC | False flags on real |
|---|---|---|---|---|
| block-8 | clean | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 11.1% |
| block-8 | jpeg40 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 4.7% |
| block-8 | resize50 | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] | 65.6% |
| block-8 | blur2 | 0.994 [0.990, 0.997] | 0.996 [0.993, 0.998] | 79.8% |
| final layer | clean | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 7.4% |
| final layer | jpeg40 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.6% |
| final layer | resize50 | 0.997 [0.995, 0.999] | 0.998 [0.997, 0.999] | 22.6% |
| final layer | blur2 | 0.989 [0.984, 0.994] | 0.996 [0.994, 0.998] | 45.9% |

## Control: real photos from sources never seen in training (share flagged as synthetic)
| Features | FIgLib frames | AoF fixed-camera frames |
|---|---|---|
| block-8 | 59.2% | 1.2% |
| final layer | 1.6% | 0.0% |

## Fire/smoke linear probe applied to generated images (threshold = 5% false positives on real D-Fire negatives)
| Head | Generated fire/smoke images flagged | Real D-Fire test positives flagged | Generated negatives flagged |
|---|---|---|---|
| flames | 91.7% (n=494) | 95.7% | 13.1% (n=756) |
| smoke | 80.1% (n=820) | 87.9% | 9.5% (n=430) |

False positives on generated hard negatives (share flagged): flames head burn_scar 82%, clear 2%, cooking_smoke 25%, dry_veg 0%, fog 0%, haze 0%, sunset 0%; smoke head burn_scar 60%, campfire 0%, clear 0%, dry_veg 0%, fog 0%, haze 1%, sunset 0%.

## Findings and caveats
- All four generator families, including the two never seen in training, are separated from real photos almost perfectly **at this resolution and compression**. This is the easy case: four open diffusion models, simple prompts, 512 px JPEG q90. It says nothing about newer or closed generators, other prompts, or higher-resolution outputs, and AUC 1.0 should not be quoted as "AI images are detected".
- The control matters. The final-layer head leaves unseen real sources almost unflagged, so it is not merely rejecting anything that is not D-Fire. The block-8 head is unstable under source shift (most FIgLib frames flagged), which argues against it.
- Detection degrades under blur and half-resolution recompression, mainly through more false flags on real images.
- **The fire head is easy to fool.** Generated fire/smoke images are flagged about as often as real ones, so the authenticity status must always be shown with the fire score. It also flags generated burn scars (a hard negative) most of the time.

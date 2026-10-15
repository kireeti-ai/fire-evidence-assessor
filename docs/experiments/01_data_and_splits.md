# 01. Data, splits and leakage control

D-Fire (Kaggle mirror, CC0): **21,527** images. Near-duplicate clustering (pHash Hamming <= 6, plus 25-frame sequence blocks for the AoF fixed cameras) gives **10,670** clusters; the largest has **8,428** images (AoF frames chain together).

**The Kaggle train/val/test splits leak:** 414 clusters (10,997 images) span more than one of them. They are not used.

## Partitions used (`partition_xsrc`, recommended)
Clusters touching AoF are held out whole as the cross-source test set; the rest are cluster-split.

| Partition | Images | Flames=1 | Smoke=1 |
|---|---|---|---|
| train | 9,169 | 3,072 | 5,281 |
| val | 1,310 | 455 | 822 |
| cal | 1,310 | 472 | 789 |
| test | 1,310 | 489 | 771 |
| xsrc_aof | 8,428 | 1,334 | 2,862 |

Manifest hash (all splits, labels and paths): `76559aeed4590d94ae4e553bebe8a3a7b9dfa2cf42cd7f48174463c3c8ccbbb1`; seed 0.

## Caveats
- Sequence block size (25) and the Hamming threshold are assumptions, not ground-truth video IDs.
- None of these images are citizen phone photos.
- D-Fire labels are derived from boxes: `flames` = any fire box, `smoke` = any smoke box.

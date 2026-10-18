"""Numerical check that the block-8 CLS tap of the LoRA checkpoint equals the frozen-base tap used to fit the authenticity head.

usage: python scripts/check_block8_equivalence.py --lora-features probe.npz --frozen-features dfire_clean.npz --out BLOCK8_EQUIVALENCE.json
Rows are matched by `paths` (never by order). Both files need `block` and `paths`. Reports max/mean abs difference, per-row cosine and
relative L2 difference, and the largest difference in units of the per-dimension feature std of the frozen file."""
import argparse, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-features", required=True); ap.add_argument("--frozen-features", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p, f = np.load(a.lora_features, allow_pickle=False), np.load(a.frozen_features, allow_pickle=False)
    idx = {str(x): i for i, x in enumerate(f["paths"].tolist())}
    rows = [idx[str(x)] for x in p["paths"].tolist()]  # KeyError = an image of the LoRA file is missing from the frozen file
    A, B = p["block"].astype(np.float64), f["block"][rows].astype(np.float64)
    d = np.abs(A - B)
    cos = (A * B).sum(1) / np.linalg.norm(A, axis=1) / np.linalg.norm(B, axis=1)
    rel = np.linalg.norm(A - B, axis=1) / np.linalg.norm(B, axis=1)
    out = {"n_images": len(rows), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()), "cosine_min": float(cos.min()),
           "cosine_median": float(np.median(cos)), "relative_l2_max": float(rel.max()), "relative_l2_median": float(np.median(rel)),
           "max_abs_diff_in_units_of_feature_std": float((d / f["block"].std(0)).max()), "feature_std_mean": float(f["block"].std(0).mean()),
           "bitwise_identical": bool(np.array_equal(p["block"], f["block"][rows]))}
    try:
        import open_clip, torch
        out["this_environment"] = {"torch": torch.__version__, "open_clip": open_clip.__version__}
    except Exception:
        pass
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

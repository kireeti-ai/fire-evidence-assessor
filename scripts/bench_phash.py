"""Stage 1 benchmark: does pHash reuse detection catch transformed copies, and how often does it
false-match distinct photos? Index = non-duplicate images (one per cluster); queries = transformed copies."""
import argparse, csv, io, json
import numpy as np
from PIL import Image, ImageEnhance
from fire_assessor.phash import phash

def jpeg(q): 
    def f(im):
        b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")
    return f
def crop(frac):
    def f(im):
        w, h = im.size; dw, dh = int(w * (1 - frac) / 2), int(h * (1 - frac) / 2); return im.crop((dw, dh, w - dw, h - dh))
    return f
TRANSFORMS = {
    "identical": lambda im: im, "resize_50": lambda im: im.resize((max(32, im.width // 2), max(32, im.height // 2))),
    "jpeg_q60": jpeg(60), "jpeg_q30": jpeg(30), "crop_90": crop(0.9), "crop_75": crop(0.75),
    "brightness_1.3": lambda im: ImageEnhance.Brightness(im).enhance(1.3), "rotate_5": lambda im: im.rotate(5, expand=False),
    "flip_h": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    "screenshot_like": lambda im: jpeg(70)(im.resize((im.width * 3 // 4, im.height * 3 // 4))),
}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--max-hamming", type=int, default=6); ap.add_argument("--out", required=True)
    a = ap.parse_args(); rng = np.random.default_rng(0)
    rows = list(csv.DictReader(open(a.manifest)))
    by = {}
    for r in rows: by.setdefault(r["cluster"], r)  # one image per cluster: no native near-duplicates in the index
    reps = list(by.values()); idx = rng.choice(len(reps), min(a.n, len(reps)), replace=False); sel = [reps[i] for i in idx]
    ims = [Image.open(f"{a.root}/{r['path']}").convert("RGB") for r in sel]
    H = np.array([int(r["phash"]) for r in sel], dtype=np.uint64)
    # false-match rate: distinct-cluster pairs within threshold
    d = np.bitwise_count(H[:, None] ^ H[None, :]); np.fill_diagonal(d, 99)
    fm_pairs = int((d <= a.max_hamming).sum() // 2); n_pairs = len(sel) * (len(sel) - 1) // 2
    res = {"n_index": len(sel), "max_hamming": a.max_hamming,
           "false_match_pairs": fm_pairs, "false_match_rate_per_pair": fm_pairs / n_pairs,
           "images_with_any_false_match": int((d.min(1) <= a.max_hamming).sum()), "transforms": {}}
    for name, t in TRANSFORMS.items():
        hit = 0; top1 = 0; dist = []
        for i, im in enumerate(ims):
            q = np.uint64(phash(t(im))); dd = np.bitwise_count(H ^ q)
            hit += int(dd[i] <= a.max_hamming); top1 += int(np.argmin(dd) == i and dd[i] <= a.max_hamming); dist.append(int(dd[i]))
        n = len(ims); p = hit / n; se = (p * (1 - p) / n) ** 0.5
        res["transforms"][name] = {"detect_rate": p, "ci95": [max(0, p - 1.96 * se), min(1, p + 1.96 * se)],
                                   "correct_top1": top1 / n, "median_hamming": float(np.median(dist))}
        print(f"{name:16s} detect {p:.3f}  top1 {top1/n:.3f}  median_d {np.median(dist):.0f}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("false-match pairs", fm_pairs, "of", n_pairs, f"({fm_pairs/n_pairs:.2e}); images affected", res["images_with_any_false_match"])

if __name__ == "__main__":
    main()

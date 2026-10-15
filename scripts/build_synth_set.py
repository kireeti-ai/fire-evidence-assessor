"""Assemble real-vs-synthetic set. Real = D-Fire originals (one per cluster, partition_xsrc, AoF excluded) and
synthetic = generator outputs; BOTH normalised to longest side 512 + JPEG q90 so size/format are not shortcuts.
Generator families sd15, sdxl are split into train/val/cal/test; pixart, kolors are held out entirely
(partition `heldout_gen`). Real images for `heldout_gen` come from the real test partition (reused: no training use)."""
import argparse, csv, hashlib, os, random
from PIL import Image

def norm(im):
    s = 512 / max(im.size); return im.convert("RGB").resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dfire-manifest", required=True); ap.add_argument("--dfire-root", required=True)
    ap.add_argument("--synth", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    rng = random.Random(0); os.makedirs(a.out + "/img", exist_ok=True); out = []
    syn = []
    for fam in ("sd15", "sdxl", "pixart", "kolors"):
        p = f"{a.synth}/manifest_{fam}.csv"
        if os.path.exists(p): syn += list(csv.DictReader(open(p)))
    counts = {}
    for r in syn:
        if r["family"] in ("pixart", "kolors"): part = "heldout_gen"
        else:  # deterministic split by prompt/seed hash (generated images are independent samples)
            u = int(hashlib.md5((r["family"] + r["seed"]).encode()).hexdigest(), 16) % 100
            part = "train" if u < 60 else "val" if u < 70 else "cal" if u < 80 else "test"
        rel = f"img/{os.path.basename(r['path'])}"
        os.replace(f"{a.synth}/{r['path']}", f"{a.out}/{rel}") if False else Image.open(f"{a.synth}/{r['path']}").save(f"{a.out}/{rel}", quality=90)
        out.append(dict(path=rel, label_synthetic=1, family=r["family"], partition=part, cluster=f"gen_{r['family']}_{r['seed']}",
                        flames=r["flames"], smoke=r["smoke"], tag=r["tag"])); counts[part] = counts.get(part, 0) + 1
    rows = [r for r in csv.DictReader(open(a.dfire_manifest)) if r["partition_xsrc"] != "xsrc_aof"]
    by = {}
    for r in rng.sample(rows, len(rows)): by.setdefault((r["partition_xsrc"], r["cluster"]), r)
    real_pool = {}
    for (part, _), r in by.items(): real_pool.setdefault(part, []).append(r)
    need = {"train": counts.get("train", 0), "val": counts.get("val", 0), "cal": counts.get("cal", 0), "test": counts.get("test", 0) + counts.get("heldout_gen", 0)}
    for part, n in need.items():
        for i, r in enumerate(real_pool[part][:n]):
            rel = f"img/real_{part}_{i:05d}.jpg"; norm(Image.open(f"{a.dfire_root}/{r['path']}")).save(f"{a.out}/{rel}", quality=90)
            out.append(dict(path=rel, label_synthetic=0, family="real", partition=part, cluster=r["cluster"], flames=r["flames"], smoke=r["smoke"], tag=r["source"]))
    with open(a.out + "/synth_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    from collections import Counter; print(Counter((r["partition"], r["family"]) for r in out))

if __name__ == "__main__":
    main()

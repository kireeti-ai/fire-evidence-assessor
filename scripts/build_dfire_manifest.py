"""Build the leakage-controlled D-Fire manifest (Stage 1).

Derives per-image flames/smoke labels from YOLO boxes (class 0 smoke, class 1 fire), computes pHash,
merges the Kaggle train/val/test splits, clusters near-duplicates + source-sequence groups, and
re-splits whole clusters into train/val/cal/test.
"""
import argparse, csv, json, os, sys
from collections import Counter
from multiprocessing import Pool

import numpy as np
from PIL import Image

from fire_assessor.metrics import prf  # noqa: F401  (import check)
from fire_assessor.phash import phash
from fire_assessor.splits import assign_partitions, cluster_ids, manifest_hash, verify_no_leakage


def one(args):
    root, split, name = args
    stem = os.path.splitext(name)[0]
    lf = f"{root}/data/{split}/labels/{stem}.txt"
    cls = {l.split()[0] for l in open(lf).read().splitlines() if l.strip()} if os.path.exists(lf) else set()
    try:
        ph = phash(Image.open(f"{root}/data/{split}/images/{name}").convert("RGB"))
    except Exception:
        return None
    src = "".join(c for c in stem if c.isalpha())
    num = int("".join(c for c in stem if c.isdigit()) or 0)
    group = f"AoF_block{num // 25}" if src == "AoF" else ""  # sequential surveillance frames: conservative proxy
    return {"path": f"data/{split}/images/{name}", "flames": int("1" in cls), "smoke": int("0" in cls),
            "orig_split": split, "source": src, "group": group, "phash": str(ph)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-hamming", type=int, default=6)
    a = ap.parse_args()
    jobs = [(a.root, sp, n) for sp in ("train", "val", "test") for n in sorted(os.listdir(f"{a.root}/data/{sp}/images"))]
    with Pool(4) as p:
        rows = [r for r in p.imap(one, jobs, chunksize=64) if r]
    print("images", len(rows), "undecodable", len(jobs) - len(rows))
    cl = cluster_ids([int(r["phash"]) for r in rows], [r["group"] for r in rows], max_hamming=a.max_hamming)
    parts = assign_partitions(cl, seed=a.seed)
    verify_no_leakage(cl, parts)
    # Second scheme: hold the whole AoF source (fixed surveillance cameras) out as a cross-source test
    # and re-split the remaining clusters, so train/val/cal/test never see AoF.
    aof_clusters = {int(c) for c, r in zip(cl, rows) if r["source"] == "AoF"}
    keep = np.array([int(c) not in aof_clusters for c in cl])  # clusters touching AoF stay together
    parts_x = np.array(["xsrc_aof"] * len(rows), dtype=object)
    parts_x[keep] = assign_partitions(cl[keep], seed=a.seed)
    verify_no_leakage(cl, parts_x)
    for r, c, pt, px in zip(rows, cl, parts, parts_x):
        r["cluster"], r["partition"], r["partition_xsrc"] = int(c), pt, px
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sizes = Counter(cl.tolist())
    # how leaky were the original Kaggle splits? clusters spanning >1 original split
    span = {}
    for r in rows: span.setdefault(r["cluster"], set()).add(r["orig_split"])
    leaky = sum(len(s) > 1 for s in span.values())
    rep = {"n_images": len(rows), "n_clusters": len(sizes), "largest_cluster": max(sizes.values()),
           "clusters_spanning_original_splits": leaky,
           "images_in_those_clusters": sum(sizes[c] for c, s in span.items() if len(s) > 1),
           "manifest_hash": manifest_hash([{k: r[k] for k in ("path", "flames", "smoke", "partition")} for r in rows]),
           "per_partition": {pt: {"n": int((parts == pt).sum()),
                                  "flames": sum(r["flames"] for r in rows if r["partition"] == pt),
                                  "smoke": sum(r["smoke"] for r in rows if r["partition"] == pt)}
                             for pt in ("train", "val", "cal", "test")}, "max_hamming": a.max_hamming, "seed": a.seed,
           "per_partition_xsrc": {pt: {"n": int((parts_x == pt).sum()),
                                  "flames": sum(r["flames"] for r in rows if r["partition_xsrc"] == pt),
                                  "smoke": sum(r["smoke"] for r in rows if r["partition_xsrc"] == pt)}
                             for pt in ("train", "val", "cal", "test", "xsrc_aof")}}
    json.dump(rep, open(a.out.replace(".csv", "_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

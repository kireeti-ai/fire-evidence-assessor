"""Stage 2: build the edited / benign-edit / untouched image set from D-Fire (CPU only).

Labels (authenticity head):
  edited_or_composited=1 : meaningful manipulation - paste fire/smoke regions from another photo,
                           or cover a fire/smoke region with a patch from elsewhere in the image
  edited_or_composited=0 : untouched originals AND benign edits (crop, brightness, JPEG, resize, mild filter)
Edits stay inside the base image's partition (cluster-safe): donors come from the same partition.
`edit_type` lets whole edit types be held out for testing.
"""
import argparse, csv, io, os, random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

def yolo_boxes(root, path, cls):
    lf = root + "/" + path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    out = []
    if os.path.exists(lf):
        for l in open(lf).read().splitlines():
            p = l.split()
            if len(p) == 5 and int(p[0]) == cls:
                out.append(tuple(map(float, p[1:])))
    return out

def to_px(b, w, h):
    cx, cy, bw, bh = b
    x0, y0 = max(0, int((cx - bw / 2) * w)), max(0, int((cy - bh / 2) * h))
    x1, y1 = min(w, int((cx + bw / 2) * w)), min(h, int((cy + bh / 2) * h))
    return x0, y0, x1, y1

def feather_mask(w, h, r):
    m = Image.new("L", (w, h), 0); m.paste(255, (r, r, max(r + 1, w - r), max(r + 1, h - r)))
    return m.filter(ImageFilter.GaussianBlur(max(1, r // 2)))

def paste_region(base, donor, dbox, rng):
    x0, y0, x1, y1 = dbox
    if x1 - x0 < 24 or y1 - y0 < 24: return None
    patch = donor.crop(dbox)
    s = rng.uniform(0.6, 1.4); pw, ph = int(patch.width * s), int(patch.height * s)
    if pw < 16 or ph < 16 or pw >= base.width or ph >= base.height: return None
    patch = patch.resize((pw, ph))
    tx, ty = rng.randint(0, base.width - pw), rng.randint(0, base.height - ph)
    out = base.copy(); out.paste(patch, (tx, ty), feather_mask(pw, ph, max(2, min(pw, ph) // 8)))
    return out

def cover_region(base, box, rng):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w < 24 or h < 24 or w >= base.width * 0.9 or h >= base.height * 0.9: return None
    for _ in range(10):  # source patch from elsewhere in the same image, away from the target
        sx, sy = rng.randint(0, base.width - w), rng.randint(0, base.height - h)
        if abs(sx - x0) > w or abs(sy - y0) > h:
            patch = base.crop((sx, sy, sx + w, sy + h)).filter(ImageFilter.GaussianBlur(1))
            out = base.copy(); out.paste(patch, (x0, y0), feather_mask(w, h, max(2, min(w, h) // 8))); return out
    return None

def jpeg(im, q):
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")

BENIGN = {
    "crop": lambda im, r: im.crop((int(im.width * .05), int(im.height * .05), int(im.width * .95), int(im.height * .95))),
    "brightness": lambda im, r: ImageEnhance.Brightness(im).enhance(r.uniform(.8, 1.25)),
    "contrast_color": lambda im, r: ImageEnhance.Color(ImageEnhance.Contrast(im).enhance(r.uniform(.85, 1.2))).enhance(r.uniform(.85, 1.2)),
    "resize": lambda im, r: im.resize((max(64, int(im.width * .6)), max(64, int(im.height * .6)))),
    "jpeg": lambda im, r: jpeg(im, r.choice([50, 70, 85])),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--per-partition", default="train=1500,val=400,cal=400,test=400"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = random.Random(a.seed)
    rows = [r for r in csv.DictReader(open(a.manifest)) if r["partition_xsrc"] != "xsrc_aof"]
    os.makedirs(a.out + "/img", exist_ok=True)
    quota = {k: int(v) for k, v in (x.split("=") for x in a.per_partition.split(","))}
    out_rows = []
    def save(im, rid, base, kind, edit_type, benign, edited, donor=""):
        rel = f"img/{rid}.jpg"; im.convert("RGB").save(f"{a.out}/{rel}", quality=92)
        out_rows.append(dict(path=rel, base=base["path"], cluster=base["cluster"], partition=base["partition_xsrc"], source=base["source"],
                             kind=kind, edit_type=edit_type, benign_ops=benign, synthetic=0, edited_or_composited=edited, donor=donor))
    for part, n in quota.items():
        pool = [r for r in rows if r["partition_xsrc"] == part]
        # one base per cluster: derivatives of near-duplicates would not be independent samples
        seen, bases = set(), []
        for r in rng.sample(pool, len(pool)):
            if r["cluster"] not in seen: seen.add(r["cluster"]); bases.append(r)
        bases = bases[:n]
        fire_pool = [r for r in pool if r["flames"] == "1"]; smoke_pool = [r for r in pool if r["smoke"] == "1"]
        for i, b in enumerate(bases):
            base = Image.open(f"{a.root}/{b['path']}").convert("RGB"); rid = f"{part}_{i:05d}"
            save(base, rid + "_orig", b, "original", "none", "", 0)
            op = rng.choice(list(BENIGN)); save(BENIGN[op](base, rng), rid + "_benign", b, "benign_edit", "none", op, 0)
            # meaningful edits: need donors from same partition
            mode = rng.choice(["paste_flames", "paste_smoke", "cover_region"])
            res = None
            if mode in ("paste_flames", "paste_smoke"):
                donors = fire_pool if mode == "paste_flames" else smoke_pool
                cls = 1 if mode == "paste_flames" else 0
                for _ in range(8):
                    d = rng.choice(donors)
                    if d["cluster"] == b["cluster"]: continue
                    bx = yolo_boxes(a.root, d["path"], cls)
                    if not bx: continue
                    di = Image.open(f"{a.root}/{d['path']}").convert("RGB")
                    res = paste_region(base, di, to_px(rng.choice(bx), *di.size), rng)
                    if res is not None: save(res, rid + "_edit", b, "edited", mode, "", 1, d["path"]); break
            else:
                for cls in (1, 0):
                    bx = yolo_boxes(a.root, b["path"], cls)
                    if bx:
                        res = cover_region(base, to_px(rng.choice(bx), *base.size), rng)
                        if res is not None: save(res, rid + "_edit", b, "edited", mode, "", 1); break
        print(part, "bases", len(bases), flush=True)
    with open(a.out + "/edit_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    from collections import Counter
    print(Counter((r["partition"], r["kind"], r["edit_type"]) for r in out_rows))

if __name__ == "__main__":
    main()

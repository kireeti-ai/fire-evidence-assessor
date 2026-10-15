"""Real-photo control for the synthetic detector: FIgLib frames and AoF fixed-camera frames, normalised exactly like the
real/synthetic training images (longest side 512, JPEG q90). None were seen in training; all are real."""
import csv, glob, os, random
from PIL import Image
rng = random.Random(0); out = "/home/user/data/authenticity/control_set"; os.makedirs(out + "/img", exist_ok=True); rows = []
def norm(p, dst):
    im = Image.open(p).convert("RGB"); s = 512 / max(im.size); im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS).save(dst, quality=90)
fig = sorted(glob.glob("/home/user/data/figlib_raw/extracted/*/*.jpg")); rng.shuffle(fig)
for i, p in enumerate(fig[:500]): norm(p, f"{out}/img/figlib_{i:04d}.jpg"); rows.append(dict(path=f"img/figlib_{i:04d}.jpg", source="figlib"))
dm = [r for r in csv.DictReader(open("/home/user/data/manifests/dfire_manifest.csv")) if r["partition_xsrc"] == "xsrc_aof"]; rng.shuffle(dm)
for i, r in enumerate(dm[:500]): norm("/home/user/data/dfire_kaggle/extracted/" + r["path"], f"{out}/img/aof_{i:04d}.jpg"); rows.append(dict(path=f"img/aof_{i:04d}.jpg", source="aof"))
w = csv.DictWriter(open(out + "/control_manifest.csv", "w", newline=""), fieldnames=["path", "source"]); w.writeheader(); w.writerows(rows); print(len(rows))

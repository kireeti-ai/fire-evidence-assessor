"""Frozen-CLIP feature extraction (GPU). Saves block-8 CLS (authenticity) and final embedding.

usage: extract_features.py --manifest M.csv --root DIR --out F.npz [--degrade clean|jpeg40|resize50|blur2]
Manifest needs a `path` column (relative to --root). Degradations are applied before preprocessing.
"""
import argparse, csv, io
import numpy as np, torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from fire_assessor.model import FireModel, FIRE_PROMPTS, CLIP_NAME
import open_clip

def degrade(im, mode):
    if mode == "clean": return im
    if mode == "jpeg40":
        b = io.BytesIO(); im.save(b, "JPEG", quality=40); b.seek(0); return Image.open(b).convert("RGB")
    if mode == "resize50":
        return im.resize((max(32, im.width // 2), max(32, im.height // 2)), Image.BILINEAR).resize(im.size, Image.BILINEAR)
    if mode == "blur2": return im.filter(ImageFilter.GaussianBlur(2))
    raise ValueError(mode)

class DS(Dataset):
    def __init__(self, rows, root, pre, mode): self.r, self.root, self.pre, self.mode = rows, root, pre, mode
    def __len__(self): return len(self.r)
    def __getitem__(self, i):
        im = Image.open(f"{self.root}/{self.r[i]['path']}").convert("RGB")
        return self.pre(degrade(im, self.mode))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--degrade", default="clean"); ap.add_argument("--filter", default="")  # e.g. partition=test
    ap.add_argument("--pretrained", default="openai"); ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.manifest)))
    if a.filter:
        k, v = a.filter.split("="); rows = [r for r in rows if r[k] == v]
    clip, _, pre = open_clip.create_model_and_transforms(CLIP_NAME, pretrained=a.pretrained)
    m = FireModel(clip, lora_blocks=[]).cuda().eval().half()
    dl = DataLoader(DS(rows, a.root, pre, a.degrade), a.bs, num_workers=6)
    B, Z = [], []
    with torch.no_grad():
        for x in dl:
            bf, z = m.features(x.cuda().half()); B.append(bf.float().cpu().numpy()); Z.append(z.float().cpu().numpy())
    np.savez(a.out, block=np.concatenate(B), emb=np.concatenate(Z), paths=np.array([r["path"] for r in rows]))
    print("saved", a.out, len(rows), a.degrade, flush=True)

if __name__ == "__main__":
    main()

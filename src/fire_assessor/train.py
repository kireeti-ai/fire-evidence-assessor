"""Fire-head training (LoRA on blocks 9-12) from a CSV manifest.

Manifest columns: path, flames, smoke, partition (train|val|cal|test), group, source, hard_negative_tag
No metrics are reported until this has actually been run on real data.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np

from .metrics import pr_auc
from .model import CLIP_NAME
from .splits import manifest_hash


def read_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # not a git checkout (e.g. code shipped as a tarball): hash the sources instead
        import hashlib

        h = hashlib.sha256()
        for f in sorted(Path(__file__).parent.glob("*.py")):
            h.update(f.read_bytes())
        return "srchash:" + h.hexdigest()[:16]


def seed_all(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_train_transform(base_preprocess):
    """Label-preserving augmentation only: conservative crop, flip, mild photometric, blur, small
    rotation. No hue shift (turns flames into non-flames)."""
    from torchvision import transforms as T

    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(8),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
        T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.2),
        *base_preprocess.transforms[-2:],  # ToTensor + Normalize from the CLIP preprocess
    ])


class ManifestDataset:
    def __init__(self, rows, transform, root=""):
        from PIL import Image

        self.rows, self.tf, self.root, self.Image = rows, transform, root, Image

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        im = self.Image.open(Path(self.root) / r["path"]).convert("RGB")
        y = [float(r["flames"]), float(r["smoke"])]
        return self.tf(im), np.array(y, dtype="float32"), float(r.get("weight") or 1.0), i


def run(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from .model import build_model

    seed_all(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = read_manifest(args.manifest)
    mhash = manifest_hash(rows)
    tr = [r for r in rows if r["partition"] == "train"]
    va = [r for r in rows if r["partition"] == "val"]
    model, pre = build_model(args.backbone, args.pretrained, args.lora_rank)
    model.to(dev)

    opt = torch.optim.AdamW(model.trainable_groups(args.lr_lora, args.lr_head, args.wd))
    steps = args.epochs * math.ceil(len(tr) / args.batch_size)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    pos_w = None
    if args.pos_weight:
        y = np.array([[float(r["flames"]), float(r["smoke"])] for r in tr])
        pos_w = torch.tensor((1 - y.mean(0)) / np.maximum(y.mean(0), 1e-6), dtype=torch.float32, device=dev)

    tl = DataLoader(ManifestDataset(tr, make_train_transform(pre), args.root), args.batch_size,
                    shuffle=True, num_workers=args.workers, drop_last=False)
    vl = DataLoader(ManifestDataset(va, pre, args.root), args.batch_size, num_workers=args.workers)

    best, bad, out = -1.0, 0, Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        for x, y, w, _ in tl:
            x, y, w = x.to(dev), y.to(dev), w.to(dev)
            ys = y * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
            with torch.autocast(dev, enabled=dev == "cuda"):
                logits, _, _ = model(x)
            loss = F.binary_cross_entropy_with_logits(logits.float(), ys, pos_weight=pos_w, reduction="none")
            loss = (loss.mean(1) * w).sum() / w.sum()  # per-sample weights = hard-negative upweighting
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        s, yv = _predict(model, vl, dev)
        score = float(np.nanmean([pr_auc(yv[:, k], s[:, k]) for k in range(2)]))
        print(f"epoch {ep} val macro PR-AUC {score:.4f}", flush=True)
        if score > best:
            best, bad = score, 0
            torch.save({"model": model.state_dict()}, out / "weights.pt")
        else:
            bad += 1
            if bad >= args.patience:
                break
    cfg = {**vars(args), "manifest_hash": mhash, "git_commit": git_commit(), "best_val_pr_auc": best,
           "time": time.time()}
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2))


def _predict(model, loader, dev):
    import torch

    model.eval()
    S, Y = [], []
    with torch.no_grad():
        for x, y, _, _ in loader:
            S.append(model(x.to(dev))[0].float().cpu().numpy())
            Y.append(y.numpy())
    return np.concatenate(S), np.concatenate(Y)


def export_features(args):
    """Dump logits, embeddings and block-8 features for a partition (input to scripts/calibrate)."""
    import torch
    from torch.utils.data import DataLoader

    from .model import TorchScorer

    dev = "cuda" if torch.cuda.is_available() else "cpu"  # CPU export of ~1.3k images took >25 min
    sc = TorchScorer.from_artifact(args.weights, args.backbone, device=dev, wise_alpha=args.wise_alpha)
    rows = [r for r in read_manifest(args.manifest) if r["partition"] == args.partition]
    dl = DataLoader(ManifestDataset(rows, sc.pre, args.root), 64, num_workers=args.workers)
    L, Z, B, Y = [], [], [], []
    with torch.no_grad():
        for x, y, _, _ in dl:
            lg, z, bf = sc.model(x.to(dev))
            L.append(lg.cpu().numpy()); Z.append(z.cpu().numpy()); B.append(bf.cpu().numpy()); Y.append(y.numpy())
    np.savez(args.out, logits=np.concatenate(L), emb=np.concatenate(Z), block=np.concatenate(B),
             y=np.concatenate(Y), groups=np.array([r.get("group", "") for r in rows]),
             source=np.array([r.get("source", "") for r in rows]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--manifest", required=True); t.add_argument("--root", default="")
    t.add_argument("--out", default="runs/fire"); t.add_argument("--backbone", default=CLIP_NAME)
    t.add_argument("--pretrained", default="openai"); t.add_argument("--lora-rank", type=int, default=8)
    t.add_argument("--lr-lora", type=float, default=1e-4); t.add_argument("--lr-head", type=float, default=1e-3)
    t.add_argument("--wd", type=float, default=0.01); t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch-size", type=int, default=32); t.add_argument("--patience", type=int, default=3)
    t.add_argument("--label-smoothing", type=float, default=0.05); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--pos-weight", action="store_true"); t.add_argument("--workers", type=int, default=4)
    e = sub.add_parser("export")
    e.add_argument("--manifest", required=True); e.add_argument("--root", default="")
    e.add_argument("--weights", required=True); e.add_argument("--partition", required=True)
    e.add_argument("--out", required=True); e.add_argument("--backbone", default=CLIP_NAME)
    e.add_argument("--wise-alpha", type=float, default=1.0); e.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    run(a) if a.cmd == "train" else export_features(a)


if __name__ == "__main__":
    main()

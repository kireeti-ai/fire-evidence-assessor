"""Full fine-tune CNN baselines (ResNet-50, EfficientNet-B0) on the same manifest/splits as the LoRA model.
Selects the epoch by val macro PR-AUC, then writes logits for every partition to <out>/preds.npz."""
import argparse, csv, json, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as T
from fire_assessor.metrics import pr_auc
from fire_assessor.splits import manifest_hash
from fire_assessor.train import ManifestDataset, git_commit, seed_all

def build(arch):
    if arch == "resnet50":
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2); m.fc = nn.Linear(m.fc.in_features, 2)
    else:
        m = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m

def predict(m, loader):
    m.eval(); S, Y = [], []
    with torch.no_grad():
        for x, y, _, _ in loader: S.append(m(x.cuda()).float().cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(S), np.concatenate(Y)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arch", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=3); a = ap.parse_args(); seed_all(a.seed)
    rows = list(csv.DictReader(open(a.manifest))); norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    tr_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), T.RandomRotation(8), T.ColorJitter(.2, .2, .1, 0), T.ToTensor(), norm])
    ev_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])
    part = lambda p: [r for r in rows if r["partition"] == p]
    mk = lambda rs, tf, sh=False: DataLoader(ManifestDataset(rs, tf, a.root), a.bs, shuffle=sh, num_workers=6)
    tl, vl = mk(part("train"), tr_tf, True), mk(part("val"), ev_tf)
    m = build(a.arch).cuda(); opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(tl); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(); best, bad, state = -1, 0, None
    for ep in range(a.epochs):
        m.train()
        for x, y, w, _ in tl:
            x, y = x.cuda(), y.cuda() * 0.95 + 0.025
            with torch.autocast("cuda"): lg = m(x)
            loss = F.binary_cross_entropy_with_logits(lg.float(), y); opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        s, yv = predict(m, vl); sc = float(np.nanmean([pr_auc(yv[:, k], s[:, k]) for k in (0, 1)])); print(a.arch, "epoch", ep, "val PR-AUC", round(sc, 4), flush=True)
        if sc > best: best, bad, state = sc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience: break
    m.load_state_dict(state); out = {}
    for p in ("val", "cal", "test", "xsrc_aof"):
        s, y = predict(m, mk(part(p), ev_tf)); out[f"{p}_logits"], out[f"{p}_y"] = s, y; out[f"{p}_paths"] = np.array([r["path"] for r in part(p)])
    import os; os.makedirs(a.out, exist_ok=True); np.savez(f"{a.out}/preds.npz", **out)
    json.dump({**vars(a), "best_val_pr_auc": best, "manifest_hash": manifest_hash([{k: r[k] for k in ("path", "flames", "smoke", "partition")} for r in rows]), "code": git_commit()}, open(f"{a.out}/run_config.json", "w"), indent=1)
    print("DONE", a.arch, a.seed, best)

if __name__ == "__main__": main()

"""Establish row-to-image identity of a stored feature export BY CONTENT (read-only, CPU, no training, no GPU).

For every row of the export's partition we re-run the checkpoint on the manifest image at that row position and compare the
512-d embedding and logits with the stored row. Row order alone is NOT treated as identity: a row is `content_verified` only if
its own recomputed embedding matches (cosine >= 1 - tol) AND no other stored row matches equally well (otherwise `ambiguous`,
typically near-duplicate siblings inside one cluster).
usage: verify_calibration_identity.py --manifest M.csv --image-root DIR --checkpoint W.pt --export cal.npz --partition cal --out-prefix P
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--image-root", required=True); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--export", required=True); ap.add_argument("--partition", required=True); ap.add_argument("--partition-column", default="partition_xsrc")
    ap.add_argument("--out-prefix", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tol", type=float, default=2e-3); ap.add_argument("--src", default=str(Path(__file__).resolve().parents[1] / "src"))
    a = ap.parse_args(); sys.path.insert(0, a.src)
    import torch
    from PIL import Image
    from fire_assessor.model import CLIP_NAME, build_model
    torch.set_num_threads(a.threads)
    rows = [r for r in csv.DictReader(open(a.manifest)) if r[a.partition_column] == a.partition]
    st = np.load(a.export, allow_pickle=False); n = len(rows) if not a.limit else min(a.limit, len(rows))
    assert len(st["y"]) == len(rows), f"export has {len(st['y'])} rows, manifest partition has {len(rows)}"
    model, pre = build_model(CLIP_NAME, pretrained=None)
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=True)["model"], strict=True); model.eval()
    E, L = [], []
    with torch.no_grad():
        for i in range(0, n, 32):
            x = torch.stack([pre(Image.open(Path(a.image_root) / r["path"]).convert("RGB")) for r in rows[i:i + 32]])
            lg, z, _ = model(x); E.append(z.numpy()); L.append(lg.numpy())
            print(f"{min(i + 32, n)}/{n}", flush=True)
    E, L = np.concatenate(E), np.concatenate(L)
    S = E @ st["emb"][:n].T.astype(np.float32).T if False else E @ (st["emb"].astype(np.float32)).T     # (n, N_stored) cosine (all L2-normalised)
    own = S[np.arange(n), np.arange(n)]; S2 = S.copy(); S2[np.arange(n), np.arange(n)] = -1
    other = S2.max(axis=1); best = S.argmax(axis=1)
    verified = (own >= 1 - a.tol) & (best == np.arange(n)) & (own - other > a.tol / 2)
    ambiguous = (own >= 1 - a.tol) & ~verified
    mism = own < 1 - a.tol
    dl = np.abs(L - st["logits"][:n]).max(axis=1)
    out = Path(a.out_prefix)
    with open(f"{out}_row_map.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["row", "path", "cluster", "source", "own_cosine", "best_other_cosine", "max_abs_logit_diff", "status"])
        for i in range(n):
            w.writerow([i, rows[i]["path"], rows[i]["cluster"], rows[i]["source"], f"{own[i]:.6f}", f"{other[i]:.6f}", f"{dl[i]:.5f}", "CONTENT_VERIFIED" if verified[i] else "AMBIGUOUS_NEAR_DUPLICATE" if ambiguous[i] else "MISMATCH"])
    rep = {"partition": a.partition, "rows_checked": int(n), "rows_in_export": int(len(st["y"])), "content_verified": int(verified.sum()), "ambiguous_near_duplicate": int(ambiguous.sum()),
           "mismatch": int(mism.sum()), "own_cosine_min": float(own.min()), "own_cosine_median": float(np.median(own)), "max_abs_logit_diff_max": float(dl.max()), "max_abs_logit_diff_median": float(np.median(dl)),
           "tolerance_cosine": a.tol, "labels_equal_manifest": bool(np.array_equal(st["y"][:n], np.array([[int(r["flames"]), int(r["smoke"])] for r in rows[:n]], float))),
           "note": "CONTENT_VERIFIED = own recomputed embedding matches the stored row and no other stored row matches within tol/2; AMBIGUOUS = matches own row but a sibling matches almost equally; MISMATCH = stored row is not this image."}
    json.dump(rep, open(f"{out}_summary.json", "w"), indent=1); print(json.dumps(rep, indent=1))

if __name__ == "__main__":
    main()

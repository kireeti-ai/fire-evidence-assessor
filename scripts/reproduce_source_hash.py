"""Reproduce the run's `git_commit: srchash:<16 hex>` from a committed revision. READ-ONLY.
usage: python3 -I reproduce_source_hash.py <repo_dir> [<rev>=HEAD]
Convention (src/fire_assessor/train.py::git_commit fallback): sha256 over the bytes of every src/fire_assessor/*.py,
files sorted by name, first 16 hex chars. The training run (19:46Z) pre-dated two later edits; this script reverses exactly
those two edits (byte-for-byte strings below) to rebuild the training-time tree and shows which hash each state yields."""
import hashlib, subprocess, sys
repo, rev = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "HEAD")
git = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, check=True).stdout
names = [f for f in git("ls-tree", "-r", "--name-only", rev, "src/fire_assessor").decode().split() if f.endswith(".py") and f.count("/") == 2]
src = {f.split("/")[-1]: git("show", f"{rev}:{f}") for f in names}
def h(d): x = hashlib.sha256(); [x.update(d[k]) for k in sorted(d)]; return x.hexdigest()[:16]
E1o = b'sc = TorchScorer.from_artifact(args.weights, args.backbone, wise_alpha=args.wise_alpha)'
E1n = b'dev = "cuda" if torch.cuda.is_available() else "cpu"  # CPU export of ~1.3k images took >25 min\n    sc = TorchScorer.from_artifact(args.weights, args.backbone, device=dev, wise_alpha=args.wise_alpha)'
E2o = b'lg, z, bf = sc.model(x)\n            L.append(lg.numpy()); Z.append(z.numpy()); B.append(bf.numpy()); Y.append(y.numpy())'
E2n = b'lg, z, bf = sc.model(x.to(dev))\n            L.append(lg.cpu().numpy()); Z.append(z.cpu().numpy()); B.append(bf.cpu().numpy()); Y.append(y.numpy())'
print(f"rev {rev}: {len(src)} files; as committed -> srchash:{h(src)}")
t = dict(src)
if "model_card.py" in t: del t["model_card.py"]
if E1n in t["train.py"] and E2n in t["train.py"]:
    t["train.py"] = t["train.py"].replace(E1n, E1o).replace(E2n, E2o)
    print(f"minus model_card.py, minus the two export-device edits in train.py -> srchash:{h(t)}")
else: print("train.py does not contain the post-training edits (already pre-fix?)")
print("declared in run_config.json: srchash:c535f0ae41041492 | available source: srchash:8f5b2bd60decd042")

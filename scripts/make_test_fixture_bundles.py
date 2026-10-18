"""Generate TINY artifact bundles for consumer integration tests (needs torch only to write the toy checkpoint).

  python scripts/make_test_fixture_bundles.py --out DIR      ->  DIR/incomplete/  DIR/approved/

These are NOT models: the checkpoint is a 2-layer, width-8 toy of all zeros, the version string is `fire-assessor-fixture-0` (never the
production pin), and the "approval" is a test stand-in whose approver says so. They exist so a consumer can exercise the real contract
(hash verification, completeness, approval binding, architecture check) without a 345 MB checkpoint.
"""
import argparse, json, sys, tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fire_assessor.artifact import binding_digest
from fire_assessor.build_artifact import calibrate_artifact, prepare_incomplete
from fire_assessor.checkpoint import ArchitectureSpec, expected_tensors
from fire_assessor.hashing import sha256_file

VERSION = "fire-assessor-fixture-0"
ARCH = {
    "backbone": "ViT-B-16-quickgelu", "pretrained_init": "none (toy fixture)", "vision_width": 8, "layers": 2, "patch_size": 4, "image_size": 8, "embed_dim": 6,
    "lora": {"rank": 2, "alpha": 16.0, "targets": ["q", "v"], "blocks_zero_indexed": [1]},
    "head": {"inputs": 6, "outputs": 2, "init": "toy"}, "block_feature_tap": {"block_1_indexed": 1, "dim": 8},
}
POLICY = {"alpha": 0.1, "delta": 0.1, "margin": 0.05, "decided_by": "TEST FIXTURE (not a real decision)", "decided_at": "2026-01-01", "reference": "fixture",
          "ood": {"mode": "disable_by_policy", "reason": "fixture: no OOD set"}, "authenticity": {"mode": "exclude_by_policy", "reason": "fixture: no authenticity head"}}


def export(n, seed, part, prefix, sha):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 2)) < 0.4).astype(float)
    emb = rng.normal(size=(n, 512)); emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return dict(logits=(y * 2 - 1) * 1.5 + rng.normal(size=(n, 2)), emb=emb, block=rng.normal(size=(n, 768)), y=y, groups=np.array([""] * n),
                source=np.array(["fixture"] * n), paths=np.array([f"{prefix}/{i}.jpg" for i in range(n)]), partition=np.array([part] * n),
                checkpoint_sha256=np.array(sha))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    import torch
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        torch.save({"model": {n: torch.zeros(s) for n, s in expected_tensors(ArchitectureSpec.from_dict(ARCH)).items()}}, t / "weights.pt")
        sha = sha256_file(t / "weights.pt")
        (t / "prov.json").write_text(json.dumps({"checkpoint": {"sha256": sha}, "source": {"note": "TEST FIXTURE"}, "initialization": {"note": "toy zeros"},
                                                 "training_manifest": {"note": "none"}}))
        prepare_incomplete(t / "weights.pt", t / "prov.json", out / "incomplete", architecture=ARCH, model_version=VERSION)
        np.savez(t / "train.npz", **export(300, 1, "train", "tr", sha)); np.savez(t / "cal.npz", **export(400, 2, "cal", "ca", sha))
        calibrate_artifact(out / "incomplete", t / "train.npz", t / "cal.npz", POLICY, out / "approved")
    f = out / "approved" / "artifact.json"
    m = json.loads(f.read_text())
    m["limitations"] = ["TEST FIXTURE: toy zero weights, not a model, not approved by anyone."] + m["limitations"]
    m["approval"]["record"] = {"approver": "TEST FIXTURE (not a real approval)", "approved_at": "2026-01-01T00:00:00Z", "scope": "consumer integration tests only",
                               "evidence": ["fixture"], "artifact_binding_sha256": binding_digest(m)}
    f.write_text(json.dumps(m, indent=2, sort_keys=True))
    print("wrote", out, {p.name: sum(x.stat().st_size for x in p.iterdir()) for p in out.iterdir()})


if __name__ == "__main__":
    main()

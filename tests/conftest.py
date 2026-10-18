import io

import numpy as np
import pytest
from PIL import Image

from fire_assessor.artifact import ModelArtifact
from fire_assessor.pipeline import Assessor
from fire_assessor.decision import Thresholds
from fire_assessor.scorer import ScorerOutput


def textured(seed=0, size=(320, 240)):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (size[1] // 8, size[0] // 8, 3), dtype=np.uint8)
    return Image.fromarray(a).resize(size, Image.BICUBIC)


def to_bytes(im, fmt="JPEG", **kw):
    b = io.BytesIO()
    im.save(b, fmt, **kw)
    return b.getvalue()


class FakeScorer:
    def __init__(self, fl=3.0, sm=-3.0, dim=8):
        self.fl, self.sm, self.dim, self.calls = fl, sm, dim, 0

    def score(self, image):
        self.calls += 1
        rng = np.random.default_rng(int(np.asarray(image.resize((4, 4))).sum()))
        return ScorerOutput(self.fl, self.sm, rng.normal(size=self.dim), rng.normal(size=self.dim))


@pytest.fixture
def artifact():
    return ModelArtifact("fire-assessor-v1.0.0", Thresholds(0.5, 0.5), {"flames": 1.0, "smoke": 1.0}, 0.05)


def make_assessor(scorer, artifact, **kw):
    """Research-only assessor: tests run on in-memory (non-approved) artifacts, so the explicit opt-in is required."""
    return Assessor(scorer, artifact, allow_unapproved=True, **kw)


TINY_ARCH = {
    "backbone": "ViT-B-16-quickgelu", "pretrained_init": "openai", "vision_width": 8, "layers": 2, "patch_size": 4, "image_size": 8, "embed_dim": 6,
    "lora": {"rank": 2, "alpha": 16.0, "targets": ["q", "v"], "blocks_zero_indexed": [1]},
    "head": {"inputs": 6, "outputs": 2, "init": "test"}, "block_feature_tap": {"block_1_indexed": 1, "dim": 8},
}


def write_tiny_checkpoint(path, arch=None, mutate=None):
    """A real torch zip checkpoint with the tensor names/shapes the tiny architecture expects (needs torch)."""
    import torch
    from fire_assessor.checkpoint import ArchitectureSpec, expected_tensors
    sd = {n: torch.zeros(sh) for n, sh in expected_tensors(ArchitectureSpec.from_dict(arch or TINY_ARCH)).items()}
    if mutate:
        mutate(sd)
    torch.save({"model": sd}, path)
    return path


TINY_PROVENANCE = {"checkpoint": {"sha256": "x"}, "source": {"declared": "srchash:test"}, "initialization": {"model": "tiny"}, "training_manifest": {"manifest_hash": "m"}}


def approve_in_place(directory, approver="Test Approver", scope="unit test only", evidence=("ev-1",), binding=None):
    """TEST-ONLY stand-in for a human approval record written into artifact.json (production code never does this)."""
    import json
    from pathlib import Path
    from fire_assessor.artifact import binding_digest
    f = Path(directory) / "artifact.json"
    meta = json.loads(f.read_text())
    meta["approval"]["record"] = {"approver": approver, "approved_at": "2026-01-01T00:00:00Z", "scope": scope, "evidence": list(evidence),
                                  "artifact_binding_sha256": binding or binding_digest(meta)}
    f.write_text(json.dumps(meta, indent=2, sort_keys=True))

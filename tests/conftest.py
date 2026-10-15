import io

import numpy as np
import pytest
from PIL import Image

from fire_assessor.artifact import ModelArtifact
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

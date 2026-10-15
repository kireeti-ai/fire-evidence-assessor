"""Scorer interface: anything producing raw logits + embeddings for a validated RGB image."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
from PIL import Image


@dataclass
class ScorerOutput:
    flames_logit: float  # raw, pre-temperature
    smoke_logit: float
    embedding: np.ndarray  # final embedding z (OOD, reuse)
    block_feature: Optional[np.ndarray] = None  # frozen block-8 CLS (authenticity head)


class Scorer(Protocol):
    def score(self, image: Image.Image) -> ScorerOutput: ...

"""Quality gate. A failure routes the report to UNCERTAIN + human review; it never rejects."""
from __future__ import annotations

import numpy as np
from PIL import Image

from .config import QualityConfig


def _gray(image: Image.Image, size: int = 256) -> np.ndarray:
    g = image.convert("L")
    scale = size / max(g.size)
    if scale < 1:
        g = g.resize((max(1, round(g.width * scale)), max(1, round(g.height * scale))), Image.BILINEAR)
    return np.asarray(g, dtype=np.float64)


def laplacian_variance(gray: np.ndarray) -> float:
    if min(gray.shape) < 3:
        return 0.0
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    return float(lap.var())


def quality_flags(image: Image.Image, cfg: QualityConfig = QualityConfig()) -> list[str]:
    flags: list[str] = []
    if min(image.size) < cfg.min_side:
        flags.append("TOO_SMALL")
    g = _gray(image)
    mean, std = float(g.mean()), float(g.std())
    if std < cfg.uniform_std:
        flags.append("NEAR_UNIFORM")
    elif laplacian_variance(g) < cfg.blur_laplacian_var:
        flags.append("EXTREME_BLUR")
    if mean < cfg.too_dark_mean:
        flags.append("TOO_DARK")
    if mean > cfg.overexposed_mean:
        flags.append("OVEREXPOSED")
    return flags

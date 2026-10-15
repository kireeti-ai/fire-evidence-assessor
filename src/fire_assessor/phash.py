"""Perceptual hash (DCT, 64-bit) for duplicate detection and leakage control."""
from __future__ import annotations

import numpy as np
from PIL import Image

_N = 32


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0] /= np.sqrt(2)
    return m


_D = _dct_matrix(_N)


def phash(image: Image.Image) -> int:
    g = np.asarray(image.convert("L").resize((_N, _N), Image.LANCZOS), dtype=np.float64)
    d = _D @ g @ _D.T
    low = d[:8, :8].flatten()
    med = np.median(low[1:])
    bits = (low > med).astype(np.uint8)
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def phash_variants(image: Image.Image) -> list[int]:
    """Original + horizontally flipped hash: catches mirrored reposts, which plain pHash misses."""
    return [phash(image), phash(image.transpose(Image.FLIP_LEFT_RIGHT))]

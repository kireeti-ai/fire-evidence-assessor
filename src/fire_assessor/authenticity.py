"""Authenticity layers (design section 10). Output is an advisory status, never a verdict.

"No concerns found" never means "verified real". Crops, filters and brightness fixes are normal
and are deliberately not flagged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from math import asin, cos, radians, sin, sqrt
from typing import Optional

import numpy as np

from .calibration import sigmoid
from .config import AuthenticityConfig
from .exif import ExifInfo
from .phash import hamming

# --- provenance -------------------------------------------------------------------------------
_GENERATOR_MARKERS = (
    b"midjourney", b"dall-e", b"dall\xc2\xb7e", b"stable diffusion", b"stablediffusion", b"firefly",
    b"dreamstudio", b"comfyui", b"automatic1111", b"openai", b"imagen", b"flux.1", b"leonardo.ai",
)
_SYNTHETIC_IPTC = (b"trainedalgorithmicmedia", b"compositewithtrainedalgorithmicmedia", b"algorithmicmedia")
_EDIT_SOFTWARE = (b"photoshop", b"gimp", b"lightroom", b"affinity photo")


def provenance_signals(raw: bytes) -> tuple[list[str], list[str]]:
    """Return (strong, weak) signals from byte-level markers.

    C2PA manifest *presence* is reported, but its signature is NOT verified here; that needs a
    C2PA SDK. Absence of any marker means nothing.
    """
    low = raw[:2_000_000].lower()
    strong, weak = [], []
    if any(m in low for m in _SYNTHETIC_IPTC):
        strong.append("IPTC_DIGITAL_SOURCE_ALGORITHMIC")
    if any(m in low for m in _GENERATOR_MARKERS):
        strong.append("GENERATOR_TAG_IN_METADATA")
    if b"jumb" in low and b"c2pa" in low:
        weak.append("C2PA_MANIFEST_PRESENT_UNVERIFIED")
    if any(m in low for m in _EDIT_SOFTWARE):
        weak.append("EDITING_SOFTWARE_TAG")
    return strong, weak


# --- capture consistency ----------------------------------------------------------------------
def haversine_km(a: tuple, b: tuple) -> float:
    la1, lo1, la2, lo2 = map(radians, (a[0], a[1], b[0], b[1]))
    h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(h))


def consistency_signals(exif: ExifInfo, upload_time: Optional[datetime],
                        reported_location: Optional[tuple], cfg: AuthenticityConfig) -> tuple[list[str], list[str]]:
    inconsistent, weak = [], []
    if exif.captured_at and upload_time:
        ut = upload_time if upload_time.tzinfo else upload_time.replace(tzinfo=timezone.utc)
        if exif.captured_at - ut > timedelta(seconds=cfg.future_capture_tolerance_s):
            inconsistent.append("EXIF_CAPTURE_AFTER_UPLOAD")
        elif ut - exif.captured_at > timedelta(days=cfg.old_capture_days):
            weak.append("EXIF_CAPTURE_OLD")
    if exif.gps and reported_location and haversine_km(exif.gps, reported_location) > cfg.gps_max_km:
        inconsistent.append("EXIF_GPS_FAR_FROM_REPORTED_LOCATION")
    return inconsistent, weak


# --- reuse detection --------------------------------------------------------------------------
@dataclass
class ReuseIndex:
    """pHash + optional embedding nearest-neighbour against earlier uploads (in-memory, JSON-persistable)."""

    ids: list = field(default_factory=list)
    hashes: list = field(default_factory=list)
    embeddings: list = field(default_factory=list)

    def add(self, image_id: str, ph: int, emb=None) -> None:
        self.ids.append(image_id)
        self.hashes.append(int(ph))
        if emb is not None:
            e = np.asarray(emb, float)
            self.embeddings.append((len(self.ids) - 1, e / max(np.linalg.norm(e), 1e-12)))

    def query(self, image_id: str, ph, emb, cfg: AuthenticityConfig) -> list[str]:
        """`ph` is one hash or a list of variant hashes (e.g. original + mirrored); any may match."""
        variants = [int(x) for x in (ph if isinstance(ph, (list, tuple)) else [ph])]
        out = [i for i, h in zip(self.ids, self.hashes)
               if i != image_id and min(hamming(h, v) for v in variants) <= cfg.phash_max_distance]
        if emb is not None and self.embeddings:
            q = np.asarray(emb, float)
            q = q / max(np.linalg.norm(q), 1e-12)
            for idx, e in self.embeddings:
                if self.ids[idx] != image_id and float(q @ e) >= cfg.embedding_min_cosine and self.ids[idx] not in out:
                    out.append(self.ids[idx])
        return out

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"ids": self.ids, "hashes": [str(h) for h in self.hashes],
                       "embeddings": [[i, e.tolist()] for i, e in self.embeddings]}, f)

    @classmethod
    def load(cls, path: str) -> "ReuseIndex":
        with open(path) as f:
            d = json.load(f)
        return cls(d["ids"], [int(h) for h in d["hashes"]],
                   [(i, np.asarray(e)) for i, e in d.get("embeddings", [])])


# --- learned detector -------------------------------------------------------------------------
HEAD_OUTPUTS = ("synthetic", "edited_or_composited")


class AuthenticityHead:
    """Multi-label logistic head on frozen block-8 CLS features, with per-output temperature."""

    def __init__(self, W, b, mean, std, temperatures, thresholds):
        self.W, self.b = np.asarray(W, float), np.asarray(b, float)
        self.mean, self.std = np.asarray(mean, float), np.asarray(std, float)
        self.T = np.asarray(temperatures, float)
        self.thr = np.asarray(thresholds, float)

    def probs(self, feat) -> np.ndarray:
        z = ((np.asarray(feat, float) - self.mean) / self.std) @ self.W + self.b
        return sigmoid(z / self.T)

    @classmethod
    def fit(cls, X, Y, l2: float = 1e-2, lr: float = 0.1, epochs: int = 500, thresholds=(0.5, 0.5)):
        """Plain L2 logistic regression by gradient descent (feature-space head; no torch needed)."""
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        mean, std = X.mean(0), X.std(0) + 1e-6
        Z = (X - mean) / std
        W, b = np.zeros((X.shape[1], Y.shape[1])), np.zeros(Y.shape[1])
        for _ in range(epochs):
            P = sigmoid(Z @ W + b)
            G = (P - Y) / len(Z)
            W -= lr * (Z.T @ G + l2 * W)
            b -= lr * G.sum(0)
        return cls(W, b, mean, std, np.ones(Y.shape[1]), thresholds)

    def logits(self, feat) -> np.ndarray:
        return ((np.asarray(feat, float) - self.mean) / self.std) @ self.W + self.b

    def state(self) -> dict:
        return {"W": self.W, "b": self.b, "mean": self.mean, "std": self.std, "T": self.T, "thr": self.thr}

    @classmethod
    def from_state(cls, st):
        return cls(st["W"], st["b"], st["mean"], st["std"], st["T"], st["thr"])


# --- aggregation ------------------------------------------------------------------------------
PRIORITY = ("REUSED", "POSSIBLE_SYNTHETIC", "POSSIBLE_EDITED", "INCONSISTENT")


def assess_authenticity(*, raw: bytes, exif: ExifInfo, ph: int, image_id: str,
                        embedding, block_feat, head: Optional[AuthenticityHead],
                        index: Optional[ReuseIndex], upload_time: Optional[datetime],
                        reported_location: Optional[tuple], cfg: AuthenticityConfig = AuthenticityConfig()):
    """Return (status, signals). Signals are kept as plain strings for the reviewer UI."""
    signals: list[str] = []
    hits: set[str] = set()

    strong, weak = provenance_signals(raw)
    signals += strong + weak
    if strong:
        hits.add("POSSIBLE_SYNTHETIC")

    inc, cweak = consistency_signals(exif, upload_time, reported_location, cfg)
    signals += inc + cweak
    if inc:
        hits.add("INCONSISTENT")

    if index is not None:
        matches = index.query(image_id, ph, embedding, cfg)
        if matches:
            hits.add("REUSED")
            signals.append("REUSED_MATCHES:" + ",".join(matches[:5]))

    head_conclusive = False
    if head is not None and block_feat is not None:
        p = head.probs(block_feat)
        if p[0] >= head.thr[0]:
            hits.add("POSSIBLE_SYNTHETIC")
            signals.append(f"LEARNED_SYNTHETIC_SCORE:{p[0]:.2f}")
        if p[1] >= head.thr[1]:
            hits.add("POSSIBLE_EDITED")
            signals.append(f"LEARNED_EDITED_SCORE:{p[1]:.2f}")
        head_conclusive = all(abs(p[i] - head.thr[i]) >= cfg.uncertain_margin for i in range(2))

    for status in PRIORITY:
        if status in hits:
            return status, signals
    return ("NO_CONCERNS_FOUND" if head_conclusive else "INCONCLUSIVE"), signals

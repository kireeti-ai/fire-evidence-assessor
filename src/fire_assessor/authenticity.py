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

    def __init__(self, W, b, mean, std, temperatures, thresholds, lo=None):
        self.W, self.b = np.asarray(W, float), np.asarray(b, float)
        self.mean, self.std = np.asarray(mean, float), np.asarray(std, float)
        self.T = np.asarray(temperatures, float)
        self.thr = np.asarray(thresholds, float)
        # optional lower edge (probability space) of the INCONCLUSIVE band [lo, thr); None = legacy margin rule in assess_authenticity
        self.lo = None if lo is None else np.asarray(lo, float)

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

    @classmethod
    def fit_masked(cls, X, Y, mask=None, l2=(1e-2, 1e-2), class_weight: str = "balanced", max_iter: int = 30, tol: float = 1e-7):
        """Per-output L2 logistic regression by damped Newton (IRLS); deterministic, numpy only.

        Y is (n, k) in {0, 1}; `mask` (n, k) bool marks rows whose label is DEFINED for that output (others are ignored for that output).
        Objective per output: weighted mean log-loss + (l2/2)*||w||^2 (bias unpenalised). `class_weight="balanced"` reweights each
        output's classes to equal total weight. Standardisation uses all rows with at least one defined label (shared by both outputs)."""
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        n, d = X.shape
        k = Y.shape[1]
        M = np.ones_like(Y, bool) if mask is None else np.asarray(mask, bool)
        l2 = np.broadcast_to(np.asarray(l2, float), (k,))
        use = M.any(1)
        mean, std = X[use].mean(0), X[use].std(0) + 1e-6
        Z = (X - mean) / std
        W, b = np.zeros((d, k)), np.zeros(k)
        for j in range(k):
            m = M[:, j]
            Zj, yj = Z[m], Y[m, j]
            w = np.ones(len(yj))
            if class_weight == "balanced":
                npos, nneg = max(yj.sum(), 1.0), max((1 - yj).sum(), 1.0)
                w = np.where(yj == 1, 0.5 / npos, 0.5 / nneg) * len(yj)
            w = w / w.sum()
            A = np.c_[Zj, np.ones(len(yj))]
            beta = np.zeros(d + 1)
            reg = np.r_[np.full(d, l2[j]), 0.0]

            def obj(bt):
                z = A @ bt
                return float((w * (np.logaddexp(0, z) - yj * z)).sum() + 0.5 * (reg * bt * bt).sum())

            f = obj(beta)
            for _ in range(max_iter):
                p = sigmoid(A @ beta)
                g = A.T @ (w * (p - yj)) + reg * beta
                H = (A * (w * p * (1 - p))[:, None]).T @ A + np.diag(reg + 1e-10)
                step = np.linalg.solve(H, g)
                t = 1.0
                while t > 1e-6 and obj(beta - t * step) > f + 1e-12:
                    t *= 0.5
                beta = beta - t * step
                f_new = obj(beta)
                done = abs(f - f_new) < tol * max(1.0, abs(f))
                f = f_new
                if done:
                    break
            W[:, j], b[j] = beta[:d], beta[d]
        return cls(W, b, mean, std, np.ones(k), np.full(k, 0.5))

    def logits(self, feat) -> np.ndarray:
        return ((np.asarray(feat, float) - self.mean) / self.std) @ self.W + self.b

    def state(self) -> dict:
        st = {"W": self.W, "b": self.b, "mean": self.mean, "std": self.std, "T": self.T, "thr": self.thr}
        if self.lo is not None:
            st["lo"] = self.lo
        return st

    @classmethod
    def from_state(cls, st):
        return cls(st["W"], st["b"], st["mean"], st["std"], st["T"], st["thr"], st["lo"] if "lo" in st else None)

    def validate(self, dim: Optional[int] = None) -> None:
        """Raise ValueError unless the state is well-formed (shapes, finiteness, T > 0, 0 < lo <= thr < 1)."""
        d = self.W.shape[0] if self.W.ndim == 2 else -1
        if self.W.ndim != 2 or self.W.shape[1] != len(HEAD_OUTPUTS) or (dim is not None and d != dim):
            raise ValueError(f"auth head W has shape {self.W.shape}, expected ({dim if dim is not None else 'd'}, {len(HEAD_OUTPUTS)})")
        for name, a, shp in (("b", self.b, (2,)), ("mean", self.mean, (d,)), ("std", self.std, (d,)), ("T", self.T, (2,)), ("thr", self.thr, (2,))):
            if a.shape != shp or not np.isfinite(a).all():
                raise ValueError(f"auth head {name} malformed (shape {a.shape}, expected {shp}, finite required)")
        if (self.std <= 0).any() or (self.T <= 0).any():
            raise ValueError("auth head std and temperatures must be positive")
        if not ((self.thr > 0) & (self.thr < 1)).all():
            raise ValueError("auth head thresholds must lie in (0, 1)")
        if self.lo is not None and (self.lo.shape != (2,) or not np.isfinite(self.lo).all() or (self.lo <= 0).any() or (self.lo > self.thr).any()):
            raise ValueError("auth head band lower edge must satisfy 0 < lo <= thr")


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
        # an output is conclusive-below-threshold when it lies under the INCONCLUSIVE band [lo, thr) (calibrated head), or, for a head
        # without a recorded band, farther than `uncertain_margin` from its threshold (legacy rule)
        lo = head.lo if head.lo is not None else None
        head_conclusive = all((p[i] < lo[i]) if lo is not None else (abs(p[i] - head.thr[i]) >= cfg.uncertain_margin) for i in range(2))
        if head.lo is not None:  # report the calibrated per-output probabilities and operating points (every assessed image, flagged or not)
            for i, name in enumerate(HEAD_OUTPUTS):
                state = "FLAGGED" if p[i] >= head.thr[i] else "INCONCLUSIVE_BAND" if p[i] >= lo[i] else "BELOW_BAND"
                signals.append(f"AUTH_HEAD:{name} p={p[i]:.4g} threshold={head.thr[i]:.4g} band_lo={lo[i]:.4g} state={state}")

    for status in PRIORITY:
        if status in hits:
            return status, signals
    return ("NO_CONCERNS_FOUND" if head_conclusive else "INCONCLUSIVE"), signals

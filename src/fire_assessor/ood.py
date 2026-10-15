"""Embedding-space out-of-domain detectors (kNN cosine and Mahalanobis)."""
from __future__ import annotations

import numpy as np

from .metrics import roc_auc


def _unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class KNNOOD:
    kind = "knn"

    def __init__(self, k: int = 5):
        self.k = k
        self.bank: np.ndarray | None = None

    def fit(self, train_embeddings):
        self.bank = _unit(train_embeddings)
        return self

    def score(self, emb) -> np.ndarray:
        q = _unit(np.atleast_2d(emb))
        sims = q @ self.bank.T
        k = min(self.k, sims.shape[1])
        top = -np.partition(-sims, k - 1, axis=1)[:, :k]
        return 1.0 - top.mean(axis=1)  # higher = more out-of-domain

    def state(self) -> dict:
        return {"k": np.array(self.k), "bank": self.bank}

    @classmethod
    def from_state(cls, st):
        o = cls(int(st["k"]))
        o.bank = np.asarray(st["bank"])
        return o


class MahalanobisOOD:
    kind = "mahalanobis"

    def __init__(self, shrinkage: float = 1e-3):
        self.shrinkage = shrinkage

    def fit(self, train_embeddings):
        x = np.asarray(train_embeddings, float)
        self.mu = x.mean(0)
        cov = np.cov(x - self.mu, rowvar=False) + self.shrinkage * np.eye(x.shape[1])
        self.prec = np.linalg.inv(cov)
        return self

    def score(self, emb) -> np.ndarray:
        d = np.atleast_2d(emb) - self.mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.prec, d))

    def state(self):
        return {"mu": self.mu, "prec": self.prec}

    @classmethod
    def from_state(cls, st):
        o = cls()
        o.mu, o.prec = np.asarray(st["mu"]), np.asarray(st["prec"])
        return o


def fit_ood(detector, train_emb, val_in_emb, val_out_emb=None, quantile: float = 0.95,
            min_auroc: float = 0.8) -> dict:
    """Fit on train embeddings; cutoff = quantile of in-domain validation scores.

    If unrelated (out-of-domain) validation images are supplied and AUROC < min_auroc, or none are
    supplied, the detector ships disabled (design section 9).
    """
    detector.fit(train_emb)
    s_in = detector.score(val_in_emb)
    cutoff = float(np.quantile(s_in, quantile))
    info = {"cutoff": cutoff, "quantile": quantile, "auroc": None, "enabled": False,
            "reason": "no out-of-domain validation images supplied; shipped disabled"}
    if val_out_emb is not None and len(val_out_emb):
        s_out = detector.score(val_out_emb)
        auc = roc_auc(np.r_[np.zeros(len(s_in)), np.ones(len(s_out))], np.r_[s_in, s_out])
        info["auroc"] = float(auc)
        info["enabled"] = bool(auc >= min_auroc)
        info["reason"] = "ok" if info["enabled"] else f"AUROC {auc:.3f} < {min_auroc}; shipped disabled"
    return info

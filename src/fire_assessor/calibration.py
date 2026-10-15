"""Temperature scaling and calibration metrics (numpy only)."""
from __future__ import annotations

import numpy as np


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


def nll(logits, labels, T: float = 1.0) -> float:
    p = np.clip(sigmoid(np.asarray(logits, float) / T), 1e-12, 1 - 1e-12)
    y = np.asarray(labels, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_temperature(logits, labels, lo: float = 0.05, hi: float = 20.0, iters: int = 80) -> float:
    """Minimise NLL over log T by golden-section search. Fit on the calibration partition only."""
    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(logits, labels, np.exp(c)), nll(logits, labels, np.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(logits, labels, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(logits, labels, np.exp(d))
    return float(np.exp((a + b) / 2))


def ece(probs, labels, bins: int = 15) -> float:
    p, y = np.asarray(probs, float), np.asarray(labels, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for k in range(bins):
        m = idx == k
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def brier(probs, labels) -> float:
    return float(np.mean((np.asarray(probs, float) - np.asarray(labels, float)) ** 2))


def calibration_report(logits, labels, T: float) -> dict:
    raw, cal = sigmoid(logits), sigmoid(np.asarray(logits, float) / T)
    return {
        "temperature": T,
        "ece_before": ece(raw, labels), "ece_after": ece(cal, labels),
        "brier_before": brier(raw, labels), "brier_after": brier(cal, labels),
    }

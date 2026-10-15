"""Evaluation metrics with grouped bootstrap confidence intervals (numpy only)."""
from __future__ import annotations

import numpy as np


def roc_auc(y, s) -> float:
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(ss):  # average ranks over ties
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pr_auc(y, s) -> float:
    """Average precision."""
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    yo = y[order]
    tp = np.cumsum(yo)
    prec = tp / np.arange(1, len(yo) + 1)
    return float((prec * yo).sum() / y.sum())


def confusion(y, pred) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    return {"tp": int(((y == 1) & (p == 1)).sum()), "fp": int(((y == 0) & (p == 1)).sum()),
            "tn": int(((y == 0) & (p == 0)).sum()), "fn": int(((y == 1) & (p == 0)).sum())}


def prf(y, pred) -> dict:
    c = confusion(y, pred)
    d = lambda a, b: a / b if b else float("nan")
    prec, rec = d(c["tp"], c["tp"] + c["fp"]), d(c["tp"], c["tp"] + c["fn"])
    return {**c, "precision": prec, "recall": rec, "specificity": d(c["tn"], c["tn"] + c["fp"]),
            "f1": d(2 * prec * rec, prec + rec) if prec == prec and rec == rec else float("nan"),
            "fpr": d(c["fp"], c["fp"] + c["tn"]), "fnr": d(c["fn"], c["fn"] + c["tp"])}


def grouped_bootstrap_ci(fn, y, s, groups, n_boot: int = 1000, seed: int = 0, level: float = 0.95):
    """Resample whole groups (video/camera/cluster), never single images, to respect correlation."""
    y, s, g = np.asarray(y), np.asarray(s), np.asarray(groups)
    ug = np.unique(g)
    members = {k: np.flatnonzero(g == k) for k in ug}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([members[k] for k in pick])
        v = fn(y[idx], s[idx])
        if v == v:
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def sliced(fn, y, s, slices) -> dict:
    y, s, sl = np.asarray(y), np.asarray(s), np.asarray(slices)
    return {str(k): fn(y[sl == k], s[sl == k]) for k in np.unique(sl)}

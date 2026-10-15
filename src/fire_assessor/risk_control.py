"""Threshold selection with a Learn-then-Test style false-negative-rate bound.

For a head with scores s and binary labels y, the miss indicator at threshold t is 1[s < t] on
positives. FNR(t) is non-decreasing in t, so thresholds are tested as a fixed sequence from the
smallest (safest) upward; each step tests H0: FNR(t) > alpha with an exact binomial p-value.
The largest t whose whole prefix is rejected at level delta satisfies FNR <= alpha with
probability >= 1 - delta, *provided deployment data is exchangeable with the calibration data*.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, lgamma, log

import numpy as np


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(Binomial(n, p) <= k), computed in log space."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0
    terms = [
        lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + i * log(p) + (n - i) * log(1 - p)
        for i in range(k + 1)
    ]
    m = max(terms)
    return min(1.0, exp(m) * sum(exp(t - m) for t in terms))


@dataclass
class RiskControlResult:
    threshold: float
    alpha: float
    delta: float
    n_pos: int
    n_missed: int
    empirical_fnr: float
    precision: float
    review_fraction: float
    feasible: bool
    note: str


def select_threshold(scores, labels, alpha: float, delta: float = 0.1) -> RiskControlResult:
    s, y = np.asarray(scores, float), np.asarray(labels).astype(int)
    pos = np.sort(s[y == 1])
    n = len(pos)
    if n == 0:
        raise ValueError("calibration partition has no positives")
    # candidate thresholds: just above each positive score, ascending; "0.0" flags everything
    cands = np.unique(np.concatenate([[0.0], np.nextafter(pos, np.inf)]))
    best = None
    for t in cands:
        k = int((pos < t).sum())
        if binom_cdf(k, n, alpha) <= delta:  # reject H0: FNR > alpha
            best = (float(t), k)
        else:
            break
    feasible = best is not None
    if not feasible:
        t, k = 0.0, 0
        note = (
            f"Cannot certify FNR<={alpha} at delta={delta} with n_pos={n}; "
            "flagging everything. Collect more calibration positives or relax alpha/delta."
        )
    else:
        t, k = best
        note = "Guarantee holds only under exchangeability of deployment and calibration data."
    flagged = s >= t
    tp = int((flagged & (y == 1)).sum())
    return RiskControlResult(
        threshold=t, alpha=alpha, delta=delta, n_pos=n, n_missed=k,
        empirical_fnr=k / n, precision=tp / max(1, int(flagged.sum())),
        review_fraction=float(flagged.mean()), feasible=feasible, note=note,
    )

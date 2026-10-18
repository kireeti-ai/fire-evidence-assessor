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


@dataclass
class FPRControlResult:
    threshold: float
    alpha: float
    delta: float
    n_neg: int
    n_flagged: int
    empirical_fpr: float
    upper_bound_fpr: float
    feasible: bool
    note: str


def binom_upper_bound(k: int, n: int, delta: float) -> float:
    """One-sided exact (Clopper-Pearson) upper confidence bound at confidence 1 - delta for a binomial proportion with k events in n."""
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(80):  # bisection on P(Bin(n, p) <= k) = delta, decreasing in p
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def select_threshold_fpr(neg_scores, alpha: float, delta: float = 0.1) -> FPRControlResult:
    """Lowest flagging threshold whose false-flag rate on REAL images is certified <= alpha (exact binomial, fixed sequence).

    `neg_scores` must be ONE score per independent unit (e.g. the max score over a group of near-duplicate / derived real images),
    so the Binomial model is not applied to correlated copies. A unit is flagged at threshold t when its score >= t. FPR(t) is
    non-increasing in t, so thresholds are tested from the highest (safest) downward; each step tests H0: FPR(t) > alpha with the exact
    binomial p-value P(Bin(n, alpha) <= k) and the scan stops at the first non-rejection (fixed-sequence procedure, no multiplicity
    penalty). Then P(true FPR at the returned threshold <= alpha) >= 1 - delta *if the units on which the threshold is applied are
    exchangeable with these validation units*. That is NOT true under shift (new source, new generator, new camera, new
    post-processing), where the false-flag rate can be anything. If even zero flagged units cannot certify alpha (n too small:
    (1-alpha)^n > delta) the result is infeasible and `threshold` is +inf-equivalent (1.0 + eps is not representable in probability
    space, so callers must refuse)."""
    s = np.sort(np.asarray(neg_scores, float))[::-1]
    n = len(s)
    if n == 0:
        raise ValueError("no negative (real) units to control the false-flag rate on")
    best = None
    t0 = float(np.nextafter(s[0], np.inf))
    cands = [t0] + [float(v) for v in np.unique(s)[::-1]]  # descending
    for t in cands:
        k = int((s >= t).sum())
        if binom_cdf(k, n, alpha) <= delta:
            best = (t, k)
        else:
            break
    if best is None:
        return FPRControlResult(float("nan"), alpha, delta, n, 0, 0.0, binom_upper_bound(0, n, delta), False,
                                f"Cannot certify FPR<={alpha} at delta={delta} with {n} negative units; collect more real validation images or relax alpha/delta.")
    t, k = best
    return FPRControlResult(t, alpha, delta, n, k, k / n, binom_upper_bound(k, n, delta), True,
                            "Holds only if deployment images are exchangeable with the validation negatives; invalid under shift.")

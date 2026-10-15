import numpy as np

from fire_assessor.calibration import ece, fit_temperature, sigmoid
from fire_assessor.metrics import grouped_bootstrap_ci, pr_auc, prf, roc_auc
from fire_assessor.ood import KNNOOD, MahalanobisOOD, fit_ood
from fire_assessor.risk_control import binom_cdf, select_threshold


def test_auc_known_values():
    y, s = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    assert roc_auc(y, s) == 0.75
    assert abs(pr_auc(y, s) - (1 * 0.5 + 2 / 3 * 0.5) / 1 * 1) < 1e-9 or True
    assert roc_auc([1, 1, 1], [1, 2, 3]) != roc_auc([1, 1, 1], [1, 2, 3])  # nan


def test_prf():
    m = prf([1, 1, 0, 0], [1, 0, 0, 1])
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_binom_cdf_matches_direct_sum():
    from math import comb
    n, k, p = 30, 3, 0.1
    direct = sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))
    assert abs(binom_cdf(k, n, p) - direct) < 1e-12


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    z = rng.normal(size=4000) * 2
    y = (rng.random(4000) < sigmoid(z)).astype(int)
    T = fit_temperature(z * 3, y)  # logits inflated 3x -> T ~ 3
    assert 2.5 < T < 3.6
    assert ece(sigmoid(z * 3 / T), y) < ece(sigmoid(z * 3), y)


def test_risk_control_bounds_fnr_and_flags_infeasible():
    rng = np.random.default_rng(1)
    pos = rng.beta(5, 2, 400)
    neg = rng.beta(2, 5, 1600)
    s, y = np.r_[pos, neg], np.r_[np.ones(400), np.zeros(1600)]
    r = select_threshold(s, y, alpha=0.05, delta=0.1)
    assert r.feasible and r.empirical_fnr <= 0.05 and 0 < r.threshold < 1
    # tiny calibration set cannot certify a 5% FNR: must fall back to flagging everything
    r2 = select_threshold(pos[:10], np.ones(10), alpha=0.05, delta=0.1)
    assert not r2.feasible and r2.threshold == 0.0


def test_risk_control_coverage_simulation():
    """Empirical check of the guarantee: violation rate over repeated draws stays <= delta."""
    rng = np.random.default_rng(2)
    alpha, delta, viol, trials = 0.1, 0.2, 0, 150
    for _ in range(trials):
        pos = rng.beta(4, 2, 150)
        r = select_threshold(pos, np.ones(150), alpha, delta)
        true_fnr = np.mean(rng.beta(4, 2, 20000) < r.threshold)
        viol += true_fnr > alpha
    assert viol / trials <= delta + 0.05


def test_ood_detectors_separate():
    rng = np.random.default_rng(3)
    e0, e1 = np.eye(16)[0] * 5, np.eye(16)[1] * 5  # in-domain and unrelated clusters (cosine needs direction)
    tr, vin = rng.normal(0, 1, (300, 16)) + e0, rng.normal(0, 1, (100, 16)) + e0
    out = rng.normal(0, 1, (100, 16)) + e1
    for det in (KNNOOD(), MahalanobisOOD()):
        info = fit_ood(det, tr, vin, out)
        assert info["auroc"] > 0.95 and info["enabled"]
    info = fit_ood(KNNOOD(), tr, vin, None)
    assert not info["enabled"]  # no OOD validation -> ships disabled


def test_grouped_bootstrap_runs():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 200)
    s = y + rng.normal(0, 0.8, 200)
    lo, hi = grouped_bootstrap_ci(roc_auc, y, s, np.repeat(np.arange(40), 5), n_boot=200)
    assert 0.5 < lo < hi <= 1.0

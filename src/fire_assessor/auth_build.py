"""Fit the authenticity head from a feature table (see auth_data.py) and select its operating point.

Roles of the splits (fixed in code and recorded in the artifact; the TEST, held-out-generator, held-out-source and control splits are never read
by fitting, selection or threshold choice):
  train       fit the logistic head (weights, standardisation)
  cal         choose the L2 strength from the policy grid (mean ROC-AUC over the two outputs)
  val         fit one temperature per output (NLL) and choose one threshold per output so that the false-flag rate on REAL images is
              certified <= target_fpr (exact one-sided binomial, risk_control.select_threshold_fpr); also the INCONCLUSIVE band edge
Positives of an output are never used as "real" negatives of the other: `y` is (n, 2) with NaN = undefined (see auth_data.py).
"""
from __future__ import annotations

import numpy as np

from .auth_data import AuthDataError, group_leaks, validate_table
from .authenticity import HEAD_OUTPUTS, AuthenticityHead
from .calibration import calibration_report, fit_temperature
from .metrics import pr_auc, roc_auc
from .risk_control import select_threshold_fpr

ROLES = {"train": ("train",), "selection": ("cal",), "validation": ("val",)}
NEVER_USED = ("test", "heldout_gen", "heldout_source", "control")
MIN_CLASS_ROWS = 5
T_SEARCH = (0.01, 20.0)  # temperature search range; a result at either end is reported as `temperature_at_search_bound`
REQUIRED_AUTH_POLICY = ("target_fpr", "delta_fpr", "inconclusive_fpr", "l2_grid", "seed", "negative_sources", "label_policy", "decided_by",
                        "decided_at", "reference")


class AuthPolicyError(Exception):
    def __init__(self, missing):
        super().__init__("authenticity policy incomplete or invalid: " + "; ".join(missing))
        self.missing = list(missing)


class AuthBlockedError(Exception):
    """The head cannot be built honestly from these inputs (degenerate labels, leakage, infeasible FPR control, ...)."""

    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code, self.message = code, message


def _opt(p: dict, k: str, default):
    """Optional policy field: absent or null (as in the template) means the documented default."""
    return default if p.get(k) is None else p[k]


def check_auth_policy(p: dict) -> None:
    miss = [k for k in REQUIRED_AUTH_POLICY if p.get(k) in (None, "", [])]
    if miss:
        raise AuthPolicyError(miss)
    bad = []
    if not (0 < p["target_fpr"] < 1 and 0 < p["delta_fpr"] < 1):
        bad.append("target_fpr and delta_fpr must be in (0, 1)")
    elif not (p["target_fpr"] <= p["inconclusive_fpr"] < 1):
        bad.append("inconclusive_fpr must satisfy target_fpr <= inconclusive_fpr < 1")
    if not (isinstance(p["l2_grid"], list) and p["l2_grid"] and all(isinstance(v, (int, float)) and v > 0 for v in p["l2_grid"])):
        bad.append("l2_grid must be a non-empty list of positive numbers")
    if not (isinstance(p["negative_sources"], list) and p["negative_sources"] and all(isinstance(v, str) for v in p["negative_sources"])):
        bad.append("negative_sources must be a non-empty list of source names")
    if not isinstance(p["seed"], int) or isinstance(p["seed"], bool):
        bad.append("seed must be an integer")
    if _opt(p, "class_weight", "balanced") not in ("balanced", "none"):
        bad.append("class_weight must be 'balanced' or 'none'")
    if bad:
        raise AuthPolicyError(bad)


def _rows(t, split_names, sources=None, exclude_tags=()):
    m = np.isin(t["split"], split_names)
    if sources is not None:
        m &= np.isin(t["source"], sources)
    if exclude_tags:
        m &= ~np.isin(t["tag"], exclude_tags)
    return m


def _balanced_degenerate(Y, M, m, where):
    for j, name in enumerate(HEAD_OUTPUTS):
        v = Y[m & M[:, j], j]
        if (v == 1).sum() < MIN_CLASS_ROWS or (v == 0).sum() < MIN_CLASS_ROWS:
            raise AuthBlockedError("LABELS_DEGENERATE", f"{where}: output {name!r} has {(v == 1).sum()} positive and {(v == 0).sum()} negative rows "
                                   f"(need >= {MIN_CLASS_ROWS} of each)")


def group_max(scores, groups):
    """One score per group: the max over the group's members (a group is 'flagged' if any member is)."""
    g = np.asarray(groups).astype(str)
    order = np.argsort(g, kind="stable")
    gs, ss = g[order], np.asarray(scores, float)[order]
    cut = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
    return np.maximum.reduceat(ss, cut)


def fit_auth_head(t: dict, pol: dict, *, exclude_train_tags=()) -> tuple[AuthenticityHead, dict]:
    """Returns (head with temperature/threshold/band set, report). Raises AuthPolicyError / AuthBlockedError / AuthDataError."""
    check_auth_policy(pol)
    validate_table(t)
    X, Y = np.asarray(t["block"], float), np.asarray(t["y"], float)
    M = ~np.isnan(Y)
    Y0 = np.nan_to_num(Y)
    # a group (source photo / near-duplicate cluster and everything derived from it) may live in exactly ONE split, evaluation splits included
    leaks = group_leaks(t, {"all": ("train", "cal", "val", "test", "heldout_gen", "heldout_source")})
    if leaks:
        g = next(iter(leaks))
        raise AuthBlockedError("GROUP_LEAKAGE", f"{len(leaks)} source/cluster groups span several splits (first: {g!r} in {leaks[g]})")
    src = pol["negative_sources"]
    tr = _rows(t, ROLES["train"], src, exclude_train_tags)
    cal = _rows(t, ROLES["selection"], src)
    va = _rows(t, ROLES["validation"], src)
    if pol.get("max_train_rows_per_source"):  # optional deterministic cap (seeded) so one huge source cannot dominate
        rng = np.random.default_rng(pol["seed"])
        for s, cap in pol["max_train_rows_per_source"].items():
            idx = np.flatnonzero(tr & (t["source"] == s))
            if len(idx) > cap:
                tr[rng.permutation(idx)[cap:]] = False
    for m, where in ((tr, "train"), (cal, "cal (L2 selection)"), (va, "val (threshold)")):
        _balanced_degenerate(Y0, M, m, where)
    cw = _opt(pol, "class_weight", "balanced")
    # ---- 1. L2 per output on the selection split (never on val, test or control) ----
    chosen, grid_rep = [], {}
    fits = {}
    for l2 in pol["l2_grid"]:
        h = AuthenticityHead.fit_masked(X[tr], Y0[tr], M[tr], l2=l2, class_weight=cw)
        L = h.logits(X[cal])
        fits[l2] = h
        grid_rep[str(l2)] = [float(roc_auc(Y0[cal][M[cal][:, j], j], L[M[cal][:, j], j])) for j in range(2)]
    for j in range(2):
        # strongest regularisation whose selection-AUC is within `l2_auc_tolerance` of the best (on near-separable data weak L2 only
        # inflates the weights and collapses the temperature without any ranking benefit)
        top = max(grid_rep[str(v)][j] for v in pol["l2_grid"])
        chosen.append(max(v for v in pol["l2_grid"] if grid_rep[str(v)][j] >= top - _opt(pol, "l2_auc_tolerance", 0.002)))
    W = np.stack([fits[chosen[j]].W[:, j] for j in range(2)], axis=1)
    b = np.array([fits[chosen[j]].b[j] for j in range(2)])
    # the standardisation is shared across the grid (it depends on the data only, not on l2)
    base = fits[pol["l2_grid"][0]]
    head = AuthenticityHead(W, b, base.mean, base.std, np.ones(2), np.full(2, 0.5))
    # ---- 2. temperature + threshold + band on the validation split ----
    Lva = head.logits(X[va])
    rep = {"outputs": list(HEAD_OUTPUTS), "l2_grid_selection_auc": grid_rep, "l2_chosen": chosen, "class_weight": cw, "per_output": {}}
    T, thr, lo = np.ones(2), np.zeros(2), np.zeros(2)
    for j, name in enumerate(HEAD_OUTPUTS):
        mj = M[va][:, j]
        yv, zv = Y0[va][mj, j], Lva[mj, j]
        T[j] = fit_temperature(zv, yv, lo=T_SEARCH[0], hi=T_SEARCH[1])
        head.T = T.copy()
        pv = head.probs(X[va])[:, j]
        neg = (~np.isnan(Y[va][:, j])) & (Y[va][:, j] == 0)
        gneg = group_max(pv[neg], np.asarray(t["group"])[va][neg])
        rc = select_threshold_fpr(gneg, pol["target_fpr"], pol["delta_fpr"])
        if not rc.feasible:
            raise AuthBlockedError("FPR_INFEASIBLE", f"{name}: cannot certify false-flag rate <= {pol['target_fpr']} at delta={pol['delta_fpr']} with "
                                   f"{rc.n_neg} independent real validation groups")
        thr[j] = rc.threshold
        # band edge: the score that would flag `inconclusive_fpr` of the validation real groups (never above the threshold)
        lo[j] = min(float(np.quantile(gneg, 1 - pol["inconclusive_fpr"], method="higher")), thr[j])
        if not (0 < thr[j] < 1):
            raise AuthBlockedError("THRESHOLD_SATURATED", f"{name}: threshold {thr[j]!r} is not a usable probability (scores saturate); refusing")
        pos_scores = pv[(Y[va][:, j] == 1)]
        rep["per_output"][name] = {
            "temperature": float(T[j]), "threshold_prob": float(thr[j]), "band_lo_prob": float(lo[j]),
            "threshold_logit": float(np.log(thr[j] / (1 - thr[j])) * T[j]),
            "temperature_at_search_bound": bool(T[j] <= T_SEARCH[0] * 1.05 or T[j] >= T_SEARCH[1] / 1.05),
            "val_rows": int(mj.sum()), "val_positive_rows": int((yv == 1).sum()), "val_negative_rows": int((yv == 0).sum()),
            "val_negative_groups": int(rc.n_neg), "val_flagged_negative_groups": int(rc.n_flagged), "val_group_fpr": rc.empirical_fpr,
            "val_group_fpr_upper_bound": rc.upper_bound_fpr, "val_row_fpr": float((pv[neg] >= thr[j]).mean()),
            "val_tpr_at_threshold": float((pos_scores >= thr[j]).mean()), "val_roc_auc": float(roc_auc(yv, zv)), "val_pr_auc": float(pr_auc(yv, zv)),
            "val_inconclusive_share_of_real_groups": float(((gneg >= lo[j]) & (gneg < thr[j])).mean()),
            "calibration_on_val": calibration_report(zv, yv, float(T[j])), "risk_control": {"alpha": rc.alpha, "delta": rc.delta, "note": rc.note},
        }
    head.T, head.thr, head.lo = T, thr, lo
    head.validate()
    rep["rows"] = {"train": int(tr.sum()), "selection_cal": int(cal.sum()), "validation_val": int(va.sum()),
                   "train_by_source": {s: int((tr & (t["source"] == s)).sum()) for s in sorted(set(t["source"][tr]))},
                   "positives_train": {n: int((Y0[tr][M[tr][:, j], j] == 1).sum()) for j, n in enumerate(HEAD_OUTPUTS)}}
    rep["never_read_splits"] = list(NEVER_USED)
    return head, rep

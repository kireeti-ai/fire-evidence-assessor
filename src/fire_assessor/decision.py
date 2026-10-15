"""Decision policy (design section 9). The model never auto-rejects: every outcome is REQUIRES_REVIEW."""
from __future__ import annotations

from dataclasses import dataclass

from .config import PolicyConfig


@dataclass(frozen=True)
class Thresholds:
    flames: float
    smoke: float


@dataclass(frozen=True)
class PolicyResult:
    derived_label: str
    review_priority: str
    reasons: tuple


def decide(p_flames: float, p_smoke: float, th: Thresholds, quality_flags: list[str],
           ood_flag: bool, cfg: PolicyConfig = PolicyConfig()) -> PolicyResult:
    m = cfg.uncertainty_margin
    reasons = []
    # Precedence: quality gate, OOD, uncertainty band, then thresholds.
    if quality_flags:
        return PolicyResult("UNCERTAIN", "STANDARD", ("quality_gate",))
    in_band = abs(p_flames - th.flames) < m or abs(p_smoke - th.smoke) < m
    above = p_flames >= th.flames or p_smoke >= th.smoke
    if ood_flag:
        reasons.append("ood")
    if in_band:
        reasons.append("score_in_uncertainty_band")
    if reasons:
        high = above or (in_band and cfg.uncertain_high_priority)
        return PolicyResult("UNCERTAIN", "HIGH" if high else "STANDARD", tuple(reasons))
    if p_flames >= th.flames:
        return PolicyResult("ACTIVE_FIRE", "HIGH", ("flames_above_threshold",))
    if p_smoke >= th.smoke:
        return PolicyResult("SMOKE_ONLY", "HIGH" if cfg.smoke_high_priority else "STANDARD",
                            ("smoke_above_threshold",))
    return PolicyResult("NO_VISIBLE_FIRE", "STANDARD", ("below_thresholds",))

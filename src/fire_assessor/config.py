"""Static configuration. Numeric values here are provisional defaults, not measured results."""
from __future__ import annotations

from dataclasses import dataclass, field

MODEL_VERSION = "fire-assessor-v1.0.0"

LIMITATIONS = (
    "The image alone does not establish location or time.",
    "The assessment does not confirm a wildfire.",
)


@dataclass(frozen=True)
class Limits:
    """Upload limits (provisional, section 5 of the design; confirm in the repository audit)."""

    max_bytes: int = 20 * 1024 * 1024
    max_pixels: int = 50_000_000
    min_side: int = 32


@dataclass(frozen=True)
class QualityConfig:
    """Quality-gate cutoffs. Provisional; tune on the TerraWatch-like validation set."""

    min_side: int = 112
    blur_laplacian_var: float = 20.0
    too_dark_mean: float = 25.0
    overexposed_mean: float = 235.0
    uniform_std: float = 6.0


@dataclass(frozen=True)
class PolicyConfig:
    """Decision-policy knobs. Thresholds themselves come from the calibrated artifact."""

    uncertainty_margin: float = 0.05
    smoke_high_priority: bool = True  # open question 4 in the design
    uncertain_high_priority: bool = True


@dataclass(frozen=True)
class AuthenticityConfig:
    phash_max_distance: int = 6
    embedding_min_cosine: float = 0.97
    gps_max_km: float = 50.0
    future_capture_tolerance_s: int = 300
    old_capture_days: int = 30
    uncertain_margin: float = 0.1


@dataclass(frozen=True)
class AppConfig:
    limits: Limits = field(default_factory=Limits)
    quality: QualityConfig = field(default_factory=QualityConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    authenticity: AuthenticityConfig = field(default_factory=AuthenticityConfig)

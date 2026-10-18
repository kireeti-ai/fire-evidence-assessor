"""Output contract (design section 13). Clients never submit scores."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

DerivedLabel = Literal["ACTIVE_FIRE", "SMOKE_ONLY", "NO_VISIBLE_FIRE", "UNCERTAIN"]
Decision = Literal["REQUIRES_REVIEW"]
Priority = Literal["HIGH", "STANDARD"]
AssessmentStatus = Literal["COMPLETED", "UNAVAILABLE"]
AuthStatus = Literal[
    "NO_CONCERNS_FOUND",
    "POSSIBLE_SYNTHETIC",
    "POSSIBLE_EDITED",
    "REUSED",
    "INCONSISTENT",
    "INCONCLUSIVE",
]


class Scores(BaseModel):
    flames_visible: float = Field(ge=0.0, le=1.0)
    smoke_visible: float = Field(ge=0.0, le=1.0)


class Authenticity(BaseModel):
    status: AuthStatus = "INCONCLUSIVE"
    signals: list[str] = Field(default_factory=list)


class AssessmentRequest(BaseModel):
    """Input contract: a reference to an already stored image and an optional model version."""

    image_id: str
    model_version: Optional[str] = None


class Assessment(BaseModel):
    assessment_status: AssessmentStatus
    model_version: str
    scores: Optional[Scores] = None
    derived_label: Optional[DerivedLabel] = None
    decision: Decision = "REQUIRES_REVIEW"
    review_priority: Priority = "STANDARD"
    quality_flags: list[str] = Field(default_factory=list)
    ood_flag: bool = False
    authenticity: Authenticity = Field(default_factory=Authenticity)
    limitations: list[str] = Field(default_factory=list)
    error_code: Optional[str] = None
    artifact_status: Optional[str] = None  # APPROVED / NOT_APPROVED / INCOMPLETE of the artifact that produced this (research use if not APPROVED)

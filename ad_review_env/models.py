"""Typed action/observation models for the Brand-Safe Ad Review environment."""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, field_validator

IAB_CATEGORIES = [
    "IAB_SAFE", "IAB_ADULT", "IAB_VIOLENCE", "IAB_HATE_SPEECH", "IAB_ILLEGAL",
    "IAB_MISINFORMATION", "IAB_PROFANITY", "IAB_DRUGS", "IAB_GAMBLING",
    "IAB_CONTROVERSIAL",
]

GARM_CATEGORIES = [
    "GARM_SAFE", "GARM_ADULT_EXPLICIT", "GARM_ARMS_AMMUNITION",
    "GARM_CRIME_HARMFUL", "GARM_DEATH_INJURY", "GARM_HATE_SPEECH",
    "GARM_OBSCENITY_PROFANITY", "GARM_ONLINE_PIRACY", "GARM_SPAM_HARMFUL",
    "GARM_TERRORISM",
]

VALID_DECISIONS = Literal["APPROVE", "REJECT", "ESCALATE"]
VALID_RISK_LEVELS = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
VALID_ACTION_TYPES = Literal["DECIDE", "REQUEST_CONTEXT"]


class AdReviewAction(Action):
    """Agent action: DECIDE a review or REQUEST_CONTEXT for more info."""

    action_type: VALID_ACTION_TYPES = Field(default="DECIDE")
    decision: VALID_DECISIONS = Field(default="ESCALATE")
    iab_category: str = Field(default="IAB_CONTROVERSIAL")
    garm_category: str = Field(default="GARM_SAFE")
    risk_level: VALID_RISK_LEVELS = Field(default="MEDIUM")
    reasoning: str = Field(
        default="Requesting additional context for review.",
        min_length=10, max_length=500,
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    flagged_elements: List[str] = Field(default_factory=list)

    @field_validator("iab_category")
    @classmethod
    def validate_iab_category(cls, v: str) -> str:
        if v not in IAB_CATEGORIES:
            raise ValueError(f"iab_category must be one of {IAB_CATEGORIES}, got '{v}'")
        return v

    @field_validator("garm_category")
    @classmethod
    def validate_garm_category(cls, v: str) -> str:
        if v not in GARM_CATEGORIES:
            raise ValueError(f"garm_category must be one of {GARM_CATEGORIES}, got '{v}'")
        return v


class AdReviewObservation(Observation):
    """Environment observation: content to review + scoring feedback."""

    content_id: str = Field(description="Unique content item ID")
    content_text: str = Field(description="UGC text to review")
    content_type: str = Field(description="post, comment, caption, or bio")
    platform: str = Field(description="instagram, tiktok, youtube, twitter")
    difficulty: str = Field(description="easy, medium, or hard")

    step_number: int = Field(default=0)
    max_steps: int = Field(default=3)
    additional_context: Optional[str] = Field(default=None)

    score_decision: float = Field(default=0.0)
    score_category: float = Field(default=0.0)
    score_reasoning: float = Field(default=0.0)
    score_efficiency: float = Field(default=0.0)
    total_score: float = Field(default=0.0)

    feedback: str = Field(default="")
    gold_decision: Optional[str] = Field(default=None)
    gold_iab_category: Optional[str] = Field(default=None)
    gold_garm_category: Optional[str] = Field(default=None)

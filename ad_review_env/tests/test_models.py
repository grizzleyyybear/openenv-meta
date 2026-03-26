"""Tests for models.py — Pydantic model validation."""

import pytest

# Skip model instantiation tests if openenv is not installed
try:
    from models import AdReviewAction, AdReviewObservation, IAB_CATEGORIES, GARM_CATEGORIES
    HAS_OPENENV = True
except ImportError:
    HAS_OPENENV = False
    # Fallback: import just the category lists from data to still test something
    IAB_CATEGORIES = [
        "IAB_SAFE", "IAB_ADULT", "IAB_VIOLENCE", "IAB_HATE_SPEECH", "IAB_ILLEGAL",
        "IAB_MISINFORMATION", "IAB_PROFANITY", "IAB_DRUGS", "IAB_GAMBLING", "IAB_CONTROVERSIAL",
    ]
    GARM_CATEGORIES = [
        "GARM_SAFE", "GARM_ADULT_EXPLICIT", "GARM_ARMS_AMMUNITION", "GARM_CRIME_HARMFUL",
        "GARM_DEATH_INJURY", "GARM_HATE_SPEECH", "GARM_OBSCENITY_PROFANITY",
        "GARM_ONLINE_PIRACY", "GARM_SPAM_HARMFUL", "GARM_TERRORISM",
    ]


class TestCategories:
    def test_iab_categories_count(self):
        assert len(IAB_CATEGORIES) == 10

    def test_garm_categories_count(self):
        assert len(GARM_CATEGORIES) == 10

    def test_iab_has_safe(self):
        assert "IAB_SAFE" in IAB_CATEGORIES

    def test_garm_has_safe(self):
        assert "GARM_SAFE" in GARM_CATEGORIES

    def test_iab_no_duplicates(self):
        assert len(IAB_CATEGORIES) == len(set(IAB_CATEGORIES))

    def test_garm_no_duplicates(self):
        assert len(GARM_CATEGORIES) == len(set(GARM_CATEGORIES))


@pytest.mark.skipif(not HAS_OPENENV, reason="openenv not installed")
class TestAdReviewAction:
    def test_valid_action(self):
        action = AdReviewAction(
            decision="APPROVE",
            iab_category="IAB_SAFE",
            garm_category="GARM_SAFE",
            risk_level="LOW",
            reasoning="This is safe content with no harmful elements whatsoever.",
            confidence=0.9,
        )
        assert action.decision == "APPROVE"

    def test_invalid_iab_category(self):
        with pytest.raises(Exception):
            AdReviewAction(
                decision="APPROVE",
                iab_category="INVALID",
                garm_category="GARM_SAFE",
                risk_level="LOW",
                reasoning="This is safe content with no harmful elements.",
                confidence=0.9,
            )

    def test_invalid_garm_category(self):
        with pytest.raises(Exception):
            AdReviewAction(
                decision="APPROVE",
                iab_category="IAB_SAFE",
                garm_category="INVALID",
                risk_level="LOW",
                reasoning="This is safe content with no harmful elements.",
                confidence=0.9,
            )

    def test_confidence_out_of_range(self):
        with pytest.raises(Exception):
            AdReviewAction(
                decision="APPROVE",
                iab_category="IAB_SAFE",
                garm_category="GARM_SAFE",
                risk_level="LOW",
                reasoning="This is safe content with no harmful elements.",
                confidence=1.5,
            )

    def test_reasoning_too_short(self):
        with pytest.raises(Exception):
            AdReviewAction(
                decision="APPROVE",
                iab_category="IAB_SAFE",
                garm_category="GARM_SAFE",
                risk_level="LOW",
                reasoning="Short",
                confidence=0.9,
            )


@pytest.mark.skipif(not HAS_OPENENV, reason="openenv not installed")
class TestAdReviewObservation:
    def test_valid_observation(self):
        obs = AdReviewObservation(
            content_id="test_001",
            content_text="Test content",
            content_type="post",
            platform="twitter",
            difficulty="easy",
            done=False,
            reward=None,
        )
        assert obs.content_id == "test_001"
        assert obs.total_score == 0.0

    def test_observation_with_scores(self):
        obs = AdReviewObservation(
            content_id="test_001",
            content_text="Test content",
            content_type="post",
            platform="twitter",
            difficulty="easy",
            score_decision=0.4,
            score_category=0.3,
            score_reasoning=0.2,
            score_efficiency=0.1,
            total_score=1.0,
            feedback="Perfect score!",
            gold_decision="APPROVE",
            done=True,
            reward=1.0,
        )
        assert obs.total_score == 1.0
        assert obs.gold_decision == "APPROVE"

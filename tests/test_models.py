"""Tests for models.py — Pydantic model validation."""

import pytest
from ad_review_env.models import AdReviewAction, AdReviewObservation, IAB_CATEGORIES, GARM_CATEGORIES


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


class TestAdReviewObservation:
    def test_valid_observation(self):
        obs = AdReviewObservation(
            content_id="test_001",
            content_text="Test content",
            content_type="post",
            platform="x",
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
            platform="x",
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


class TestMultiStepModels:
    """Tests for multi-step episode model support."""

    def test_action_default_is_decide(self):
        action = AdReviewAction(
            decision="APPROVE",
            iab_category="IAB_SAFE",
            garm_category="GARM_SAFE",
            risk_level="LOW",
            reasoning="Standard safe content review.",
            confidence=0.9,
        )
        assert action.action_type == "DECIDE"

    def test_action_request_context(self):
        action = AdReviewAction(action_type="REQUEST_CONTEXT")
        assert action.action_type == "REQUEST_CONTEXT"
        assert action.decision == "ESCALATE"  # default

    def test_action_invalid_action_type(self):
        with pytest.raises(Exception):
            AdReviewAction(action_type="INVALID")

    def test_observation_has_step_fields(self):
        obs = AdReviewObservation(
            content_id="test_001",
            content_text="Test",
            content_type="post",
            platform="x",
            difficulty="easy",
            step_number=1,
            max_steps=3,
            additional_context="[Context 1] Author has clean record.",
            done=False,
            reward=None,
        )
        assert obs.step_number == 1
        assert obs.max_steps == 3
        assert "Author has clean record" in obs.additional_context

    def test_observation_defaults_step_zero(self):
        obs = AdReviewObservation(
            content_id="test_001",
            content_text="Test",
            content_type="post",
            platform="x",
            difficulty="easy",
            done=False,
            reward=None,
        )
        assert obs.step_number == 0
        assert obs.max_steps == 3
        assert obs.additional_context is None

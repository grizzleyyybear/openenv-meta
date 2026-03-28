"""Tests for grader.py — scoring logic correctness."""

from ad_review_env.grader import grade


def _make_gold(decision="APPROVE", iab="IAB_SAFE", garm="GARM_SAFE",
               risk="LOW", difficulty="easy"):
    return {
        "gold_decision": decision,
        "gold_iab_category": iab,
        "gold_garm_category": garm,
        "gold_risk_level": risk,
        "difficulty": difficulty,
    }


def _make_action(decision="APPROVE", iab="IAB_SAFE", garm="GARM_SAFE",
                 risk="LOW", reasoning="This is a safe post with no harmful content at all.",
                 confidence=0.9, flagged=None):
    return {
        "decision": decision,
        "iab_category": iab,
        "garm_category": garm,
        "risk_level": risk,
        "reasoning": reasoning,
        "confidence": confidence,
        "flagged_elements": flagged or [],
    }


class TestPerfectScore:
    def test_all_correct_easy(self):
        gold = _make_gold()
        action = _make_action(reasoning="A" * 100)
        total, scores, feedback = grade(action, gold)
        assert total > 0.9
        assert scores["decision"] == 0.4

    def test_all_correct_hard(self):
        gold = _make_gold(difficulty="hard")
        action = _make_action(reasoning="A" * 100)
        total, scores, feedback = grade(action, gold)
        # Hard tasks get 1.1x multiplier
        assert total >= 0.99


class TestDecisionScoring:
    def test_correct_decision(self):
        gold = _make_gold(decision="REJECT", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY")
        action = _make_action(decision="REJECT", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY")
        total, scores, _ = grade(action, gold)
        assert scores["decision"] == 0.4

    def test_wrong_decision(self):
        gold = _make_gold(decision="REJECT", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY")
        action = _make_action(decision="APPROVE", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY")
        total, scores, _ = grade(action, gold)
        assert scores["decision"] == 0.0

    def test_adjacent_decision_partial_credit(self):
        gold = _make_gold(decision="REJECT", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY")
        action = _make_action(decision="ESCALATE", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY")
        total, scores, _ = grade(action, gold)
        # Adjacent: ESCALATE when gold is REJECT → 0.4 * 0.4 = 0.16
        assert scores["decision"] == round(0.4 * 0.4, 4)

    def test_approve_escalate_not_adjacent(self):
        gold = _make_gold(decision="APPROVE")
        action = _make_action(decision="ESCALATE")
        total, scores, _ = grade(action, gold)
        assert scores["decision"] == 0.0


class TestCategoryScoring:
    def test_both_correct(self):
        gold = _make_gold(iab="IAB_VIOLENCE", garm="GARM_DEATH_INJURY")
        action = _make_action(iab="IAB_VIOLENCE", garm="GARM_DEATH_INJURY")
        _, scores, _ = grade(action, gold)
        assert scores["category"] == 0.3

    def test_iab_only_correct(self):
        gold = _make_gold(iab="IAB_VIOLENCE", garm="GARM_DEATH_INJURY")
        action = _make_action(iab="IAB_VIOLENCE", garm="GARM_SAFE")
        _, scores, _ = grade(action, gold)
        assert scores["category"] == round(0.3 * 0.5, 4)

    def test_garm_only_correct(self):
        gold = _make_gold(iab="IAB_VIOLENCE", garm="GARM_DEATH_INJURY")
        action = _make_action(iab="IAB_SAFE", garm="GARM_DEATH_INJURY")
        _, scores, _ = grade(action, gold)
        assert scores["category"] == round(0.3 * 0.5, 4)

    def test_both_wrong(self):
        gold = _make_gold(iab="IAB_VIOLENCE", garm="GARM_DEATH_INJURY")
        action = _make_action(iab="IAB_SAFE", garm="GARM_SAFE")
        _, scores, _ = grade(action, gold)
        assert scores["category"] == 0.0


class TestReasoningScoring:
    def test_long_reasoning(self):
        gold = _make_gold()
        action = _make_action(reasoning="A" * 100)
        _, scores, _ = grade(action, gold)
        assert scores["reasoning"] == round(0.2 * 1.0, 4)

    def test_medium_reasoning(self):
        gold = _make_gold()
        action = _make_action(reasoning="A" * 50)
        _, scores, _ = grade(action, gold)
        assert scores["reasoning"] == round(0.2 * 0.6, 4)

    def test_short_reasoning(self):
        gold = _make_gold()
        action = _make_action(reasoning="A" * 10)
        _, scores, _ = grade(action, gold)
        assert scores["reasoning"] == round(0.2 * 0.3, 4)

    def test_flagged_elements_bonus(self):
        gold = _make_gold(decision="REJECT", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY")
        action = _make_action(
            decision="REJECT", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY",
            reasoning="A" * 50, flagged=["f-word", "violent language"]
        )
        _, scores, _ = grade(action, gold)
        # 0.6 + 0.2 = 0.8 → 0.2 * 0.8 = 0.16
        assert scores["reasoning"] == round(0.2 * 0.8, 4)


class TestEfficiencyScoring:
    def test_correct_high_confidence(self):
        gold = _make_gold()
        action = _make_action(confidence=0.95)
        _, scores, _ = grade(action, gold)
        assert scores["efficiency"] == round(0.1 * 0.95, 4)

    def test_correct_low_confidence(self):
        gold = _make_gold()
        action = _make_action(confidence=0.3)
        _, scores, _ = grade(action, gold)
        assert scores["efficiency"] == round(0.1 * 0.3, 4)

    def test_wrong_high_confidence_penalized(self):
        gold = _make_gold(decision="REJECT")
        action = _make_action(decision="APPROVE", confidence=0.95)
        _, scores, _ = grade(action, gold)
        # Wrong + high confidence: 1.0 - 0.95 = 0.05
        assert scores["efficiency"] == round(0.1 * 0.05, 4)

    def test_wrong_low_confidence_rewarded(self):
        gold = _make_gold(decision="REJECT")
        action = _make_action(decision="APPROVE", confidence=0.3)
        _, scores, _ = grade(action, gold)
        # Wrong + low confidence: 1.0 - 0.3 = 0.7
        assert scores["efficiency"] == round(0.1 * 0.7, 4)


class TestDifficultyMultiplier:
    def test_easy_no_multiplier(self):
        gold = _make_gold(difficulty="easy")
        action = _make_action(reasoning="A" * 100)
        total, _, _ = grade(action, gold)
        # Raw total: 0.4 + 0.3 + 0.2 + 0.09 = 0.99 * 1.0 = 0.99
        assert 0.98 <= total <= 1.0

    def test_medium_1_05_multiplier(self):
        gold = _make_gold(difficulty="medium")
        action = _make_action(reasoning="A" * 100)
        total, _, _ = grade(action, gold)
        # Should be capped at 1.0
        assert total == 1.0

    def test_hard_1_1_multiplier(self):
        gold = _make_gold(difficulty="hard")
        action = _make_action(reasoning="A" * 100)
        total, _, _ = grade(action, gold)
        assert total == 1.0


class TestFeedback:
    def test_feedback_is_string(self):
        gold = _make_gold()
        action = _make_action()
        _, _, feedback = grade(action, gold)
        assert isinstance(feedback, str)
        assert len(feedback) > 0

    def test_correct_feedback_has_checkmark(self):
        gold = _make_gold()
        action = _make_action()
        _, _, feedback = grade(action, gold)
        assert "correct" in feedback.lower()

    def test_wrong_feedback_has_cross(self):
        gold = _make_gold(decision="REJECT")
        action = _make_action(decision="APPROVE")
        _, _, feedback = grade(action, gold)
        assert "wrong" in feedback.lower()


class TestStepEfficiency:
    """Test that multi-step episodes adjust efficiency scoring."""

    def test_one_step_full_efficiency(self):
        gold = _make_gold()
        action = _make_action(confidence=0.9)
        _, scores_1, _ = grade(action, gold, steps_taken=1)
        # 1 step: efficiency = 0.1 * 0.9 * 1.0
        assert scores_1["efficiency"] == round(0.1 * 0.9 * 1.0, 4)

    def test_two_steps_reduced_efficiency(self):
        gold = _make_gold()
        action = _make_action(confidence=0.9)
        _, scores_2, _ = grade(action, gold, steps_taken=2)
        # 2 steps: efficiency = 0.1 * 0.9 * 0.7
        assert scores_2["efficiency"] == round(0.1 * 0.9 * 0.7, 4)

    def test_three_steps_low_efficiency(self):
        gold = _make_gold()
        action = _make_action(confidence=0.9)
        _, scores_3, _ = grade(action, gold, steps_taken=3)
        # 3 steps: efficiency = 0.1 * 0.9 * 0.4
        assert scores_3["efficiency"] == round(0.1 * 0.9 * 0.4, 4)

    def test_fewer_steps_higher_total(self):
        gold = _make_gold()
        action = _make_action(confidence=0.9)
        total_1, _, _ = grade(action, gold, steps_taken=1)
        total_2, _, _ = grade(action, gold, steps_taken=2)
        total_3, _, _ = grade(action, gold, steps_taken=3)
        assert total_1 > total_2 > total_3

    def test_default_steps_is_one(self):
        gold = _make_gold()
        action = _make_action(confidence=0.9)
        _, scores_default, _ = grade(action, gold)
        _, scores_explicit, _ = grade(action, gold, steps_taken=1)
        assert scores_default == scores_explicit

    def test_feedback_includes_step_count(self):
        gold = _make_gold()
        action = _make_action()
        _, _, feedback = grade(action, gold, steps_taken=2)
        assert "2 steps" in feedback


class TestContextLayers:
    """Test that data items have context layers for multi-step episodes."""

    def test_all_items_have_context_layers(self):
        from ad_review_env.data import CONTENT_ITEMS
        for item in CONTENT_ITEMS:
            assert "context_layer_1" in item, f"{item['content_id']} missing layer 1"
            assert "context_layer_2" in item, f"{item['content_id']} missing layer 2"

    def test_context_layers_are_strings(self):
        from ad_review_env.data import CONTENT_ITEMS
        for item in CONTENT_ITEMS:
            assert isinstance(item["context_layer_1"], str)
            assert isinstance(item["context_layer_2"], str)
            assert len(item["context_layer_1"]) > 20
            assert len(item["context_layer_2"]) > 20

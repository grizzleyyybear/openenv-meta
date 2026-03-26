"""Tests for grader.py — scoring logic correctness."""

from grader import grade


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
        assert "✓" in feedback

    def test_wrong_feedback_has_cross(self):
        gold = _make_gold(decision="REJECT")
        action = _make_action(decision="APPROVE")
        _, _, feedback = grade(action, gold)
        assert "✗" in feedback

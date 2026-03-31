"""Tests for grader.py -- scoring logic correctness."""

from ad_review_env.grader import grade


def _make_gold(decision="APPROVE", iab="IAB_SAFE", garm="GARM_SAFE",
               risk="LOW", difficulty="easy", age_rating="TEEN"):
    return {
        "gold_decision": decision,
        "gold_iab_category": iab,
        "gold_garm_category": garm,
        "gold_risk_level": risk,
        "gold_age_rating": age_rating,
        "difficulty": difficulty,
    }


def _make_action(decision="APPROVE", iab="IAB_SAFE", garm="GARM_SAFE",
                 risk="LOW", reasoning="This is a safe post with no harmful content at all.",
                 confidence=0.9, flagged=None, age_rating="TEEN"):
    return {
        "decision": decision,
        "iab_category": iab,
        "garm_category": garm,
        "risk_level": risk,
        "reasoning": reasoning,
        "confidence": confidence,
        "age_rating": age_rating,
        "flagged_elements": flagged or [],
    }


class TestPerfectScore:
    def test_all_correct_easy(self):
        gold = _make_gold()
        action = _make_action(reasoning="A" * 120)
        total, scores, feedback = grade(action, gold)
        assert total > 0.9
        assert scores["decision"] == 0.30

    def test_all_correct_hard(self):
        gold = _make_gold(difficulty="hard")
        action = _make_action(reasoning="A" * 120)
        total, scores, feedback = grade(action, gold)
        assert total >= 0.99

    def test_perfect_with_confidence_1(self):
        gold = _make_gold()
        action = _make_action(reasoning="A" * 120, confidence=1.0)
        total, scores, _ = grade(action, gold)
        assert abs(total - 1.0) < 1e-9


class TestDecisionScoring:
    def test_correct_decision(self):
        gold = _make_gold(decision="REJECT", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY")
        action = _make_action(decision="REJECT", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY")
        _, scores, _ = grade(action, gold)
        assert scores["decision"] == 0.30

    def test_wrong_decision_approve_vs_reject(self):
        gold = _make_gold(decision="REJECT", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY")
        action = _make_action(decision="APPROVE", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY")
        _, scores, _ = grade(action, gold)
        assert scores["decision"] == 0.0

    def test_adjacent_escalate_reject_partial(self):
        gold = _make_gold(decision="REJECT", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY")
        action = _make_action(decision="ESCALATE", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY")
        _, scores, _ = grade(action, gold)
        assert scores["decision"] == round(0.30 * 0.4, 4)

    def test_adjacent_reject_escalate_partial(self):
        gold = _make_gold(decision="ESCALATE")
        action = _make_action(decision="REJECT")
        _, scores, _ = grade(action, gold)
        assert scores["decision"] == round(0.30 * 0.4, 4)

    def test_adjacent_approve_escalate_partial(self):
        """APPROVE<->ESCALATE now gives partial credit of 0.15."""
        gold = _make_gold(decision="APPROVE")
        action = _make_action(decision="ESCALATE")
        _, scores, _ = grade(action, gold)
        assert scores["decision"] == round(0.30 * 0.15, 4)

    def test_adjacent_escalate_approve_partial(self):
        gold = _make_gold(decision="ESCALATE")
        action = _make_action(decision="APPROVE")
        _, scores, _ = grade(action, gold)
        assert scores["decision"] == round(0.30 * 0.15, 4)

    def test_approve_reject_no_credit(self):
        gold = _make_gold(decision="APPROVE")
        action = _make_action(decision="REJECT")
        _, scores, _ = grade(action, gold)
        assert scores["decision"] == 0.0

    def test_reject_approve_no_credit(self):
        gold = _make_gold(decision="REJECT")
        action = _make_action(decision="APPROVE")
        _, scores, _ = grade(action, gold)
        assert scores["decision"] == 0.0


class TestCategoryScoring:
    def test_both_correct(self):
        gold = _make_gold(iab="IAB_VIOLENCE", garm="GARM_DEATH_INJURY")
        action = _make_action(iab="IAB_VIOLENCE", garm="GARM_DEATH_INJURY")
        _, scores, _ = grade(action, gold)
        assert scores["category"] == 0.20

    def test_iab_only_correct(self):
        gold = _make_gold(iab="IAB_VIOLENCE", garm="GARM_DEATH_INJURY")
        action = _make_action(iab="IAB_VIOLENCE", garm="GARM_SAFE")
        _, scores, _ = grade(action, gold)
        assert scores["category"] == 0.12

    def test_garm_only_correct(self):
        gold = _make_gold(iab="IAB_VIOLENCE", garm="GARM_DEATH_INJURY")
        action = _make_action(iab="IAB_SAFE", garm="GARM_DEATH_INJURY")
        _, scores, _ = grade(action, gold)
        assert scores["category"] == 0.12

    def test_both_wrong_with_risk_match(self):
        """Both categories wrong but risk matches -> still get risk contribution."""
        gold = _make_gold(iab="IAB_VIOLENCE", garm="GARM_DEATH_INJURY")
        action = _make_action(iab="IAB_SAFE", garm="GARM_SAFE")
        _, scores, _ = grade(action, gold)
        assert scores["category"] == 0.04

    def test_risk_same_level(self):
        gold = _make_gold(risk="MEDIUM")
        action = _make_action(risk="MEDIUM")
        _, scores, _ = grade(action, gold)
        assert scores["category"] == 0.20

    def test_risk_one_level_apart(self):
        gold = _make_gold(risk="LOW")
        action = _make_action(risk="MEDIUM")
        _, scores, _ = grade(action, gold)
        # risk_dist=1, risk_score=0.65, cat=0.4+0.4+0.2*0.65=0.93
        assert scores["category"] == round(0.20 * 0.93, 4)

    def test_risk_two_levels_apart(self):
        gold = _make_gold(risk="LOW")
        action = _make_action(risk="HIGH")
        _, scores, _ = grade(action, gold)
        # risk_dist=2, risk_score=0.3, cat=0.4+0.4+0.2*0.3=0.86
        assert scores["category"] == round(0.20 * 0.86, 4)

    def test_risk_three_levels_apart(self):
        gold = _make_gold(risk="LOW")
        action = _make_action(risk="CRITICAL")
        _, scores, _ = grade(action, gold)
        # risk_dist=3, risk_score=0.0, cat=0.4+0.4+0.0=0.8
        assert scores["category"] == 0.16

    def test_risk_medium_to_high(self):
        gold = _make_gold(risk="MEDIUM")
        action = _make_action(risk="HIGH")
        _, scores, _ = grade(action, gold)
        assert scores["category"] == round(0.20 * 0.93, 4)

    def test_risk_high_to_critical(self):
        gold = _make_gold(risk="HIGH")
        action = _make_action(risk="CRITICAL")
        _, scores, _ = grade(action, gold)
        assert scores["category"] == round(0.20 * 0.93, 4)

    def test_all_wrong_with_risk_mismatch(self):
        gold = _make_gold(iab="IAB_VIOLENCE", garm="GARM_DEATH_INJURY", risk="LOW")
        action = _make_action(iab="IAB_SAFE", garm="GARM_SAFE", risk="CRITICAL")
        _, scores, _ = grade(action, gold)
        assert scores["category"] == 0.0


class TestReasoningScoring:
    def test_120_chars_perfect(self):
        gold = _make_gold()
        action = _make_action(reasoning="A" * 120)
        _, scores, _ = grade(action, gold)
        assert scores["reasoning"] == round(0.18 * 1.0, 4)

    def test_200_chars(self):
        gold = _make_gold()
        action = _make_action(reasoning="A" * 200)
        _, scores, _ = grade(action, gold)
        assert scores["reasoning"] == round(0.18 * 1.0, 4)

    def test_119_chars_not_perfect(self):
        gold = _make_gold()
        action = _make_action(reasoning="A" * 119)
        _, scores, _ = grade(action, gold)
        assert scores["reasoning"] == round(0.18 * 0.75, 4)

    def test_80_chars(self):
        gold = _make_gold()
        action = _make_action(reasoning="A" * 80)
        _, scores, _ = grade(action, gold)
        assert scores["reasoning"] == round(0.18 * 0.75, 4)

    def test_79_chars(self):
        gold = _make_gold()
        action = _make_action(reasoning="A" * 79)
        _, scores, _ = grade(action, gold)
        assert scores["reasoning"] == round(0.18 * 0.5, 4)

    def test_40_chars(self):
        gold = _make_gold()
        action = _make_action(reasoning="A" * 40)
        _, scores, _ = grade(action, gold)
        assert scores["reasoning"] == round(0.18 * 0.5, 4)

    def test_39_chars(self):
        gold = _make_gold()
        action = _make_action(reasoning="A" * 39)
        _, scores, _ = grade(action, gold)
        assert scores["reasoning"] == round(0.18 * 0.25, 4)

    def test_10_chars(self):
        gold = _make_gold()
        action = _make_action(reasoning="A" * 10)
        _, scores, _ = grade(action, gold)
        assert scores["reasoning"] == round(0.18 * 0.25, 4)

    def test_9_chars_zero(self):
        gold = _make_gold()
        action = _make_action(reasoning="A" * 9)
        _, scores, _ = grade(action, gold)
        assert scores["reasoning"] == 0.0

    def test_empty_reasoning(self):
        gold = _make_gold()
        action = _make_action(reasoning="")
        _, scores, _ = grade(action, gold)
        assert scores["reasoning"] == 0.0

    def test_flagged_elements_bonus(self):
        gold = _make_gold(decision="REJECT", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY")
        action = _make_action(
            decision="REJECT", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY",
            reasoning="A" * 50, flagged=["f-word", "violent language"]
        )
        _, scores, _ = grade(action, gold)
        # length 0.5 + flagging min(0.3, 0.2) = 0.7
        assert scores["reasoning"] == round(0.18 * 0.7, 4)

    def test_flagged_three_elements(self):
        gold = _make_gold(decision="REJECT", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY")
        action = _make_action(
            decision="REJECT", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY",
            reasoning="A" * 50, flagged=["a", "b", "c"]
        )
        _, scores, _ = grade(action, gold)
        # length 0.5 + flagging min(0.3, 0.3) = 0.8
        assert scores["reasoning"] == round(0.18 * 0.8, 4)

    def test_flagged_four_elements_capped(self):
        gold = _make_gold(decision="REJECT", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY")
        action = _make_action(
            decision="REJECT", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY",
            reasoning="A" * 50, flagged=["a", "b", "c", "d"]
        )
        _, scores, _ = grade(action, gold)
        # length 0.5 + flagging min(0.3, 0.4)=0.3 -> 0.8
        assert scores["reasoning"] == round(0.18 * 0.8, 4)

    def test_flagged_for_approve_no_bonus(self):
        gold = _make_gold(decision="APPROVE")
        action = _make_action(decision="APPROVE", reasoning="A" * 50, flagged=["a", "b"])
        _, scores, _ = grade(action, gold)
        assert scores["reasoning"] == round(0.18 * 0.5, 4)

    def test_specificity_bonus_profanity(self):
        gold = _make_gold()
        action = _make_action(reasoning="This ad contains profanity which is bad for brand safety")
        _, scores, _ = grade(action, gold)
        # len=56 >= 40 -> 0.5, "profanity" keyword -> +0.1 -> 0.6
        assert scores["reasoning"] == round(0.18 * 0.6, 4)

    def test_specificity_bonus_violence(self):
        gold = _make_gold()
        action = _make_action(reasoning="This content depicts violence and should be flagged now")
        _, scores, _ = grade(action, gold)
        assert scores["reasoning"] == round(0.18 * 0.6, 4)

    def test_specificity_bonus_hate(self):
        gold = _make_gold()
        action = _make_action(reasoning="Detected hate speech targeting minorities in this ad")
        _, scores, _ = grade(action, gold)
        assert scores["reasoning"] == round(0.18 * 0.6, 4)

    def test_specificity_no_keywords(self):
        gold = _make_gold()
        action = _make_action(reasoning="A" * 50)
        _, scores, _ = grade(action, gold)
        assert scores["reasoning"] == round(0.18 * 0.5, 4)

    def test_specificity_requires_min_length(self):
        """Specificity bonus requires reasoning_len >= 10."""
        gold = _make_gold()
        action = _make_action(reasoning="hate bad")
        _, scores, _ = grade(action, gold)
        assert scores["reasoning"] == 0.0

    def test_combined_flagging_and_specificity(self):
        gold = _make_gold(decision="REJECT", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY")
        action = _make_action(
            decision="REJECT", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY",
            reasoning="This ad contains profanity and is harmful to brand safety",
            flagged=["f-word"]
        )
        _, scores, _ = grade(action, gold)
        # len=57 >= 40 -> 0.5, flag 0.1, specificity 0.1 -> 0.7
        assert scores["reasoning"] == round(0.18 * 0.7, 4)

    def test_reasoning_capped_at_one(self):
        gold = _make_gold(decision="REJECT", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY")
        action = _make_action(
            decision="REJECT", iab="IAB_PROFANITY", garm="GARM_OBSCENITY_PROFANITY",
            reasoning="A" * 120 + " profanity",
            flagged=["a", "b", "c"]
        )
        _, scores, _ = grade(action, gold)
        # 1.0 + 0.3 + 0.1 -> capped at 1.0
        assert scores["reasoning"] == round(0.18 * 1.0, 4)


class TestEfficiencyScoring:
    def test_one_step(self):
        gold = _make_gold()
        action = _make_action()
        _, scores, _ = grade(action, gold, steps_taken=1)
        assert scores["efficiency"] == round(0.1 * 1.0, 4)

    def test_two_steps(self):
        gold = _make_gold()
        action = _make_action()
        _, scores, _ = grade(action, gold, steps_taken=2)
        assert scores["efficiency"] == round(0.1 * 0.7, 4)

    def test_three_steps(self):
        gold = _make_gold()
        action = _make_action()
        _, scores, _ = grade(action, gold, steps_taken=3)
        assert scores["efficiency"] == round(0.1 * 0.4, 4)

    def test_confidence_does_not_affect_efficiency(self):
        """Efficiency depends only on step_mult, not confidence."""
        gold = _make_gold()
        action_high = _make_action(confidence=0.95)
        action_low = _make_action(confidence=0.1)
        _, scores_h, _ = grade(action_high, gold)
        _, scores_l, _ = grade(action_low, gold)
        assert scores_h["efficiency"] == scores_l["efficiency"]

    def test_steps_clamped_high(self):
        gold = _make_gold()
        action = _make_action()
        _, scores, _ = grade(action, gold, steps_taken=5)
        assert scores["efficiency"] == round(0.1 * 0.4, 4)


class TestCalibration:
    def test_correct_high_confidence(self):
        gold = _make_gold()
        action = _make_action(confidence=0.9)
        _, scores, _ = grade(action, gold)
        assert scores["calibration"] == round(0.10 * 0.9, 4)

    def test_correct_low_confidence(self):
        gold = _make_gold()
        action = _make_action(confidence=0.3)
        _, scores, _ = grade(action, gold)
        assert scores["calibration"] == round(0.10 * 0.3, 4)

    def test_correct_confidence_one(self):
        gold = _make_gold()
        action = _make_action(confidence=1.0)
        _, scores, _ = grade(action, gold)
        assert scores["calibration"] == round(0.10 * 1.0, 4)

    def test_correct_confidence_zero(self):
        gold = _make_gold()
        action = _make_action(confidence=0.0)
        _, scores, _ = grade(action, gold)
        assert scores["calibration"] == round(0.10 * 0.0, 4)

    def test_correct_medium_confidence(self):
        gold = _make_gold()
        action = _make_action(confidence=0.5)
        _, scores, _ = grade(action, gold)
        assert scores["calibration"] == round(0.10 * 0.5, 4)

    def test_wrong_low_confidence(self):
        gold = _make_gold(decision="REJECT")
        action = _make_action(decision="APPROVE", confidence=0.3)
        _, scores, _ = grade(action, gold)
        assert scores["calibration"] == round(0.10 * 0.7, 4)

    def test_wrong_medium_confidence(self):
        gold = _make_gold(decision="REJECT")
        action = _make_action(decision="APPROVE", confidence=0.5)
        _, scores, _ = grade(action, gold)
        assert scores["calibration"] == round(0.10 * 0.5, 4)

    def test_wrong_confidence_zero(self):
        gold = _make_gold(decision="REJECT")
        action = _make_action(decision="APPROVE", confidence=0.0)
        _, scores, _ = grade(action, gold)
        assert scores["calibration"] == round(0.10 * 1.0, 4)

    def test_wrong_exactly_0_9_no_extra_penalty(self):
        """Confidence exactly 0.9 does NOT trigger the >0.9 penalty."""
        gold = _make_gold(decision="REJECT")
        action = _make_action(decision="APPROVE", confidence=0.9)
        _, scores, _ = grade(action, gold)
        assert scores["calibration"] == round(0.10 * 0.1, 4)

    def test_overconfident_wrong_0_95(self):
        """Confidence 0.95 when wrong -> extra 0.5x penalty."""
        gold = _make_gold(decision="REJECT")
        action = _make_action(decision="APPROVE", confidence=0.95)
        _, scores, _ = grade(action, gold)
        # (1 - 0.95) * 0.5 = 0.025
        assert scores["calibration"] == round(0.10 * 0.025, 4)

    def test_overconfident_wrong_0_99(self):
        gold = _make_gold(decision="REJECT")
        action = _make_action(decision="APPROVE", confidence=0.99)
        _, scores, _ = grade(action, gold)
        # (1 - 0.99) * 0.5 = 0.005
        assert scores["calibration"] == round(0.10 * 0.005, 4)

    def test_overconfident_wrong_0_91(self):
        gold = _make_gold(decision="REJECT")
        action = _make_action(decision="APPROVE", confidence=0.91)
        _, scores, _ = grade(action, gold)
        # (1 - 0.91) * 0.5 = 0.045
        assert scores["calibration"] == round(0.10 * 0.045, 4)

    def test_overconfident_much_worse_than_normal_wrong(self):
        gold = _make_gold(decision="REJECT")
        action_over = _make_action(decision="APPROVE", confidence=0.95)
        action_norm = _make_action(decision="APPROVE", confidence=0.5)
        _, scores_o, _ = grade(action_over, gold)
        _, scores_n, _ = grade(action_norm, gold)
        assert scores_o["calibration"] < scores_n["calibration"]


class TestDifficultyMultiplier:
    def test_easy_no_multiplier(self):
        gold = _make_gold(difficulty="easy")
        action = _make_action(reasoning="A" * 120)
        total, _, _ = grade(action, gold)
        assert 0.98 <= total <= 1.0

    def test_medium_1_05_multiplier(self):
        gold = _make_gold(difficulty="medium")
        action = _make_action(reasoning="A" * 120)
        total, _, _ = grade(action, gold)
        assert abs(total - 1.0) < 1e-9

    def test_hard_1_1_multiplier(self):
        gold = _make_gold(difficulty="hard")
        action = _make_action(reasoning="A" * 120)
        total, _, _ = grade(action, gold)
        assert abs(total - 1.0) < 1e-9

    def test_medium_boosts_raw_total(self):
        gold = _make_gold(decision="REJECT", difficulty="medium")
        action = _make_action(decision="APPROVE", reasoning="A" * 120, confidence=0.5)
        total_med, _, _ = grade(action, gold)
        gold_easy = _make_gold(decision="REJECT", difficulty="easy")
        total_easy, _, _ = grade(action, gold_easy)
        assert total_med > total_easy

    def test_unknown_difficulty_no_multiplier(self):
        gold = _make_gold(difficulty="unknown")
        action = _make_action(reasoning="A" * 120)
        total, _, _ = grade(action, gold)
        assert 0.98 <= total <= 1.0


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
        assert scores_1["efficiency"] == round(0.1 * 1.0, 4)

    def test_two_steps_reduced_efficiency(self):
        gold = _make_gold()
        action = _make_action(confidence=0.9)
        _, scores_2, _ = grade(action, gold, steps_taken=2)
        assert scores_2["efficiency"] == round(0.1 * 0.7, 4)

    def test_three_steps_low_efficiency(self):
        gold = _make_gold()
        action = _make_action(confidence=0.9)
        _, scores_3, _ = grade(action, gold, steps_taken=3)
        assert scores_3["efficiency"] == round(0.1 * 0.4, 4)

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


class TestEdgeCases:
    def test_empty_decision(self):
        gold = _make_gold(decision="APPROVE")
        action = _make_action(decision="")
        _, scores, _ = grade(action, gold)
        assert scores["decision"] == 0.0

    def test_missing_flagged_elements_defaults(self):
        gold = _make_gold()
        action = {"decision": "APPROVE", "iab_category": "IAB_SAFE",
                  "garm_category": "GARM_SAFE", "risk_level": "LOW",
                  "reasoning": "A" * 120, "confidence": 0.9}
        total, scores, _ = grade(action, gold)
        assert total > 0.9

    def test_invalid_confidence_clamped_high(self):
        gold = _make_gold()
        action = _make_action(confidence=1.5)
        _, scores, _ = grade(action, gold)
        assert scores["calibration"] == 0.10

    def test_invalid_confidence_clamped_low(self):
        gold = _make_gold()
        action = _make_action(confidence=-0.5)
        _, scores, _ = grade(action, gold)
        assert scores["calibration"] == 0.0

    def test_flagged_elements_not_list_ignored(self):
        gold = _make_gold()
        action = _make_action()
        action["flagged_elements"] = "not a list"
        _, scores, _ = grade(action, gold)
        assert scores["reasoning"] >= 0.0

    def test_whitespace_reasoning_trimmed(self):
        gold = _make_gold()
        action = _make_action(reasoning="   " + "A" * 40 + "   ")
        _, scores, _ = grade(action, gold)
        assert scores["reasoning"] == round(0.18 * 0.5, 4)

    def test_total_is_capped_at_one(self):
        gold = _make_gold(difficulty="hard")
        action = _make_action(reasoning="A" * 120, confidence=1.0)
        total, _, _ = grade(action, gold)
        assert total <= 1.0

    def test_total_is_non_negative(self):
        gold = _make_gold(decision="REJECT")
        action = _make_action(decision="APPROVE", reasoning="", confidence=0.99)
        total, _, _ = grade(action, gold)
        assert total >= 0.0

    def test_component_scores_sum_to_total_easy(self):
        gold = _make_gold(difficulty="easy")
        action = _make_action(reasoning="A" * 120)
        total, scores, _ = grade(action, gold)
        component_sum = (scores["decision"] + scores["category"] +
                        scores["reasoning"] + scores["age_rating"] +
                        scores["efficiency"] + scores["calibration"])
        assert abs(total - component_sum) < 0.001

    def test_returns_three_values(self):
        gold = _make_gold()
        action = _make_action()
        result = grade(action, gold)
        assert len(result) == 3

    def test_scores_dict_has_all_keys(self):
        gold = _make_gold()
        action = _make_action()
        _, scores, _ = grade(action, gold)
        expected_keys = {"decision", "category", "reasoning", "age_rating", "efficiency", "calibration", "total"}
        assert set(scores.keys()) == expected_keys


class TestIntegration:
    def test_all_wrong_minimum_score(self):
        gold = _make_gold(decision="REJECT", iab="IAB_VIOLENCE", garm="GARM_DEATH_INJURY", risk="HIGH")
        action = _make_action(decision="APPROVE", iab="IAB_SAFE", garm="GARM_SAFE", risk="LOW",
                             reasoning="", confidence=0.99)
        total, scores, feedback = grade(action, gold)
        assert 0.0 <= total <= 1.0
        assert isinstance(feedback, str)

    def test_adjacent_with_wrong_categories(self):
        gold = _make_gold(decision="ESCALATE", iab="IAB_VIOLENCE", garm="GARM_DEATH_INJURY")
        action = _make_action(decision="REJECT", iab="IAB_SAFE", garm="GARM_SAFE",
                             reasoning="A" * 120, confidence=0.5)
        total, scores, _ = grade(action, gold)
        assert scores["decision"] == round(0.30 * 0.4, 4)
        assert 0.58 <= total <= 0.65


class TestAgeRatingScoring:
    """Test the age_rating scoring component (weight 0.12)."""

    def test_exact_match_all_ages(self):
        gold = _make_gold(age_rating="ALL_AGES")
        action = _make_action(age_rating="ALL_AGES")
        _, scores, _ = grade(action, gold)
        assert scores["age_rating"] == round(0.12 * 1.0, 4)

    def test_exact_match_teen(self):
        gold = _make_gold(age_rating="TEEN")
        action = _make_action(age_rating="TEEN")
        _, scores, _ = grade(action, gold)
        assert scores["age_rating"] == round(0.12 * 1.0, 4)

    def test_exact_match_mature(self):
        gold = _make_gold(age_rating="MATURE")
        action = _make_action(age_rating="MATURE")
        _, scores, _ = grade(action, gold)
        assert scores["age_rating"] == round(0.12 * 1.0, 4)

    def test_exact_match_adult(self):
        gold = _make_gold(age_rating="ADULT")
        action = _make_action(age_rating="ADULT")
        _, scores, _ = grade(action, gold)
        assert scores["age_rating"] == round(0.12 * 1.0, 4)

    def test_off_by_one_all_ages_vs_teen(self):
        gold = _make_gold(age_rating="ALL_AGES")
        action = _make_action(age_rating="TEEN")
        _, scores, _ = grade(action, gold)
        assert scores["age_rating"] == round(0.12 * 0.5, 4)

    def test_off_by_one_teen_vs_mature(self):
        gold = _make_gold(age_rating="TEEN")
        action = _make_action(age_rating="MATURE")
        _, scores, _ = grade(action, gold)
        assert scores["age_rating"] == round(0.12 * 0.5, 4)

    def test_off_by_one_mature_vs_adult(self):
        gold = _make_gold(age_rating="MATURE")
        action = _make_action(age_rating="ADULT")
        _, scores, _ = grade(action, gold)
        assert scores["age_rating"] == round(0.12 * 0.5, 4)

    def test_off_by_two_all_ages_vs_mature(self):
        gold = _make_gold(age_rating="ALL_AGES")
        action = _make_action(age_rating="MATURE")
        _, scores, _ = grade(action, gold)
        assert scores["age_rating"] == round(0.12 * 0.15, 4)

    def test_off_by_two_teen_vs_adult(self):
        gold = _make_gold(age_rating="TEEN")
        action = _make_action(age_rating="ADULT")
        _, scores, _ = grade(action, gold)
        assert scores["age_rating"] == round(0.12 * 0.15, 4)

    def test_off_by_three_all_ages_vs_adult(self):
        gold = _make_gold(age_rating="ALL_AGES")
        action = _make_action(age_rating="ADULT")
        _, scores, _ = grade(action, gold)
        assert scores["age_rating"] == round(0.12 * 0.0, 4)

    def test_off_by_three_adult_vs_all_ages(self):
        gold = _make_gold(age_rating="ADULT")
        action = _make_action(age_rating="ALL_AGES")
        _, scores, _ = grade(action, gold)
        assert scores["age_rating"] == round(0.12 * 0.0, 4)

    def test_default_age_rating_is_teen(self):
        """When age_rating is not specified, defaults produce exact match."""
        gold = _make_gold()
        action = _make_action()
        _, scores, _ = grade(action, gold)
        assert scores["age_rating"] == round(0.12 * 1.0, 4)

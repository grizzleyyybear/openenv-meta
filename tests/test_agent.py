"""Tests for agent.py — smart agent scoring and correctness."""

from ad_review_env.data import CONTENT_ITEMS
from ad_review_env.grader import grade
from ad_review_env.agent import smart_agent, evaluate_all


class TestSmartAgentOverall:
    def test_mean_score_above_075(self):
        result = evaluate_all(CONTENT_ITEMS, grade)
        assert result["aggregate"]["mean_score"] > 0.75

    def test_decision_accuracy_above_90(self):
        result = evaluate_all(CONTENT_ITEMS, grade)
        assert result["aggregate"]["decision_accuracy"] > 0.90

    def test_all_50_items_processed(self):
        result = evaluate_all(CONTENT_ITEMS, grade)
        assert result["aggregate"]["total_items"] == 50
        assert len(result["results"]) == 50


class TestSmartAgentDecisions:
    def test_safe_content_approved(self):
        text = "Just made chocolate chip cookies! Recipe in bio. #baking #homemade"
        result = smart_agent(text)
        assert result["decision"] == "APPROVE"
        assert result["iab_category"] == "IAB_SAFE"

    def test_profanity_rejected(self):
        text = "I want to f***ing kill someone she ate my pizza"
        result = smart_agent(text)
        assert result["decision"] == "REJECT"

    def test_adult_content_rejected(self):
        text = "18+ ONLY 🔞 Full uncensored content available. DM me for access."
        result = smart_agent(text)
        assert result["decision"] == "REJECT"
        assert result["iab_category"] == "IAB_ADULT"

    def test_violence_rejected(self):
        text = "GRAPHIC: Compilation of the most brutal street fight knockouts"
        result = smart_agent(text)
        assert result["decision"] == "REJECT"
        assert result["iab_category"] == "IAB_VIOLENCE"

    def test_hate_speech_rejected(self):
        text = "These [ethnic slur] are ruining our country. #deportthem"
        result = smart_agent(text)
        assert result["decision"] == "REJECT"

    def test_illegal_drugs_rejected(self):
        text = "Buy cheap Xanax no prescription needed!! DM for prices."
        result = smart_agent(text)
        assert result["decision"] == "REJECT"
        assert result["iab_category"] == "IAB_ILLEGAL"

    def test_piracy_rejected(self):
        text = "Pirated the new Marvel movie, quality is great. DM for the link"
        result = smart_agent(text)
        assert result["decision"] == "REJECT"

    def test_drugs_personal_escalated(self):
        text = "Microdosing psilocybin has genuinely helped my anxiety"
        result = smart_agent(text)
        assert result["decision"] == "ESCALATE"

    def test_gambling_escalated(self):
        text = "Vegas baby!! Hit the jackpot at the Bellagio 🎰💰 #gambling"
        result = smart_agent(text)
        assert result["decision"] == "ESCALATE"

    def test_satire_approved(self):
        text = "Satire: Breaking: Local man discovers that working hard leads to success. Scientists baffled"
        result = smart_agent(text)
        assert result["decision"] == "APPROVE"

    def test_advocacy_escalated(self):
        text = "As a nurse, I see patients die from gun violence. We need common sense reform NOW. #guncontrol"
        result = smart_agent(text)
        assert result["decision"] == "ESCALATE"


class TestSmartAgentOutputFormat:
    def test_has_required_fields(self):
        result = smart_agent("Test content")
        required = ["decision", "iab_category", "garm_category", "risk_level",
                     "reasoning", "confidence", "flagged_elements"]
        for field in required:
            assert field in result

    def test_confidence_in_range(self):
        result = smart_agent("Some random text")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_reasoning_length_valid(self):
        result = smart_agent("Some random text")
        assert 10 <= len(result["reasoning"]) <= 500

    def test_decision_values_valid(self):
        for item in CONTENT_ITEMS:
            result = smart_agent(item["content_text"])
            assert result["decision"] in ("APPROVE", "REJECT", "ESCALATE")

    def test_flagged_elements_max_5(self):
        for item in CONTENT_ITEMS:
            result = smart_agent(item["content_text"])
            assert len(result["flagged_elements"]) <= 5


class TestEvaluateAll:
    def test_returns_aggregate(self):
        result = evaluate_all(CONTENT_ITEMS[:5], grade)
        assert "aggregate" in result
        assert "results" in result
        assert result["aggregate"]["total_items"] == 5

    def test_empty_input(self):
        result = evaluate_all([], grade)
        assert result["aggregate"]["total_items"] == 0
        assert len(result["results"]) == 0

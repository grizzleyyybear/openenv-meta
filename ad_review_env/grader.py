"""Deterministic multi-signal reward grader."""

from typing import Any, Dict, List, Tuple

_STEP_EFFICIENCY = {1: 1.0, 2: 0.7, 3: 0.4}

_ADJACENCY = {
    ("ESCALATE", "REJECT"): 0.4,
    ("REJECT", "ESCALATE"): 0.4,
    ("APPROVE", "ESCALATE"): 0.15,
    ("ESCALATE", "APPROVE"): 0.15,
}

_RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

_AGE_ORDER = {"ALL_AGES": 0, "TEEN": 1, "MATURE": 2, "ADULT": 3}

# Age adjacency: off-by-one is partial credit, off-by-two+ is zero
_AGE_DISTANCE_SCORE = {0: 1.0, 1: 0.5, 2: 0.15, 3: 0.0}


def grade(
    action_data: Dict[str, Any],
    gold: Dict[str, Any],
    steps_taken: int = 1,
) -> Tuple[float, Dict[str, float], str]:
    decision = action_data.get("decision", "")
    iab_category = action_data.get("iab_category", "")
    garm_category = action_data.get("garm_category", "")
    risk_level = action_data.get("risk_level", "MEDIUM")
    reasoning = action_data.get("reasoning", "")
    confidence = max(0.0, min(1.0, float(action_data.get("confidence", 0.5))))
    age_rating = action_data.get("age_rating", "TEEN")
    flagged_elements = action_data.get("flagged_elements", [])
    if not isinstance(flagged_elements, list):
        flagged_elements = []

    gold_decision = gold["gold_decision"]
    gold_iab = gold["gold_iab_category"]
    gold_garm = gold["gold_garm_category"]
    gold_risk = gold.get("gold_risk_level", "MEDIUM")
    gold_age = gold.get("gold_age_rating", "TEEN")
    difficulty = gold["difficulty"]
    correct = decision == gold_decision

    # Decision score (weight 0.30): exact match, adjacency, or miss
    if correct:
        decision_score = 1.0
    else:
        decision_score = _ADJACENCY.get((decision, gold_decision), 0.0)

    # Category score (weight 0.20): IAB + GARM + risk proximity
    iab_score = 1.0 if iab_category == gold_iab else 0.0
    garm_score = 1.0 if garm_category == gold_garm else 0.0
    risk_dist = abs(_RISK_ORDER.get(risk_level, 1) - _RISK_ORDER.get(gold_risk, 1))
    risk_score = max(0.0, 1.0 - risk_dist * 0.35)
    category_score = 0.4 * iab_score + 0.4 * garm_score + 0.2 * risk_score

    # Reasoning score (weight 0.18): length + flagged elements + specificity
    reasoning_text = reasoning.strip()
    reasoning_len = len(reasoning_text)
    if reasoning_len >= 120:
        length_score = 1.0
    elif reasoning_len >= 80:
        length_score = 0.75
    elif reasoning_len >= 40:
        length_score = 0.5
    elif reasoning_len >= 10:
        length_score = 0.25
    else:
        length_score = 0.0

    flagging_bonus = 0.0
    if gold_decision in ("REJECT", "ESCALATE") and len(flagged_elements) > 0:
        flagging_bonus = min(0.3, len(flagged_elements) * 0.1)

    specificity_bonus = 0.0
    if reasoning_len >= 10 and any(word in reasoning_text.lower() for word in
            ["profanity", "hate", "misinformation", "violence", "drug", "gambling",
             "adult", "piracy", "terrorism", "scam", "mlm", "extremist", "satire",
             "advocacy", "political", "brand-safe", "unsafe"]):
        specificity_bonus = 0.1

    reasoning_score = min(1.0, length_score + flagging_bonus + specificity_bonus)

    # Age rating score (weight 0.12): exact or adjacent
    age_dist = abs(_AGE_ORDER.get(age_rating, 1) - _AGE_ORDER.get(gold_age, 1))
    age_rating_score = _AGE_DISTANCE_SCORE.get(age_dist, 0.0)

    # Efficiency score (weight 0.10): step-efficiency multiplier
    step_mult = _STEP_EFFICIENCY.get(min(max(steps_taken, 1), 3), 0.4)
    efficiency_score = step_mult

    # Calibration score (weight 0.10): rewards well-calibrated confidence
    if correct:
        calibration_score = confidence
    else:
        calibration_score = 1.0 - confidence
        if confidence > 0.9:
            calibration_score *= 0.5  # extra penalty for overconfident wrong answers

    # Difficulty multiplier
    difficulty_multiplier = {"easy": 1.0, "medium": 1.05, "hard": 1.1}.get(difficulty, 1.0)

    raw_total = (
        0.30 * decision_score
        + 0.20 * category_score
        + 0.18 * reasoning_score
        + 0.12 * age_rating_score
        + 0.10 * efficiency_score
        + 0.10 * calibration_score
    )
    total = min(1.0, max(0.0, raw_total * difficulty_multiplier))

    component_scores = {
        "decision": round(0.30 * decision_score, 4),
        "category": round(0.20 * category_score, 4),
        "reasoning": round(0.18 * reasoning_score, 4),
        "age_rating": round(0.12 * age_rating_score, 4),
        "efficiency": round(0.10 * efficiency_score, 4),
        "calibration": round(0.10 * calibration_score, 4),
        "total": round(total, 4),
    }

    feedback = _build_feedback(
        decision, gold_decision, iab_category, gold_iab,
        garm_category, gold_garm, risk_level, gold_risk,
        age_rating, gold_age, component_scores, difficulty, steps_taken
    )

    return total, component_scores, feedback


def _build_feedback(
    decision: str, gold_decision: str,
    iab: str, gold_iab: str,
    garm: str, gold_garm: str,
    risk: str, gold_risk: str,
    age: str, gold_age: str,
    scores: Dict[str, float],
    difficulty: str,
    steps_taken: int = 1,
) -> str:
    parts: List[str] = []
    if decision == gold_decision:
        parts.append(f"Decision '{decision}' correct.")
    elif (decision, gold_decision) in _ADJACENCY:
        parts.append(f"Decision '{decision}' adjacent to gold '{gold_decision}' (partial credit).")
    else:
        parts.append(f"Decision '{decision}' wrong. Gold: '{gold_decision}'.")

    if iab != gold_iab:
        parts.append(f"IAB '{iab}' incorrect, expected '{gold_iab}'.")
    if garm != gold_garm:
        parts.append(f"GARM '{garm}' incorrect, expected '{gold_garm}'.")
    if risk != gold_risk:
        parts.append(f"Risk '{risk}' vs gold '{gold_risk}'.")
    if age != gold_age:
        parts.append(f"Age rating '{age}' vs gold '{gold_age}'.")

    step_note = f" ({steps_taken} step{'s' if steps_taken != 1 else ''})"
    parts.append(
        f"Scores: decision={scores['decision']:.2f} "
        f"category={scores['category']:.2f} "
        f"reasoning={scores['reasoning']:.2f} "
        f"age_rating={scores['age_rating']:.2f} "
        f"efficiency={scores['efficiency']:.2f} "
        f"calibration={scores['calibration']:.2f} "
        f"| total={scores['total']:.3f} [{difficulty}]{step_note}"
    )
    return " ".join(parts)

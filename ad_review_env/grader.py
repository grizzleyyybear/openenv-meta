"""
Deterministic grader for the Brand-Safe Ad Review environment.

Reward (max 1.0) = 0.4×decision + 0.3×category + 0.2×reasoning + 0.1×efficiency
Step-efficiency: 1 step → 1.0×, 2 steps → 0.7×, 3 steps → 0.4×
"""

from typing import Any, Dict, Tuple

_STEP_EFFICIENCY = {1: 1.0, 2: 0.7, 3: 0.4}


def grade(
    action_data: Dict[str, Any],
    gold: Dict[str, Any],
    steps_taken: int = 1,
) -> Tuple[float, Dict[str, float], str]:
    """Grade an action against gold labels. Returns (total, scores, feedback)."""
    decision = action_data.get("decision", "")
    iab_category = action_data.get("iab_category", "")
    garm_category = action_data.get("garm_category", "")
    reasoning = action_data.get("reasoning", "")
    confidence = float(action_data.get("confidence", 0.5))
    flagged_elements = action_data.get("flagged_elements", [])

    gold_decision = gold["gold_decision"]
    gold_iab = gold["gold_iab_category"]
    gold_garm = gold["gold_garm_category"]
    difficulty = gold["difficulty"]

    # Decision score (weight 0.4)
    if decision == gold_decision:
        decision_score = 1.0
    elif _is_adjacent_decision(decision, gold_decision):
        decision_score = 0.4
    else:
        decision_score = 0.0

    # Category score (weight 0.3): 50% IAB + 50% GARM
    iab_score = 1.0 if iab_category == gold_iab else 0.0
    garm_score = 1.0 if garm_category == gold_garm else 0.0
    category_score = 0.5 * iab_score + 0.5 * garm_score

    # Reasoning score (weight 0.2)
    reasoning_len = len(reasoning.strip())
    if reasoning_len >= 100:
        length_score = 1.0
    elif reasoning_len >= 50:
        length_score = 0.6
    elif reasoning_len >= 10:
        length_score = 0.3
    else:
        length_score = 0.0

    flagging_bonus = 0.0
    if gold_decision in ("REJECT", "ESCALATE") and len(flagged_elements) > 0:
        flagging_bonus = min(0.3, len(flagged_elements) * 0.1)
    reasoning_score = min(1.0, length_score + flagging_bonus)

    # Efficiency score (weight 0.1): calibration × step-efficiency
    correct = decision == gold_decision
    raw_efficiency = confidence if correct else 1.0 - confidence
    step_mult = _STEP_EFFICIENCY.get(min(steps_taken, 3), 0.4)
    efficiency_score = raw_efficiency * step_mult

    # Difficulty multiplier
    difficulty_multiplier = {"easy": 1.0, "medium": 1.05, "hard": 1.1}.get(difficulty, 1.0)

    raw_total = (
        0.4 * decision_score
        + 0.3 * category_score
        + 0.2 * reasoning_score
        + 0.1 * efficiency_score
    )
    total = min(1.0, raw_total * difficulty_multiplier)

    component_scores = {
        "decision": round(0.4 * decision_score, 4),
        "category": round(0.3 * category_score, 4),
        "reasoning": round(0.2 * reasoning_score, 4),
        "efficiency": round(0.1 * efficiency_score, 4),
        "total": round(total, 4),
    }

    feedback = _build_feedback(
        decision, gold_decision, iab_category, gold_iab,
        garm_category, gold_garm, component_scores, difficulty, steps_taken
    )

    return total, component_scores, feedback


def _is_adjacent_decision(predicted: str, gold: str) -> bool:
    return (predicted, gold) in {("ESCALATE", "REJECT"), ("REJECT", "ESCALATE")}


def _build_feedback(
    decision: str, gold_decision: str,
    iab: str, gold_iab: str,
    garm: str, gold_garm: str,
    scores: Dict[str, float],
    difficulty: str,
    steps_taken: int = 1,
) -> str:
    parts = []
    if decision == gold_decision:
        parts.append(f"Decision '{decision}' correct.")
    elif _is_adjacent_decision(decision, gold_decision):
        parts.append(f"Decision '{decision}' adjacent to gold '{gold_decision}' (partial credit).")
    else:
        parts.append(f"Decision '{decision}' wrong. Gold: '{gold_decision}'.")

    if iab != gold_iab:
        parts.append(f"IAB '{iab}' incorrect, expected '{gold_iab}'.")
    if garm != gold_garm:
        parts.append(f"GARM '{garm}' incorrect, expected '{gold_garm}'.")

    step_note = f" ({steps_taken} step{'s' if steps_taken != 1 else ''})"
    parts.append(
        f"Scores: decision={scores['decision']:.2f} "
        f"category={scores['category']:.2f} "
        f"reasoning={scores['reasoning']:.2f} "
        f"efficiency={scores['efficiency']:.2f} "
        f"| total={scores['total']:.3f} [{difficulty}]{step_note}"
    )
    return " ".join(parts)

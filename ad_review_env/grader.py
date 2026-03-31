"""Deterministic multi-signal reward grader with platform-aware scoring."""

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

_AGE_DISTANCE_SCORE = {0: 1.0, 1: 0.5, 2: 0.15, 3: 0.0}

# Platform-specific weight profiles: each platform adjusts component emphasis.
# TikTok/Instagram skew young → age_rating matters more
# LinkedIn is professional → category/reasoning matters more
# Reddit is community-moderated → decision/reasoning matters more
# X/Threads are fast-moving → efficiency matters more
_PLATFORM_WEIGHTS: Dict[str, Dict[str, float]] = {
    "tiktok":    {"decision": 0.26, "category": 0.18, "reasoning": 0.16, "age_rating": 0.18, "efficiency": 0.10, "calibration": 0.12},
    "instagram": {"decision": 0.27, "category": 0.19, "reasoning": 0.17, "age_rating": 0.16, "efficiency": 0.10, "calibration": 0.11},
    "youtube":   {"decision": 0.28, "category": 0.20, "reasoning": 0.18, "age_rating": 0.14, "efficiency": 0.10, "calibration": 0.10},
    "x":         {"decision": 0.30, "category": 0.18, "reasoning": 0.18, "age_rating": 0.10, "efficiency": 0.14, "calibration": 0.10},
    "threads":   {"decision": 0.30, "category": 0.18, "reasoning": 0.18, "age_rating": 0.10, "efficiency": 0.14, "calibration": 0.10},
    "reddit":    {"decision": 0.28, "category": 0.16, "reasoning": 0.22, "age_rating": 0.10, "efficiency": 0.12, "calibration": 0.12},
    "linkedin":  {"decision": 0.26, "category": 0.22, "reasoning": 0.22, "age_rating": 0.10, "efficiency": 0.10, "calibration": 0.10},
    "facebook":  {"decision": 0.28, "category": 0.20, "reasoning": 0.18, "age_rating": 0.14, "efficiency": 0.10, "calibration": 0.10},
}
_DEFAULT_WEIGHTS = {"decision": 0.30, "category": 0.20, "reasoning": 0.18, "age_rating": 0.12, "efficiency": 0.10, "calibration": 0.10}

# Platform-specific adjacency overrides: some platforms are stricter
# On young-skewing platforms, approving harmful content is worse
_PLATFORM_APPROVE_PENALTY: Dict[str, float] = {
    "tiktok": 0.8,     # approving harmful on TikTok is 80% worse
    "instagram": 0.7,
    "youtube": 0.5,
}

# Platform-specific age rating strictness: young-skewing platforms
# penalize age under-rating more heavily
_PLATFORM_AGE_STRICTNESS: Dict[str, float] = {
    "tiktok": 1.5,     # 50% stricter on age under-rating
    "instagram": 1.3,
    "youtube": 1.2,
    "linkedin": 0.8,   # less strict (professional audience)
    "reddit": 0.7,     # community-moderated, more permissive
}


def _get_weights(platform: str) -> Dict[str, float]:
    return _PLATFORM_WEIGHTS.get(platform.lower().strip(), _DEFAULT_WEIGHTS)


def _safe_str(val: Any, default: str = "") -> str:
    if val is None:
        return default
    return str(val).strip()


def _safe_float(val: Any, default: float = 0.5, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, f))


def _safe_int(val: Any, default: int = 1, lo: int = 1, hi: int = 3) -> int:
    try:
        i = int(val)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, i))


def grade(
    action_data: Dict[str, Any],
    gold: Dict[str, Any],
    steps_taken: int = 1,
) -> Tuple[float, Dict[str, float], str]:
    # --- Extract and sanitize action fields ---
    decision = _safe_str(action_data.get("decision"))
    iab_category = _safe_str(action_data.get("iab_category"))
    garm_category = _safe_str(action_data.get("garm_category"))
    risk_level = _safe_str(action_data.get("risk_level"), "MEDIUM")
    reasoning = _safe_str(action_data.get("reasoning"))
    confidence = _safe_float(action_data.get("confidence"), 0.5)
    age_rating = _safe_str(action_data.get("age_rating"), "TEEN")
    flagged_elements = action_data.get("flagged_elements", [])
    if not isinstance(flagged_elements, list):
        flagged_elements = []

    # --- Extract gold labels with safe defaults ---
    gold_decision = _safe_str(gold.get("gold_decision"))
    gold_iab = _safe_str(gold.get("gold_iab_category"))
    gold_garm = _safe_str(gold.get("gold_garm_category"))
    gold_risk = _safe_str(gold.get("gold_risk_level"), "MEDIUM")
    gold_age = _safe_str(gold.get("gold_age_rating"), "TEEN")
    difficulty = _safe_str(gold.get("difficulty"), "easy")
    platform = _safe_str(gold.get("platform"), "")
    steps_taken = _safe_int(steps_taken)

    correct = decision == gold_decision
    weights = _get_weights(platform)

    # === Decision score ===
    if correct:
        decision_score = 1.0
    else:
        decision_score = _ADJACENCY.get((decision, gold_decision), 0.0)
        # Platform penalty: approving harmful content on young platforms is worse
        if decision == "APPROVE" and gold_decision == "REJECT":
            penalty_mult = _PLATFORM_APPROVE_PENALTY.get(platform, 0.0)
            decision_score *= (1.0 - penalty_mult)

    # === Category score: IAB + GARM + risk proximity ===
    iab_score = 1.0 if iab_category == gold_iab else 0.0
    garm_score = 1.0 if garm_category == gold_garm else 0.0
    risk_dist = abs(_RISK_ORDER.get(risk_level, 1) - _RISK_ORDER.get(gold_risk, 1))
    risk_score = max(0.0, 1.0 - risk_dist * 0.35)
    category_score = 0.4 * iab_score + 0.4 * garm_score + 0.2 * risk_score

    # === Reasoning score: length + flagged elements + specificity ===
    reasoning_len = len(reasoning)
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
    if reasoning_len >= 10 and any(word in reasoning.lower() for word in
            ["profanity", "hate", "misinformation", "violence", "drug", "gambling",
             "adult", "piracy", "terrorism", "scam", "mlm", "extremist", "satire",
             "advocacy", "political", "brand-safe", "unsafe", "age", "minor",
             "young", "explicit", "threat", "illegal"]):
        specificity_bonus = 0.1

    reasoning_score = min(1.0, length_score + flagging_bonus + specificity_bonus)

    # === Age rating score: exact or adjacent, platform-adjusted ===
    age_dist = abs(_AGE_ORDER.get(age_rating, 1) - _AGE_ORDER.get(gold_age, 1))
    age_rating_score = _AGE_DISTANCE_SCORE.get(age_dist, 0.0)
    # Platform strictness: on young platforms, under-rating age is penalized more
    strictness = _PLATFORM_AGE_STRICTNESS.get(platform, 1.0)
    predicted_age_ord = _AGE_ORDER.get(age_rating, 1)
    gold_age_ord = _AGE_ORDER.get(gold_age, 1)
    if predicted_age_ord < gold_age_ord:
        # Under-rated: said ALL_AGES when it should be TEEN/MATURE/ADULT
        penalty = (1.0 - age_rating_score) * (strictness - 1.0)
        age_rating_score = max(0.0, age_rating_score - penalty)

    # === Efficiency score ===
    step_mult = _STEP_EFFICIENCY.get(steps_taken, 0.4)
    efficiency_score = step_mult

    # === Calibration score ===
    if correct:
        calibration_score = confidence
    else:
        calibration_score = 1.0 - confidence
        if confidence > 0.9:
            calibration_score *= 0.5

    # === Difficulty multiplier ===
    difficulty_multiplier = {"easy": 1.0, "medium": 1.05, "hard": 1.1}.get(difficulty, 1.0)

    # === Weighted total ===
    raw_total = (
        weights["decision"] * decision_score
        + weights["category"] * category_score
        + weights["reasoning"] * reasoning_score
        + weights["age_rating"] * age_rating_score
        + weights["efficiency"] * efficiency_score
        + weights["calibration"] * calibration_score
    )
    total = min(1.0, max(0.0, raw_total * difficulty_multiplier))

    component_scores = {
        "decision": round(weights["decision"] * decision_score, 4),
        "category": round(weights["category"] * category_score, 4),
        "reasoning": round(weights["reasoning"] * reasoning_score, 4),
        "age_rating": round(weights["age_rating"] * age_rating_score, 4),
        "efficiency": round(weights["efficiency"] * efficiency_score, 4),
        "calibration": round(weights["calibration"] * calibration_score, 4),
        "total": round(total, 4),
    }

    feedback = _build_feedback(
        decision, gold_decision, iab_category, gold_iab,
        garm_category, gold_garm, risk_level, gold_risk,
        age_rating, gold_age, component_scores, difficulty,
        steps_taken, platform
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
    platform: str = "",
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

    platform_note = f" [{platform}]" if platform else ""
    step_note = f" ({steps_taken} step{'s' if steps_taken != 1 else ''})"
    parts.append(
        f"Scores: decision={scores['decision']:.2f} "
        f"category={scores['category']:.2f} "
        f"reasoning={scores['reasoning']:.2f} "
        f"age_rating={scores['age_rating']:.2f} "
        f"efficiency={scores['efficiency']:.2f} "
        f"calibration={scores['calibration']:.02f} "
        f"| total={scores['total']:.3f} [{difficulty}]{platform_note}{step_note}"
    )
    return " ".join(parts)

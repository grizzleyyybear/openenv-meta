"""
Baseline keyword agent — run across all 30 items as a benchmark floor.

Usage:
    python baseline.py
"""

import argparse
import statistics
from typing import Dict, List

from data import CONTENT_ITEMS
from grader import grade


def run_baseline_agent(text: str) -> Dict:
    """Keyword-based heuristic agent."""
    text_lower = text.lower()

    critical_kw = ["kill", "murder", "terrorist", "bomb", "rape", "slur"]
    reject_kw = [
        "f***", "fuck", "shit", "bitch", "hate", "illegal", "pirat", "xanax",
        "no prescription", "18+", "adult content", "explicit", "graphic violence",
        "knockouts", "brutal", "proud boys", "cancer cure", "doctors don't want"
    ]
    escalate_kw = [
        "psilocybin", "microdosing", "crypto", "gambling", "jackpot",
        "leaked", "government lying", "wake up sheeple", "gun control", "abortion"
    ]

    flagged = [kw for kw in critical_kw + reject_kw + escalate_kw if kw in text_lower]

    if any(kw in text_lower for kw in critical_kw):
        decision, iab, garm, risk, conf = "REJECT", "IAB_HATE_SPEECH", "GARM_HATE_SPEECH", "CRITICAL", 0.9
    elif any(kw in text_lower for kw in reject_kw):
        decision, iab, garm, risk, conf = "REJECT", "IAB_PROFANITY", "GARM_OBSCENITY_PROFANITY", "HIGH", 0.75
    elif any(kw in text_lower for kw in escalate_kw):
        decision, iab, garm, risk, conf = "ESCALATE", "IAB_CONTROVERSIAL", "GARM_SAFE", "MEDIUM", 0.6
    else:
        decision, iab, garm, risk, conf = "APPROVE", "IAB_SAFE", "GARM_SAFE", "LOW", 0.8

    return {
        "decision": decision,
        "iab_category": iab,
        "garm_category": garm,
        "risk_level": risk,
        "reasoning": f"Keyword scan: {'flagged ' + str(flagged[:3]) if flagged else 'no red-flag keywords'}.",
        "confidence": conf,
        "flagged_elements": flagged[:5],
    }


def main():
    parser = argparse.ArgumentParser(description="Run baseline agent evaluation")
    parser.add_argument("--url", default=None, help="Server URL (if None, runs locally)")
    args = parser.parse_args()

    scores_by_difficulty: Dict[str, List[float]] = {"easy": [], "medium": [], "hard": []}
    all_scores: List[float] = []

    print(f"\n{'='*60}")
    print("Brand-Safe Ad Review — Baseline Agent Evaluation")
    print(f"{'='*60}\n")

    for item in CONTENT_ITEMS:
        action = run_baseline_agent(item["content_text"])
        total, scores, feedback = grade(action, item)

        all_scores.append(total)
        scores_by_difficulty[item["difficulty"]].append(total)

        status = "✓" if action["decision"] == item["gold_decision"] else "✗"
        print(f"[{status}] {item['content_id']:12s} | {item['difficulty']:6s} | "
              f"pred={action['decision']:8s} gold={item['gold_decision']:8s} | "
              f"score={total:.3f}")

    print(f"\n{'='*60}")
    print(f"Overall mean score:  {statistics.mean(all_scores):.3f}")
    print(f"Overall median:      {statistics.median(all_scores):.3f}")
    for diff, scores in scores_by_difficulty.items():
        print(f"  {diff:6s} mean:     {statistics.mean(scores):.3f}")
    print(f"{'='*60}\n")
    print("A well-tuned LLM agent should score > 0.70 overall.")


if __name__ == "__main__":
    main()

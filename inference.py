"""
Inference Script — Brand-Safe Ad Review Environment
====================================================
Required env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN
Must be named inference.py at project root.
Uses OpenAI Client for all LLM calls.
"""

import json
import os
import re
import statistics
import textwrap
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# --- Configuration ---

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
TEMPERATURE = 0.0
MAX_TOKENS = 2048

# --- Valid values (must match ad_review_env/models.py) ---

VALID_DECISIONS = {"APPROVE", "REJECT", "ESCALATE"}
VALID_IAB = {
    "IAB_SAFE", "IAB_ADULT", "IAB_VIOLENCE", "IAB_HATE_SPEECH", "IAB_ILLEGAL",
    "IAB_MISINFORMATION", "IAB_PROFANITY", "IAB_DRUGS", "IAB_GAMBLING",
    "IAB_CONTROVERSIAL",
}
VALID_GARM = {
    "GARM_SAFE", "GARM_ADULT_EXPLICIT", "GARM_ARMS_AMMUNITION",
    "GARM_CRIME_HARMFUL", "GARM_DEATH_INJURY", "GARM_HATE_SPEECH",
    "GARM_OBSCENITY_PROFANITY", "GARM_ONLINE_PIRACY", "GARM_SPAM_HARMFUL",
    "GARM_TERRORISM",
}
VALID_RISK = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

# --- LLM system prompt ---

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert content moderator reviewing user-generated content (UGC)
for brand-safe ad placement using IAB Content Taxonomy 3.0 and GARM Brand
Safety Floor standards.

Respond with ONLY a JSON object — no other text, no explanations, no markdown:
{"decision":"...","iab_category":"...","garm_category":"...","risk_level":"...","reasoning":"...","confidence":0.0,"flagged_elements":["..."]}

DECISION RULES (follow strictly):

REJECT — content is clearly harmful or unsafe for ANY brand:
  - Explicit profanity, slurs, violent threats (even hyperbolic like "I want to kill X")
  - Hate speech, discrimination, extremist group promotion
  - Illegal activity: drug sales, piracy, fraud
  - Health misinformation: anti-vax, fake cancer cures, dangerous weight loss claims
  - Political attacks, conspiracy theories, inflammatory political content
  - Graphic violence compilations, explicit adult content

ESCALATE — content is ambiguous, dual-use, or brand-dependent:
  - Legitimate advocacy on divisive topics (gun control, reproductive rights) by credible sources
  - Personal drug experiences framed as mental health discussion (NOT promotion)
  - Gambling content in personal/travel context (not promotion)
  - Financial risk content targeting young audiences (crypto, day trading)
  - Leaked documents / whistleblower content (could be real or fabricated)
  - Content where context would change the decision

APPROVE — content is clearly brand-safe:
  - Lifestyle, fitness, food, pets, travel, education, DIY
  - Clearly labeled satire/parody with no harmful content
  - Historical remembrance, charitable/community content
  - Gaming reviews (fictional violence is fine)
  - Personal growth narratives referencing past mistakes
  - Wine/alcohol in aspirational travel/dining context
  - Evidence-based health content (intermittent fasting, nutrition)

IAB categories: IAB_SAFE, IAB_ADULT, IAB_VIOLENCE, IAB_HATE_SPEECH,
  IAB_ILLEGAL, IAB_MISINFORMATION, IAB_PROFANITY, IAB_DRUGS,
  IAB_GAMBLING, IAB_CONTROVERSIAL

GARM categories: GARM_SAFE, GARM_ADULT_EXPLICIT, GARM_ARMS_AMMUNITION,
  GARM_CRIME_HARMFUL, GARM_DEATH_INJURY, GARM_HATE_SPEECH,
  GARM_OBSCENITY_PROFANITY, GARM_ONLINE_PIRACY, GARM_SPAM_HARMFUL,
  GARM_TERRORISM

EXAMPLES:

Content: "I want to f***ing kill my sister she ate my last slice of pizza"
→ {"decision":"REJECT","iab_category":"IAB_PROFANITY","garm_category":"GARM_OBSCENITY_PROFANITY","risk_level":"HIGH","reasoning":"Explicit profanity with violent language directed at a family member, even if hyperbolic.","confidence":0.92,"flagged_elements":["f***ing","kill"]}

Content: "As a nurse, I see patients die from gun violence every week. We need reform NOW."
→ {"decision":"ESCALATE","iab_category":"IAB_CONTROVERSIAL","garm_category":"GARM_DEATH_INJURY","risk_level":"MEDIUM","reasoning":"Legitimate professional advocacy on politically divisive topic. Credible but brand-sensitive.","confidence":0.80,"flagged_elements":["gun violence","die"]}

Content: "Microdosing psilocybin has helped my anxiety more than any prescription."
→ {"decision":"ESCALATE","iab_category":"IAB_DRUGS","garm_category":"GARM_CRIME_HARMFUL","risk_level":"MEDIUM","reasoning":"Personal mental health narrative involving controlled substance. Not promotion, but substance discussion requires human review.","confidence":0.75,"flagged_elements":["psilocybin","microdosing"]}

Content: "This politician is destroying democracy. Can't believe people vote for this clown."
→ {"decision":"REJECT","iab_category":"IAB_CONTROVERSIAL","garm_category":"GARM_SPAM_HARMFUL","risk_level":"MEDIUM","reasoning":"Political attack content with inflammatory language. Brand risk regardless of political target.","confidence":0.85,"flagged_elements":["destroying democracy","clown"]}

Content: "POV: explaining to parents why you dropped out to trade crypto full time"
→ {"decision":"ESCALATE","iab_category":"IAB_CONTROVERSIAL","garm_category":"GARM_SAFE","risk_level":"MEDIUM","reasoning":"Financial risk content targeting young audience. Normalizes high-risk financial behavior for impressionable viewers.","confidence":0.78,"flagged_elements":["dropped out","crypto"]}

Respond with ONLY the JSON object. No other text.""")

# --- Fallback action ---

FALLBACK_ACTION: Dict[str, Any] = {
    "decision": "ESCALATE",
    "iab_category": "IAB_CONTROVERSIAL",
    "garm_category": "GARM_SAFE",
    "risk_level": "MEDIUM",
    "reasoning": "Unable to parse LLM response. Escalating for human review as a safety precaution.",
    "confidence": 0.3,
    "flagged_elements": [],
}


# --- Helpers ---

def extract_json(text: str) -> Optional[Dict]:
    """Extract JSON from LLM output, handling <think> tags and code blocks."""
    # Strip reasoning model thinking tokens (<think>...</think>)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip any other XML-like tags that reasoning models emit
    text = re.sub(r"<\|.*?\|>", "", text)
    text = text.strip()

    # Try markdown code block first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find the last JSON object (models sometimes emit multiple)
    json_matches = list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL))
    for m in reversed(json_matches):
        try:
            obj = json.loads(m.group(0))
            if "decision" in obj:
                return obj
        except json.JSONDecodeError:
            continue

    # Try the full text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def validate_action(raw: Dict) -> Dict[str, Any]:
    """Validate and sanitize an action dict from LLM output."""
    action: Dict[str, Any] = {}

    decision = str(raw.get("decision", "ESCALATE")).upper().strip()
    action["decision"] = decision if decision in VALID_DECISIONS else "ESCALATE"

    iab = str(raw.get("iab_category", "IAB_CONTROVERSIAL")).upper().strip()
    action["iab_category"] = iab if iab in VALID_IAB else "IAB_CONTROVERSIAL"

    garm = str(raw.get("garm_category", "GARM_SAFE")).upper().strip()
    action["garm_category"] = garm if garm in VALID_GARM else "GARM_SAFE"

    risk = str(raw.get("risk_level", "MEDIUM")).upper().strip()
    action["risk_level"] = risk if risk in VALID_RISK else "MEDIUM"

    reasoning = str(raw.get("reasoning", ""))
    if len(reasoning) < 10:
        reasoning = "Content reviewed by LLM agent. Decision based on contextual analysis of the content."
    action["reasoning"] = reasoning[:500]

    try:
        conf = float(raw.get("confidence", 0.5))
        action["confidence"] = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        action["confidence"] = 0.5

    flagged = raw.get("flagged_elements", [])
    if isinstance(flagged, list):
        action["flagged_elements"] = [str(e) for e in flagged[:5]]
    else:
        action["flagged_elements"] = []

    return action


def call_llm(
    client: OpenAI, content_text: str, content_type: str, platform: str,
    additional_context: Optional[str] = None, retries: int = 2,
) -> Dict[str, Any]:
    """Use the LLM to review a UGC content item."""
    user_msg = (
        f"Platform: {platform}\n"
        f"Content type: {content_type}\n"
        f"\nContent:\n{content_text}"
    )

    if additional_context:
        user_msg += f"\n\nAdditional moderator context:\n{additional_context}"

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw_text = response.choices[0].message.content or ""
            raw_text = raw_text.strip()
            parsed = extract_json(raw_text)

            if parsed is not None:
                return validate_action(parsed)

            if attempt < retries - 1:
                print(f"    ⚠ Parse failed (attempt {attempt+1}), retrying...")
                continue

            print("    ⚠ Failed to parse LLM JSON after retries, using fallback")
            return FALLBACK_ACTION.copy()

        except Exception as e:
            if attempt < retries - 1:
                print(f"    ⚠ LLM error (attempt {attempt+1}): {e}, retrying...")
                continue
            print(f"    ⚠ LLM call failed after retries: {e}, using fallback")
            return FALLBACK_ACTION.copy()

    return FALLBACK_ACTION.copy()


def should_request_context(action: Dict[str, Any], difficulty: str) -> bool:
    """Request context for hard/medium items when the LLM is uncertain."""
    if difficulty == "easy":
        return False
    if action.get("confidence", 1.0) < 0.7:
        return True
    if difficulty == "hard" and action.get("decision") == "ESCALATE":
        return True
    return False


# --- Multi-step episode execution ---

def run_episode_via_api(client: OpenAI, env_url: str, task: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single multi-step episode against the environment."""
    # Reset to start a new episode (with seed for reproducibility)
    reset_resp = requests.post(f"{env_url}/reset", timeout=10)
    reset_resp.raise_for_status()
    obs = reset_resp.json().get("observation", reset_resp.json())

    content_text = obs.get("content_text", task["content_text"])
    content_type = obs.get("content_type", task["content_type"])
    platform = obs.get("platform", task["platform"])
    difficulty = task["difficulty"]
    steps_taken = 0
    additional_context = None

    # First LLM call — initial assessment
    action = call_llm(client, content_text, content_type, platform)
    steps_taken = 1

    # Multi-step: request context for uncertain/hard items
    if should_request_context(action, difficulty):
        ctx_payload = {"action": {
            "action_type": "REQUEST_CONTEXT",
            "reasoning": "Requesting author history and community signals for better assessment.",
        }}
        ctx_resp = requests.post(
            f"{env_url}/step", json=ctx_payload, timeout=15,
        )
        if ctx_resp.ok:
            ctx_obs = ctx_resp.json().get("observation", ctx_resp.json())
            additional_context = ctx_obs.get("additional_context")
            if not ctx_obs.get("done", False) and additional_context:
                # Re-assess with context
                action = call_llm(
                    client, content_text, content_type, platform,
                    additional_context=additional_context,
                )
                steps_taken = 2

    return {"action": action, "steps_taken": steps_taken}


# --- Evaluation ---

def run_evaluation(client: OpenAI, env_url: str) -> Dict[str, Any]:
    """Run inference across all 30 tasks and collect scores."""

    # 1. Verify environment is reachable
    try:
        health = requests.get(f"{env_url}/health", timeout=10)
        health.raise_for_status()
        print(f"✓ Environment healthy at {env_url}\n")
    except Exception as e:
        raise RuntimeError(f"Cannot reach environment at {env_url}: {e}") from e

    # 2. Fetch all 30 tasks (seed=42 for reproducibility)
    resp = requests.get(f"{env_url}/tasks", params={"n": 30, "seed": 42}, timeout=30)
    resp.raise_for_status()
    tasks = resp.json()["tasks"]
    print(f"Fetched {len(tasks)} tasks. Running inference...\n")

    scores_by_difficulty: Dict[str, List[float]] = {"easy": [], "medium": [], "hard": []}
    all_scores: List[float] = []
    results: List[Dict[str, Any]] = []

    for i, task in enumerate(tasks, 1):
        content_id = task["content_id"]
        difficulty = task["difficulty"]

        # LLM reviews the content (multi-step if needed)
        action = call_llm(
            client, task["content_text"], task["content_type"], task["platform"],
        )
        steps_taken = 1

        # Multi-step: request context for uncertain/hard items
        if should_request_context(action, difficulty):
            action = call_llm(
                client, task["content_text"], task["content_type"], task["platform"],
                additional_context="(Enriched context would be provided by the environment in a live episode.)",
            )
            steps_taken = 2

        # Grade via the environment's grader endpoint
        grader_payload = {"content_id": content_id, "steps_taken": steps_taken, **action}
        grade_resp = requests.post(
            f"{env_url}/grader", json=grader_payload, timeout=30,
        )
        grade_resp.raise_for_status()
        result = grade_resp.json()

        score = result["total_reward"]
        all_scores.append(score)
        scores_by_difficulty[difficulty].append(score)

        status = "✓" if result["your_decision"] == result["gold_decision"] else "✗"
        print(
            f"  [{status}] {i:2d}/{len(tasks)}  {content_id:12s} | "
            f"{difficulty:6s} | pred={result['your_decision']:8s} "
            f"gold={result['gold_decision']:8s} | score={score:.3f}"
        )

        results.append({
            "content_id": content_id,
            "difficulty": difficulty,
            "decision": result["your_decision"],
            "gold_decision": result["gold_decision"],
            "score": score,
        })

    return {
        "results": results,
        "all_scores": all_scores,
        "scores_by_difficulty": scores_by_difficulty,
    }


def print_report(eval_data: Dict[str, Any]) -> None:
    """Print the final evaluation report."""
    all_scores = eval_data["all_scores"]
    scores_by_difficulty = eval_data["scores_by_difficulty"]
    results = eval_data["results"]

    correct = sum(1 for r in results if r["decision"] == r["gold_decision"])
    total = len(results)

    print(f"\n{'=' * 60}")
    print("Brand-Safe Ad Review — LLM Inference Results")
    print(f"{'=' * 60}")
    print(f"  Model:             {MODEL_NAME}")
    print(f"  Items evaluated:   {total}")
    print(f"  Correct decisions: {correct}/{total} ({100 * correct / total:.1f}%)")
    print(f"  Overall mean:      {statistics.mean(all_scores):.4f}")
    print(f"  Overall median:    {statistics.median(all_scores):.4f}")
    if len(all_scores) > 1:
        print(f"  Std deviation:     {statistics.stdev(all_scores):.4f}")

    print()
    for diff in ["easy", "medium", "hard"]:
        scores = scores_by_difficulty.get(diff, [])
        if scores:
            diff_correct = sum(
                1 for r in results
                if r["difficulty"] == diff and r["decision"] == r["gold_decision"]
            )
            print(
                f"  {diff:6s}:  mean={statistics.mean(scores):.4f}  "
                f"correct={diff_correct}/{len(scores)}"
            )

    print(f"\n{'=' * 60}")


# --- Entry point ---

def main():
    assert API_KEY, (
        "Missing API key. Set HF_TOKEN or API_KEY environment variable."
    )
    assert MODEL_NAME, (
        "Missing model name. Set MODEL_NAME environment variable."
    )

    print(f"\n{'=' * 60}")
    print("Brand-Safe Ad Review — LLM Inference")
    print(f"{'=' * 60}")
    print(f"  API base:  {API_BASE_URL}")
    print(f"  Model:     {MODEL_NAME}")
    print(f"  Env URL:   {ENV_URL}")
    print(f"{'=' * 60}\n")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    eval_data = run_evaluation(client, ENV_URL)
    print_report(eval_data)

    return eval_data


if __name__ == "__main__":
    main()

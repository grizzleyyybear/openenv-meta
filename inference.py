"""Inference script for Brand-Safe Ad Review. Env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN."""

import json
import os
import re
import statistics
import textwrap
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
TEMPERATURE = 0.0
MAX_TOKENS = 2048

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

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert content moderator for brand-safe ad placement using IAB
Content Taxonomy 3.0 and GARM Brand Safety Floor standards.

Respond with ONLY a JSON object:
{"decision":"...","iab_category":"...","garm_category":"...","risk_level":"...","reasoning":"...","confidence":0.0,"flagged_elements":["..."]}

REJECT — harmful, deceptive, or unsafe for ANY brand:
  - Explicit profanity, slurs, violent threats (even hyperbolic)
  - Hate speech, discrimination, coded extremist language ("great replacement", ethno-nationalism)
  - Illegal activity: drug sales, piracy, fraud
  - Health misinformation: miracle cures, "doctors don't want you to know", anti-vax
  - MLM/pyramid scheme recruitment: "join my team", "DM me", income claims, "#girlboss #financialfreedom"
  - Gambling/casino PROMOTION: affiliate codes, "free spins", casino links, gambling ads
  - Weight loss scams: "one weird trick", unrealistic transformation claims with sales links
  - Political attacks, conspiracy theories
  - Graphic violence, explicit adult content

ESCALATE — ambiguous, divisive, or brand-dependent:
  - Political humor/memes about specific politicians or government (even lighthearted)
  - Geopolitical advocacy: Israel/Palestine, peace activism, war-related content
  - Legitimate advocacy on divisive topics by credible sources
  - Personal drug experiences as mental health discussion (not promotion)
  - Supplement promotion without clear fraud signals (borderline health claims)
  - Financial risk content targeting young audiences (crypto, day trading)
  - Content where decision depends heavily on author credibility or audience context

APPROVE — clearly brand-safe:
  - Lifestyle, fitness, food, pets, travel, education, DIY
  - Labeled satire/parody, historical remembrance, charity
  - Gaming reviews (fictional violence is fine)
  - Recovery/growth narratives, wine in dining context
  - Evidence-based health content from credible sources

IAB: IAB_SAFE, IAB_ADULT, IAB_VIOLENCE, IAB_HATE_SPEECH, IAB_ILLEGAL, IAB_MISINFORMATION, IAB_PROFANITY, IAB_DRUGS, IAB_GAMBLING, IAB_CONTROVERSIAL
GARM: GARM_SAFE, GARM_ADULT_EXPLICIT, GARM_ARMS_AMMUNITION, GARM_CRIME_HARMFUL, GARM_DEATH_INJURY, GARM_HATE_SPEECH, GARM_OBSCENITY_PROFANITY, GARM_ONLINE_PIRACY, GARM_SPAM_HARMFUL, GARM_TERRORISM

EXAMPLES:

Content: "I want to f***ing kill my sister she ate my last slice of pizza"
{"decision":"REJECT","iab_category":"IAB_PROFANITY","garm_category":"GARM_OBSCENITY_PROFANITY","risk_level":"HIGH","reasoning":"Explicit profanity with violent language, even if hyperbolic.","confidence":0.92,"flagged_elements":["f***ing","kill"]}

Content: "Lost 30lbs in 30 days with this ONE weird trick! Link in bio"
{"decision":"REJECT","iab_category":"IAB_MISINFORMATION","garm_category":"GARM_SPAM_HARMFUL","risk_level":"HIGH","reasoning":"Classic weight loss scam with unrealistic claims and sales link. Deceptive health misinformation.","confidence":0.95,"flagged_elements":["ONE weird trick","30lbs in 30 days","Link in bio"]}

Content: "Use code SPIN50 for 50 free spins! I just won $2K in 10 minutes #gambling #ad"
{"decision":"REJECT","iab_category":"IAB_GAMBLING","garm_category":"GARM_CRIME_HARMFUL","risk_level":"HIGH","reasoning":"Gambling promotion with affiliate code and misleading ease-of-winning claims.","confidence":0.93,"flagged_elements":["free spins","affiliate code","$2K in 10 minutes"]}

Content: "Join my team and start earning $10K/month from your phone! DM 'FREEDOM' #girlboss"
{"decision":"REJECT","iab_category":"IAB_MISINFORMATION","garm_category":"GARM_SPAM_HARMFUL","risk_level":"HIGH","reasoning":"MLM/pyramid scheme recruitment with unrealistic income claims.","confidence":0.94,"flagged_elements":["$10K/month","join my team","DM","#girlboss"]}

Content: "We need to secure a future for our people. The great replacement is real. Stay based, kings"
{"decision":"REJECT","iab_category":"IAB_HATE_SPEECH","garm_category":"GARM_HATE_SPEECH","risk_level":"CRITICAL","reasoning":"Coded white-nationalist rhetoric using known extremist phrases.","confidence":0.95,"flagged_elements":["great replacement","secure a future for our people","based"]}

Content: "lol whoever made this meme of the senator falling asleep deserves an award #politics"
{"decision":"ESCALATE","iab_category":"IAB_CONTROVERSIAL","garm_category":"GARM_SAFE","risk_level":"MEDIUM","reasoning":"Political humor about specific politician. Lighthearted but brand-sensitive political content.","confidence":0.80,"flagged_elements":["senator","politics"]}

Content: "As a peace activist, I've sat with families on both sides who've lost children. Peace isn't weakness."
{"decision":"ESCALATE","iab_category":"IAB_CONTROVERSIAL","garm_category":"GARM_SAFE","risk_level":"MEDIUM","reasoning":"Geopolitical peace advocacy on highly divisive conflict. Credible but brand-sensitive.","confidence":0.78,"flagged_elements":["both sides","lost children"]}

Content: "Microdosing psilocybin has helped my anxiety more than any prescription."
{"decision":"ESCALATE","iab_category":"IAB_DRUGS","garm_category":"GARM_CRIME_HARMFUL","risk_level":"MEDIUM","reasoning":"Personal mental health narrative involving controlled substance. Not promotion but requires review.","confidence":0.75,"flagged_elements":["psilocybin","microdosing"]}

Respond with ONLY the JSON object.""")

FALLBACK_ACTION: Dict[str, Any] = {
    "decision": "ESCALATE",
    "iab_category": "IAB_CONTROVERSIAL",
    "garm_category": "GARM_SAFE",
    "risk_level": "MEDIUM",
    "reasoning": "Unable to parse LLM response. Escalating for human review as safety precaution.",
    "confidence": 0.3,
    "flagged_elements": [],
}


def extract_json(text: str) -> Optional[Dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|.*?\|>", "", text)
    text = text.strip()

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    for m in reversed(list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL))):
        try:
            obj = json.loads(m.group(0))
            if "decision" in obj:
                return obj
        except json.JSONDecodeError:
            continue

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def validate_action(raw: Dict) -> Dict[str, Any]:
    decision = str(raw.get("decision", "ESCALATE")).upper().strip()
    iab = str(raw.get("iab_category", "IAB_CONTROVERSIAL")).upper().strip()
    garm = str(raw.get("garm_category", "GARM_SAFE")).upper().strip()
    risk = str(raw.get("risk_level", "MEDIUM")).upper().strip()
    reasoning = str(raw.get("reasoning", ""))
    if len(reasoning) < 10:
        reasoning = "Content reviewed. Decision based on contextual analysis."

    try:
        conf = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5

    flagged = raw.get("flagged_elements", [])

    return {
        "decision": decision if decision in VALID_DECISIONS else "ESCALATE",
        "iab_category": iab if iab in VALID_IAB else "IAB_CONTROVERSIAL",
        "garm_category": garm if garm in VALID_GARM else "GARM_SAFE",
        "risk_level": risk if risk in VALID_RISK else "MEDIUM",
        "reasoning": reasoning[:500],
        "confidence": conf,
        "flagged_elements": [str(e) for e in flagged[:5]] if isinstance(flagged, list) else [],
    }


def call_llm(
    client: OpenAI, content_text: str, content_type: str, platform: str,
    additional_context: Optional[str] = None, retries: int = 2,
) -> Dict[str, Any]:
    user_msg = f"Platform: {platform}\nContent type: {content_type}\n\nContent:\n{content_text}"
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
            parsed = extract_json(response.choices[0].message.content or "")
            if parsed is not None:
                return validate_action(parsed)
            if attempt < retries - 1:
                continue
            return FALLBACK_ACTION.copy()
        except Exception:
            if attempt < retries - 1:
                continue
            return FALLBACK_ACTION.copy()
    return FALLBACK_ACTION.copy()


def should_request_context(difficulty: str) -> bool:
    """Always request context for medium/hard items — improves accuracy significantly."""
    return difficulty != "easy"


def run_episode_via_api(client: OpenAI, env_url: str, task: Dict[str, Any]) -> Dict[str, Any]:
    reset_resp = requests.post(f"{env_url}/reset", timeout=10)
    reset_resp.raise_for_status()
    obs = reset_resp.json().get("observation", reset_resp.json())

    content_text = obs.get("content_text", task["content_text"])
    content_type = obs.get("content_type", task["content_type"])
    platform = obs.get("platform", task["platform"])
    difficulty = task["difficulty"]

    action = call_llm(client, content_text, content_type, platform)
    steps_taken = 1

    if should_request_context(difficulty):
        ctx_payload = {"action": {
            "action_type": "REQUEST_CONTEXT",
            "reasoning": "Requesting author history and community signals for better assessment.",
        }}
        ctx_resp = requests.post(f"{env_url}/step", json=ctx_payload, timeout=15)
        if ctx_resp.ok:
            ctx_obs = ctx_resp.json().get("observation", ctx_resp.json())
            additional_context = ctx_obs.get("additional_context")
            if not ctx_obs.get("done", False) and additional_context:
                action = call_llm(client, content_text, content_type, platform,
                                  additional_context=additional_context)
                steps_taken = 2

    return {"action": action, "steps_taken": steps_taken}


def run_evaluation(client: OpenAI, env_url: str) -> Dict[str, Any]:
    try:
        requests.get(f"{env_url}/health", timeout=10).raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Cannot reach environment at {env_url}: {e}") from e

    resp = requests.get(f"{env_url}/tasks", params={"n": 50, "seed": 42}, timeout=30)
    resp.raise_for_status()
    tasks = resp.json()["tasks"]
    print(f"Fetched {len(tasks)} tasks\n")

    scores_by_difficulty: Dict[str, List[float]] = {"easy": [], "medium": [], "hard": []}
    all_scores: List[float] = []
    results: List[Dict[str, Any]] = []

    for i, task in enumerate(tasks, 1):
        cid, diff = task["content_id"], task["difficulty"]

        action = call_llm(client, task["content_text"], task["content_type"], task["platform"])
        steps_taken = 1

        if should_request_context(diff):
            action = call_llm(
                client, task["content_text"], task["content_type"], task["platform"],
                additional_context="(Context provided by environment in live episode.)",
            )
            steps_taken = 2

        grade_resp = requests.post(
            f"{env_url}/grader",
            json={"content_id": cid, "steps_taken": steps_taken, **action},
            timeout=30,
        )
        grade_resp.raise_for_status()
        result = grade_resp.json()

        score = result["total_reward"]
        all_scores.append(score)
        scores_by_difficulty[diff].append(score)

        ok = "+" if result["your_decision"] == result["gold_decision"] else "-"
        print(f"  [{ok}] {i:2d}/{len(tasks)} {cid:12s} {diff:6s} "
              f"pred={result['your_decision']:8s} gold={result['gold_decision']:8s} "
              f"score={score:.3f}")

        results.append({
            "content_id": cid, "difficulty": diff,
            "decision": result["your_decision"],
            "gold_decision": result["gold_decision"],
            "score": score,
        })

    return {"results": results, "all_scores": all_scores, "scores_by_difficulty": scores_by_difficulty}


def print_report(eval_data: Dict[str, Any]) -> None:
    scores = eval_data["all_scores"]
    by_diff = eval_data["scores_by_difficulty"]
    results = eval_data["results"]
    correct = sum(1 for r in results if r["decision"] == r["gold_decision"])
    total = len(results)

    print(f"\n{'='*60}")
    print(f"  Model:     {MODEL_NAME}")
    print(f"  Correct:   {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"  Mean:      {statistics.mean(scores):.4f}")
    for diff in ["easy", "medium", "hard"]:
        ds = by_diff.get(diff, [])
        if ds:
            dc = sum(1 for r in results if r["difficulty"] == diff and r["decision"] == r["gold_decision"])
            print(f"  {diff:6s}:   mean={statistics.mean(ds):.4f}  correct={dc}/{len(ds)}")
    print(f"{'='*60}")


def main():
    assert API_KEY, "Set HF_TOKEN or API_KEY env var."
    assert MODEL_NAME, "Set MODEL_NAME env var."

    print(f"\nBrand-Safe Ad Review Inference")
    print(f"  API: {API_BASE_URL}  Model: {MODEL_NAME}  Env: {ENV_URL}\n")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    eval_data = run_evaluation(client, ENV_URL)
    print_report(eval_data)
    return eval_data


if __name__ == "__main__":
    main()

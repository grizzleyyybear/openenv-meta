"""
FastAPI application for the Brand-Safe Ad Review environment.

Auto-generated: POST /reset, POST /step, GET /state, GET /schema, GET /health, WS /ws
Hackathon:      GET /tasks, POST /grader, GET /baseline, GET /evaluate, GET /web
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: uv sync") from e

try:
    from ..models import AdReviewAction, AdReviewObservation
    from ..data import CONTENT_ITEMS, CONTENT_INDEX
    from ..grader import grade
    from ..agent import smart_agent, evaluate_agent
    from .environment import AdReviewEnvironment
except ImportError:
    from models import AdReviewAction, AdReviewObservation
    from data import CONTENT_ITEMS, CONTENT_INDEX
    from grader import grade
    from agent import smart_agent, evaluate_agent
    from server.environment import AdReviewEnvironment


app = create_app(
    AdReviewEnvironment, AdReviewAction, AdReviewObservation,
    env_name="ad_review_env", max_concurrent_envs=4,
)


# --- Helpers ---

def _filter_by_difficulty(pool: List[Dict[str, Any]], difficulty: Optional[str]) -> List[Dict[str, Any]]:
    if not difficulty:
        return pool
    if difficulty not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail="difficulty must be easy, medium, or hard")
    return [item for item in pool if item["difficulty"] == difficulty]


def _lookup_content(content_id: str) -> Dict[str, Any]:
    item = CONTENT_INDEX.get(content_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"content_id '{content_id}' not found")
    return item


def _strip_gold_labels(item: Dict[str, Any]) -> Dict[str, Any]:
    return {k: item[k] for k in ("content_id", "content_text", "content_type", "platform", "difficulty")}


# --- Baseline agent ---

_CRITICAL_KW = ["kill", "murder", "terrorist", "bomb", "rape", "child porn", "cp ", "slur"]
_REJECT_KW = [
    "f***", "fuck", "shit", "bitch", "hate", "illegal", "pirat", "xanax",
    "no prescription", "18+", "adult content", "explicit", "graphic violence",
    "knockouts", "brutal", "proud boys", "cancer cure", "doctors don't want",
]
_ESCALATE_KW = [
    "psilocybin", "microdosing", "crypto", "gambling", "jackpot", "controversial",
    "leaked", "government lying", "wake up sheeple", "gun control", "abortion",
]


def baseline_agent(text: str, content_type: str = "", platform: str = "") -> Dict[str, Any]:
    """Simple keyword-based baseline agent."""
    text_lower = text.lower()
    flagged = [kw for kw in _CRITICAL_KW + _REJECT_KW + _ESCALATE_KW if kw in text_lower]

    if any(kw in text_lower for kw in _CRITICAL_KW):
        decision, iab, garm, risk, confidence = (
            "REJECT", "IAB_HATE_SPEECH", "GARM_HATE_SPEECH", "CRITICAL", 0.9
        )
    elif any(kw in text_lower for kw in _REJECT_KW):
        decision, iab, garm, risk, confidence = (
            "REJECT", "IAB_PROFANITY", "GARM_OBSCENITY_PROFANITY", "HIGH", 0.75
        )
    elif any(kw in text_lower for kw in _ESCALATE_KW):
        decision, iab, garm, risk, confidence = (
            "ESCALATE", "IAB_CONTROVERSIAL", "GARM_SAFE", "MEDIUM", 0.6
        )
    else:
        decision, iab, garm, risk, confidence = (
            "APPROVE", "IAB_SAFE", "GARM_SAFE", "LOW", 0.8
        )

    reasoning = (
        f"Keyword scan: {'flagged ' + str(flagged[:3]) if flagged else 'no red-flag keywords detected'}. "
        f"Decision based on heuristic rules."
    )

    return {
        "decision": decision, "iab_category": iab, "garm_category": garm,
        "risk_level": risk, "reasoning": reasoning, "confidence": confidence,
        "flagged_elements": flagged[:5],
    }


# --- GET /tasks ---

@app.get("/tasks", tags=["Hackathon"])
def get_tasks(n: int = 5, difficulty: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    """Return a sample of UGC content items for agent evaluation."""
    if not 1 <= n <= 30:
        raise HTTPException(status_code=400, detail="n must be between 1 and 30")

    pool = _filter_by_difficulty(CONTENT_ITEMS, difficulty)
    rng = random.Random(seed)
    sample = rng.sample(pool, min(n, len(pool)))

    return {
        "tasks": [_strip_gold_labels(item) for item in sample],
        "count": len(sample),
        "total_available": len(pool),
    }


# --- POST /grader ---

class GraderRequest(BaseModel):
    content_id: str = Field(..., description="Content ID from /tasks")
    decision: str = Field(..., description="APPROVE, REJECT, or ESCALATE")
    iab_category: str = Field(..., description="IAB Content Taxonomy category")
    garm_category: str = Field(..., description="GARM Brand Safety Floor category")
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH, or CRITICAL")
    reasoning: str = Field(..., description="Explanation of the decision")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0,1]")
    flagged_elements: List[str] = Field(default_factory=list)
    steps_taken: int = Field(default=1, ge=1, le=3)


@app.post("/grader", tags=["Hackathon"])
def grader_endpoint(request: GraderRequest) -> Dict[str, Any]:
    """Grade a review decision against gold labels."""
    gold = _lookup_content(request.content_id)

    action_data = request.model_dump(exclude={"content_id", "steps_taken"})
    total_reward, scores, feedback = grade(action_data, gold, steps_taken=request.steps_taken)

    return {
        "content_id": request.content_id,
        "difficulty": gold["difficulty"],
        "your_decision": request.decision,
        "gold_decision": gold["gold_decision"],
        "scores": scores,
        "total_reward": total_reward,
        "feedback": feedback,
        "steps_taken": request.steps_taken,
    }


# --- GET /baseline ---

@app.get("/baseline", tags=["Hackathon"])
def baseline_demo(content_id: Optional[str] = None, seed: Optional[int] = 42) -> Dict[str, Any]:
    """Run the keyword-based baseline agent on a content item."""
    if content_id:
        item = _lookup_content(content_id)
    else:
        item = random.Random(seed).choice(CONTENT_ITEMS)

    action = baseline_agent(item["content_text"])
    total_reward, scores, feedback = grade(action, item, steps_taken=1)

    return {
        "content_id": item["content_id"],
        "content_text": item["content_text"],
        "difficulty": item["difficulty"],
        "baseline_action": action,
        "gold_decision": item["gold_decision"],
        "scores": scores,
        "total_reward": total_reward,
        "feedback": feedback,
        "note": "Keyword-based baseline. A well-tuned LLM agent should significantly outperform this.",
    }


# --- GET /evaluate ---

_AGENTS = {"smart": smart_agent, "baseline": baseline_agent}


@app.get("/evaluate", tags=["Hackathon"])
def evaluate_endpoint(
    agent: str = "smart",
    difficulty: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a batch evaluation of an agent across all content items."""
    if agent not in _AGENTS:
        raise HTTPException(
            status_code=400,
            detail=f"agent must be one of {list(_AGENTS.keys())}",
        )

    pool = _filter_by_difficulty(CONTENT_ITEMS, difficulty)
    result = evaluate_agent(pool, grade, agent_fn=_AGENTS[agent])
    result["agent"] = agent
    return result


# --- GET /web ---

_DASHBOARD_PATH = Path(__file__).parent / "dashboard.html"


@app.get("/web", tags=["Dashboard"], response_class=HTMLResponse)
def web_dashboard():
    """Interactive web dashboard for the Brand-Safe Ad Review environment."""
    return _DASHBOARD_PATH.read_text(encoding="utf-8")


# --- Entry point ---

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)

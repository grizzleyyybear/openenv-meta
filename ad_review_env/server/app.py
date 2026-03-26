# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Brand-Safe Ad Review Environment.

Auto-generated endpoints (via create_app):
    POST /reset       — start a new review episode
    POST /step        — submit a review decision
    GET  /state       — current episode state
    GET  /schema      — action/observation JSON schemas
    GET  /health      — health check
    WS   /ws          — WebSocket for persistent sessions

Hackathon endpoints (bolted on):
    GET  /tasks       — sample tasks for evaluation
    POST /grader      — standalone grader endpoint
    GET  /baseline    — baseline agent demonstration
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import AdReviewAction, AdReviewObservation
    from ..data import CONTENT_ITEMS, CONTENT_INDEX
    from ..grader import grade
    from ..agent import smart_agent, evaluate_all
    from .environment import AdReviewEnvironment
except ImportError:
    from models import AdReviewAction, AdReviewObservation
    from data import CONTENT_ITEMS, CONTENT_INDEX
    from grader import grade
    from agent import smart_agent, evaluate_all
    from server.environment import AdReviewEnvironment

import random
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core app — auto-generates /reset /step /state /schema /health /ws /docs
# ---------------------------------------------------------------------------
app = create_app(
    AdReviewEnvironment,
    AdReviewAction,
    AdReviewObservation,
    env_name="ad_review_env",
    max_concurrent_envs=4,
)


# ---------------------------------------------------------------------------
# Hackathon endpoint 1: GET /tasks
# Returns a sample of content items for evaluation
# ---------------------------------------------------------------------------
@app.get("/tasks", tags=["Hackathon"])
def get_tasks(
    n: int = 5,
    difficulty: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Return a sample of UGC content items for agent evaluation.

    Args:
        n: Number of tasks to return (1-30, default 5)
        difficulty: Filter by 'easy', 'medium', or 'hard' (optional)
        seed: Random seed for reproducible sampling (optional)
    """
    if not 1 <= n <= 30:
        raise HTTPException(status_code=400, detail="n must be between 1 and 30")

    pool = CONTENT_ITEMS
    if difficulty:
        if difficulty not in ("easy", "medium", "hard"):
            raise HTTPException(status_code=400, detail="difficulty must be easy, medium, or hard")
        pool = [item for item in pool if item["difficulty"] == difficulty]

    rng = random.Random(seed)
    sample = rng.sample(pool, min(n, len(pool)))

    # Strip gold labels from task presentation
    tasks = [
        {
            "content_id": item["content_id"],
            "content_text": item["content_text"],
            "content_type": item["content_type"],
            "platform": item["platform"],
            "difficulty": item["difficulty"],
        }
        for item in sample
    ]

    return {
        "tasks": tasks,
        "count": len(tasks),
        "total_available": len(pool),
    }


# ---------------------------------------------------------------------------
# Hackathon endpoint 2: POST /grader
# Standalone grader — accepts action + content_id, returns scored result
# ---------------------------------------------------------------------------
class GraderRequest(BaseModel):
    content_id: str = Field(..., description="Content ID from /tasks")
    decision: str = Field(..., description="APPROVE, REJECT, or ESCALATE")
    iab_category: str = Field(..., description="IAB Content Taxonomy category")
    garm_category: str = Field(..., description="GARM Brand Safety Floor category")
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH, or CRITICAL")
    reasoning: str = Field(..., description="Explanation of the decision")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0,1]")
    flagged_elements: List[str] = Field(default_factory=list)


@app.post("/grader", tags=["Hackathon"])
def grader_endpoint(request: GraderRequest) -> Dict[str, Any]:
    """
    Grade a review decision against gold labels.

    Submit your decision for a content_id and receive a detailed score breakdown.
    """
    gold = CONTENT_INDEX.get(request.content_id)
    if gold is None:
        raise HTTPException(
            status_code=404,
            detail=f"content_id '{request.content_id}' not found. Use GET /tasks to get valid IDs.",
        )

    action_data = {
        "decision": request.decision,
        "iab_category": request.iab_category,
        "garm_category": request.garm_category,
        "risk_level": request.risk_level,
        "reasoning": request.reasoning,
        "confidence": request.confidence,
        "flagged_elements": request.flagged_elements,
    }

    total_reward, scores, feedback = grade(action_data, gold)

    return {
        "content_id": request.content_id,
        "difficulty": gold["difficulty"],
        "your_decision": request.decision,
        "gold_decision": gold["gold_decision"],
        "scores": scores,
        "total_reward": total_reward,
        "feedback": feedback,
    }


# ---------------------------------------------------------------------------
# Hackathon endpoint 3: GET /baseline
# Demonstrates a simple rule-based baseline agent
# ---------------------------------------------------------------------------
@app.get("/baseline", tags=["Hackathon"])
def baseline_demo(content_id: Optional[str] = None, seed: Optional[int] = 42) -> Dict[str, Any]:
    """
    Run a simple keyword-based baseline agent on a content item.

    Demonstrates the environment API and provides a performance benchmark.
    If content_id is not provided, a random item is selected.
    """
    if content_id:
        item = CONTENT_INDEX.get(content_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"content_id '{content_id}' not found")
    else:
        rng = random.Random(seed)
        item = rng.choice(CONTENT_ITEMS)

    # Simple keyword-based baseline
    action = _baseline_agent(item["content_text"])

    action_data = {
        "decision": action["decision"],
        "iab_category": action["iab_category"],
        "garm_category": action["garm_category"],
        "risk_level": action["risk_level"],
        "reasoning": action["reasoning"],
        "confidence": action["confidence"],
        "flagged_elements": action["flagged_elements"],
    }

    total_reward, scores, feedback = grade(action_data, item)

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


def _baseline_agent(text: str) -> Dict[str, Any]:
    """
    Simple keyword-based baseline agent.
    Scans for red-flag keywords and makes a heuristic decision.
    """
    text_lower = text.lower()

    # Red-flag keyword sets
    critical_keywords = ["kill", "murder", "terrorist", "bomb", "rape", "child porn", "cp ", "slur"]
    reject_keywords = [
        "f***", "fuck", "shit", "bitch", "hate", "illegal", "pirat", "xanax",
        "no prescription", "18+", "adult content", "explicit", "graphic violence",
        "knockouts", "brutal", "proud boys", "cancer cure", "doctors don't want"
    ]
    escalate_keywords = [
        "psilocybin", "microdosing", "crypto", "gambling", "jackpot", "controversial",
        "leaked", "government lying", "wake up sheeple", "gun control", "abortion"
    ]

    flagged = []

    for kw in critical_keywords:
        if kw in text_lower:
            flagged.append(kw)

    for kw in reject_keywords:
        if kw in text_lower:
            flagged.append(kw)

    for kw in escalate_keywords:
        if kw in text_lower:
            flagged.append(kw)

    if any(kw in text_lower for kw in critical_keywords):
        decision = "REJECT"
        iab = "IAB_HATE_SPEECH"
        garm = "GARM_HATE_SPEECH"
        risk = "CRITICAL"
        confidence = 0.9
    elif any(kw in text_lower for kw in reject_keywords):
        decision = "REJECT"
        iab = "IAB_PROFANITY"
        garm = "GARM_OBSCENITY_PROFANITY"
        risk = "HIGH"
        confidence = 0.75
    elif any(kw in text_lower for kw in escalate_keywords):
        decision = "ESCALATE"
        iab = "IAB_CONTROVERSIAL"
        garm = "GARM_SAFE"
        risk = "MEDIUM"
        confidence = 0.6
    else:
        decision = "APPROVE"
        iab = "IAB_SAFE"
        garm = "GARM_SAFE"
        risk = "LOW"
        confidence = 0.8

    reasoning = (
        f"Keyword scan: {'flagged ' + str(flagged[:3]) if flagged else 'no red-flag keywords detected'}. "
        f"Decision based on heuristic rules."
    )

    return {
        "decision": decision,
        "iab_category": iab,
        "garm_category": garm,
        "risk_level": risk,
        "reasoning": reasoning,
        "confidence": confidence,
        "flagged_elements": flagged[:5],
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for uv run or python -m."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


# ---------------------------------------------------------------------------
# Hackathon endpoint 4: GET /evaluate
# Batch evaluation — runs an agent across all 30 items
# ---------------------------------------------------------------------------
@app.get("/evaluate", tags=["Hackathon"])
def evaluate_endpoint(
    agent: str = "smart",
    difficulty: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a batch evaluation of an agent across all content items.

    Args:
        agent: Which agent to evaluate — 'smart' (default) or 'baseline'
        difficulty: Filter by 'easy', 'medium', or 'hard' (optional)
    """
    if agent not in ("smart", "baseline"):
        raise HTTPException(status_code=400, detail="agent must be 'smart' or 'baseline'")

    pool = CONTENT_ITEMS
    if difficulty:
        if difficulty not in ("easy", "medium", "hard"):
            raise HTTPException(status_code=400, detail="difficulty must be easy, medium, or hard")
        pool = [item for item in pool if item["difficulty"] == difficulty]

    if agent == "smart":
        result = evaluate_all(pool, grade)
    else:
        # Run baseline agent
        from ad_review_env.server.app import _baseline_agent
        results_list = []
        all_scores = []
        scores_by_difficulty: Dict[str, list] = {"easy": [], "medium": [], "hard": []}
        correct = 0
        for item in pool:
            action = _baseline_agent(item["content_text"])
            total, scores, feedback = grade(action, item)
            all_scores.append(total)
            scores_by_difficulty[item["difficulty"]].append(total)
            if action["decision"] == item["gold_decision"]:
                correct += 1
            results_list.append({
                "content_id": item["content_id"],
                "difficulty": item["difficulty"],
                "predicted_decision": action["decision"],
                "gold_decision": item["gold_decision"],
                "correct": action["decision"] == item["gold_decision"],
                "total_score": round(total, 4),
                "scores": scores,
                "feedback": feedback,
            })
        import statistics
        mean_score = statistics.mean(all_scores) if all_scores else 0.0
        median_score = statistics.median(all_scores) if all_scores else 0.0
        result = {
            "results": results_list,
            "aggregate": {
                "mean_score": round(mean_score, 4),
                "median_score": round(median_score, 4),
                "decision_accuracy": round(correct / len(pool), 4) if pool else 0.0,
                "total_items": len(pool),
                "correct_decisions": correct,
                "by_difficulty": {
                    d: round(statistics.mean(s), 4) if s else 0.0
                    for d, s in scores_by_difficulty.items()
                },
            },
        }

    result["agent"] = agent
    return result


# ---------------------------------------------------------------------------
# Hackathon endpoint 5: GET /web — Interactive dashboard
# ---------------------------------------------------------------------------
from fastapi.responses import HTMLResponse

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Brand-Safe Ad Review — Dashboard</title>
<style>
  :root { --bg: #0f172a; --card: #1e293b; --accent: #3b82f6; --green: #22c55e;
           --red: #ef4444; --yellow: #eab308; --text: #e2e8f0; --muted: #94a3b8; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg);
         color: var(--text); line-height: 1.6; padding: 1rem; }
  .container { max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 1.8rem; margin-bottom: 0.25rem; }
  h1 span { color: var(--accent); }
  .subtitle { color: var(--muted); margin-bottom: 1.5rem; font-size: 0.9rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
  .stat-card { background: var(--card); border-radius: 12px; padding: 1.2rem; text-align: center; }
  .stat-value { font-size: 2rem; font-weight: 700; }
  .stat-label { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  .stat-value.green { color: var(--green); }
  .stat-value.blue { color: var(--accent); }
  .controls { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; align-items: center; }
  select, button { background: var(--card); color: var(--text); border: 1px solid #334155;
                   padding: 0.5rem 1rem; border-radius: 8px; font-size: 0.9rem; cursor: pointer; }
  button { background: var(--accent); border: none; font-weight: 600; }
  button:hover { opacity: 0.9; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .results-table { width: 100%; border-collapse: collapse; background: var(--card); border-radius: 12px; overflow: hidden; }
  .results-table th, .results-table td { padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #334155; }
  .results-table th { background: #334155; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
  .badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
  .badge-approve { background: #166534; color: #bbf7d0; }
  .badge-reject { background: #991b1b; color: #fecaca; }
  .badge-escalate { background: #854d0e; color: #fef08a; }
  .badge-easy { background: #164e63; color: #a5f3fc; }
  .badge-medium { background: #713f12; color: #fef08a; }
  .badge-hard { background: #7f1d1d; color: #fecaca; }
  .score-bar { width: 100px; height: 8px; background: #334155; border-radius: 4px; overflow: hidden; display: inline-block; vertical-align: middle; }
  .score-fill { height: 100%; border-radius: 4px; }
  .status { font-size: 1.1rem; }
  .panel { background: var(--card); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }
  .panel h2 { font-size: 1.1rem; margin-bottom: 1rem; }
  .try-it { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .try-it textarea, .try-it select, .try-it input { width: 100%; background: var(--bg); color: var(--text);
    border: 1px solid #334155; padding: 0.5rem; border-radius: 8px; font-size: 0.85rem; }
  .try-it textarea { grid-column: 1 / -1; min-height: 80px; resize: vertical; }
  .try-it button { grid-column: 1 / -1; }
  .try-result { grid-column: 1 / -1; background: var(--bg); border-radius: 8px; padding: 1rem;
    font-family: monospace; font-size: 0.85rem; white-space: pre-wrap; max-height: 300px; overflow-y: auto; }
  .comparison { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .comparison .stat-card { text-align: left; }
  .loading { text-align: center; padding: 2rem; color: var(--muted); }
  @media (max-width: 768px) { .try-it { grid-template-columns: 1fr; } .comparison { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="container">
  <h1>🛡️ Brand-Safe <span>Ad Review</span></h1>
  <p class="subtitle">UGC Content Moderation Environment — IAB 3.0 + GARM Brand Safety Floor</p>

  <!-- Agent Comparison -->
  <div class="panel">
    <h2>📊 Agent Comparison</h2>
    <div class="controls">
      <button id="btnCompare" onclick="runComparison()">Run Comparison</button>
      <span id="compareStatus" style="color: var(--muted); font-size: 0.85rem;"></span>
    </div>
    <div class="comparison" id="comparisonGrid">
      <div class="stat-card"><div class="stat-label">Smart Agent</div><div class="stat-value green" id="smartScore">—</div><div class="stat-label" id="smartAcc">—</div></div>
      <div class="stat-card"><div class="stat-label">Baseline Agent</div><div class="stat-value blue" id="baselineScore">—</div><div class="stat-label" id="baselineAcc">—</div></div>
    </div>
  </div>

  <!-- Stats -->
  <div class="grid" id="statsGrid">
    <div class="stat-card"><div class="stat-value green" id="meanScore">—</div><div class="stat-label">Mean Score</div></div>
    <div class="stat-card"><div class="stat-value blue" id="accuracy">—</div><div class="stat-label">Decision Accuracy</div></div>
    <div class="stat-card"><div class="stat-value" id="easyScore">—</div><div class="stat-label">Easy</div></div>
    <div class="stat-card"><div class="stat-value" id="medScore">—</div><div class="stat-label">Medium</div></div>
    <div class="stat-card"><div class="stat-value" id="hardScore">—</div><div class="stat-label">Hard</div></div>
  </div>

  <!-- Controls -->
  <div class="controls">
    <select id="agentSelect"><option value="smart">Smart Agent</option><option value="baseline">Baseline Agent</option></select>
    <select id="diffSelect"><option value="">All Difficulties</option><option value="easy">Easy</option><option value="medium">Medium</option><option value="hard">Hard</option></select>
    <button id="btnEval" onclick="runEval()">▶ Run Evaluation</button>
  </div>

  <!-- Results Table -->
  <div style="overflow-x:auto; border-radius:12px;">
    <table class="results-table" id="resultsTable">
      <thead><tr><th></th><th>ID</th><th>Difficulty</th><th>Predicted</th><th>Gold</th><th>Score</th><th>Feedback</th></tr></thead>
      <tbody id="resultsBody"><tr><td colspan="7" class="loading">Click "Run Evaluation" to start</td></tr></tbody>
    </table>
  </div>

  <!-- Try It -->
  <div class="panel" style="margin-top:1.5rem;">
    <h2>🧪 Try It Yourself</h2>
    <div class="try-it">
      <textarea id="tryText" placeholder="Paste UGC content here...">Just made the most amazing chocolate chip cookies! 🍪 Recipe in bio. #baking #homemade #foodie</textarea>
      <select id="tryDecision"><option value="APPROVE">APPROVE</option><option value="REJECT">REJECT</option><option value="ESCALATE">ESCALATE</option></select>
      <select id="tryIAB">
        <option value="IAB_SAFE">IAB_SAFE</option><option value="IAB_ADULT">IAB_ADULT</option>
        <option value="IAB_VIOLENCE">IAB_VIOLENCE</option><option value="IAB_HATE_SPEECH">IAB_HATE_SPEECH</option>
        <option value="IAB_ILLEGAL">IAB_ILLEGAL</option><option value="IAB_MISINFORMATION">IAB_MISINFORMATION</option>
        <option value="IAB_PROFANITY">IAB_PROFANITY</option><option value="IAB_DRUGS">IAB_DRUGS</option>
        <option value="IAB_GAMBLING">IAB_GAMBLING</option><option value="IAB_CONTROVERSIAL">IAB_CONTROVERSIAL</option>
      </select>
      <select id="tryGARM">
        <option value="GARM_SAFE">GARM_SAFE</option><option value="GARM_ADULT_EXPLICIT">GARM_ADULT_EXPLICIT</option>
        <option value="GARM_ARMS_AMMUNITION">GARM_ARMS_AMMUNITION</option><option value="GARM_CRIME_HARMFUL">GARM_CRIME_HARMFUL</option>
        <option value="GARM_DEATH_INJURY">GARM_DEATH_INJURY</option><option value="GARM_HATE_SPEECH">GARM_HATE_SPEECH</option>
        <option value="GARM_OBSCENITY_PROFANITY">GARM_OBSCENITY_PROFANITY</option><option value="GARM_ONLINE_PIRACY">GARM_ONLINE_PIRACY</option>
        <option value="GARM_SPAM_HARMFUL">GARM_SPAM_HARMFUL</option><option value="GARM_TERRORISM">GARM_TERRORISM</option>
      </select>
      <select id="tryRisk"><option value="LOW">LOW</option><option value="MEDIUM">MEDIUM</option><option value="HIGH">HIGH</option><option value="CRITICAL">CRITICAL</option></select>
      <input id="tryConf" type="number" min="0" max="1" step="0.05" value="0.85" placeholder="Confidence [0-1]">
      <button onclick="tryGrade()">Grade My Decision</button>
      <div class="try-result" id="tryResult" style="display:none;"></div>
    </div>
  </div>

  <!-- Sample Tasks -->
  <div class="panel" style="margin-top:1.5rem;">
    <h2>📋 Sample Tasks</h2>
    <div class="controls">
      <select id="taskDiff"><option value="">All</option><option value="easy">Easy</option><option value="medium">Medium</option><option value="hard">Hard</option></select>
      <button onclick="loadTasks()">Load Tasks</button>
    </div>
    <div id="tasksList" style="font-size:0.85rem;"></div>
  </div>
</div>

<script>
const API = window.location.origin;

async function runEval() {
  const agent = document.getElementById('agentSelect').value;
  const diff = document.getElementById('diffSelect').value;
  document.getElementById('btnEval').disabled = true;
  document.getElementById('resultsBody').innerHTML = '<tr><td colspan="7" class="loading">Running evaluation...</td></tr>';
  try {
    const url = new URL(API + '/evaluate');
    url.searchParams.set('agent', agent);
    if (diff) url.searchParams.set('difficulty', diff);
    const res = await fetch(url);
    const data = await res.json();
    const agg = data.aggregate;
    document.getElementById('meanScore').textContent = agg.mean_score.toFixed(4);
    document.getElementById('accuracy').textContent = (agg.decision_accuracy * 100).toFixed(1) + '%';
    document.getElementById('easyScore').textContent = (agg.by_difficulty.easy || 0).toFixed(4);
    document.getElementById('medScore').textContent = (agg.by_difficulty.medium || 0).toFixed(4);
    document.getElementById('hardScore').textContent = (agg.by_difficulty.hard || 0).toFixed(4);
    let html = '';
    for (const r of data.results) {
      const icon = r.correct ? '✅' : '❌';
      const predClass = r.predicted_decision.toLowerCase();
      const goldClass = r.gold_decision.toLowerCase();
      const diffClass = r.difficulty;
      const pct = Math.round(r.total_score * 100);
      const color = pct >= 80 ? '#22c55e' : pct >= 50 ? '#eab308' : '#ef4444';
      html += '<tr>' +
        '<td class="status">' + icon + '</td>' +
        '<td><code>' + r.content_id + '</code></td>' +
        '<td><span class="badge badge-' + diffClass + '">' + r.difficulty + '</span></td>' +
        '<td><span class="badge badge-' + predClass + '">' + r.predicted_decision + '</span></td>' +
        '<td><span class="badge badge-' + goldClass + '">' + r.gold_decision + '</span></td>' +
        '<td><div class="score-bar"><div class="score-fill" style="width:' + pct + '%;background:' + color + '"></div></div> ' + r.total_score.toFixed(3) + '</td>' +
        '<td style="font-size:0.8rem;color:var(--muted);max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="' + escapeHtml(r.feedback) + '">' + escapeHtml(r.feedback.substring(0, 80)) + '</td></tr>';
    }
    document.getElementById('resultsBody').innerHTML = html;
  } catch (e) {
    document.getElementById('resultsBody').innerHTML = '<tr><td colspan="7" class="loading">Error: ' + e.message + '</td></tr>';
  }
  document.getElementById('btnEval').disabled = false;
}

async function runComparison() {
  document.getElementById('btnCompare').disabled = true;
  document.getElementById('compareStatus').textContent = 'Running...';
  try {
    const [smartRes, baseRes] = await Promise.all([
      fetch(API + '/evaluate?agent=smart').then(r => r.json()),
      fetch(API + '/evaluate?agent=baseline').then(r => r.json()),
    ]);
    document.getElementById('smartScore').textContent = smartRes.aggregate.mean_score.toFixed(4);
    document.getElementById('smartAcc').textContent = (smartRes.aggregate.decision_accuracy * 100).toFixed(1) + '% accuracy';
    document.getElementById('baselineScore').textContent = baseRes.aggregate.mean_score.toFixed(4);
    document.getElementById('baselineAcc').textContent = (baseRes.aggregate.decision_accuracy * 100).toFixed(1) + '% accuracy';
    document.getElementById('compareStatus').textContent = 'Done!';
  } catch (e) {
    document.getElementById('compareStatus').textContent = 'Error: ' + e.message;
  }
  document.getElementById('btnCompare').disabled = false;
}

async function tryGrade() {
  const el = id => document.getElementById(id);
  const contentId = 'easy_001'; // Use a known ID for grading
  const body = {
    content_id: contentId,
    decision: el('tryDecision').value,
    iab_category: el('tryIAB').value,
    garm_category: el('tryGARM').value,
    risk_level: el('tryRisk').value,
    reasoning: 'User-submitted decision via web dashboard for content review evaluation.',
    confidence: parseFloat(el('tryConf').value) || 0.5,
    flagged_elements: [],
  };
  try {
    const res = await fetch(API + '/grader', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    const data = await res.json();
    el('tryResult').style.display = 'block';
    el('tryResult').textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    el('tryResult').style.display = 'block';
    el('tryResult').textContent = 'Error: ' + e.message;
  }
}

async function loadTasks() {
  const diff = document.getElementById('taskDiff').value;
  const url = new URL(API + '/tasks');
  url.searchParams.set('n', '10');
  if (diff) url.searchParams.set('difficulty', diff);
  try {
    const res = await fetch(url);
    const data = await res.json();
    let html = '';
    for (const t of data.tasks) {
      html += '<div style="background:var(--bg);padding:0.75rem;border-radius:8px;margin-bottom:0.5rem;">' +
        '<div><span class="badge badge-' + t.difficulty + '">' + t.difficulty + '</span> ' +
        '<code>' + t.content_id + '</code> · ' + t.platform + ' · ' + t.content_type + '</div>' +
        '<div style="margin-top:0.3rem;">' + escapeHtml(t.content_text) + '</div></div>';
    }
    document.getElementById('tasksList').innerHTML = html;
  } catch (e) {
    document.getElementById('tasksList').innerHTML = '<div class="loading">Error: ' + e.message + '</div>';
  }
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
</script>
</body>
</html>"""


@app.get("/web", tags=["Dashboard"], response_class=HTMLResponse)
def web_dashboard():
    """Interactive web dashboard for the Brand-Safe Ad Review environment."""
    return DASHBOARD_HTML


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)

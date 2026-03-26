---
title: Brand-Safe Ad Review Environment
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - content-moderation
  - brand-safety
  - ugc
---

# Brand-Safe Ad Review Environment

An RL environment for training agents to review user-generated content (UGC) for brand-safe ad placement, using **IAB Content Taxonomy 3.0** and **GARM Brand Safety Floor** standards.

## The Task

Each episode presents one UGC content item (post, caption, comment, or bio) from a social platform. The agent must:

1. **Decide**: `APPROVE` (brand-safe), `REJECT` (unsafe), or `ESCALATE` (human review needed)
2. **Classify**: Assign an IAB Content Taxonomy category and a GARM Brand Safety Floor category
3. **Assess**: Assign a risk level (`LOW` / `MEDIUM` / `HIGH` / `CRITICAL`)
4. **Explain**: Provide reasoning and flag specific harmful elements
5. **Calibrate**: Express a confidence score `[0.0, 1.0]`

Episodes are **single-step** — `done=True` after every `step()`.

## Reward Formula (max 1.0)

| Component | Weight | Description |
|-----------|--------|-------------|
| Decision accuracy | 0.4 | Correct APPROVE/REJECT/ESCALATE |
| Category accuracy | 0.3 | Correct IAB (0.15) + GARM (0.15) |
| Reasoning quality | 0.2 | Length + flagged elements |
| Confidence calibration | 0.1 | Penalizes overconfidence on wrong answers |

Hard tasks get a 1.1× difficulty multiplier.

## Quick Start

```python
from ad_review_env import AdReviewAction, AdReviewEnv

with AdReviewEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(result.observation.content_text)

    action = AdReviewAction(
        decision="REJECT",
        iab_category="IAB_PROFANITY",
        garm_category="GARM_OBSCENITY_PROFANITY",
        risk_level="HIGH",
        reasoning="Content contains explicit profanity and violent language directed at a family member.",
        confidence=0.92,
        flagged_elements=["f***", "violent language"],
    )
    result = env.step(action)
    print(f"Score: {result.observation.total_score:.3f}")
    print(result.observation.feedback)
```

## Dataset

30 curated UGC items across 3 difficulty tiers:

- **Easy (10)**: Clear-cut cases — obvious safe content or obvious violations
- **Medium (10)**: Requires contextual judgment — alcohol, political content, gambling
- **Hard (10)**: Subtle cases — satire, advocacy, coded language, dual-use content

## IAB Categories

`IAB_SAFE`, `IAB_ADULT`, `IAB_VIOLENCE`, `IAB_HATE_SPEECH`, `IAB_ILLEGAL`,
`IAB_MISINFORMATION`, `IAB_PROFANITY`, `IAB_DRUGS`, `IAB_GAMBLING`, `IAB_CONTROVERSIAL`

## GARM Categories

`GARM_SAFE`, `GARM_ADULT_EXPLICIT`, `GARM_ARMS_AMMUNITION`, `GARM_CRIME_HARMFUL`,
`GARM_DEATH_INJURY`, `GARM_HATE_SPEECH`, `GARM_OBSCENITY_PROFANITY`,
`GARM_ONLINE_PIRACY`, `GARM_SPAM_HARMFUL`, `GARM_TERRORISM`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new review episode |
| `/step` | POST | Submit a review decision |
| `/state` | GET | Current episode state |
| `/schema` | GET | Action/observation JSON schemas |
| `/health` | GET | Health check |
| `/ws` | WS | WebSocket for persistent sessions |
| `/tasks` | GET | Sample tasks for evaluation |
| `/grader` | POST | Standalone grader endpoint |
| `/baseline` | GET | Baseline agent demonstration |
| `/evaluate` | GET | Batch evaluation (smart or baseline agent) |
| `/web` | GET | Interactive web dashboard |
| `/docs` | GET | Interactive API documentation |

## Running Locally

```bash
cd ad_review_env
uv pip install -e .
uvicorn server.app:app --reload --port 8000
```

## Baseline Performance

The keyword-based baseline agent scores approximately **0.45–0.55** overall.

The smart contextual agent scores **~0.996** (30/30 correct decisions).

A well-tuned LLM agent should target **> 0.75**.

```bash
python baseline.py
```

## Smart Agent

The smart agent (`agent.py`) uses multi-signal contextual analysis instead of flat keyword matching:

- **Per-category pattern libraries** with weighted signals (IAB + GARM categories)
- **Context modifiers** — satire detection, personal narrative, advocacy recognition
- **Calibrated confidence** — adjusts based on signal clarity
- **Rich reasoning** — generates detailed explanations with specific flagged elements

```bash
# Run batch evaluation via API
curl http://localhost:8000/evaluate?agent=smart
curl http://localhost:8000/evaluate?agent=baseline
```

## Tests

```bash
cd ad_review_env
python -m pytest tests/ -v
```

70 tests covering data integrity, grader logic, smart agent decisions, and model validation.

## Project Structure

```
ad_review_env/
├── __init__.py          # Module exports
├── models.py            # AdReviewAction + AdReviewObservation (Pydantic)
├── data.py              # 30 curated UGC items with gold labels
├── grader.py            # Deterministic reward grader
├── agent.py             # Smart contextual review agent (~0.996 score)
├── client.py            # AdReviewEnv WebSocket client
├── baseline.py          # Keyword-based baseline agent
├── openenv.yaml         # OpenEnv manifest
├── pyproject.toml       # Project metadata
├── server/
│   ├── environment.py   # Core RL environment logic
│   ├── app.py           # FastAPI app + hackathon endpoints + web dashboard
│   └── Dockerfile       # Container image
└── tests/
    ├── conftest.py      # Test configuration + openenv stubs
    ├── test_data.py     # Dataset integrity tests
    ├── test_grader.py   # Grader scoring logic tests
    ├── test_agent.py    # Smart agent correctness tests
    └── test_models.py   # Pydantic model validation tests
```

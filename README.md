# Brand-Safe Ad Review Environment

An OpenEnv RL environment that trains agents to moderate user-generated content for brand-safe ad placement — the way real trust & safety teams do it.

Agents review UGC posts, classify them using **IAB Content Taxonomy 3.0** and **GARM Brand Safety Floor** standards, and decide: approve, reject, or escalate to human review. Multi-step episodes let agents request author history and community signals before making a final call.

**Live demo:** [grizzleyyybear-ad-review-env.hf.space](https://grizzleyyybear-ad-review-env.hf.space/web)

## Why This Matters

Every major platform employs content moderators to decide what's brand-safe. It's a $10B+ market, and the decisions are nuanced — a nurse advocating for gun reform is different from a political attack post, even though both mention violence. This environment captures that nuance with 30 real-world-style UGC items spanning obvious violations to subtle edge cases that challenge frontier models.

## How It Works

Each episode presents one UGC item (post, caption, comment, or bio). The agent must:

1. **Decide** — `APPROVE`, `REJECT`, or `ESCALATE`
2. **Classify** — assign IAB + GARM categories
3. **Assess** — risk level and confidence
4. **Explain** — reasoning with specific flagged elements

### Multi-Step Episodes

Agents can `REQUEST_CONTEXT` before deciding. Each request reveals a new layer:

| Step | What happens |
|------|-------------|
| 0 | Agent sees raw content → `DECIDE` or `REQUEST_CONTEXT` |
| 1 | Author history revealed → `DECIDE` or `REQUEST_CONTEXT` |
| 2 | Community signals revealed → must `DECIDE` (auto-escalate otherwise) |

Fewer steps = higher efficiency score. Easy items should be nailed in one step.

### Reward Formula (max 1.0)

| Component | Weight | Signal |
|-----------|--------|--------|
| Decision accuracy | 40% | Correct APPROVE/REJECT/ESCALATE (partial credit for adjacent) |
| Category accuracy | 30% | Correct IAB (15%) + GARM (15%) |
| Reasoning quality | 20% | Explanation length + specific flagged elements |
| Efficiency | 10% | Confidence calibration × step-efficiency multiplier |

Step-efficiency: 1 step → 1.0×, 2 steps → 0.7×, 3 steps → 0.4×. Hard tasks get a 1.1× difficulty multiplier.

## Tasks (3 tiers, 30 items)

| Tier | Count | What's tested | Example |
|------|-------|--------------|---------|
| **Easy** | 10 | Obvious safe/unsafe signals | Cookie recipe (approve) · Drug sales bio (reject) |
| **Medium** | 10 | Context-dependent judgment | Wine tasting in Tuscany · Gambling in Vegas trip context |
| **Hard** | 10 | Subtle edge cases | Nurse advocating gun reform · Labeled satire · Crypto targeting teens |

Each item has gold labels, context layers, and deterministic grading.

## Action & Observation Spaces

**Action** (`AdReviewAction`):
- `action_type`: `DECIDE` or `REQUEST_CONTEXT`
- `decision`: `APPROVE`, `REJECT`, or `ESCALATE`
- `iab_category`: one of 10 IAB categories
- `garm_category`: one of 10 GARM categories
- `risk_level`: `LOW` / `MEDIUM` / `HIGH` / `CRITICAL`
- `reasoning`: 10–500 char explanation
- `confidence`: float [0, 1]
- `flagged_elements`: list of specific harmful fragments

**Observation** (`AdReviewObservation`):
- Content fields: `content_id`, `content_text`, `content_type`, `platform`, `difficulty`
- Episode state: `step_number`, `max_steps`, `additional_context`
- Scoring: `score_decision`, `score_category`, `score_reasoning`, `score_efficiency`, `total_score`
- Feedback: `feedback`, `gold_decision`, `gold_iab_category`, `gold_garm_category`

## Quick Start

### Docker (recommended)

```bash
cd ad_review_env
docker build -t ad-review-env .
docker run -p 8000:8000 ad-review-env
```

### Local

```bash
cd ad_review_env
uv pip install -e ".[core]"
uvicorn server.app:app --port 8000
```

Then: [localhost:8000/web](http://localhost:8000/web) for the dashboard, [localhost:8000/docs](http://localhost:8000/docs) for API docs.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode |
| `/step` | POST | Submit action (DECIDE or REQUEST_CONTEXT) |
| `/state` | GET | Current episode state |
| `/health` | GET | Health check |
| `/tasks` | GET | Fetch content items for evaluation |
| `/grader` | POST | Grade a decision against gold labels |
| `/evaluate` | GET | Batch evaluation (smart or baseline agent) |
| `/baseline` | GET | Baseline agent demo |
| `/web` | GET | Interactive dashboard |
| `/schema` | GET | Action/observation JSON schemas |
| `/docs` | GET | OpenAPI documentation |

## Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="deepseek/deepseek-r1-0528"
export HF_TOKEN="your-token"
export ENV_URL="https://grizzleyyybear-ad-review-env.hf.space"

pip install openai requests
python inference.py
```

The script fetches all 30 tasks, calls the LLM for each, grades via `/grader`, and reports scores. Runs under 20 minutes on 2 vCPU / 8GB RAM.

## Baseline Scores

| Agent | Mean Score | Accuracy | Easy | Medium | Hard |
|-------|-----------|----------|------|--------|------|
| Keyword baseline | 0.52 | 53% | 0.55 | 0.49 | 0.51 |
| Smart rule-based | 0.996 | 100% | 0.99 | 0.99 | 1.00 |
| DeepSeek R1 (LLM) | 0.82 | 73% | 0.89 | 0.74 | 0.83 |

The smart agent uses multi-signal contextual analysis with per-category pattern libraries, context modifiers (satire/narrative/advocacy detection), and calibrated confidence scoring.

## Tests

```bash
python -m pytest tests/ -v
```

83 tests covering data integrity, grader logic, agent decisions, model validation, and multi-step episodes.

## Project Structure

```
openenv-meta/
├── inference.py               # LLM inference script (OpenAI client)
├── conftest.py                # Test infrastructure (openenv stubs)
├── tests/                     # 83 tests
│   ├── test_data.py
│   ├── test_grader.py
│   ├── test_agent.py
│   └── test_models.py
└── ad_review_env/
    ├── Dockerfile             # Docker (HF Spaces compatible)
    ├── openenv.yaml           # OpenEnv manifest (3 task definitions)
    ├── pyproject.toml
    ├── models.py              # Typed Pydantic action/observation models
    ├── data.py                # 30 UGC items with gold labels + context layers
    ├── grader.py              # Deterministic reward grader
    ├── agent.py               # Smart contextual agent (~0.996)
    ├── client.py              # WebSocket client
    ├── baseline.py            # Keyword baseline
    └── server/
        ├── environment.py     # Core RL environment (multi-step)
        ├── app.py             # FastAPI application
        └── dashboard.html     # Interactive web dashboard
```

## License

BSD — see [LICENSE](LICENSE).

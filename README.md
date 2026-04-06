---
title: Brand-Safe Ad Review
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# Brand-Safe Ad Review Environment

An OpenEnv RL environment for UGC content moderation. Agents classify posts using **IAB Content Taxonomy 3.0** and **GARM Brand Safety Floor** standards, deciding to approve, reject, or escalate. Multi-step episodes let agents request author history and community signals before deciding.

**Live:** [https://huggingface.co/spaces/Tnmae/openenv-ad-review](https://huggingface.co/spaces/Tnmae/openenv-ad-review)

## How It Works

Each episode presents one UGC item. The agent must:

1. **Decide** — `APPROVE`, `REJECT`, or `ESCALATE`
2. **Classify** — assign IAB + GARM categories
3. **Assess** — risk level and confidence
4. **Explain** — reasoning with flagged elements

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
| Decision accuracy | 35% | Correct APPROVE/REJECT/ESCALATE (graded adjacency) |
| Category accuracy | 25% | IAB (40%) + GARM (40%) + risk proximity (20%) |
| Reasoning quality | 20% | Length + flagged elements + domain-specific terms |
| Step efficiency | 10% | Fewer steps = higher score |
| Calibration | 10% | Confidence aligned with correctness (penalizes overconfident wrong answers) |

Adjacency: REJECT↔ESCALATE gets 0.4 partial credit, APPROVE↔ESCALATE gets 0.15. Step-efficiency: 1 step → 1.0×, 2 → 0.7×, 3 → 0.4×. Hard tasks get 1.1× multiplier.

## Tasks (3 tiers, 50 items)

| Tier | Count | What's tested | Example |
|------|-------|--------------|---------|
| **Easy** | 15 | Obvious safe/unsafe signals | Cookie recipe (approve) · Drug sales bio (reject) |
| **Medium** | 18 | Context-dependent judgment | Wine tasting in Tuscany · Gambling in Vegas trip context |
| **Hard** | 17 | Subtle edge cases | Nurse advocating gun reform · Labeled satire · Crypto targeting teens |

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
- Scoring: `score_decision`, `score_category`, `score_reasoning`, `score_efficiency`, `score_calibration`, `total_score`
- Feedback: `feedback`, `gold_decision`, `gold_iab_category`, `gold_garm_category`

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/Tnmae/openenv-meta.git
cd openenv-meta
docker build -t ad-review-env .
docker run -p 8000:8000 ad-review-env
```

### Local (pip)

```bash
git clone https://github.com/Tnmae/openenv-meta.git
cd openenv-meta/ad_review_env
pip install -e ".[inference]"
uvicorn server.app:app --port 8000
```

### Local (uv)

```bash
git clone https://github.com/Tnmae/openenv-meta.git
cd openenv-meta/ad_review_env
uv pip install --system -e ".[inference]"
uvicorn server.app:app --port 8000
```

**Requirements:** Python ≥ 3.10, pip or uv. Docker alternative needs only Docker.

Once running, visit:
- Dashboard: [localhost:8000/web](http://localhost:8000/web)
- API docs: [localhost:8000/docs](http://localhost:8000/docs)
- Health check: [localhost:8000/health](http://localhost:8000/health)

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
| `/analyze` | POST | Classify arbitrary text with the smart agent |
| `/baseline` | GET | Baseline agent demo |
| `/web` | GET | Interactive dashboard |
| `/schema` | GET | Action/observation JSON schemas |
| `/docs` | GET | OpenAPI documentation |

## Running Inference

The inference script runs episodes using the standard `POST /reset` → `POST /step` loop.

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `API_BASE_URL` | Yes | OpenAI-compatible API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Yes | Model identifier (e.g. `qwen2.5-coder:7b`) | — |
| `HF_TOKEN` or `API_KEY` | Yes | API authentication key | — |
| `ENV_URL` | No | Environment server URL | `http://localhost:8000` |

### With HuggingFace Router

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="deepseek-ai/DeepSeek-R1"
export HF_TOKEN="hf_your_token"
export ENV_URL="http://localhost:8000"
python inference.py
```

### With Local Ollama

```bash
# Install and run a model first: ollama pull qwen2.5-coder:7b
export API_BASE_URL="http://localhost:11434/v1"
export API_KEY="ollama"
export MODEL_NAME="qwen2.5-coder:7b"
export ENV_URL="http://localhost:8000"
python inference.py
```

### With Any OpenAI-Compatible API

```bash
export API_BASE_URL="https://api.openai.com/v1"  # or any compatible endpoint
export API_KEY="sk-your-key"
export MODEL_NAME="gpt-4o"
export ENV_URL="http://localhost:8000"
python inference.py
```

### Windows (PowerShell)

```powershell
$env:API_BASE_URL = "http://localhost:11434/v1"
$env:API_KEY = "ollama"
$env:MODEL_NAME = "qwen2.5-coder:7b"
$env:ENV_URL = "http://localhost:8000"
python inference.py
```

## Baseline Scores

| Agent | Mean Score | Accuracy | Easy | Medium | Hard |
|-------|-----------|----------|------|--------|------|
| Keyword baseline | 0.896 | 82% | 0.85 | 0.91 | 0.93 |
| Smart rule-based | 0.9971 | 100% | 0.990 | 1.000 | 1.000 |
| Qwen 2.5-coder 7B | 0.847 | 76% | 0.91 | 0.85 | 0.78 |

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

189 tests covering data integrity, grader logic (6 components + calibration), agent decisions, model validation, edge cases, and multi-step episodes.

## Project Structure

```
openenv-meta/
├── Dockerfile                 # Docker (builds from repo root)
├── openenv.yaml               # OpenEnv manifest (3 task definitions)
├── inference.py               # LLM inference script (OpenAI client)
├── conftest.py                # Test infrastructure (openenv stubs)
├── README.md
├── LICENSE
├── tests/                     # 83 tests
│   ├── test_data.py
│   ├── test_grader.py
│   ├── test_agent.py
│   └── test_models.py
└── ad_review_env/             # The environment package
    ├── pyproject.toml
    ├── models.py              # Typed Pydantic action/observation models
    ├── data.py                # 50 UGC items with gold labels + context layers
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

"""Core environment — multi-step UGC content moderation episodes (max 3 steps).

Uses class-level shared state so that the openenv-core HTTP framework (which
creates a fresh instance per request) can persist episode state across
reset → step → step calls.
"""

import random
import threading
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import AdReviewAction, AdReviewObservation
    from ..data import CONTENT_ITEMS, CONTENT_INDEX
    from ..grader import grade
except ImportError:
    from models import AdReviewAction, AdReviewObservation
    from data import CONTENT_ITEMS, CONTENT_INDEX
    from grader import grade

MAX_STEPS = 3


class _SharedState:
    """Thread-safe shared state store for HTTP-mode episode persistence."""

    def __init__(self):
        self._lock = threading.Lock()
        self._item: Optional[dict] = None
        self._context: list[str] = []
        self._step_count: int = 0
        self._episode_id: str = str(uuid4())
        self._rng = random.Random()

    def reset(self, seed: Optional[int], episode_id: Optional[str]) -> dict:
        with self._lock:
            if seed is not None:
                self._rng.seed(seed)
            self._episode_id = episode_id or str(uuid4())
            self._item = self._rng.choice(CONTENT_ITEMS)
            self._context = []
            self._step_count = 0
            return self._item.copy()

    def get(self):
        with self._lock:
            return self._item, list(self._context), self._step_count, self._episode_id

    def increment_step(self) -> int:
        with self._lock:
            self._step_count += 1
            return self._step_count

    def add_context(self, ctx: str):
        with self._lock:
            self._context.append(ctx)

    def clear_item(self):
        with self._lock:
            self._item = None

    @property
    def state(self) -> State:
        with self._lock:
            return State(episode_id=self._episode_id, step_count=self._step_count)


_shared = _SharedState()


class AdReviewEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> AdReviewObservation:
        item = _shared.reset(seed, episode_id)
        return self._make_obs(item, [], step_number=0, done=False, reward=None)

    def step(self, action: AdReviewAction, timeout_s: Optional[float] = None, **kwargs) -> AdReviewObservation:
        item, context, _, _ = _shared.get()
        if item is None:
            raise RuntimeError("Call reset() before step().")

        step_num = _shared.increment_step()

        if action.action_type == "REQUEST_CONTEXT" and step_num < MAX_STEPS:
            key = f"context_layer_{len(context) + 1}"
            ctx = item.get(key, "No additional context available.")
            _shared.add_context(ctx)
            _, updated_ctx, _, _ = _shared.get()
            return self._make_obs(item, updated_ctx, step_number=step_num, done=False, reward=None)

        if action.action_type == "REQUEST_CONTEXT" and step_num >= MAX_STEPS:
            action_data = {
                "decision": "ESCALATE", "iab_category": "IAB_CONTROVERSIAL",
                "garm_category": "GARM_SAFE", "risk_level": "MEDIUM",
                "age_rating": "TEEN",
                "reasoning": "Max steps reached without decision. Auto-escalated.",
                "confidence": 0.2, "flagged_elements": [],
            }
        else:
            action_data = {
                "decision": action.decision, "iab_category": action.iab_category,
                "garm_category": action.garm_category, "risk_level": action.risk_level,
                "age_rating": action.age_rating,
                "reasoning": action.reasoning, "confidence": action.confidence,
                "flagged_elements": action.flagged_elements,
            }

        total_reward, scores, feedback = grade(action_data, item, steps_taken=step_num)

        _, final_ctx, _, _ = _shared.get()
        obs = self._make_obs(
            item, final_ctx, step_number=step_num, done=True, reward=total_reward,
            score_decision=scores["decision"], score_category=scores["category"],
            score_reasoning=scores["reasoning"], score_age_rating=scores["age_rating"],
            score_efficiency=scores["efficiency"],
            score_calibration=scores.get("calibration", 0.0),
            total_score=scores["total"], feedback=feedback,
            gold_decision=item["gold_decision"],
            gold_iab_category=item["gold_iab_category"],
            gold_garm_category=item["gold_garm_category"],
            gold_age_rating=item.get("gold_age_rating"),
        )
        _shared.clear_item()
        return obs

    def _make_obs(self, item: dict, context: list, step_number: int, done: bool,
                  reward, **extra) -> AdReviewObservation:
        formatted_ctx = "\n".join(f"[Context {i}] {c}" for i, c in enumerate(context, 1)) if context else None
        return AdReviewObservation(
            content_id=item["content_id"],
            content_text=item["content_text"],
            content_type=item["content_type"],
            platform=item["platform"],
            difficulty=item["difficulty"],
            step_number=step_number, max_steps=MAX_STEPS,
            additional_context=formatted_ctx,
            done=done, reward=reward, **extra,
        )

    @property
    def state(self) -> State:
        return _shared.state

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name="ad_review_env",
            description="Brand-Safe Ad Review: UGC content moderation with IAB/GARM standards.",
            version="1.1.0",
            author="Meta OpenEnv Hackathon",
        )

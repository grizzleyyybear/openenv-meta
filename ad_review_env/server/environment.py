"""Core environment — multi-step UGC content moderation episodes (max 3 steps)."""

import random
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


class AdReviewEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_item: Optional[dict] = None
        self._rng = random.Random()
        self._revealed_context: list[str] = []

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> AdReviewObservation:
        if seed is not None:
            self._rng.seed(seed)

        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._current_item = self._rng.choice(CONTENT_ITEMS)
        self._revealed_context = []

        return self._make_obs(step_number=0, done=False, reward=None)

    def step(self, action: AdReviewAction, timeout_s: Optional[float] = None, **kwargs) -> AdReviewObservation:
        if self._current_item is None:
            self.reset()

        self._state.step_count += 1
        step_num = self._state.step_count

        if action.action_type == "REQUEST_CONTEXT" and step_num < MAX_STEPS:
            self._get_next_context()
            return self._make_obs(step_number=step_num, done=False, reward=None)

        # DECIDE (or forced at max steps)
        if action.action_type == "REQUEST_CONTEXT" and step_num >= MAX_STEPS:
            action_data = {
                "decision": "ESCALATE", "iab_category": "IAB_CONTROVERSIAL",
                "garm_category": "GARM_SAFE", "risk_level": "MEDIUM",
                "reasoning": "Max steps reached without decision. Auto-escalated.",
                "confidence": 0.2, "flagged_elements": [],
            }
        else:
            action_data = {
                "decision": action.decision, "iab_category": action.iab_category,
                "garm_category": action.garm_category, "risk_level": action.risk_level,
                "reasoning": action.reasoning, "confidence": action.confidence,
                "flagged_elements": action.flagged_elements,
            }

        total_reward, scores, feedback = grade(action_data, self._current_item, steps_taken=step_num)

        return self._make_obs(
            step_number=step_num, done=True, reward=total_reward,
            score_decision=scores["decision"], score_category=scores["category"],
            score_reasoning=scores["reasoning"], score_efficiency=scores["efficiency"],
            total_score=scores["total"], feedback=feedback,
            gold_decision=self._current_item["gold_decision"],
            gold_iab_category=self._current_item["gold_iab_category"],
            gold_garm_category=self._current_item["gold_garm_category"],
        )

    def _make_obs(self, step_number: int, done: bool, reward, **extra) -> AdReviewObservation:
        return AdReviewObservation(
            content_id=self._current_item["content_id"],
            content_text=self._current_item["content_text"],
            content_type=self._current_item["content_type"],
            platform=self._current_item["platform"],
            difficulty=self._current_item["difficulty"],
            step_number=step_number, max_steps=MAX_STEPS,
            additional_context=self._format_all_context() if self._revealed_context else None,
            done=done, reward=reward, **extra,
        )

    def _get_next_context(self) -> str:
        key = f"context_layer_{len(self._revealed_context) + 1}"
        ctx = self._current_item.get(key, "No additional context available.")
        self._revealed_context.append(ctx)
        return ctx

    def _format_all_context(self) -> str:
        return "\n".join(f"[Context {i}] {c}" for i, c in enumerate(self._revealed_context, 1))

    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name="ad_review_env",
            description="Brand-Safe Ad Review: UGC content moderation with IAB/GARM standards.",
            version="1.1.0",
            author="Meta OpenEnv Hackathon",
        )

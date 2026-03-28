"""
Core environment logic — multi-step episodes for UGC content moderation.

Agents can REQUEST_CONTEXT to get enriched observations (author history,
community signals) before making a final DECIDE. Max 3 steps per episode.
"""

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
    """Each episode presents one UGC item. The agent reviews and classifies it."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_item: Optional[dict] = None
        self._rng = random.Random()
        self._revealed_context: list[str] = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> AdReviewObservation:
        """Reset: sample a new content item to review."""
        if seed is not None:
            self._rng.seed(seed)

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        self._current_item = self._rng.choice(CONTENT_ITEMS)
        self._revealed_context = []

        return AdReviewObservation(
            content_id=self._current_item["content_id"],
            content_text=self._current_item["content_text"],
            content_type=self._current_item["content_type"],
            platform=self._current_item["platform"],
            difficulty=self._current_item["difficulty"],
            step_number=0,
            max_steps=MAX_STEPS,
            additional_context=None,
            done=False,
            reward=None,
        )

    def step(
        self,
        action: AdReviewAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> AdReviewObservation:
        """Execute one review step."""
        if self._current_item is None:
            self.reset()

        self._state.step_count += 1
        step_num = self._state.step_count

        # --- REQUEST_CONTEXT path ---
        if action.action_type == "REQUEST_CONTEXT" and step_num < MAX_STEPS:
            context = self._get_next_context()
            return AdReviewObservation(
                content_id=self._current_item["content_id"],
                content_text=self._current_item["content_text"],
                content_type=self._current_item["content_type"],
                platform=self._current_item["platform"],
                difficulty=self._current_item["difficulty"],
                step_number=step_num,
                max_steps=MAX_STEPS,
                additional_context=self._format_all_context(),
                done=False,
                reward=None,
            )

        # --- DECIDE path (or forced at max steps) ---
        if action.action_type == "REQUEST_CONTEXT" and step_num >= MAX_STEPS:
            # Max steps reached on a context request → force ESCALATE
            action_data = {
                "decision": "ESCALATE",
                "iab_category": "IAB_CONTROVERSIAL",
                "garm_category": "GARM_SAFE",
                "risk_level": "MEDIUM",
                "reasoning": "Max steps reached without decision. Auto-escalated for human review.",
                "confidence": 0.2,
                "flagged_elements": [],
            }
        else:
            action_data = {
                "decision": action.decision,
                "iab_category": action.iab_category,
                "garm_category": action.garm_category,
                "risk_level": action.risk_level,
                "reasoning": action.reasoning,
                "confidence": action.confidence,
                "flagged_elements": action.flagged_elements,
            }

        total_reward, scores, feedback = grade(
            action_data, self._current_item, steps_taken=step_num,
        )

        return AdReviewObservation(
            content_id=self._current_item["content_id"],
            content_text=self._current_item["content_text"],
            content_type=self._current_item["content_type"],
            platform=self._current_item["platform"],
            difficulty=self._current_item["difficulty"],
            step_number=step_num,
            max_steps=MAX_STEPS,
            additional_context=self._format_all_context() if self._revealed_context else None,
            # Scoring
            score_decision=scores["decision"],
            score_category=scores["category"],
            score_reasoning=scores["reasoning"],
            score_efficiency=scores["efficiency"],
            total_score=scores["total"],
            # Feedback
            feedback=feedback,
            gold_decision=self._current_item["gold_decision"],
            gold_iab_category=self._current_item["gold_iab_category"],
            gold_garm_category=self._current_item["gold_garm_category"],
            # Terminal
            done=True,
            reward=total_reward,
        )

    def _get_next_context(self) -> str:
        layer_num = len(self._revealed_context) + 1
        key = f"context_layer_{layer_num}"
        context = self._current_item.get(key, "No additional context available.")
        self._revealed_context.append(context)
        return context

    def _format_all_context(self) -> str:
        return "\n".join(f"[Context {i}] {ctx}" for i, ctx in enumerate(self._revealed_context, 1))

    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name="ad_review_env",
            description=(
                "Brand-Safe Ad Review environment for UGC content moderation. "
                "Agents review user-generated content and decide APPROVE/REJECT/ESCALATE "
                "using IAB Content Taxonomy 3.0 and GARM Brand Safety Floor standards. "
                "Multi-step episodes: agents can REQUEST_CONTEXT before deciding."
            ),
            version="1.1.0",
            author="Meta OpenEnv Hackathon",
        )

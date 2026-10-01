"""The simulated interviewer.

It runs the question script and decides follow-ups **from the answer text
alone**. It never sees the truth sheet or ``disclosed_fact_ids`` — a live agent
would not have them, and letting it peek would turn the follow-up policy into an
oracle and make recall-by-disclosure meaningless.

Follow-up policy (spec 3.7): ask one, up to ``max_followups``, when the answer is
shorter than ``followup_trigger_words`` **or** a cheap judge call reports that a
target field went unaddressed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import RoleConfig
from ..hashing import sha256_hex
from ..llm import call
from ..models import Question, word_count
from ..providers.base import Message, Provider
from ..role_schemas import INTERVIEWER_FOLLOWUP_SCHEMA

PROMPT_PATH = Path("prompts/interviewer_followup.txt")


@dataclass(frozen=True)
class FollowupDecision:
    targets_addressed: list[str]
    followup: str | None
    #: Set when the decision call failed. The simulation continues without a
    #: follow-up rather than aborting, and the failure is recorded.
    error: str | None = None

    @property
    def unaddressed(self) -> bool:
        return self.followup is not None


class Interviewer:
    def __init__(
        self,
        *,
        provider: Provider,
        model: str,
        role_cfg: RoleConfig | None = None,
        template: str | None = None,
        followup_trigger_words: int = 40,
        cache_root=None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.cfg = role_cfg or RoleConfig()
        self.template = template if template is not None else PROMPT_PATH.read_text()
        self.prompt_hash = sha256_hex(self.template)
        self.followup_trigger_words = followup_trigger_words
        self.cache_root = cache_root

    async def decide(self, question: Question, answer: str) -> FollowupDecision:
        """Decide whether to follow up.

        A short answer is reason enough on its own, but the judge call still
        runs, because `targets_addressed` is what makes "which field did this
        question actually reach" reportable rather than guessed.
        """
        result = await call(
            role="interviewer_followup",
            provider=self.provider,
            system=self.template.format(
                question=question.text,
                targets=", ".join(question.targets),
                answer=answer,
            ),
            messages=[Message(role="user", content="Decide.")],
            model=self.model,
            role_cfg=self.cfg,
            json_schema=INTERVIEWER_FOLLOWUP_SCHEMA,
            cache_root=self.cache_root,
        )
        if not result.ok:
            return FollowupDecision([], None, error=result.error or result.reason)

        payload = result.parsed or {}
        addressed = [t for t in (payload.get("targets_addressed") or []) if t in question.targets]
        followup = payload.get("followup")
        if isinstance(followup, str):
            followup = followup.strip() or None

        too_short = word_count(answer) < self.followup_trigger_words
        missing_target = bool(set(question.targets) - set(addressed))
        if not (too_short or missing_target):
            followup = None
        elif followup is None and (too_short or missing_target):
            # The policy says to probe, but the model declined to write one.
            # Record the decision honestly rather than inventing a question.
            followup = None
        return FollowupDecision(addressed, followup)

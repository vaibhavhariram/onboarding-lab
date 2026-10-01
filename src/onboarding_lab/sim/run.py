"""Run one simulated interview per persona.

Simulation is **serial within a persona** — turn N+1 depends on turn N — and
parallel across personas. That shape is why wall clock, not cost, is the
binding constraint on a full run.

Every disclosed fact is attributed to the turn that elicited it, which is what
makes per-question yield and follow-up value computable rather than estimated.
Follow-up turns carry the **parent** ``question_id`` with ``is_followup=True``;
there is no separate follow-up id scheme (PLAN.md M15).
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from ..config import Config, RoleConfig
from ..hashing import hash_obj
from ..models import (
    ProvenanceStamp,
    QuestionScript,
    Transcript,
    TruthSheet,
    Turn,
    word_count,
)
from ..paths import transcript_stem
from ..providers.base import Provider
from .interviewer import Interviewer
from .user_agent import UserAgent

SimStatus = Literal["ok", "provider_error", "refusal", "truncated"]


@dataclass(frozen=True)
class SimResult:
    persona_id: str
    status: SimStatus
    transcript: Transcript | None = None
    detail: str | None = None
    #: Fact ids the user agent reported that are not on its own sheet. Dropped,
    #: not trusted, and surfaced because `disclosed_fact_ids` is the ground
    #: truth for every recall and yield number (PLAN.md M10).
    invented_fact_ids: list[str] = field(default_factory=list)
    #: Follow-up decisions that failed. The interview continues without a
    #: follow-up; it is recorded rather than hidden.
    followup_errors: int = 0

    def as_failure_record(self) -> dict:
        return {
            "stage": "sim",
            "persona_id": self.persona_id,
            "status": self.status,
            "detail": self.detail,
        }


def ordered_questions(script: QuestionScript) -> list:
    """Script order, or a seeded shuffle when ``order_seed`` is set."""
    questions = list(script.questions)
    if script.order_seed is not None:
        random.Random(f"{script.order_seed}:order").shuffle(questions)
    return questions


def script_hash(script: QuestionScript) -> str:
    return hash_obj(script.model_dump(mode="json"))


async def simulate_one(
    sheet: TruthSheet,
    script: QuestionScript,
    *,
    provider: Provider,
    model: str,
    sim_seed: int,
    user_cfg: RoleConfig | None = None,
    interviewer_cfg: RoleConfig | None = None,
    code_version: str = "dev",
    cache_root=None,
    now: datetime | None = None,
) -> SimResult:
    agent = UserAgent(
        sheet,
        provider=provider,
        model=model,
        role_cfg=user_cfg,
        cache_root=cache_root,
    )
    interviewer = Interviewer(
        provider=provider,
        model=model,
        role_cfg=interviewer_cfg,
        followup_trigger_words=script.followup_trigger_words,
        cache_root=cache_root,
    )

    turns: list[Turn] = []
    followup_errors = 0

    def add_turn(
        speaker: str,
        text: str,
        question_id: str,
        *,
        is_followup: bool,
        disclosed: list[str] | None = None,
    ) -> None:
        turns.append(
            Turn(
                turn_id=f"t{len(turns) + 1:03d}",
                speaker=speaker,
                question_id=question_id,
                is_followup=is_followup,
                text=text,
                word_count=word_count(text),
                disclosed_fact_ids=disclosed or [],
            )
        )

    for question in ordered_questions(script):
        add_turn("interviewer", question.text, question.question_id, is_followup=False)
        utterance, disclosed, error = await agent.answer(turns)
        if error is not None:
            return SimResult(sheet.persona_id, _status_for(error), detail=error)
        add_turn("user", utterance, question.question_id, is_followup=False, disclosed=disclosed)

        last_answer = utterance
        for _ in range(question.max_followups):
            decision = await interviewer.decide(question, last_answer)
            if decision.error is not None:
                followup_errors += 1
                break
            if decision.followup is None:
                break
            add_turn("interviewer", decision.followup, question.question_id, is_followup=True)
            utterance, disclosed, error = await agent.answer(turns)
            if error is not None:
                return SimResult(sheet.persona_id, _status_for(error), detail=error)
            add_turn(
                "user",
                utterance,
                question.question_id,
                is_followup=True,
                disclosed=disclosed,
            )
            last_answer = utterance

    transcript = Transcript(
        transcript_id=transcript_stem(
            sheet.persona_id, sim_seed, 0.0, shuffled=script.order_seed is not None
        ),
        persona_id=sheet.persona_id,
        script_hash=script_hash(script),
        sim_seed=sim_seed,
        turns=turns,
        simulated_minutes=round(sum(t.word_count for t in turns) / 150.0, 4),
        provenance=ProvenanceStamp(
            model=model,
            prompt_hash=agent.prompt_hash,
            params={
                "user_agent": (user_cfg or RoleConfig()).model_dump(),
                "interviewer_followup": (interviewer_cfg or RoleConfig()).model_dump(),
                "sim_seed": sim_seed,
            },
            code_version=code_version,
            created_at=now or datetime.now(UTC),
        ),
    )
    return SimResult(
        sheet.persona_id,
        "ok",
        transcript=transcript,
        invented_fact_ids=list(agent.invented_fact_ids),
        followup_errors=followup_errors,
    )


_STATUS_HINTS: tuple[tuple[str, SimStatus], ...] = (
    ("refusal", "refusal"),
    ("max_tokens", "truncated"),
    ("model_context_window_exceeded", "truncated"),
)


def _status_for(error: str) -> SimStatus:
    for needle, status in _STATUS_HINTS:
        if needle in error:
            return status
    return "provider_error"


async def simulate_all(
    sheets: list[TruthSheet],
    script: QuestionScript,
    *,
    provider: Provider,
    model: str,
    config: Config | None = None,
    sim_seed: int | None = None,
    code_version: str = "dev",
    cache_root=None,
    now: datetime | None = None,
) -> list[SimResult]:
    """Simulate every persona, bounded by ``concurrency``.

    Results are re-ordered by persona id rather than returned in completion
    order: joining by position after a gather is exactly the misalignment bug
    this project exists to avoid.
    """
    cfg = config or Config()
    gate = asyncio.Semaphore(cfg.concurrency)
    seed = cfg.sim.seed if sim_seed is None else sim_seed

    async def one(sheet: TruthSheet) -> SimResult:
        async with gate:
            return await simulate_one(
                sheet,
                script,
                provider=provider,
                model=model,
                sim_seed=seed,
                user_cfg=cfg.role("user_agent"),
                interviewer_cfg=cfg.role("interviewer_followup"),
                code_version=code_version,
                cache_root=cache_root,
                now=now,
            )

    results = await asyncio.gather(*(one(s) for s in sheets))
    by_id = {r.persona_id: r for r in results}
    return [by_id[s.persona_id] for s in sheets]

"""The simulated interview.

Two properties matter beyond "it produces a transcript": the interviewer must
stay blind to the truth sheet, and the user agent's system prompt must be
byte-stable so it actually caches.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from onboarding_lab import llm as llm_module
from onboarding_lab.models import QuestionScript, Transcript
from onboarding_lab.providers.fake import FakeProvider
from onboarding_lab.sim.run import ordered_questions, simulate_all, simulate_one
from onboarding_lab.sim.user_agent import STYLE_WORD_RANGES, render_fact_sheet, render_system

ANSWER = "I work shifts and spend my evenings reading. " * 6
FOLLOWUP = "What draws you to that?"


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(llm_module.asyncio, "sleep", _instant)


@pytest.fixture
def script() -> QuestionScript:
    return QuestionScript.model_validate(yaml.safe_load(Path("questions.yaml").read_text()))


def cooperative_provider(disclosed: list[str], *, followup: str | None = FOLLOWUP) -> FakeProvider:
    return FakeProvider(
        {
            "role:user_agent": {"utterance": ANSWER, "disclosed_fact_ids": disclosed},
            "role:interviewer_followup": {"targets_addressed": [], "followup": followup},
        }
    )


def run(sheet, script, provider, **kw):
    return asyncio.run(
        simulate_one(sheet, script, provider=provider, model="fake-model", sim_seed=1, **kw)
    )


def test_produces_a_valid_transcript(truth_sheet, script, tmp_path) -> None:
    result = run(truth_sheet, script, cooperative_provider(["f06"]), cache_root=tmp_path)
    assert result.status == "ok"
    t = result.transcript
    assert isinstance(t, Transcript)
    assert Transcript.model_validate_json(t.model_dump_json()) == t
    assert t.noise_rate == 0.0
    assert t.source_transcript_id is None
    assert t.transcript_id == "p001__s1__n000"


def test_turn_ids_are_sequential(truth_sheet, script, tmp_path) -> None:
    t = run(truth_sheet, script, cooperative_provider([]), cache_root=tmp_path).transcript
    assert [turn.turn_id for turn in t.turns] == [f"t{i + 1:03d}" for i in range(len(t.turns))]


def test_followups_carry_the_parent_question_id(truth_sheet, script, tmp_path) -> None:
    """Per-question yield aggregates main and follow-up turns under one id."""
    t = run(truth_sheet, script, cooperative_provider([]), cache_root=tmp_path).transcript
    followups = [turn for turn in t.turns if turn.is_followup]
    assert followups, "the cooperative provider should have triggered follow-ups"
    for turn in followups:
        assert turn.question_id is not None
        assert turn.question_id in {q.question_id for q in script.questions}
    # Both speakers' follow-up turns are marked, so via_followup is derivable.
    assert {turn.speaker for turn in followups} == {"interviewer", "user"}


def test_no_followup_when_the_interviewer_declines(truth_sheet, script, tmp_path) -> None:
    t = run(
        truth_sheet, script, cooperative_provider([], followup=None), cache_root=tmp_path
    ).transcript
    assert not any(turn.is_followup for turn in t.turns)
    assert len(t.turns) == 2 * len(script.questions)


def test_invented_fact_ids_are_dropped_and_surfaced(truth_sheet, script, tmp_path) -> None:
    """disclosed_fact_ids is the ground truth for recall, so it is not trusted."""
    result = run(truth_sheet, script, cooperative_provider(["f06", "f99"]), cache_root=tmp_path)
    for turn in result.transcript.turns:
        assert "f99" not in turn.disclosed_fact_ids
    assert "f99" in result.invented_fact_ids


def test_duplicate_disclosures_are_collapsed(truth_sheet, script, tmp_path) -> None:
    result = run(truth_sheet, script, cooperative_provider(["f06", "f06"]), cache_root=tmp_path)
    for turn in result.transcript.turns:
        assert len(set(turn.disclosed_fact_ids)) == len(turn.disclosed_fact_ids)


def test_interviewer_never_sees_the_truth_sheet(truth_sheet, script, tmp_path) -> None:
    """A peeking interviewer would turn the follow-up policy into an oracle and
    make recall-by-disclosure meaningless."""
    provider = cooperative_provider(["f06"])
    run(truth_sheet, script, provider, cache_root=tmp_path)
    interviewer_prompts = [
        c["system"] for c in provider.calls if c["role"] == "interviewer_followup"
    ]
    assert interviewer_prompts
    for prompt in interviewer_prompts:
        for fact in truth_sheet.facts:
            assert fact.fact_id not in prompt
        assert truth_sheet.bio.strip()[:40] not in prompt
        assert "needs_followup" not in prompt


def test_user_agent_system_prompt_is_byte_stable(truth_sheet, script, tmp_path) -> None:
    """It is the one cacheable prefix in the pipeline; any volatile byte in it
    multiplies the simulation bill (PLAN.md P7)."""
    provider = cooperative_provider([])
    run(truth_sheet, script, provider, cache_root=tmp_path)
    prompts = {c["system"] for c in provider.calls if c["role"] == "user_agent"}
    assert len(prompts) == 1, "the persona's system prompt changed between turns"


def test_user_agent_requests_the_cache_breakpoint(truth_sheet, script, tmp_path) -> None:
    from onboarding_lab.config import Config

    cfg = Config.load("lab.yaml")
    provider = cooperative_provider([])
    run(
        truth_sheet,
        script,
        provider,
        user_cfg=cfg.role("user_agent"),
        interviewer_cfg=cfg.role("interviewer_followup"),
        cache_root=tmp_path,
    )
    user_calls = [c for c in provider.calls if c["role"] == "user_agent"]
    assert all(c["cache_system"] for c in user_calls)
    assert not any(c["cache_system"] for c in provider.calls if c["role"] != "user_agent")


def test_fact_sheet_rendering_is_sorted_and_deterministic(truth_sheet) -> None:
    once = render_fact_sheet(truth_sheet.facts)
    twice = render_fact_sheet(list(reversed(truth_sheet.facts)))
    assert once == twice
    ids = [line.split(" | ")[0] for line in once.splitlines()]
    assert ids == sorted(ids)


def test_style_word_range_reaches_the_prompt(truth_sheet) -> None:
    template = Path("prompts/user_agent.txt").read_text()
    rendered = render_system(truth_sheet, template)
    low, high = STYLE_WORD_RANGES[truth_sheet.style]
    assert str(low) in rendered and str(high) in rendered
    assert "{bio}" not in rendered


def test_tangential_style_gets_its_digression_instruction(truth_sheet) -> None:
    template = Path("prompts/user_agent.txt").read_text()
    sheet = truth_sheet.model_copy(update={"style": "tangential"})
    assert "digression" in render_system(sheet, template)
    assert "digression" not in render_system(truth_sheet, template)


def test_order_seed_shuffles_deterministically(script) -> None:
    as_written = [q.question_id for q in ordered_questions(script)]
    shuffled = script.model_copy(update={"order_seed": 3})
    first = [q.question_id for q in ordered_questions(shuffled)]
    second = [q.question_id for q in ordered_questions(shuffled)]
    assert first == second
    assert first != as_written
    assert sorted(first) == sorted(as_written)


def test_shuffled_run_is_marked_in_the_transcript_id(truth_sheet, script, tmp_path) -> None:
    shuffled = script.model_copy(update={"order_seed": 3})
    t = run(truth_sheet, shuffled, cooperative_provider([]), cache_root=tmp_path).transcript
    assert "shuf" in t.transcript_id


def test_user_agent_failure_is_a_recorded_status(truth_sheet, script, tmp_path) -> None:
    provider = FakeProvider(
        {
            "role:user_agent": {"__error__": True, "reason": "refusal", "error": "refusal: no"},
            "role:interviewer_followup": {"targets_addressed": [], "followup": None},
        }
    )
    result = run(truth_sheet, script, provider, cache_root=tmp_path)
    assert result.status == "refusal"
    assert result.transcript is None
    assert result.as_failure_record()["stage"] == "sim"


def test_followup_failure_does_not_abort_the_interview(truth_sheet, script, tmp_path) -> None:
    provider = FakeProvider(
        {
            "role:user_agent": {"utterance": ANSWER, "disclosed_fact_ids": []},
            "role:interviewer_followup": {"__error__": True, "error": "boom"},
        }
    )
    result = run(truth_sheet, script, provider, cache_root=tmp_path)
    assert result.status == "ok"
    assert result.followup_errors == len(script.questions)


def test_simulate_all_returns_persona_order(truth_sheet, script, tmp_path) -> None:
    sheets = [
        truth_sheet.model_copy(update={"persona_id": pid}) for pid in ("p001", "p002", "p003")
    ]
    results = asyncio.run(
        simulate_all(
            sheets,
            script,
            provider=cooperative_provider([]),
            model="fake-model",
            cache_root=tmp_path,
        )
    )
    assert [r.persona_id for r in results] == ["p001", "p002", "p003"]

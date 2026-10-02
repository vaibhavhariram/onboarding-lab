"""Builders for valid contract objects.

Tests construct these and then break one thing, so the builders must produce
something that passes every validator.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from onboarding_lab.config import API_KEY_VAR, FORBIDDEN_KEY_VAR
from onboarding_lab.models import (
    Alignment,
    Claim,
    Extraction,
    Fact,
    FactCoverage,
    ProvenanceStamp,
    Transcript,
    TruthSheet,
    Turn,
    word_count,
)


@pytest.fixture(autouse=True)
def _no_credentials_and_no_network(monkeypatch):
    """Make `make test` structurally unable to reach the network.

    Two separate holes, both closed here:

    1. ``config.load_dotenv`` writes into the real ``os.environ``. ``monkeypatch``
       only reverts variables it set itself, so a test that exercises ``.env``
       loading used to leak a credential into every later test in the process --
       and on a machine with a real ``.env`` that meant `make test` would build a
       live client and spend money.
    2. Even with no credential, nothing should be able to construct a real
       client by accident. Any attempt raises instead.

    A test that deliberately sets a key still works: its own ``monkeypatch``
    applies after this fixture.
    """
    for var in (API_KEY_VAR, FORBIDDEN_KEY_VAR):
        monkeypatch.delenv(var, raising=False)

    import anthropic

    def _blocked(*args, **kwargs):
        raise AssertionError(
            "a test tried to construct a real Anthropic client; the suite must "
            "run entirely on FakeProvider"
        )

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _blocked)
    monkeypatch.setattr(anthropic, "Anthropic", _blocked, raising=False)


@pytest.fixture
def stamp() -> ProvenanceStamp:
    return ProvenanceStamp(
        model="fake-model",
        prompt_hash="deadbeef",
        params={"effort": "low", "thinking": "between_tools"},
        code_version="dev",
        created_at=datetime(2026, 9, 30, 12, 0, tzinfo=UTC),
    )


def make_turn(
    turn_id: str,
    speaker: str,
    text: str,
    *,
    question_id: str | None = "q01",
    is_followup: bool = False,
    disclosed: list[str] | None = None,
) -> Turn:
    return Turn(
        turn_id=turn_id,
        speaker=speaker,
        question_id=question_id,
        is_followup=is_followup,
        text=text,
        word_count=word_count(text),
        disclosed_fact_ids=disclosed or [],
    )


def make_facts() -> list[Fact]:
    """One fact per exact-scored field, plus a couple of judged ones."""
    return [
        Fact(
            fact_id="f01",
            field="age",
            value=34,
            specificity="specific",
            disclosure="volunteer",
            importance="core",
        ),
        Fact(
            fact_id="f02",
            field="relationship_goal",
            value="long_term",
            specificity="generic",
            disclosure="volunteer",
            importance="core",
        ),
        Fact(
            fact_id="f03",
            field="wants_kids",
            value="yes",
            specificity="generic",
            disclosure="needs_followup",
            importance="dealbreaker",
        ),
        Fact(
            fact_id="f04",
            field="has_kids",
            value=False,
            specificity="generic",
            disclosure="volunteer",
            importance="dealbreaker",
        ),
        Fact(
            fact_id="f05",
            field="religion_importance",
            value="low",
            specificity="generic",
            disclosure="hedge",
            importance="dealbreaker",
        ),
        Fact(
            fact_id="f06",
            field="occupation",
            value="hospital systems administrator",
            specificity="specific",
            disclosure="volunteer",
            importance="core",
        ),
        Fact(
            fact_id="f07",
            field="interests",
            value="restoring a 1972 motorcycle",
            specificity="specific",
            disclosure="needs_followup",
            importance="color",
        ),
    ]


@pytest.fixture
def truth_sheet(stamp: ProvenanceStamp) -> TruthSheet:
    return TruthSheet(
        persona_id="p001",
        seed=7,
        style="balanced",
        demographics={"age": 34, "city": "Portland", "occupation": "administrator"},
        facts=make_facts(),
        bio="I am a hospital systems administrator in Portland. " * 4,
        provenance=stamp,
    )


@pytest.fixture
def transcript(stamp: ProvenanceStamp) -> Transcript:
    turns = [
        make_turn("t001", "interviewer", "What do you do for work?"),
        make_turn("t002", "user", "I am a hospital systems administrator.", disclosed=["f06"]),
    ]
    return Transcript(
        transcript_id="p001__s1__n000",
        persona_id="p001",
        script_hash="abc123",
        sim_seed=1,
        turns=turns,
        simulated_minutes=round(sum(t.word_count for t in turns) / 150.0, 4),
        provenance=stamp,
    )


@pytest.fixture
def extraction(stamp: ProvenanceStamp) -> Extraction:
    return Extraction(
        extraction_id="p001__s1__n000:v1",
        transcript_id="p001__s1__n000",
        tag="v1",
        prompt_hash="p" * 8,
        schema_hash="s" * 8,
        model="fake-model",
        status="ok",
        profile={"occupation": "hospital systems administrator"},
        claims=[Claim(claim_id="c01", field="occupation", value="hospital systems administrator")],
        raw='{"occupation": "hospital systems administrator"}',
        provenance=stamp,
    )


@pytest.fixture
def supported_alignment() -> Alignment:
    return Alignment(
        claim_id="c01",
        verdict="supported",
        method="judge",
        matched_fact_id="f06",
        span_turn_id="t002",
        span_text="hospital systems administrator",
        specificity="specific",
    )


@pytest.fixture
def coverage() -> FactCoverage:
    return FactCoverage(
        fact_id="f06",
        captured=True,
        by_claim_id="c01",
        disclosed_turn_id="t002",
        question_id="q01",
        via_followup=False,
    )

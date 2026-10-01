"""Hand-built artifacts with a known answer for every metric key.

The fixture below is one persona scored end to end: twelve claims covering every
verdict including both error states, eight facts covering every disclosure type,
and one fact the interview never elicited. Every expected number in this file is
written as the arithmetic it comes from, so a changed denominator shows up as a
failure rather than as a new number nobody checked.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from onboarding_lab import metrics_schema
from onboarding_lab.models import (
    DEALBREAKER_FIELDS,
    EXACT_FIELDS,
    SCORED_FIELDS,
    Alignment,
    Claim,
    Extraction,
    Fact,
    FactCoverage,
    JudgeHealth,
    ProvenanceStamp,
    Scores,
    Transcript,
    TruthSheet,
    Turn,
    word_count,
)
from onboarding_lab.score import metrics
from onboarding_lab.score.stats import count, rate, ratio

STAMP = ProvenanceStamp(
    model="fake-model",
    prompt_hash="deadbeef",
    params={"effort": "low"},
    code_version="dev",
    created_at=datetime(2026, 9, 30, 12, 0, tzinfo=UTC),
)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def fact(
    fact_id: str,
    field: str,
    value: str | bool | int,
    *,
    specificity: str = "generic",
    disclosure: str = "volunteer",
    importance: str = "core",
) -> Fact:
    return Fact(
        fact_id=fact_id,
        field=field,
        value=value,
        specificity=specificity,
        disclosure=disclosure,
        importance=importance,
    )


def turn(
    turn_id: str,
    speaker: str,
    text: str,
    *,
    question_id: str | None,
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


def transcript(turns: list[Turn], **kwargs: object) -> Transcript:
    return Transcript(
        transcript_id=str(kwargs.pop("transcript_id", "p001__s1__n000")),
        persona_id=str(kwargs.pop("persona_id", "p001")),
        script_hash="abc123",
        sim_seed=1,
        turns=turns,
        simulated_minutes=round(sum(t.word_count for t in turns) / 150.0, 4),
        provenance=STAMP,
        **kwargs,  # type: ignore[arg-type]
    )


def sheet(facts: list[Fact], *, style: str = "terse", persona_id: str = "p001") -> TruthSheet:
    return TruthSheet(
        persona_id=persona_id,
        seed=7,
        style=style,
        demographics={"age": 34},
        facts=facts,
        bio="bio",
        provenance=STAMP,
    )


def exact(claim_id: str, verdict: str, fact_id: str | None = None) -> Alignment:
    return Alignment(claim_id=claim_id, verdict=verdict, method="exact", matched_fact_id=fact_id)


def judged(
    claim_id: str,
    verdict: str,
    *,
    fact_id: str | None = None,
    span_turn_id: str | None = None,
    span_text: str | None = None,
) -> Alignment:
    return Alignment(
        claim_id=claim_id,
        verdict=verdict,
        method="judge",
        matched_fact_id=fact_id,
        span_turn_id=span_turn_id,
        span_text=span_text,
        specificity="specific" if span_turn_id else None,
    )


# --------------------------------------------------------------------------- #
# The scored persona
# --------------------------------------------------------------------------- #

#: ``f06`` carries ``importance="dealbreaker"`` on a field that is **not** a
#: dealbreaker field, and ``f08`` the reverse. That is what makes the M14 cut
#: visible: both dealbreaker metrics must follow ``DEALBREAKER_FIELDS``.
FACTS = [
    fact("f01", "age", 34, specificity="specific"),
    fact("f02", "relationship_goal", "long_term"),
    fact("f03", "wants_kids", "yes", disclosure="needs_followup"),
    fact("f04", "has_kids", False, importance="dealbreaker"),
    fact("f05", "religion_importance", "low", disclosure="hedge", importance="dealbreaker"),
    fact(
        "f06",
        "occupation",
        "hospital systems administrator",
        specificity="specific",
        importance="dealbreaker",
    ),
    fact(
        "f07",
        "interests",
        "restoring a 1972 motorcycle",
        specificity="specific",
        disclosure="needs_followup",
        importance="color",
    ),
    fact("f08", "dealbreakers", "smoking", importance="color"),
]

TURNS = [
    turn("t001", "interviewer", "What do you do for work?", question_id="q01"),
    turn(
        "t002",
        "user",
        "I am a hospital systems administrator.",
        question_id="q01",
        disclosed=["f06"],
    ),
    turn(
        "t003",
        "interviewer",
        "Tell me about your interests and what you are looking for.",
        question_id="q02",
    ),
    turn(
        "t004",
        "user",
        "I am thirty four and looking for something long term with no kids yet "
        "and I cannot stand smoking.",
        question_id="q02",
        disclosed=["f01", "f02", "f04", "f08"],
    ),
    turn(
        "t005",
        "interviewer",
        "What have you been up to lately?",
        question_id="q02",
        is_followup=True,
    ),
    turn(
        "t006",
        "user",
        "Lately I have been restoring a 1972 motorcycle in the garage and I do want kids.",
        question_id="q02",
        is_followup=True,
        disclosed=["f03", "f07"],
    ),
]

CLAIMS = [
    Claim(claim_id="c01", field="occupation", value="hospital systems administrator"),
    Claim(claim_id="c02", field="interests", value="restoring a 1972 motorcycle"),
    Claim(claim_id="c03", field="age", value=34),
    Claim(claim_id="c04", field="relationship_goal", value="long_term"),
    Claim(claim_id="c05", field="wants_kids", value="unsure"),
    Claim(claim_id="c06", field="has_kids", value=True),
    Claim(claim_id="c07", field="religion_importance", value="high"),
    Claim(claim_id="c08", field="location", value="Portland"),
    Claim(claim_id="c09", field="values", value="honesty"),
    Claim(claim_id="c10", field="interests", value="jazz"),
    Claim(claim_id="c11", field="dealbreakers", value="smoking"),
    Claim(claim_id="c12", field="communication_style", value="direct"),
]

ALIGNMENTS = [
    judged("c01", "supported", fact_id="f06", span_turn_id="t002", span_text="administrator"),
    judged("c02", "supported", fact_id="f07", span_turn_id="t006", span_text="motorcycle"),
    exact("c03", "supported", "f01"),
    exact("c04", "supported", "f02"),
    # The model answered "unsure" where the truth is "yes": a declined answer,
    # not a hallucination (PLAN.md M4).
    exact("c05", "abstained"),
    exact("c06", "unsupported"),
    exact("c07", "unsupported"),
    judged("c08", "judge_error"),
    judged("c09", "span_invalid"),
    judged("c10", "supported_offsheet", span_turn_id="t004", span_text="jazz"),
    judged("c11", "supported", fact_id="f08", span_turn_id="t004", span_text="smoking"),
    judged("c12", "unsupported"),
]

COVERAGE = [
    FactCoverage(
        fact_id="f01",
        captured=True,
        by_claim_id="c03",
        disclosed_turn_id="t004",
        question_id="q02",
        via_followup=False,
    ),
    FactCoverage(
        fact_id="f02",
        captured=True,
        by_claim_id="c04",
        disclosed_turn_id="t004",
        question_id="q02",
        via_followup=False,
    ),
    FactCoverage(
        fact_id="f03",
        captured=False,
        disclosed_turn_id="t006",
        question_id="q02",
        via_followup=True,
    ),
    FactCoverage(
        fact_id="f04",
        captured=False,
        disclosed_turn_id="t004",
        question_id="q02",
        via_followup=False,
    ),
    # Never elicited: no follow-up probed it, so extraction could not capture it.
    FactCoverage(fact_id="f05", captured=False),
    FactCoverage(
        fact_id="f06",
        captured=True,
        by_claim_id="c01",
        disclosed_turn_id="t002",
        question_id="q01",
        via_followup=False,
    ),
    FactCoverage(
        fact_id="f07",
        captured=True,
        by_claim_id="c02",
        disclosed_turn_id="t006",
        question_id="q02",
        via_followup=True,
    ),
    FactCoverage(
        fact_id="f08",
        captured=True,
        by_claim_id="c11",
        disclosed_turn_id="t004",
        question_id="q02",
        via_followup=False,
    ),
]

JUDGE = JudgeHealth(calls=2, judged_claims=7, errors=1, span_invalid=1, span_invalid_first_pass=2)

#: ``q01`` is one interviewer turn plus one user turn; ``q02`` adds a follow-up
#: pair that carries the parent question id.
WORDS_Q01 = 6 + 6
WORDS_Q02 = 11 + 19 + 7 + 16


@pytest.fixture
def truth() -> TruthSheet:
    return sheet(FACTS)


@pytest.fixture
def clean() -> Transcript:
    return transcript(TURNS)


@pytest.fixture
def extraction() -> Extraction:
    return Extraction(
        extraction_id="p001__s1__n000:v1",
        transcript_id="p001__s1__n000",
        tag="v1",
        prompt_hash="p" * 8,
        schema_hash="s" * 8,
        model="fake-model",
        status="ok",
        profile={"occupation": "hospital systems administrator"},
        claims=CLAIMS,
        raw="{}",
        provenance=STAMP,
    )


@pytest.fixture
def scores(truth: TruthSheet, clean: Transcript, extraction: Extraction) -> Scores:
    return metrics.score_extraction(
        truth=truth,
        transcript=clean,
        extraction=extraction,
        alignments=ALIGNMENTS,
        coverage=COVERAGE,
        judge=JUDGE,
        claims_deduped=1,
    )


# --------------------------------------------------------------------------- #
# Precision family
# --------------------------------------------------------------------------- #


def test_the_word_counts_in_this_fixture_are_what_the_assertions_assume() -> None:
    by_question: dict[str, int] = {}
    for t in TURNS:
        assert t.question_id is not None
        by_question[t.question_id] = by_question.get(t.question_id, 0) + t.word_count
    assert by_question == {"q01": WORDS_Q01, "q02": WORDS_Q02}


def test_precision_excludes_both_error_states_and_abstentions(scores: Scores) -> None:
    """12 claims, minus one judge_error, one span_invalid and one abstention."""
    assert scores.n["precision"] == 9
    assert scores.metrics["precision"] == pytest.approx(6 / 9)
    assert scores.metrics["hallucination_rate"] == pytest.approx(3 / 9)
    assert scores.n["hallucination_rate"] == 9


def test_abstention_keeps_itself_in_its_own_denominator(scores: Scores) -> None:
    """Otherwise the rate could exceed 1."""
    assert scores.n["abstention_rate"] == 10
    assert scores.metrics["abstention_rate"] == pytest.approx(1 / 10)


def test_offsheet_rate_is_over_judged_claims_the_judge_actually_scored(
    scores: Scores,
) -> None:
    """Same cut as precision: seven judged claims, minus the two error states."""
    assert scores.n["offsheet_rate"] == 5
    assert scores.metrics["offsheet_rate"] == pytest.approx(1 / 5)


def test_a_field_where_everything_was_wrong_reports_a_real_zero(scores: Scores) -> None:
    """0.0 with a denominator is a finding; 0.0 with none is the bug (M12)."""
    assert scores.metrics["precision_by_field.communication_style"] == 0.0
    assert scores.n["precision_by_field.communication_style"] == 1


def test_judge_health_is_over_judged_claims_not_all_claims(scores: Scores) -> None:
    """A denominator including exact claims the judge never saw would deflate it
    and weaken the 5% gate (PLAN.md M11)."""
    assert scores.n["judge_error_rate"] == 7
    assert scores.metrics["judge_error_rate"] == pytest.approx(2 / 7)
    assert JUDGE.error_rate == pytest.approx(2 / 7)


def test_both_span_invalid_rates_are_reported(scores: Scores) -> None:
    """The aligner retries once, so the first pass and the final differ (M5)."""
    assert scores.metrics["span_invalid_rate_first_pass"] == pytest.approx(2 / 7)
    assert scores.metrics["span_invalid_rate_final"] == pytest.approx(1 / 7)


# --------------------------------------------------------------------------- #
# Errors are never verdicts
# --------------------------------------------------------------------------- #


def test_error_verdicts_change_no_rate_denominator(truth: TruthSheet, clean: Transcript) -> None:
    """The regression test for the bug class this project is defined against.

    Adding a judge error and an invalid span must move nothing: not a rate, not
    a denominator. They are states, counted in judge health only.
    """
    base = metrics.score_extraction(
        truth=truth,
        transcript=clean,
        extraction=_extraction(CLAIMS),
        alignments=ALIGNMENTS,
        coverage=COVERAGE,
        judge=JUDGE,
        claims_deduped=1,
    )
    extra_claims = [
        *CLAIMS,
        Claim(claim_id="c13", field="location", value="Seattle"),
        Claim(claim_id="c14", field="values", value="loyalty"),
    ]
    noisier = metrics.score_extraction(
        truth=truth,
        transcript=clean,
        extraction=_extraction(extra_claims),
        alignments=[*ALIGNMENTS, judged("c13", "judge_error"), judged("c14", "span_invalid")],
        coverage=COVERAGE,
        judge=JUDGE,
        claims_deduped=1,
    )
    assert noisier.n == base.n
    assert noisier.metrics == base.metrics


def _extraction(claims: list[Claim]) -> Extraction:
    return Extraction(
        extraction_id="p001__s1__n000:v1",
        transcript_id="p001__s1__n000",
        tag="v1",
        prompt_hash="p" * 8,
        schema_hash="s" * 8,
        model="fake-model",
        status="ok",
        profile={},
        claims=claims,
        raw="{}",
        provenance=STAMP,
    )


# --------------------------------------------------------------------------- #
# Recall family
# --------------------------------------------------------------------------- #


def test_recall_and_elicited_recall_differ_on_the_fact_nobody_asked_about(
    scores: Scores,
) -> None:
    """The stage attribution the harness exists to provide (PLAN.md M6)."""
    assert scores.n["recall"] == 8
    assert scores.metrics["recall"] == pytest.approx(5 / 8)
    assert scores.n["recall_elicited"] == 7
    assert scores.metrics["recall_elicited"] == pytest.approx(5 / 7)
    assert scores.metrics["recall"] < scores.metrics["recall_elicited"]


def test_an_abstention_counts_as_not_captured(scores: Scores) -> None:
    assert scores.metrics["recall_by_field.wants_kids"] == 0.0
    assert scores.n["recall_by_field.wants_kids"] == 1


def test_a_capture_citing_an_abstained_claim_is_not_a_capture(
    truth: TruthSheet, clean: Transcript, extraction: Extraction
) -> None:
    """The metrics layer re-reads the verdict rather than trusting ``captured``."""
    coverage = [
        FactCoverage(
            fact_id="f03",
            captured=True,
            by_claim_id="c05",
            disclosed_turn_id="t006",
            question_id="q02",
            via_followup=True,
        ),
        *[c for c in COVERAGE if c.fact_id != "f03"],
    ]
    scored = metrics.score_extraction(
        truth=truth,
        transcript=clean,
        extraction=extraction,
        alignments=ALIGNMENTS,
        coverage=coverage,
        judge=JUDGE,
    )
    assert scored.metrics["recall"] == pytest.approx(5 / 8)
    assert scored.metrics["recall_by_field.wants_kids"] == 0.0


def test_recall_by_disclosure_covers_every_type_in_the_sheet(scores: Scores) -> None:
    assert scores.metrics["recall_by_disclosure.volunteer"] == pytest.approx(4 / 5)
    assert scores.metrics["recall_by_disclosure.needs_followup"] == pytest.approx(1 / 2)
    assert scores.metrics["recall_by_disclosure.hedge"] == 0.0
    assert scores.n["recall_by_disclosure.hedge"] == 1


def test_exact_field_recall_and_its_per_field_breakdown(scores: Scores) -> None:
    assert scores.metrics["recall_exact"] == pytest.approx(2 / 5)
    assert scores.n["recall_exact"] == len(EXACT_FIELDS)
    assert scores.metrics["recall_exact_by_field.age"] == 1.0
    assert scores.metrics["recall_exact_by_field.has_kids"] == 0.0


# --------------------------------------------------------------------------- #
# Dealbreakers cut the same population (PLAN.md M14)
# --------------------------------------------------------------------------- #


def test_both_dealbreaker_metrics_cut_by_field_not_by_importance(
    scores: Scores,
) -> None:
    by_field = [f for f in FACTS if f.field in DEALBREAKER_FIELDS]
    by_importance = [f for f in FACTS if f.importance == "dealbreaker"]
    assert {f.fact_id for f in by_field} != {f.fact_id for f in by_importance}

    assert scores.n["recall_dealbreakers"] == len(by_field) == 4
    assert scores.metrics["recall_dealbreakers"] == pytest.approx(1 / 4)

    scorable_dealbreaker_claims = [
        c for c in CLAIMS if c.field in DEALBREAKER_FIELDS and c.claim_id not in {"c05"}
    ]
    assert scores.n["precision_dealbreakers"] == len(scorable_dealbreaker_claims) == 3
    assert scores.metrics["precision_dealbreakers"] == pytest.approx(1 / 3)


def test_the_dealbreaker_fact_on_a_non_dealbreaker_field_is_in_neither(
    scores: Scores,
) -> None:
    """``f06``/``c01`` is occupation: excluded from both despite its importance."""
    assert "occupation" not in DEALBREAKER_FIELDS
    assert scores.metrics["recall_by_field.occupation"] == 1.0
    assert scores.metrics["recall_dealbreakers"] == pytest.approx(1 / 4)


# --------------------------------------------------------------------------- #
# Zero denominators
# --------------------------------------------------------------------------- #


def test_a_zero_denominator_rate_is_absent_with_zero_in_n(scores: Scores) -> None:
    """Never 0.0, never NaN (PLAN.md M12, CLAUDE.md rule 6)."""
    for key in (
        "precision_by_field.wants_kids",
        "precision_by_field.location",
        "precision_by_field.values",
        "recall_by_field.life_events",
        "recall_by_field.partner_preferences",
    ):
        assert key not in scores.metrics, f"{key} must be omitted, not reported as 0.0"
        assert scores.n[key] == 0


def test_every_reported_rate_has_a_non_zero_denominator(scores: Scores) -> None:
    for key, value in scores.metrics.items():
        if metrics_schema.needs_denominator(key):
            assert scores.n[key] > 0, key
            assert value == value, f"{key} is NaN"


def test_a_failed_extraction_scores_zero_recall_and_no_precision(
    truth: TruthSheet, clean: Transcript
) -> None:
    """``schema_invalid`` personas are excluded from precision denominators and
    counted as zero recall (PLAN.md M15)."""
    broken = Extraction(
        extraction_id="p001__s1__n000:v1",
        transcript_id="p001__s1__n000",
        tag="v1",
        prompt_hash="p" * 8,
        schema_hash="s" * 8,
        model="fake-model",
        status="schema_invalid",
        profile=None,
        claims=[],
        raw="not json",
        provenance=STAMP,
    )
    scored = metrics.score_extraction(
        truth=truth,
        transcript=clean,
        extraction=broken,
        alignments=[],
        coverage=[],
        judge=JudgeHealth(calls=0, judged_claims=0, errors=0, span_invalid=0),
    )
    assert "precision" not in scored.metrics
    assert scored.n["precision"] == 0
    assert scored.metrics["recall"] == 0.0
    assert scored.n["recall"] == 8


# --------------------------------------------------------------------------- #
# Yield and follow-up share
# --------------------------------------------------------------------------- #


def test_yield_counts_captured_facts_per_question(scores: Scores) -> None:
    assert scores.metrics["yield.q01"] == 1.0
    assert scores.metrics["yield.q02"] == 4.0
    assert "yield.q01" not in scores.n, "a count carries no denominator"


def test_yield_per_minute_includes_interviewer_words(scores: Scores) -> None:
    """It measures interview time, and a verbose question costs minutes (M15)."""
    assert scores.n["yield_per_minute.q01"] == WORDS_Q01
    user_words_only = next(t.word_count for t in TURNS if t.turn_id == "t002")
    assert scores.n["yield_per_minute.q01"] != user_words_only
    assert scores.metrics["yield_per_minute.q01"] == pytest.approx(1 * 150 / WORDS_Q01)
    assert scores.metrics["yield_per_minute.q02"] == pytest.approx(4 * 150 / WORDS_Q02)


def test_a_follow_up_turn_folds_into_its_parent_question(scores: Scores) -> None:
    """Follow-ups carry the parent ``question_id``; there is no second bucket."""
    assert {k for k in scores.metrics if k.startswith("yield.")} == {"yield.q01", "yield.q02"}
    assert scores.n["yield_per_minute.q02"] == WORDS_Q02


def test_followup_share_is_broken_down_by_style(scores: Scores) -> None:
    """The headline is largely an artifact of the sampled style mix (M15)."""
    assert scores.n["followup_share"] == 5
    assert scores.metrics["followup_share"] == pytest.approx(1 / 5)
    assert scores.metrics["followup_share_by_style.terse"] == pytest.approx(1 / 5)
    assert "followup_share_by_style.rambling" not in scores.metrics


def test_time_metrics_read_the_clean_source_not_the_noised_copy(
    truth: TruthSheet, clean: Transcript, extraction: Extraction
) -> None:
    """Noise drops words and inserts fillers; yield-per-minute must not move for
    reasons unrelated to extraction (PLAN.md M8)."""
    noised_turns = [
        turn(
            t.turn_id,
            t.speaker,
            t.text + " um like you know" if t.speaker == "user" else t.text,
            question_id=t.question_id,
            is_followup=t.is_followup,
            disclosed=list(t.disclosed_fact_ids),
        )
        for t in TURNS
    ]
    noised = transcript(
        noised_turns,
        transcript_id="p001__s1__n010",
        noise_rate=0.1,
        noise_seed=3,
        source_transcript_id=clean.transcript_id,
    )
    assert sum(t.word_count for t in noised.turns) != sum(t.word_count for t in clean.turns)
    scored = metrics.score_extraction(
        truth=truth,
        transcript=noised,
        extraction=extraction,
        alignments=ALIGNMENTS,
        coverage=COVERAGE,
        judge=JUDGE,
        clean_transcript=clean,
    )
    assert scored.n["yield_per_minute.q01"] == WORDS_Q01
    assert scored.metrics["yield_per_minute.q02"] == pytest.approx(4 * 150 / WORDS_Q02)


def test_scoring_a_noised_transcript_without_its_source_is_refused(
    truth: TruthSheet, clean: Transcript, extraction: Extraction
) -> None:
    noised = transcript(
        list(TURNS),
        transcript_id="p001__s1__n010",
        noise_rate=0.1,
        noise_seed=3,
        source_transcript_id=clean.transcript_id,
    )
    with pytest.raises(ValueError, match="pass its clean source"):
        metrics.score_extraction(
            truth=truth,
            transcript=noised,
            extraction=extraction,
            alignments=ALIGNMENTS,
            coverage=COVERAGE,
            judge=JUDGE,
        )


# --------------------------------------------------------------------------- #
# The disclosure checks feed back into yield and follow-up share (M10)
# --------------------------------------------------------------------------- #


def test_an_untagged_capture_leaves_both_yield_and_followup_share(
    truth: TruthSheet, extraction: Extraction, scores: Scores
) -> None:
    """As specified these drop out of ``yield`` but stay in the
    ``followup_share`` denominator; they are excluded from both.

    Here the tag for ``f06`` moves to a later turn than the span the judge
    cited, so the capture precedes the first tagged turn.
    """
    turns = [
        turn("t001", "interviewer", TURNS[0].text, question_id="q01"),
        turn("t002", "user", TURNS[1].text, question_id="q01"),
        turn("t003", "interviewer", TURNS[2].text, question_id="q02"),
        turn(
            "t004",
            "user",
            TURNS[3].text,
            question_id="q02",
            disclosed=["f01", "f02", "f04", "f06", "f08"],
        ),
        turn("t005", "interviewer", TURNS[4].text, question_id="q02", is_followup=True),
        turn(
            "t006",
            "user",
            TURNS[5].text,
            question_id="q02",
            is_followup=True,
            disclosed=["f03", "f07"],
        ),
    ]
    coverage = [
        FactCoverage(
            fact_id="f06",
            captured=True,
            by_claim_id="c01",
            disclosed_turn_id="t004",
            question_id="q02",
            via_followup=False,
        )
        if c.fact_id == "f06"
        else c
        for c in COVERAGE
    ]
    scored = metrics.score_extraction(
        truth=truth,
        transcript=transcript(turns),
        extraction=extraction,
        alignments=ALIGNMENTS,
        coverage=coverage,
        judge=JUDGE,
    )
    assert scored.metrics["untagged_capture"] == 1.0
    # f06 is still a capture for recall; it is only uncreditable to a question.
    assert scored.metrics["recall"] == pytest.approx(5 / 8)
    assert scored.metrics["yield.q01"] == 0.0
    assert scored.metrics["yield.q02"] == 4.0
    assert scored.n["followup_share"] == 4, "excluded from the denominator, not just the top"
    assert scored.metrics["followup_share"] == pytest.approx(1 / 4)


def test_the_disclosure_mismatch_rate_is_reported_with_its_n(scores: Scores) -> None:
    """Two judged-field ``specific`` facts are tagged; both left a trace."""
    assert scores.n["disclosure_mismatch"] == 2
    assert scores.metrics["disclosure_mismatch"] == 0.0


# --------------------------------------------------------------------------- #
# Joins
# --------------------------------------------------------------------------- #


def test_an_alignment_for_an_unknown_claim_is_an_error(
    truth: TruthSheet, clean: Transcript, extraction: Extraction
) -> None:
    with pytest.raises(ValueError, match="unknown claims"):
        metrics.score_extraction(
            truth=truth,
            transcript=clean,
            extraction=extraction,
            alignments=[*ALIGNMENTS, exact("c99", "supported", "f01")],
            coverage=COVERAGE,
            judge=JUDGE,
        )


def test_a_claim_with_no_alignment_is_an_error(
    truth: TruthSheet, clean: Transcript, extraction: Extraction
) -> None:
    with pytest.raises(ValueError, match="no alignment"):
        metrics.score_extraction(
            truth=truth,
            transcript=clean,
            extraction=extraction,
            alignments=ALIGNMENTS[:-1],
            coverage=COVERAGE,
            judge=JUDGE,
        )


def test_coverage_for_a_fact_outside_the_truth_sheet_is_an_error(
    truth: TruthSheet, clean: Transcript, extraction: Extraction
) -> None:
    with pytest.raises(ValueError, match="absent from truth sheet"):
        metrics.score_extraction(
            truth=truth,
            transcript=clean,
            extraction=extraction,
            alignments=ALIGNMENTS,
            coverage=[*COVERAGE, FactCoverage(fact_id="f99", captured=False)],
            judge=JUDGE,
        )


def test_every_emitted_key_is_in_the_registry(scores: Scores) -> None:
    for key in list(scores.metrics) + list(scores.n):
        assert metrics_schema.is_known(key), key


# --------------------------------------------------------------------------- #
# The exact-field majority-class floor (PLAN.md M9)
# --------------------------------------------------------------------------- #

FLOOR_SHEETS = [
    sheet(
        [
            fact("a1", "age", age),
            fact("a2", "relationship_goal", goal),
            fact("a3", "wants_kids", "yes"),
            fact("a4", "has_kids", kids),
            fact("a5", "religion_importance", "low"),
        ],
        persona_id=f"p{i:03d}",
    )
    for i, (age, goal, kids) in enumerate(
        [
            (34, "long_term", True),
            (34, "long_term", True),
            (35, "casual", True),
            (36, "marriage", False),
        ]
    )
]


def test_the_floor_is_the_majority_class_share_of_the_runs_own_sheets() -> None:
    floor = metrics.exact_field_floor(FLOOR_SHEETS)
    assert floor["exact_floor.has_kids"].numerator == 3
    assert floor["exact_floor.has_kids"].denominator == 4
    assert floor["exact_floor.relationship_goal"].value == pytest.approx(2 / 4)
    assert floor["exact_floor.age"].value == pytest.approx(2 / 4)


def test_a_field_with_one_class_floors_at_one() -> None:
    """A constant guesser scores perfectly, so the recall number means nothing."""
    floor = metrics.exact_field_floor(FLOOR_SHEETS)
    assert floor["exact_floor.wants_kids"].value == 1.0
    assert floor["exact_floor.religion_importance"].value == 1.0


def test_the_floor_distinguishes_a_boolean_from_its_string() -> None:
    """``True`` and ``"True"`` are different classes, never silently merged."""
    sheets = [
        sheet(
            [
                fact("a1", "age", 34),
                fact("a2", "relationship_goal", "casual"),
                fact("a3", "wants_kids", "yes"),
                fact("a4", "has_kids", value),
                fact("a5", "religion_importance", "low"),
            ],
            persona_id=f"q{i:03d}",
        )
        for i, value in enumerate([True, False])
    ]
    assert metrics.exact_field_floor(sheets)["exact_floor.has_kids"].value == 0.5


def test_no_sheets_means_no_floor_rather_than_zero() -> None:
    floor = metrics.exact_field_floor([])
    assert floor["exact_floor.has_kids"].value is None


def test_lift_over_floor_is_absent_where_either_side_is() -> None:
    lift = metrics.lift_over_floor(
        {"recall_exact_by_field.has_kids": 0.9, "exact_floor.has_kids": 0.6}
    )
    assert lift["has_kids"] == pytest.approx(0.3)
    assert lift["age"] is None


# --------------------------------------------------------------------------- #
# Aggregation (PLAN.md M13)
# --------------------------------------------------------------------------- #


def _persona_scores(
    extraction_id: str,
    metric_tallies: dict[str, object],
    judge: JudgeHealth | None = None,
) -> Scores:
    values: dict[str, float] = {}
    denominators: dict[str, int] = {}
    for key, tally in metric_tallies.items():
        if tally.kind != "count":  # type: ignore[attr-defined]
            denominators[key] = tally.denominator  # type: ignore[attr-defined]
        value = tally.value  # type: ignore[attr-defined]
        if value is not None:
            values[key] = value
    return Scores(
        extraction_id=extraction_id,
        alignments=[],
        coverage=[],
        judge=judge or JudgeHealth(calls=1, judged_claims=0, errors=0, span_invalid=0),
        metrics=values,
        n=denominators,
    )


def test_aggregation_pools_counts_rather_than_averaging_ratios() -> None:
    """A persona with one claim must not weigh like one with nine (M13)."""
    thin = _persona_scores("p001:v1", {"precision": rate(0, 1)})
    fat = _persona_scores("p002:v1", {"precision": rate(9, 9)})
    agg = metrics.aggregate([thin, fat])
    assert agg.metrics["precision"] == pytest.approx(0.9)
    assert agg.n["precision"] == 10
    assert agg.per_persona["precision"] == [0.0, 1.0]
    assert agg.metrics["precision"] != pytest.approx(0.5), "that would be a mean of ratios"


def test_the_aggregate_keeps_the_per_persona_vector_and_its_ids() -> None:
    a = _persona_scores("p001:v1", {"precision": rate(1, 2)})
    b = _persona_scores("p002:v1", {"precision": rate(3, 4)})
    agg = metrics.aggregate([a, b])
    assert agg.extraction_ids == ("p001:v1", "p002:v1")
    assert agg.per_persona["precision"] == [0.5, 0.75]
    assert agg.n_personas == 2


def test_a_key_one_persona_omitted_still_pools_from_the_other() -> None:
    a = _persona_scores("p001:v1", {"recall_by_field.values": rate(0, 0)})
    b = _persona_scores("p002:v1", {"recall_by_field.values": rate(2, 2)})
    agg = metrics.aggregate([a, b])
    assert agg.metrics["recall_by_field.values"] == 1.0
    assert agg.n["recall_by_field.values"] == 2
    assert agg.per_persona["recall_by_field.values"] == [None, 1.0]


def test_wilson_is_printed_for_proportions_and_withheld_from_ratios() -> None:
    scores = _persona_scores(
        "p001:v1",
        {
            "precision": rate(8, 10),
            "yield_per_minute.q01": ratio(1 * 150, 12),
            "claims_deduped": count(2),
        },
    )
    agg = metrics.aggregate([scores])
    assert agg.ci["precision"] is not None
    assert agg.ci["yield_per_minute.q01"] is None, "Wilson needs a proportion"
    assert agg.ci["claims_deduped"] is None
    low, high = agg.ci["precision"]
    assert low < agg.metrics["precision"] < high


def test_a_ratio_pools_as_numerator_sum_over_denominator_sum() -> None:
    """Facts per minute: total facts over total minutes, not a mean of rates."""
    a = _persona_scores("p001:v1", {"yield_per_minute.q01": ratio(1 * 150, 12)})
    b = _persona_scores("p002:v1", {"yield_per_minute.q01": ratio(2 * 150, 30)})
    agg = metrics.aggregate([a, b])
    assert agg.n["yield_per_minute.q01"] == 42
    assert agg.metrics["yield_per_minute.q01"] == pytest.approx(3 * 150 / 42)


def test_counts_sum() -> None:
    a = _persona_scores("p001:v1", {"claims_deduped": count(2)})
    b = _persona_scores("p002:v1", {"claims_deduped": count(3)})
    agg = metrics.aggregate([a, b])
    assert agg.metrics["claims_deduped"] == 5.0
    assert "claims_deduped" not in agg.n


def test_degraded_trips_on_pooled_counts_not_averaged_booleans() -> None:
    """PLAN.md M11. The mean of the per-persona rates is below the threshold and
    only one persona is individually degraded; the pooled rate is above it."""
    clean_persona = _persona_scores(
        "p001:v1",
        {"judge_error_rate": rate(0, 10)},
        judge=JudgeHealth(calls=1, judged_claims=10, errors=0, span_invalid=0),
    )
    bad_persona = _persona_scores(
        "p002:v1",
        {"judge_error_rate": rate(6, 90)},
        judge=JudgeHealth(calls=1, judged_claims=90, errors=6, span_invalid=0),
    )
    per_persona_mean = (0 / 10 + 6 / 90) / 2
    assert per_persona_mean <= 0.05
    assert [clean_persona.judge.degraded, bad_persona.judge.degraded] == [False, True]

    agg = metrics.aggregate([clean_persona, bad_persona])
    assert agg.judge.judged_claims == 100
    assert agg.judge.errors == 6
    assert agg.metrics["judge_error_rate"] == pytest.approx(0.06)
    assert agg.judge.error_rate == pytest.approx(0.06)
    assert agg.judge.degraded


def test_the_aggregate_carries_the_full_expected_key_set() -> None:
    """Per-persona files legitimately omit keys; the report never has a hole."""
    expected = metrics_schema.expected_keys(
        fields=SCORED_FIELDS,
        exact_fields=EXACT_FIELDS,
        disclosures=("volunteer", "needs_followup", "hedge"),
        styles=("terse", "balanced", "rambling", "tangential"),
        question_ids=("q01", "q02"),
    )
    thin = _persona_scores("p001:v1", {"precision": rate(1, 2)})
    agg = metrics.aggregate([thin], truth_sheets=FLOOR_SHEETS, expected=expected)
    missing = expected - set(agg.metrics)
    assert not missing, f"aggregate is missing {sorted(missing)}"


def test_an_undefined_aggregate_is_null_never_zero() -> None:
    """``None`` in JSON is honest; ``0.0`` would read as a measured failure."""
    expected = frozenset({"precision", "recall_elicited"})
    thin = _persona_scores("p001:v1", {"precision": rate(1, 2)})
    agg = metrics.aggregate([thin], expected=expected)
    assert agg.metrics["recall_elicited"] is None
    assert agg.n["recall_elicited"] == 0
    assert agg.ci["recall_elicited"] is None


def test_the_floor_and_its_lift_appear_in_the_aggregate() -> None:
    thin = _persona_scores("p001:v1", {"recall_exact_by_field.has_kids": rate(1, 1)})
    agg = metrics.aggregate([thin], truth_sheets=FLOOR_SHEETS)
    assert agg.metrics["exact_floor.has_kids"] == pytest.approx(3 / 4)
    assert agg.lift_over_floor["has_kids"] == pytest.approx(1 - 3 / 4)


def test_excluded_personas_are_counted_for_the_report() -> None:
    """ "n personas excluded: schema_invalid" needs a number to print (M15)."""
    ok = _persona_scores("p001:v1", {"precision": rate(1, 2)})
    broken = _persona_scores("p002:v1", {"recall": rate(0, 8)})
    agg = metrics.aggregate([ok, broken], statuses={"p002:v1": "schema_invalid"})
    assert agg.excluded == {"schema_invalid": 1}
    assert agg.n_excluded == 1
    assert agg.to_dict()["excluded"] == {"schema_invalid": 1}


def test_aggregating_the_scored_persona_round_trips_its_numbers(
    scores: Scores,
) -> None:
    """``Scores`` stores quotients; pooling has to recover the parts exactly."""
    agg = metrics.aggregate([scores])
    assert agg.n["precision"] == scores.n["precision"]
    assert agg.metrics["precision"] == pytest.approx(scores.metrics["precision"])
    assert agg.metrics["yield.q02"] == 4.0
    assert agg.metrics["yield_per_minute.q02"] == pytest.approx(
        scores.metrics["yield_per_minute.q02"]
    )
    recovered = metrics.tallies_from_scores(scores)
    assert recovered["precision"].numerator == 6
    assert recovered["yield_per_minute.q02"].numerator == 4 * 150


def test_an_unknown_key_in_an_aggregate_is_an_error() -> None:
    with pytest.raises(ValueError, match="not in metrics_schema"):
        metrics.aggregate(
            [_persona_scores("p001:v1", {"precision": rate(1, 2)})], expected={"made_up"}
        )


def test_the_aggregate_serialises_without_losing_the_distinction() -> None:
    thin = _persona_scores("p001:v1", {"precision": rate(1, 2)})
    payload = metrics.aggregate([thin], expected={"precision", "recall"}).to_dict()
    assert payload["metrics"]["precision"] == 0.5
    assert payload["metrics"]["recall"] is None
    assert payload["n"]["recall"] == 0
    assert payload["judge"]["degraded"] is False
    assert payload["per_persona"] == [
        {
            "persona_id": "p001:v1",
            "extraction_id": "p001:v1",
            "metrics": {"precision": 0.5},
            "n": {"precision": 2},
        }
    ]


def test_the_persona_records_name_the_persona_not_just_the_extraction() -> None:
    """The report's distribution table is per persona; the id has to be in it."""
    thin = _persona_scores("p001__s1__n000:v1", {"precision": rate(1, 2)})
    agg = metrics.aggregate([thin], persona_ids={"p001__s1__n000:v1": "p001"})
    assert agg.persona_ids == ("p001",)
    assert agg.persona_records()[0]["persona_id"] == "p001"


def test_a_persona_that_omitted_a_key_omits_it_from_its_record() -> None:
    """Rather than carrying a zero nobody measured."""
    a = _persona_scores("p001:v1", {"recall_by_field.values": rate(0, 0)})
    b = _persona_scores("p002:v1", {"recall_by_field.values": rate(2, 2)})
    records = metrics.aggregate([a, b]).persona_records()
    assert "recall_by_field.values" not in records[0]["metrics"]
    assert records[0]["n"]["recall_by_field.values"] == 0
    assert records[1]["metrics"]["recall_by_field.values"] == 1.0


def test_the_measured_floor_is_carried_through_for_the_report() -> None:
    """The gate is only as sound as the floor, so the floor gets printed (P5)."""
    from onboarding_lab.score import floor as fl

    floors = fl.compute_floor([{"precision": 0.80}, {"precision": 0.82}])
    thin = _persona_scores("p001:v1", {"precision": rate(1, 2)})
    payload = metrics.aggregate([thin], floors=floors).to_dict()
    assert payload["floor"]["precision"] == pytest.approx(floors["precision"].floor)
    assert metrics.aggregate([thin]).to_dict()["floor"] == {}

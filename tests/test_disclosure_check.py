"""The disclosure self-report audit (PLAN.md M10).

``disclosed_fact_ids`` is the simulator's own word and the ground truth behind
every recall and yield number, so these are the tests that keep it honest: one
fact stated verbatim, one paraphrased (which must **not** read as a mismatch),
and one that left no trace at all.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from onboarding_lab.models import (
    Alignment,
    Fact,
    FactCoverage,
    ProvenanceStamp,
    Transcript,
    TruthSheet,
    Turn,
    word_count,
)
from onboarding_lab.score import disclosure_check as dc

STAMP = ProvenanceStamp(
    model="fake-model",
    prompt_hash="deadbeef",
    params={"effort": "low"},
    code_version="dev",
    created_at=datetime(2026, 9, 30, 12, 0, tzinfo=UTC),
)

#: The five exact-scored fields every truth sheet must carry exactly once.
EXACT_BASE = (
    ("x1", "age", 34),
    ("x2", "relationship_goal", "long_term"),
    ("x3", "wants_kids", "yes"),
    ("x4", "has_kids", False),
    ("x5", "religion_importance", "low"),
)


def sheet(*facts: Fact, style: str = "balanced") -> TruthSheet:
    base = [
        Fact(
            fact_id=fid,
            field=field,
            value=value,
            specificity="generic",
            disclosure="volunteer",
            importance="core",
        )
        for fid, field, value in EXACT_BASE
    ]
    return TruthSheet(
        persona_id="p001",
        seed=7,
        style=style,
        demographics={"age": 34},
        facts=base + list(facts),
        bio="bio",
        provenance=STAMP,
    )


def judged_fact(fact_id: str, field: str, value: str, *, specificity: str = "specific") -> Fact:
    return Fact(
        fact_id=fact_id,
        field=field,
        value=value,
        specificity=specificity,
        disclosure="volunteer",
        importance="core",
    )


def turn(
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


def transcript(turns: list[Turn], **kwargs: object) -> Transcript:
    return Transcript(
        transcript_id=str(kwargs.pop("transcript_id", "p001__s1__n000")),
        persona_id="p001",
        script_hash="abc123",
        sim_seed=1,
        turns=turns,
        simulated_minutes=round(sum(t.word_count for t in turns) / 150.0, 4),
        provenance=STAMP,
        **kwargs,  # type: ignore[arg-type]
    )


# -- anchor tokens ------------------------------------------------------------


def test_anchors_pick_up_numbers_proper_nouns_and_content_words() -> None:
    assert dc.anchor_tokens("restoring a 1972 motorcycle") == frozenset(
        {"1972", "motorcycle", "restoring"}
    )


def test_a_short_proper_noun_anchors_despite_its_length() -> None:
    """The length rule alone would drop it, and a place name is strong evidence."""
    assert "rio" in dc.anchor_tokens("a summer in Rio")


def test_a_value_made_of_common_words_has_no_anchor() -> None:
    """Reported as unverifiable, never as a mismatch."""
    assert dc.anchor_tokens("a lot of time") == frozenset()
    assert dc.anchor_tokens(True) == frozenset()


def test_plurals_normalise_on_both_sides() -> None:
    assert dc.anchor_tokens("hospital systems") == dc.anchor_tokens("hospital system")


# -- disclosure_mismatch ------------------------------------------------------


def test_a_verbatim_disclosure_is_a_match() -> None:
    truth = sheet(judged_fact("f06", "occupation", "hospital systems administrator"))
    clean = transcript(
        [
            turn("t001", "interviewer", "What do you do for work?"),
            turn("t002", "user", "I am a hospital systems administrator.", disclosed=["f06"]),
        ]
    )
    report = dc.check_disclosure_mismatch(truth=truth, clean_transcript=clean)
    assert report.checked == ("f06",)
    assert report.mismatched == ()
    assert report.mismatch.value == 0.0


def test_a_paraphrase_is_not_a_mismatch() -> None:
    """The agent states facts in its own voice; anchors survive, wording does not."""
    truth = sheet(judged_fact("f07", "interests", "volunteers at a community garden on weekends"))
    clean = transcript(
        [
            turn("t001", "interviewer", "What do you do with your spare time?"),
            turn(
                "t002",
                "user",
                "I help out at the community garden most Saturdays.",
                disclosed=["f07"],
            ),
        ]
    )
    report = dc.check_disclosure_mismatch(truth=truth, clean_transcript=clean)
    assert report.checked == ("f07",)
    assert report.mismatched == ()


def test_a_fact_that_left_no_trace_is_a_mismatch() -> None:
    truth = sheet(judged_fact("f08", "location", "a suburb outside Lisbon"))
    clean = transcript(
        [
            turn("t001", "interviewer", "Where did you grow up?"),
            turn("t002", "user", "I moved around a lot when I was a kid.", disclosed=["f08"]),
        ]
    )
    report = dc.check_disclosure_mismatch(truth=truth, clean_transcript=clean)
    assert report.mismatched == ("f08",)
    assert report.mismatch.value == 1.0
    assert "f08" in report.notes


def test_all_three_together_report_one_mismatch_over_two_checked() -> None:
    truth = sheet(
        judged_fact("f06", "occupation", "hospital systems administrator"),
        judged_fact("f07", "interests", "volunteers at a community garden on weekends"),
        judged_fact("f08", "location", "a suburb outside Lisbon"),
    )
    clean = transcript(
        [
            turn("t001", "interviewer", "Tell me about yourself."),
            turn(
                "t002",
                "user",
                "I am a hospital systems administrator and I help out at the "
                "community garden most Saturdays.",
                disclosed=["f06", "f07"],
            ),
            turn("t003", "interviewer", "Where did you grow up?"),
            turn("t004", "user", "I moved around a lot when I was a kid.", disclosed=["f08"]),
        ]
    )
    report = dc.check_disclosure_mismatch(truth=truth, clean_transcript=clean)
    assert set(report.checked) == {"f06", "f07", "f08"}
    assert report.mismatched == ("f08",)
    assert report.mismatch.numerator == 1
    assert report.mismatch.denominator == 3


def test_enum_and_boolean_fields_are_not_checked() -> None:
    """An exact field's value cannot be looked for in prose."""
    truth = sheet()
    clean = transcript(
        [
            turn("t001", "interviewer", "Do you have children?"),
            turn("t002", "user", "No, not yet.", disclosed=["x4", "x3"]),
        ]
    )
    report = dc.check_disclosure_mismatch(truth=truth, clean_transcript=clean)
    assert report.checked == ()
    assert report.mismatch.value is None, "a zero denominator has no rate"


def test_a_generic_fact_is_not_checked() -> None:
    """``hedge`` facts are stated vaguely by design; checking them measures nothing."""
    truth = sheet(
        judged_fact("f09", "values", "family comes first", specificity="generic"),
    )
    clean = transcript(
        [
            turn("t001", "interviewer", "What matters to you?"),
            turn("t002", "user", "Hard to say really.", disclosed=["f09"]),
        ]
    )
    assert dc.check_disclosure_mismatch(truth=truth, clean_transcript=clean).checked == ()


def test_an_untagged_fact_has_no_self_report_to_audit() -> None:
    truth = sheet(judged_fact("f06", "occupation", "hospital systems administrator"))
    clean = transcript([turn("t001", "interviewer", "What do you do for work?")])
    report = dc.check_disclosure_mismatch(truth=truth, clean_transcript=clean)
    assert report.checked == ()
    assert report.mismatched == ()


def test_a_value_with_no_anchor_is_unverifiable_not_mismatched() -> None:
    truth = sheet(judged_fact("f10", "values", "a lot of time"))
    clean = transcript(
        [
            turn("t001", "interviewer", "What matters to you?"),
            turn("t002", "user", "Family and nothing else.", disclosed=["f10"]),
        ]
    )
    report = dc.check_disclosure_mismatch(truth=truth, clean_transcript=clean)
    assert report.unverifiable == ("f10",)
    assert report.checked == ()
    assert report.mismatched == ()


def test_the_noised_copy_is_refused() -> None:
    """ASR noise drops the very words being looked for and would inflate the rate."""
    truth = sheet(judged_fact("f06", "occupation", "hospital systems administrator"))
    noised = transcript(
        [turn("t002", "user", "I am a hosptal systems administrator.", disclosed=["f06"])],
        transcript_id="p001__s1__n010",
        noise_rate=0.1,
        noise_seed=3,
        source_transcript_id="p001__s1__n000",
    )
    with pytest.raises(ValueError, match="noised copy"):
        dc.check_disclosure_mismatch(truth=truth, clean_transcript=noised)


# -- untagged_capture ---------------------------------------------------------


def _captured(fact_id: str, claim_id: str) -> FactCoverage:
    return FactCoverage(
        fact_id=fact_id,
        captured=True,
        by_claim_id=claim_id,
        disclosed_turn_id=None,
        question_id="q01",
        via_followup=False,
    )


def _judge_supported(claim_id: str, fact_id: str, turn_id: str) -> Alignment:
    return Alignment(
        claim_id=claim_id,
        verdict="supported",
        method="judge",
        matched_fact_id=fact_id,
        span_turn_id=turn_id,
        span_text="something",
        specificity="specific",
    )


def test_a_capture_no_turn_ever_tagged_is_untagged() -> None:
    clean = transcript(
        [
            turn("t001", "interviewer", "Tell me about yourself."),
            turn("t002", "user", "I work in hospitals.", disclosed=["f06"]),
            turn("t003", "user", "I also ride motorcycles.", disclosed=[]),
        ]
    )
    report = dc.check_untagged_capture(
        transcript=clean,
        coverage=[_captured("f06", "c01"), _captured("f07", "c02")],
        alignments=[_judge_supported("c01", "f06", "t002"), _judge_supported("c02", "f07", "t003")],
    )
    assert report.untagged == ("f07",)
    assert report.untagged_count == 1


def test_a_span_before_the_first_tagged_turn_is_untagged() -> None:
    """The judge cannot have read the fact out of a turn the agent had not spoken to."""
    clean = transcript(
        [
            turn("t001", "interviewer", "Tell me about yourself."),
            turn("t002", "user", "Not much to say.", disclosed=[]),
            turn("t003", "user", "I work in hospitals.", disclosed=["f06"]),
        ]
    )
    report = dc.check_untagged_capture(
        transcript=clean,
        coverage=[_captured("f06", "c01")],
        alignments=[_judge_supported("c01", "f06", "t002")],
    )
    assert report.untagged == ("f06",)
    assert report.first_tagged_index == 2


def test_an_exact_field_capture_is_never_untagged() -> None:
    """Exact comparison cites no span, so the check does not apply to it."""
    clean = transcript([turn("t002", "user", "I am thirty four.", disclosed=["f06"])])
    report = dc.check_untagged_capture(
        transcript=clean,
        coverage=[_captured("x1", "c03")],
        alignments=[
            Alignment(claim_id="c03", verdict="supported", method="exact", matched_fact_id="x1")
        ],
    )
    assert report.untagged == ()


def test_an_uncaptured_fact_is_not_untagged_capture() -> None:
    clean = transcript([turn("t002", "user", "I work in hospitals.", disclosed=["f06"])])
    report = dc.check_untagged_capture(
        transcript=clean,
        coverage=[FactCoverage(fact_id="f07", captured=False)],
        alignments=[],
    )
    assert report.untagged == ()


def test_a_span_citing_a_turn_outside_the_transcript_is_an_error() -> None:
    """Join by id: a span naming no turn means the artifacts do not match."""
    clean = transcript([turn("t002", "user", "I work in hospitals.", disclosed=["f06"])])
    with pytest.raises(ValueError, match="not in transcript"):
        dc.check_untagged_capture(
            transcript=clean,
            coverage=[_captured("f06", "c01")],
            alignments=[_judge_supported("c01", "f06", "t099")],
        )


def test_both_checks_combine_into_one_report() -> None:
    truth = sheet(judged_fact("f06", "occupation", "hospital systems administrator"))
    clean = transcript(
        [
            turn("t001", "interviewer", "What do you do for work?"),
            turn("t002", "user", "I am a hospital systems administrator.", disclosed=["f06"]),
            turn("t003", "user", "I also ride motorcycles.", disclosed=[]),
        ]
    )
    report = dc.check_disclosures(
        truth=truth,
        clean_transcript=clean,
        coverage=[_captured("f06", "c01"), _captured("f07", "c02")],
        alignments=[
            _judge_supported("c01", "f06", "t002"),
            _judge_supported("c02", "f07", "t003"),
        ],
    )
    assert report.checked == ("f06",)
    assert report.mismatched == ()
    assert report.untagged_fact_ids == frozenset({"f07"})

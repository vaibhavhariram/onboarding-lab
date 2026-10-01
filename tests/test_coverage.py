"""Fact coverage: what was captured, and which turn elicited it."""

from __future__ import annotations

from onboarding_lab.models import Alignment
from onboarding_lab.score.coverage import build_coverage, first_disclosure


def supported(claim_id: str, fact_id: str) -> Alignment:
    return Alignment(
        claim_id=claim_id,
        verdict="supported",
        method="judge",
        matched_fact_id=fact_id,
        span_turn_id="t002",
        span_text="hospital systems administrator",
    )


def test_supported_claim_captures_its_fact(truth_sheet, transcript) -> None:
    coverage = build_coverage(truth_sheet.facts, [supported("c01", "f06")], transcript)
    by_id = {c.fact_id: c for c in coverage}
    assert by_id["f06"].captured is True
    assert by_id["f06"].by_claim_id == "c01"


def test_eliciting_turn_question_and_followup_flag_come_from_the_transcript(
    truth_sheet, transcript
) -> None:
    coverage = build_coverage(truth_sheet.facts, [supported("c01", "f06")], transcript)
    f06 = next(c for c in coverage if c.fact_id == "f06")
    assert f06.disclosed_turn_id == "t002"
    assert f06.question_id == "q01"
    assert f06.via_followup is False


def test_never_disclosed_fact_has_no_eliciting_turn(truth_sheet, transcript) -> None:
    """None here means the interview never asked, which is what separates
    pipeline recall from extraction recall (PLAN.md M6)."""
    coverage = build_coverage(truth_sheet.facts, [], transcript)
    f07 = next(c for c in coverage if c.fact_id == "f07")
    assert f07.disclosed_turn_id is None
    assert f07.captured is False
    assert f07.by_claim_id is None


def test_unsupported_claims_do_not_capture(truth_sheet, transcript) -> None:
    alignment = Alignment(claim_id="c01", verdict="unsupported", method="judge")
    coverage = build_coverage(truth_sheet.facts, [alignment], transcript)
    assert all(c.captured is False for c in coverage)


def test_offsheet_claims_do_not_capture(truth_sheet, transcript) -> None:
    alignment = Alignment(
        claim_id="c01",
        verdict="supported_offsheet",
        method="judge",
        span_turn_id="t002",
        span_text="hospital systems administrator",
    )
    coverage = build_coverage(truth_sheet.facts, [alignment], transcript)
    assert all(c.captured is False for c in coverage)


def test_duplicate_matches_capture_once(truth_sheet, transcript) -> None:
    coverage = build_coverage(
        truth_sheet.facts, [supported("c01", "f06"), supported("c02", "f06")], transcript
    )
    f06 = next(c for c in coverage if c.fact_id == "f06")
    assert f06.by_claim_id == "c01"


def test_every_fact_gets_exactly_one_row(truth_sheet, transcript) -> None:
    coverage = build_coverage(truth_sheet.facts, [], transcript)
    assert len(coverage) == len(truth_sheet.facts)
    assert len({c.fact_id for c in coverage}) == len(truth_sheet.facts)


def test_first_disclosure_prefers_the_earliest_turn(truth_sheet, transcript) -> None:
    extra = transcript.turns[1].model_copy(
        update={"turn_id": "t003", "disclosed_fact_ids": ["f06"]}
    )
    t = transcript.model_copy(
        update={
            "turns": [*transcript.turns, extra],
            "simulated_minutes": round(
                sum(x.word_count for x in [*transcript.turns, extra]) / 150, 4
            ),
        }
    )
    assert first_disclosure(t, "f06").turn_id == "t002"

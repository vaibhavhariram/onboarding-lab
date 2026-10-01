"""The frozen contracts, and the invariants that keep errors from becoming verdicts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from onboarding_lab.models import (
    EXACT_CLASSES,
    EXACT_FIELDS,
    JUDGED_FIELDS,
    SCORED_FIELDS,
    UNSCORED_FIELDS,
    Alignment,
    Claim,
    Fact,
    FactCoverage,
    JudgeHealth,
    Question,
    QuestionScript,
    Scores,
    Transcript,
    TruthSheet,
    Turn,
    word_count,
)
from onboarding_lab.role_schemas import load_profile_schema

# -- round trips --------------------------------------------------------------


def test_round_trip_truth_sheet(truth_sheet: TruthSheet) -> None:
    assert TruthSheet.model_validate_json(truth_sheet.model_dump_json()) == truth_sheet


def test_round_trip_transcript(transcript: Transcript) -> None:
    assert Transcript.model_validate_json(transcript.model_dump_json()) == transcript


def test_round_trip_extraction(extraction) -> None:
    assert type(extraction).model_validate_json(extraction.model_dump_json()) == extraction


def test_round_trip_alignment(supported_alignment: Alignment) -> None:
    assert Alignment.model_validate_json(supported_alignment.model_dump_json()) == (
        supported_alignment
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, bool), (False, bool), (35, int), ("x", str), ("35", str)],
)
def test_fact_value_union_preserves_type(value, expected) -> None:
    """A bool must not land as "True" nor an int as "35".

    Exact scoring compares across a JSON round trip, so a silent coercion here
    would turn a correct answer into a hallucination or vice versa.
    """
    f = Fact(
        fact_id="f01",
        field="has_kids",
        value=value,
        specificity="generic",
        disclosure="volunteer",
        importance="core",
    )
    assert isinstance(f.value, expected)
    assert isinstance(Fact.model_validate_json(f.model_dump_json()).value, expected)
    assert isinstance(Claim(claim_id="c01", field="has_kids", value=value).value, expected)


# -- field taxonomy matches schema.json ---------------------------------------


def test_field_taxonomy_matches_schema() -> None:
    schema_fields = set(load_profile_schema()["properties"])
    assert set(SCORED_FIELDS) | set(UNSCORED_FIELDS) == schema_fields
    assert not set(EXACT_FIELDS) & set(JUDGED_FIELDS)


def test_exact_classes_match_schema_enums() -> None:
    props = load_profile_schema()["properties"]
    for field, classes in EXACT_CLASSES.items():
        if "enum" in props[field]:
            assert set(classes) == set(props[field]["enum"]), field


def test_profile_schema_rejects_required() -> None:
    """A `required` array collapses the eval: every partial profile fails."""
    with pytest.raises(ValueError, match="required"):
        load_profile_schema.__wrapped__("tests/fixtures/schema_with_required.json")


# -- Fact / TruthSheet --------------------------------------------------------


def test_decoy_value_requires_contradict() -> None:
    with pytest.raises(ValidationError, match="decoy_value"):
        Fact(
            fact_id="f01",
            field="occupation",
            value="nurse",
            specificity="generic",
            disclosure="volunteer",
            importance="core",
            decoy_value="teacher",
        )


def test_truth_sheet_requires_exactly_one_fact_per_exact_field(truth_sheet: TruthSheet) -> None:
    facts = [f for f in truth_sheet.facts if f.field != "wants_kids"]
    with pytest.raises(ValidationError, match="wants_kids"):
        truth_sheet.model_copy(update={"facts": facts}).model_validate(
            {**truth_sheet.model_dump(), "facts": [f.model_dump() for f in facts]}
        )


def test_truth_sheet_rejects_duplicate_fact_ids(truth_sheet: TruthSheet) -> None:
    facts = [f.model_dump() for f in truth_sheet.facts]
    facts.append({**facts[-1]})
    with pytest.raises(ValidationError, match="duplicate fact_ids"):
        TruthSheet.model_validate({**truth_sheet.model_dump(), "facts": facts})


# -- Turn / Transcript --------------------------------------------------------


def test_turn_word_count_must_match_text() -> None:
    with pytest.raises(ValidationError, match="word_count"):
        Turn(
            turn_id="t001",
            speaker="user",
            question_id="q01",
            text="three words here",
            word_count=99,
        )


def test_only_user_turns_disclose_facts() -> None:
    with pytest.raises(ValidationError, match="only user turns"):
        Turn(
            turn_id="t001",
            speaker="interviewer",
            question_id="q01",
            text="So what do you do?",
            word_count=5,
            disclosed_fact_ids=["f01"],
        )


def test_followup_turn_carries_parent_question_id() -> None:
    with pytest.raises(ValidationError, match="parent question_id"):
        Turn(
            turn_id="t003",
            speaker="interviewer",
            question_id=None,
            is_followup=True,
            text="Say more about that?",
            word_count=4,
        )


def test_noised_transcript_needs_seed_and_source(transcript: Transcript) -> None:
    data = transcript.model_dump()
    with pytest.raises(ValidationError, match="noise_seed"):
        Transcript.model_validate({**data, "noise_rate": 0.1})
    with pytest.raises(ValidationError, match="source_transcript_id"):
        Transcript.model_validate({**data, "noise_rate": 0.1, "noise_seed": 1})


def test_clean_transcript_rejects_source_id(transcript: Transcript) -> None:
    with pytest.raises(ValidationError, match="noised copies only"):
        Transcript.model_validate({**transcript.model_dump(), "source_transcript_id": "other"})


def test_simulated_minutes_must_match_turns(transcript: Transcript) -> None:
    with pytest.raises(ValidationError, match="simulated_minutes"):
        Transcript.model_validate({**transcript.model_dump(), "simulated_minutes": 99.0})


def test_word_count_definition() -> None:
    assert word_count("  two   words  ") == 2
    assert word_count("") == 0


# -- Extraction ---------------------------------------------------------------


def test_extraction_ok_iff_profile_present(extraction) -> None:
    data = extraction.model_dump()
    with pytest.raises(ValidationError, match="inconsistent"):
        type(extraction).model_validate({**data, "profile": None})
    with pytest.raises(ValidationError, match="inconsistent"):
        type(extraction).model_validate({**data, "status": "schema_invalid"})


def test_failed_extraction_has_no_claims(extraction) -> None:
    data = extraction.model_dump()
    with pytest.raises(ValidationError, match="only produced when status"):
        type(extraction).model_validate(
            {**data, "status": "refusal", "profile": None, "claims": data["claims"]}
        )


@pytest.mark.parametrize("status", ["schema_invalid", "provider_error", "refusal", "truncated"])
def test_error_states_are_representable(extraction, status: str) -> None:
    """Refusal and truncation are first-class statuses, not parsed as output."""
    obj = type(extraction).model_validate(
        {**extraction.model_dump(), "status": status, "profile": None, "claims": []}
    )
    assert obj.status == status


# -- Alignment: the judge-only verdicts --------------------------------------


def test_span_text_requires_span_turn_id() -> None:
    with pytest.raises(ValidationError, match="span_text without span_turn_id"):
        Alignment(
            claim_id="c01",
            verdict="supported",
            method="judge",
            matched_fact_id="f01",
            span_text="a quote",
        )


@pytest.mark.parametrize("verdict", ["supported_offsheet", "judge_error", "span_invalid"])
def test_exact_method_cannot_produce_judge_only_verdicts(verdict: str) -> None:
    with pytest.raises(ValidationError, match="judge-only"):
        Alignment(claim_id="c01", verdict=verdict, method="exact")


def test_exact_method_never_cites_a_span() -> None:
    with pytest.raises(ValidationError, match="never cite a span"):
        Alignment(
            claim_id="c01",
            verdict="supported",
            method="exact",
            matched_fact_id="f01",
            span_turn_id="t002",
        )


def test_supported_requires_matched_fact() -> None:
    with pytest.raises(ValidationError, match="requires matched_fact_id"):
        Alignment(claim_id="c01", verdict="supported", method="exact")


def test_judged_supported_requires_span() -> None:
    with pytest.raises(ValidationError, match="requires a cited span"):
        Alignment(claim_id="c01", verdict="supported", method="judge", matched_fact_id="f01")


def test_offsheet_means_no_matched_fact() -> None:
    with pytest.raises(ValidationError, match="no matched fact"):
        Alignment(
            claim_id="c01",
            verdict="supported_offsheet",
            method="judge",
            matched_fact_id="f01",
            span_turn_id="t002",
            span_text="a quote",
        )


def test_offsheet_requires_a_span() -> None:
    with pytest.raises(ValidationError, match="requires a valid span"):
        Alignment(claim_id="c01", verdict="supported_offsheet", method="judge")


def test_abstained_is_an_exact_verdict() -> None:
    """Declining to answer is its own verdict, not a hallucination."""
    a = Alignment(claim_id="c01", verdict="abstained", method="exact")
    assert a.verdict == "abstained"


# -- Coverage / JudgeHealth / Scores -----------------------------------------


def test_uncaptured_fact_has_no_claim() -> None:
    with pytest.raises(ValidationError, match="by_claim_id set on an uncaptured fact"):
        FactCoverage(fact_id="f01", captured=False, by_claim_id="c01")


def test_judge_health_is_derived_not_supplied() -> None:
    """A caller cannot assert its own error rate or degraded flag."""
    h = JudgeHealth(calls=10, judged_claims=100, errors=3, span_invalid=3)
    assert h.error_rate == pytest.approx(0.06)
    assert h.degraded is True
    with pytest.raises(ValidationError):
        JudgeHealth.model_validate(
            {"calls": 10, "judged_claims": 100, "errors": 0, "span_invalid": 0, "degraded": False}
        )


def test_degraded_threshold_is_strictly_above_five_percent() -> None:
    assert JudgeHealth(calls=1, judged_claims=100, errors=5, span_invalid=0).degraded is False
    assert JudgeHealth(calls=1, judged_claims=100, errors=6, span_invalid=0).degraded is True


def test_judge_health_zero_claims_is_zero_not_nan() -> None:
    h = JudgeHealth(calls=0, judged_claims=0, errors=0, span_invalid=0)
    assert h.error_rate == 0.0
    assert h.degraded is False


def test_scores_rejects_unknown_metric_keys(supported_alignment, coverage) -> None:
    base = {
        "extraction_id": "e1",
        "alignments": [supported_alignment.model_dump()],
        "coverage": [coverage.model_dump()],
        "judge": {"calls": 1, "judged_claims": 1, "errors": 0, "span_invalid": 0},
    }
    with pytest.raises(ValidationError, match="unknown metric key"):
        Scores.model_validate({**base, "metrics": {"made_up": 1.0}, "n": {}})


def test_scores_requires_denominator_for_rates(supported_alignment, coverage) -> None:
    base = {
        "extraction_id": "e1",
        "alignments": [supported_alignment.model_dump()],
        "coverage": [coverage.model_dump()],
        "judge": {"calls": 1, "judged_claims": 1, "errors": 0, "span_invalid": 0},
    }
    with pytest.raises(ValidationError, match="no denominator"):
        Scores.model_validate({**base, "metrics": {"precision": 1.0}, "n": {}})
    ok = Scores.model_validate({**base, "metrics": {"precision": 1.0}, "n": {"precision": 1}})
    assert ok.metrics["precision"] == 1.0


def test_counts_need_no_denominator(supported_alignment, coverage) -> None:
    ok = Scores.model_validate(
        {
            "extraction_id": "e1",
            "alignments": [supported_alignment.model_dump()],
            "coverage": [coverage.model_dump()],
            "judge": {"calls": 1, "judged_claims": 1, "errors": 0, "span_invalid": 0},
            "metrics": {"yield.q01": 3.0},
            "n": {},
        }
    )
    assert ok.metrics["yield.q01"] == 3.0


# -- QuestionScript -----------------------------------------------------------


def test_question_script_rejects_unknown_targets() -> None:
    with pytest.raises(ValidationError, match="unknown targets"):
        QuestionScript(
            script_id="s1",
            questions=[Question(question_id="q01", text="?", targets=["not_a_field"])],
        )


def test_question_script_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate question_ids"):
        QuestionScript(
            script_id="s1",
            questions=[
                Question(question_id="q01", text="a", targets=["values"]),
                Question(question_id="q01", text="b", targets=["values"]),
            ],
        )

"""Exact-field scoring: no model, no type coercion."""

from __future__ import annotations

import pytest

from onboarding_lab.models import Claim
from onboarding_lab.score.exact import is_abstention, score_exact, values_match


def claim(field: str, value) -> Claim:
    return Claim(claim_id="c01", field=field, value=value)


def test_matching_value_is_supported(truth_sheet) -> None:
    [a] = score_exact([claim("wants_kids", "yes")], truth_sheet.facts)
    assert a.verdict == "supported"
    assert a.method == "exact"
    assert a.matched_fact_id == "f03"
    assert a.span_turn_id is None


def test_differing_value_is_unsupported(truth_sheet) -> None:
    [a] = score_exact([claim("wants_kids", "no")], truth_sheet.facts)
    assert a.verdict == "unsupported"


def test_declining_is_abstained_not_unsupported(truth_sheet) -> None:
    """The prompt tells the model to answer 'unsure' when the user did not say;
    scoring that as a hallucination would be wrong."""
    [a] = score_exact([claim("wants_kids", "unsure")], truth_sheet.facts)
    assert a.verdict == "abstained"
    assert a.matched_fact_id is None


def test_unsure_is_supported_when_unsure_is_the_truth(truth_sheet) -> None:
    facts = [
        f.model_copy(update={"value": "unsure"}) if f.field == "wants_kids" else f
        for f in truth_sheet.facts
    ]
    [a] = score_exact([claim("wants_kids", "unsure")], facts)
    assert a.verdict == "supported"


def test_no_abstention_for_fields_without_an_unsure_member(truth_sheet) -> None:
    assert not is_abstention("religion_importance", "none", "low")
    [a] = score_exact([claim("religion_importance", "none")], truth_sheet.facts)
    assert a.verdict == "unsupported"


def test_string_integer_is_not_coerced_into_agreement(truth_sheet) -> None:
    """The schema is the type gate; scoring never coerces (PLAN.md M3)."""
    assert not values_match("34", 34)
    [a] = score_exact([claim("age", "34")], truth_sheet.facts)
    assert a.verdict == "unsupported"


def test_bool_and_int_are_distinguished() -> None:
    """True == 1 in Python, so has_kids=1 must not count as correct."""
    assert not values_match(1, False)
    assert not values_match(True, 1)
    assert values_match(False, False)


def test_enum_comparison_folds_case_and_whitespace(truth_sheet) -> None:
    [a] = score_exact([claim("wants_kids", "  YES ")], truth_sheet.facts)
    assert a.verdict == "supported"


def test_judged_fields_are_ignored(truth_sheet) -> None:
    assert score_exact([claim("occupation", "nurse")], truth_sheet.facts) == []


def test_missing_truth_fact_fails_loudly(truth_sheet) -> None:
    facts = [f for f in truth_sheet.facts if f.field != "has_kids"]
    with pytest.raises(KeyError, match="has_kids"):
        score_exact([claim("has_kids", True)], facts)

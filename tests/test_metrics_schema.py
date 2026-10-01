"""The metric key registry.

Keys live in one place so the aggregate, the report, and the e2e completeness
assertion cannot drift apart.
"""

from __future__ import annotations

import pytest

from onboarding_lab import metrics_schema as ms
from onboarding_lab.models import EXACT_FIELDS, SCORED_FIELDS


def test_static_rates_classify_as_rates() -> None:
    for key in ms.STATIC_RATES:
        assert ms.classify(key) == "rate"
        assert ms.needs_denominator(key)
        assert ms.supports_wilson(key)


def test_counts_need_no_denominator_and_get_no_interval() -> None:
    for key in ms.STATIC_COUNTS:
        assert ms.classify(key) == "count"
        assert not ms.needs_denominator(key)
        assert not ms.supports_wilson(key)


def test_yield_per_minute_is_a_ratio_without_an_interval() -> None:
    """Wilson needs a proportion; facts per minute is not one."""
    assert ms.classify("yield_per_minute.q01") == "ratio"
    assert ms.needs_denominator("yield_per_minute.q01")
    assert not ms.supports_wilson("yield_per_minute.q01")


def test_unknown_keys_are_rejected() -> None:
    assert not ms.is_known("made_up")
    assert not ms.is_known("yield")  # bare prefix, no suffix
    assert not ms.is_known("yield.")
    with pytest.raises(KeyError):
        ms.classify("made_up")


def test_gated_keys_are_real_rates() -> None:
    for key in ms.GATED_KEYS:
        assert ms.classify(key) == "rate"


def test_gate_uses_elicited_recall() -> None:
    """The gate should measure extraction, not whether the interview asked."""
    assert "recall_elicited" in ms.GATED_KEYS
    assert "recall" not in ms.GATED_KEYS


def test_expected_keys_cover_every_field_and_question() -> None:
    keys = ms.expected_keys(
        fields=SCORED_FIELDS,
        exact_fields=EXACT_FIELDS,
        disclosures=("volunteer", "needs_followup", "hedge"),
        styles=("terse", "balanced", "rambling", "tangential"),
        question_ids=("q01", "q02"),
    )
    assert "precision" in keys
    assert "recall_by_field.occupation" in keys
    assert "exact_floor.has_kids" in keys
    assert "recall_by_disclosure.hedge" in keys
    assert "followup_share_by_style.terse" in keys
    assert "yield.q01" in keys and "yield_per_minute.q02" in keys
    for key in keys:
        assert ms.is_known(key), key


def test_v1_cuts_are_absent() -> None:
    """Deferred to v1.1; their absence should be deliberate, not accidental."""
    assert not ms.is_known("decoy_rate")
    assert not ms.is_known("specificity_match")

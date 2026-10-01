"""Tallies, Wilson intervals, and the pooling helpers.

The point estimate for an empty sample is the thing being tested here: it must
be absent, not zero (PLAN.md M12).
"""

from __future__ import annotations

import pytest

from onboarding_lab.score import stats


def test_a_rate_with_a_zero_denominator_has_no_value() -> None:
    tally = stats.rate(0, 0)
    assert tally.value is None
    assert not tally.defined
    assert tally.interval() is None


def test_a_count_is_defined_even_at_zero() -> None:
    """Zero occurrences is an observation; zero out of zero is not."""
    tally = stats.count(0)
    assert tally.defined
    assert tally.value == 0.0
    assert tally.interval() is None


def test_a_proportion_cannot_exceed_its_denominator() -> None:
    """The symptom of using the wrong denominator, caught at construction."""
    with pytest.raises(ValueError, match="exceeds denominator"):
        stats.rate(3, 2)


def test_a_ratio_may_exceed_one() -> None:
    """Facts per minute is not a proportion."""
    assert stats.ratio(300, 60).value == 5.0


def test_a_count_takes_no_denominator() -> None:
    with pytest.raises(ValueError, match="no denominator"):
        stats.Tally("count", 1, 2)


def test_negative_parts_are_rejected() -> None:
    with pytest.raises(ValueError, match="negative numerator"):
        stats.rate(-1, 4)
    with pytest.raises(ValueError, match="negative denominator"):
        stats.Tally("ratio", 1, -4)


def test_wilson_is_absent_at_n_zero_rather_than_a_point_at_zero() -> None:
    assert stats.wilson_interval(0, 0) is None


def test_wilson_matches_the_published_interval() -> None:
    """1 of 10, two-sided 95%: an external reference value, not a self-check."""
    low, high = stats.wilson_interval(1, 10)
    assert low == pytest.approx(0.0179, abs=1e-3)
    assert high == pytest.approx(0.4042, abs=1e-3)


def test_wilson_stays_inside_the_unit_range_at_the_extremes() -> None:
    """Where the normal approximation would leave it."""
    low, high = stats.wilson_interval(0, 8)
    assert low == 0.0
    assert 0.0 < high < 1.0
    low, high = stats.wilson_interval(8, 8)
    assert high == 1.0
    assert 0.0 < low < 1.0


def test_wilson_is_centred_at_a_half_and_narrows_with_n() -> None:
    low, high = stats.wilson_interval(5, 10)
    assert (low + high) / 2 == pytest.approx(0.5)
    wide = high - low
    low, high = stats.wilson_interval(500, 1000)
    assert high - low < wide


def test_wilson_rejects_more_successes_than_trials() -> None:
    with pytest.raises(ValueError, match="exceeds n"):
        stats.wilson_interval(5, 4)


def test_pooling_sums_the_parts_rather_than_averaging_the_ratios() -> None:
    """PLAN.md M13: a persona with one claim must not weigh like one with nine."""
    pooled = stats.pool([stats.rate(0, 1), stats.rate(9, 9)])
    assert pooled.value == 0.9
    assert pooled.denominator == 10
    mean_of_ratios = stats.mean([0 / 1, 9 / 9])
    assert mean_of_ratios == 0.5


def test_pooling_rejects_mixed_kinds_and_empty_input() -> None:
    with pytest.raises(ValueError, match="mixed kinds"):
        stats.pool([stats.rate(1, 2), stats.count(3)])
    with pytest.raises(ValueError, match="empty"):
        stats.pool([])


def test_mean_and_stdev_are_absent_rather_than_fabricated() -> None:
    assert stats.mean([]) is None
    assert stats.stdev([]) is None
    assert stats.stdev([0.5]) is None, "one observation is not a variance estimate"


def test_stdev_is_the_sample_estimator() -> None:
    assert stats.stdev([1, 2, 3, 4, 5]) == pytest.approx(1.5811388300841898)
    assert stats.stdev([0.5, 0.5, 0.5]) == 0.0


def test_pairwise_jaccard_pools_over_pairs() -> None:
    sets = [frozenset({"a", "b"}), frozenset({"a", "b"}), frozenset({"a", "c"})]
    tally = stats.pairwise_jaccard(sets)
    # pairs: (ab,ab) 2/2, (ab,ac) 1/3, (ab,ac) 1/3
    assert tally.numerator == 4
    assert tally.denominator == 8
    assert tally.value == 0.5


def test_pairwise_jaccard_needs_two_runs_with_something_in_them() -> None:
    assert stats.pairwise_jaccard([frozenset({"a"})]).value is None
    assert stats.pairwise_jaccard([frozenset(), frozenset()]).value is None

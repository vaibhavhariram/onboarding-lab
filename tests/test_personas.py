"""Skeleton sampling and persona generation.

The sampler is the project's only fully-deterministic oracle, so its balance and
independence properties are asserted directly rather than inferred from
downstream numbers.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path

import pytest

from onboarding_lab import llm as llm_module
from onboarding_lab.models import (
    DEALBREAKER_FIELDS,
    EXACT_CLASSES,
    EXACT_FIELDS,
    TruthSheet,
)
from onboarding_lab.personas.generate import generate_all, generate_one, render_prompt
from onboarding_lab.personas.sample_skeleton import (
    AGE_MAX,
    AGE_MIN,
    JUDGED_DISCLOSURE_MIX,
    STYLES,
    balanced_classes,
    sample_skeletons,
    split_counts,
)
from onboarding_lab.providers.fake import FakeProvider

N = 12
SEED = 7


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Skip retry backoff. The tests assert attempt counts, not wall clock."""

    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(llm_module.asyncio, "sleep", _instant)


def column(skeletons, field: str) -> list:
    return [next(f.value for f in s.exact_facts if f.field == field) for s in skeletons]


# -- balanced_classes ---------------------------------------------------------


def test_balance_is_exact_when_n_divides() -> None:
    counts = Counter(balanced_classes(("a", "b", "c"), 9, seed=1, field="x"))
    assert set(counts.values()) == {3}


def test_balance_is_within_one_when_n_does_not_divide() -> None:
    counts = Counter(balanced_classes(("a", "b", "c"), 10, seed=1, field="x"))
    assert max(counts.values()) - min(counts.values()) <= 1
    assert sum(counts.values()) == 10


def test_per_field_seeding_breaks_correlation() -> None:
    """Two fields with the same class count must not move together.

    `class = i % k` keeps the counts balanced but makes every same-k field
    identical, which would silently make relationship_goal and
    religion_importance the same variable (PLAN.md amendment 4).
    """
    a = balanced_classes(("w", "x", "y", "z"), 12, seed=SEED, field="field_a")
    b = balanced_classes(("w", "x", "y", "z"), 12, seed=SEED, field="field_b")
    assert a != b
    fan_out = {}
    for left, right in zip(a, b, strict=True):
        fan_out.setdefault(left, set()).add(right)
    assert all(len(v) > 1 for v in fan_out.values()), "fields are perfectly correlated"


def test_same_field_and_seed_is_reproducible() -> None:
    assert balanced_classes(STYLES, 12, seed=3, field="style") == balanced_classes(
        STYLES, 12, seed=3, field="style"
    )


def test_empty_classes_rejected() -> None:
    with pytest.raises(ValueError, match="no classes"):
        balanced_classes((), 5, seed=1, field="x")


# -- split_counts -------------------------------------------------------------


@pytest.mark.parametrize("total", [0, 1, 7, 12, 13, 16, 101])
def test_split_counts_sums_exactly(total: int) -> None:
    """Largest-remainder, so rounding never loses or invents a fact."""
    counts = split_counts(total, JUDGED_DISCLOSURE_MIX)
    assert sum(counts.values()) == total
    assert all(v >= 0 for v in counts.values())


# -- sample_skeletons ---------------------------------------------------------


def test_every_categorical_exact_field_is_exactly_balanced() -> None:
    skeletons = sample_skeletons(N, SEED)
    for field, classes in EXACT_CLASSES.items():
        counts = Counter(column(skeletons, field))
        assert set(counts) == set(classes), field
        assert set(counts.values()) == {N // len(classes)}, f"{field}: {dict(counts)}"


def test_exact_fields_are_mutually_independent() -> None:
    skeletons = sample_skeletons(N, SEED)
    goal = column(skeletons, "relationship_goal")
    religion = column(skeletons, "religion_importance")
    fan_out = {}
    for left, right in zip(goal, religion, strict=True):
        fan_out.setdefault(left, set()).add(right)
    assert all(len(v) > 1 for v in fan_out.values())


def test_one_complete_fact_per_exact_field() -> None:
    for s in sample_skeletons(N, SEED):
        fields = [f.field for f in s.exact_facts]
        assert fields == list(EXACT_FIELDS)
        assert len(set(f.fact_id for f in s.exact_facts)) == len(EXACT_FIELDS)


def test_dealbreaker_importance_follows_the_field_list() -> None:
    for s in sample_skeletons(N, SEED):
        for f in s.exact_facts:
            expected = "dealbreaker" if f.field in DEALBREAKER_FIELDS else "core"
            assert f.importance == expected, f.field


def test_exact_fields_are_never_hedged() -> None:
    """An enum has no vague partial form, so hedging one is a guaranteed miss
    and would make recall_by_disclosure.hedge unreadable."""
    for s in sample_skeletons(N, SEED):
        assert all(f.disclosure != "hedge" for f in s.exact_facts)


def test_ages_are_adults_only() -> None:
    ages = column(sample_skeletons(N, SEED), "age")
    assert all(AGE_MIN <= a <= AGE_MAX for a in ages)


def test_disclosure_counts_sum_to_the_judged_total() -> None:
    for s in sample_skeletons(N, SEED):
        assert s.n_volunteer + s.n_followup + s.n_hedge == s.n_judged_facts
        assert s.n_specific <= s.n_judged_facts


def test_sampling_is_deterministic_and_seed_sensitive() -> None:
    assert sample_skeletons(N, SEED) == sample_skeletons(N, SEED)
    assert sample_skeletons(N, SEED) != sample_skeletons(N, SEED + 1)


def test_persona_ids_are_sequential() -> None:
    assert [s.persona_id for s in sample_skeletons(3, SEED)] == ["p001", "p002", "p003"]


def test_n_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        sample_skeletons(0, SEED)


def test_prompt_payload_key_order_is_stable() -> None:
    """The rendered prompt must be byte-stable or it cannot be cached."""
    s = sample_skeletons(1, SEED)[0]
    assert render_prompt(s, Path("prompts/persona_gen.txt").read_text()) == render_prompt(
        s, Path("prompts/persona_gen.txt").read_text()
    )


def test_prompt_renders_without_leftover_placeholders() -> None:
    s = sample_skeletons(1, SEED)[0]
    rendered = render_prompt(s, Path("prompts/persona_gen.txt").read_text())
    assert "{skeleton}" not in rendered
    assert str(s.n_specific) in rendered
    assert "relationship_goal" in rendered


# -- generation ---------------------------------------------------------------


def judged_payload(n: int = 10) -> dict:
    fields = ["occupation", "location", "values", "interests", "life_events"]
    return {
        "facts": [
            {
                "field": fields[i % len(fields)],
                "value": f"judged value {i}",
                "specificity": "specific" if i % 2 == 0 else "generic",
                "disclosure": ("volunteer", "needs_followup", "hedge")[i % 3],
                "importance": ("core", "color")[i % 2],
            }
            for i in range(n)
        ],
        "bio": "I am a person with a life. " * 20,
    }


def run_one(provider, skeleton, **kw):
    return asyncio.run(generate_one(skeleton, provider=provider, model="fake-model", **kw))


def test_generation_merges_fixed_anchors_with_model_facts(tmp_path: Path) -> None:
    skeleton = sample_skeletons(1, SEED)[0]
    provider = FakeProvider({"role:persona_gen": judged_payload()})
    result = run_one(provider, skeleton, cache_root=tmp_path)

    assert result.status == "ok"
    sheet = result.sheet
    assert isinstance(sheet, TruthSheet)
    # The exact-scored values survive untouched: the model never chose them.
    for anchor in skeleton.exact_facts:
        assert sheet.fact(anchor.fact_id).value == anchor.value
    assert sheet.style == skeleton.style
    assert sheet.demographics == skeleton.demographics


def test_fact_ids_are_assigned_here_and_are_unique(tmp_path: Path) -> None:
    """Ids are the only join key, so the model is not trusted with them."""
    skeleton = sample_skeletons(1, SEED)[0]
    provider = FakeProvider({"role:persona_gen": judged_payload(10)})
    sheet = run_one(provider, skeleton, cache_root=tmp_path).sheet
    ids = [f.fact_id for f in sheet.facts]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
    assert ids[: len(EXACT_FIELDS)] == [f.fact_id for f in skeleton.exact_facts]


def test_refusal_is_a_recorded_status_not_a_sheet(tmp_path: Path) -> None:
    skeleton = sample_skeletons(1, SEED)[0]
    provider = FakeProvider(
        {"role:persona_gen": {"__error__": True, "reason": "refusal", "error": "declined"}}
    )
    result = run_one(provider, skeleton, cache_root=tmp_path)
    assert result.status == "refusal"
    assert result.sheet is None
    assert result.as_failure_record()["stage"] == "gen"


def test_truncation_is_distinguished_from_a_generic_error(tmp_path: Path) -> None:
    skeleton = sample_skeletons(1, SEED)[0]
    provider = FakeProvider(
        {"role:persona_gen": {"__error__": True, "reason": "max_tokens", "error": "cut off"}}
    )
    assert run_one(provider, skeleton, cache_root=tmp_path).status == "truncated"


def test_unusable_payload_fails_the_persona_rather_than_the_run(tmp_path: Path) -> None:
    """A fact for an exact-scored field would duplicate a fixed anchor."""
    skeleton = sample_skeletons(1, SEED)[0]
    bad = judged_payload()
    bad["facts"][0]["field"] = "wants_kids"
    provider = FakeProvider({"role:persona_gen": bad})
    result = run_one(provider, skeleton, cache_root=tmp_path)
    assert result.status == "schema_invalid"
    assert result.sheet is None
    assert result.detail


def test_missing_canned_response_is_a_provider_error(tmp_path: Path) -> None:
    skeleton = sample_skeletons(1, SEED)[0]
    result = run_one(FakeProvider({}), skeleton, cache_root=tmp_path)
    assert result.status == "provider_error"


def test_generate_all_returns_skeleton_order(tmp_path: Path) -> None:
    """Joining by position after a gather is the misalignment bug this project
    exists to avoid, so order is rebuilt from ids."""
    skeletons = sample_skeletons(5, SEED)
    provider = FakeProvider({"role:persona_gen": judged_payload()})
    results = asyncio.run(
        generate_all(
            skeletons,
            provider=provider,
            model="fake-model",
            concurrency=4,
            cache_root=tmp_path,
        )
    )
    assert [r.persona_id for r in results] == [s.persona_id for s in skeletons]
    assert all(r.status == "ok" for r in results)

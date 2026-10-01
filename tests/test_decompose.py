"""Profile to claims."""

from __future__ import annotations

from onboarding_lab.extract.decompose import decompose


def test_scalars_become_one_claim_each() -> None:
    d = decompose({"age": 34, "occupation": "nurse", "has_kids": False})
    assert [(c.field, c.value) for c in d.claims] == [
        ("age", 34),
        ("has_kids", False),
        ("occupation", "nurse"),
    ]


def test_arrays_become_one_claim_per_item() -> None:
    d = decompose({"interests": ["cycling", "baking", "chess"]})
    assert len(d.claims) == 3
    assert all(c.field == "interests" for c in d.claims)


def test_summary_is_not_scored() -> None:
    d = decompose({"summary": "a paragraph", "occupation": "nurse"})
    assert [c.field for c in d.claims] == ["occupation"]
    assert d.unscored_present == ["summary"]


def test_duplicates_are_collapsed_and_counted() -> None:
    """Three 'travel' entries would otherwise earn three supported claims."""
    d = decompose({"interests": ["travel", "Travel", " travel ", "chess"]})
    assert [c.value for c in d.claims] == ["travel", "chess"]
    assert d.deduped == 2


def test_missing_and_empty_values_are_not_claims() -> None:
    d = decompose({"age": None, "occupation": "   ", "interests": []})
    assert d.claims == []


def test_claim_ids_are_sequential_and_unique() -> None:
    d = decompose({"interests": ["a", "b"], "values": ["c"]})
    assert [c.claim_id for c in d.claims] == ["c01", "c02", "c03"]


def test_unknown_fields_are_ignored() -> None:
    d = decompose({"not_a_field": "x", "occupation": "nurse"})
    assert [c.field for c in d.claims] == ["occupation"]

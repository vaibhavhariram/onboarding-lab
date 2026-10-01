"""Exact-field scoring. No model in the loop.

Enum, boolean, and integer fields are compared with ``==`` after schema
validation. The JSON Schema is the type gate: a string in an integer field is a
``schema_invalid`` extraction, never silently coerced into agreement (which
would manufacture a correct answer) nor into a mismatch (which would
manufacture a hallucination). Nothing here coerces types (PLAN.md M3).

``abstained`` exists because ``extract.txt`` tells the model to answer ``unsure``
when the user did not say, while ``unsure`` is *also* a real truth value for
``relationship_goal`` and ``wants_kids`` — and under the Python-sampled
skeletons it is a guaranteed truth value for a known share of personas. Without
this verdict, a model that correctly declines would be scored as having
hallucinated (PLAN.md M4).
"""

from __future__ import annotations

from ..models import EXACT_FIELDS, UNSURE_MEMBER, Alignment, Claim, Fact


def values_match(claim_value: str | bool | int, truth_value: str | bool | int) -> bool:
    """Equality, with case and whitespace folded for strings only.

    ``True == 1`` is true in Python, so booleans are compared by type as well:
    a claim of ``1`` for ``has_kids`` is not a correct answer.
    """
    if isinstance(claim_value, bool) != isinstance(truth_value, bool):
        return False
    if isinstance(claim_value, str) and isinstance(truth_value, str):
        return " ".join(claim_value.split()).casefold() == " ".join(truth_value.split()).casefold()
    if isinstance(claim_value, str) != isinstance(truth_value, str):
        return False
    return claim_value == truth_value


def is_abstention(field: str, claim_value: str | bool | int, truth_value: str | bool | int) -> bool:
    """The claim says "not stated" and the truth is something else."""
    unsure = UNSURE_MEMBER.get(field)
    if unsure is None:
        return False
    return values_match(claim_value, unsure) and not values_match(truth_value, unsure)


def score_exact(claims: list[Claim], facts: list[Fact]) -> list[Alignment]:
    """One alignment per exact-field claim."""
    truth_by_field = {f.field: f for f in facts if f.field in EXACT_FIELDS}
    alignments: list[Alignment] = []

    for claim in claims:
        if claim.field not in EXACT_FIELDS:
            continue
        fact = truth_by_field.get(claim.field)
        if fact is None:
            # Every truth sheet carries one fact per exact field, enforced by the
            # TruthSheet validator, so this is a programming error rather than a
            # model failure. Fail loudly instead of scoring it.
            raise KeyError(
                f"no truth fact for exact-scored field {claim.field!r}; "
                f"the truth sheet is incomplete"
            )

        if values_match(claim.value, fact.value):
            alignments.append(
                Alignment(
                    claim_id=claim.claim_id,
                    verdict="supported",
                    method="exact",
                    matched_fact_id=fact.fact_id,
                )
            )
        elif is_abstention(claim.field, claim.value, fact.value):
            alignments.append(
                Alignment(claim_id=claim.claim_id, verdict="abstained", method="exact")
            )
        else:
            alignments.append(
                Alignment(claim_id=claim.claim_id, verdict="unsupported", method="exact")
            )
    return alignments

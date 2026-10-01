"""Turn an extracted profile into claims.

Scalar fields become one claim; array fields become one claim per item;
``summary`` becomes none (it is shown in the report but never scored).

Duplicates are collapsed. A model that lists "travel" three times under
``interests`` would otherwise earn three ``supported`` claims, because coverage
dedupes through the singular ``FactCoverage.by_claim_id`` while precision does
not — so the duplicates would inflate precision only (PLAN.md M15). The number
collapsed is reported as ``claims_deduped`` rather than discarded quietly.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import ARRAY_FIELDS, SCORED_FIELDS, UNSCORED_FIELDS, Claim


def normalize_for_dedupe(value: str | bool | int) -> str | bool | int:
    """Case and whitespace only. Types are never coerced."""
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    return value


@dataclass(frozen=True)
class Decomposition:
    claims: list[Claim]
    deduped: int
    #: Fields present in the profile that the schema does not score. Not an
    #: error (the schema already forbids unknown properties), but surfaced so a
    #: schema change cannot silently drop claims.
    unscored_present: list[str]


def decompose(profile: dict) -> Decomposition:
    claims: list[Claim] = []
    seen: set[tuple[str, object]] = set()
    deduped = 0
    unscored_present: list[str] = []

    for field in SCORED_FIELDS:
        if field not in profile:
            continue
        raw = profile[field]
        if raw is None:
            continue
        values = raw if field in ARRAY_FIELDS and isinstance(raw, list) else [raw]
        for value in values:
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            key = (field, normalize_for_dedupe(value))
            if key in seen:
                deduped += 1
                continue
            seen.add(key)
            claims.append(Claim(claim_id=f"c{len(claims) + 1:02d}", field=field, value=value))

    unscored_present = [f for f in UNSCORED_FIELDS if f in profile]
    return Decomposition(claims=claims, deduped=deduped, unscored_present=unscored_present)

"""Seed-sampled persona skeletons.

Everything the persona-generation model is *not* trusted to decide happens here,
in plain Python with a seeded RNG. In particular every exact-scored field's truth
value is fixed here (PLAN.md M2), for two reasons:

- It removes an LLM-generated oracle. The exact-scored path — the one compared
  with ``==`` and never sent to the judge — is then deterministic end to end.
- It pins the majority-class floor at ``1/k`` by construction. A model that
  always guesses the majority class scores well on exact-field recall with zero
  evidence, so that floor is what makes exact-field recall mean anything
  (PLAN.md M9). If the generating model chose these values, the floor would sit
  wherever that model happened to skew.

"Seed" here always means a Python-side ``random.Random``. No seed is ever sent to
a model; the Messages API has no such parameter.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

from ..models import (
    DEALBREAKER_FIELDS,
    EXACT_CLASSES,
    EXACT_FIELDS,
    Disclosure,
    Fact,
    Style,
)

STYLES: tuple[Style, ...] = ("terse", "balanced", "rambling", "tangential")

#: Content guardrail: adults only (CLAUDE.md rule 13).
AGE_MIN, AGE_MAX = 25, 45

#: Diversity comes from seed-sampled demographics, not from caricature. These are
#: deliberately plain: the generating model writes the person, and nothing about
#: their personality is allowed to follow from these values.
CITIES: tuple[str, ...] = (
    "Portland",
    "Austin",
    "Minneapolis",
    "Philadelphia",
    "Denver",
    "Pittsburgh",
    "Sacramento",
    "Nashville",
    "Providence",
    "Tucson",
    "Milwaukee",
    "Richmond",
)

OCCUPATIONS: tuple[str, ...] = (
    "nurse",
    "structural engineer",
    "high school teacher",
    "line cook",
    "accountant",
    "physical therapist",
    "graphic designer",
    "electrician",
    "research librarian",
    "logistics coordinator",
    "veterinary technician",
    "software developer",
)

EDUCATION: tuple[str, ...] = (
    "high school",
    "trade certificate",
    "associate degree",
    "bachelor's degree",
    "master's degree",
)

#: Disclosure mix for judged-field facts. ``contradict`` is deferred to v1.1, so
#: the mix is three ways; the scorer does not yet account for decoy values.
JUDGED_DISCLOSURE_MIX: tuple[tuple[Disclosure, float], ...] = (
    ("volunteer", 0.6),
    ("needs_followup", 0.3),
    ("hedge", 0.1),
)

#: Exact-scored fields are never hedged. An enum or boolean has no vague partial
#: form: "hedging" one would just guarantee a miss, which would conflate
#: "the user was vague" with "the value was impossible to state" and make
#: ``recall_by_disclosure.hedge`` unreadable.
EXACT_DISCLOSURES: tuple[Disclosure, ...] = ("volunteer", "needs_followup")

#: Roughly this share of judged facts carry a concrete place, named thing,
#: number, or duration. Specific facts are what make the eval non-trivial.
SPECIFIC_SHARE = 0.6

JUDGED_FACTS_MIN, JUDGED_FACTS_MAX = 12, 16


@dataclass(frozen=True)
class Skeleton:
    """Fixed inputs handed to persona generation as constraints."""

    persona_id: str
    seed: int
    style: Style
    demographics: dict[str, Any]
    #: One complete ``Fact`` per exact-scored field. The model may not alter,
    #: omit, or duplicate these.
    exact_facts: list[Fact] = dataclass_field(default_factory=list)
    n_judged_facts: int = 0
    n_specific: int = 0
    n_volunteer: int = 0
    n_followup: int = 0
    n_hedge: int = 0

    def next_fact_index(self) -> int:
        """Judged facts continue the id sequence after the exact ones."""
        return len(self.exact_facts) + 1

    def as_prompt_payload(self) -> dict[str, Any]:
        """What ``persona_gen.txt`` is shown. Deterministic key order so the
        rendered prompt is byte-stable and therefore cacheable."""
        return {
            "persona_id": self.persona_id,
            "style": self.style,
            "demographics": dict(sorted(self.demographics.items())),
            "fixed_exact_fields": {
                f.field: f.value for f in sorted(self.exact_facts, key=lambda f: f.field)
            },
        }


def balanced_classes(classes: tuple[Any, ...], n: int, *, seed: int, field: str) -> list[Any]:
    """Assign ``n`` personas across ``classes`` as evenly as possible.

    Repeat the classes up to length ``n``, then shuffle with an RNG seeded on
    ``(seed, field)``.

    The per-field seed is the whole point. Assigning ``class = i % k`` would keep
    the counts balanced but make any two fields with the same ``k`` perfectly
    correlated — ``relationship_goal`` and ``religion_importance`` are both
    4-way, so every persona wanting marriage would also be highly religious, and
    ``has_kids`` (2-way) would be a function of both. Shuffling each field
    independently keeps the exact balance and destroys the correlation
    (PLAN.md amendment 4).
    """
    if not classes:
        raise ValueError(f"{field}: no classes to sample from")
    if n < 0:
        raise ValueError(f"{field}: n must not be negative")
    assignment = [classes[i % len(classes)] for i in range(n)]
    random.Random(f"{seed}:{field}").shuffle(assignment)
    return assignment


def split_counts(total: int, mix: tuple[tuple[Any, float], ...]) -> dict[Any, int]:
    """Split ``total`` across a weighted mix so the parts sum to exactly ``total``.

    Largest-remainder, so rounding never loses or invents a fact.
    """
    exact = {name: total * weight for name, weight in mix}
    counts = {name: int(value) for name, value in exact.items()}
    shortfall = total - sum(counts.values())
    by_remainder = sorted(exact, key=lambda name: exact[name] - counts[name], reverse=True)
    for name in by_remainder[:shortfall]:
        counts[name] += 1
    return counts


def _exact_fact(fact_id: str, field: str, value: Any, disclosure: Disclosure) -> Fact:
    return Fact(
        fact_id=fact_id,
        field=field,
        value=value,
        # Age is a number, so it is concrete by nature; the enums and the boolean
        # have no specific form.
        specificity="specific" if field == "age" else "generic",
        disclosure=disclosure,
        importance="dealbreaker" if field in DEALBREAKER_FIELDS else "core",
    )


def sample_skeletons(n: int, seed: int) -> list[Skeleton]:
    """Sample ``n`` skeletons. Fully determined by ``(n, seed)``."""
    if n <= 0:
        raise ValueError("n must be positive")

    styles = balanced_classes(STYLES, n, seed=seed, field="style")
    cities = balanced_classes(CITIES, n, seed=seed, field="city")
    occupations = balanced_classes(OCCUPATIONS, n, seed=seed, field="occupation")
    education = balanced_classes(EDUCATION, n, seed=seed, field="education")

    # Categorical exact fields get exact per-field balance.
    exact_values = {
        field: balanced_classes(classes, n, seed=seed, field=field)
        for field, classes in EXACT_CLASSES.items()
    }
    # Age has 21 classes, so balance across a run of 12 is not meaningful; sample
    # it uniformly instead. Track C computes its floor empirically from the
    # truth sheets rather than assuming 1/k.
    age_rng = random.Random(f"{seed}:age")
    ages = [age_rng.randint(AGE_MIN, AGE_MAX) for _ in range(n)]

    exact_disclosure = {
        field: balanced_classes(EXACT_DISCLOSURES, n, seed=seed, field=f"disclosure:{field}")
        for field in EXACT_FIELDS
    }

    count_rng = random.Random(f"{seed}:fact_counts")

    skeletons: list[Skeleton] = []
    for i in range(n):
        persona_id = f"p{i + 1:03d}"
        values: dict[str, Any] = {"age": ages[i]}
        for field in EXACT_CLASSES:
            values[field] = exact_values[field][i]

        exact_facts = [
            _exact_fact(f"f{j + 1:02d}", field, values[field], exact_disclosure[field][i])
            for j, field in enumerate(EXACT_FIELDS)
        ]

        n_judged = count_rng.randint(JUDGED_FACTS_MIN, JUDGED_FACTS_MAX)
        disclosure_counts = split_counts(n_judged, JUDGED_DISCLOSURE_MIX)

        skeletons.append(
            Skeleton(
                persona_id=persona_id,
                seed=seed,
                style=styles[i],
                demographics={
                    "age": ages[i],
                    "city": cities[i],
                    "occupation": occupations[i],
                    "education": education[i],
                },
                exact_facts=exact_facts,
                n_judged_facts=n_judged,
                n_specific=round(SPECIFIC_SHARE * n_judged),
                n_volunteer=disclosure_counts["volunteer"],
                n_followup=disclosure_counts["needs_followup"],
                n_hedge=disclosure_counts["hedge"],
            )
        )
    return skeletons

"""Frozen Pydantic contracts between pipeline stages.

Every stage reads and writes these; modules are built against them and changes
go through the main thread with an explicit note (CLAUDE.md rule 8).

Two invariants are enforced here rather than in the stages that depend on them,
because enforcing them downstream is how they get silently violated:

- **Errors are states, not verdicts.** A refusal, truncation, provider error,
  invalid schema, or non-verbatim span is its own status and never a scoring
  verdict.
- **Join by id, never by position.** Ids are the only join keys.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, computed_field, model_validator

from . import metrics_schema

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #

Disclosure = Literal["volunteer", "needs_followup", "hedge", "contradict"]
Importance = Literal["dealbreaker", "core", "color"]
Specificity = Literal["generic", "specific"]
Style = Literal["terse", "balanced", "rambling", "tangential"]

#: Role tags for ``llm.call``. A Literal, not a bare string, because the call
#: site, FakeProvider's keying, and per-role config in lab.yaml would otherwise
#: drift as three independent spellings (PLAN.md P9).
Role = Literal["persona_gen", "user_agent", "interviewer_followup", "extract", "aligner"]

#: ``abstained`` exists because ``extract.txt`` instructs the model to answer
#: ``unsure`` when the user did not say, while ``unsure`` is *also* a legitimate
#: truth value for ``relationship_goal`` and ``wants_kids``. Without it, a model
#: that correctly declines is recorded as having hallucinated (PLAN.md M4).
Verdict = Literal[
    "supported",
    "supported_offsheet",
    "unsupported",
    "abstained",
    "judge_error",
    "span_invalid",
]

#: Judge-only verdicts: an exact-field comparison can never produce these.
JUDGE_ONLY_VERDICTS: frozenset[str] = frozenset(
    {"supported_offsheet", "judge_error", "span_invalid"}
)

ExtractionStatus = Literal["ok", "schema_invalid", "provider_error", "refusal", "truncated"]

# --------------------------------------------------------------------------- #
# Profile field taxonomy (mirrors schema.json; asserted equal in tests)
# --------------------------------------------------------------------------- #

#: Compared with ``==`` after schema validation. Never sent to the judge.
EXACT_FIELDS: tuple[str, ...] = (
    "age",
    "relationship_goal",
    "wants_kids",
    "has_kids",
    "religion_importance",
)

#: Sent to the judge, which must cite a verbatim span from a user turn.
JUDGED_FIELDS: tuple[str, ...] = (
    "location",
    "occupation",
    "values",
    "interests",
    "life_events",
    "dealbreakers",
    "partner_preferences",
    "communication_style",
)

#: One claim per item, not one claim per field.
ARRAY_FIELDS: tuple[str, ...] = (
    "values",
    "interests",
    "life_events",
    "dealbreakers",
    "partner_preferences",
)

#: Shown in the report, excluded from precision and recall.
UNSCORED_FIELDS: tuple[str, ...] = ("summary",)

#: Field-based, not persona-``importance``-based, so ``precision_dealbreakers``
#: and ``recall_dealbreakers`` cut the same population (PLAN.md M14).
DEALBREAKER_FIELDS: tuple[str, ...] = (
    "wants_kids",
    "has_kids",
    "religion_importance",
    "dealbreakers",
)

SCORED_FIELDS: tuple[str, ...] = EXACT_FIELDS + JUDGED_FIELDS
ALL_FIELDS: tuple[str, ...] = SCORED_FIELDS + UNSCORED_FIELDS

#: Classes for each categorical exact field, sampled in Python rather than by the
#: persona-generation model (PLAN.md M2). Also the denominator basis for the
#: majority-class floor (PLAN.md M9).
EXACT_CLASSES: dict[str, tuple[str | bool, ...]] = {
    "relationship_goal": ("casual", "long_term", "marriage", "unsure"),
    "wants_kids": ("yes", "no", "unsure"),
    "has_kids": (True, False),
    "religion_importance": ("none", "low", "medium", "high"),
}

#: The enum member that means "the user did not say" — the ``abstained`` trigger.
UNSURE_MEMBER: dict[str, str] = {
    "relationship_goal": "unsure",
    "wants_kids": "unsure",
}

# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


class ProvenanceStamp(BaseModel):
    """What produced an artifact, so a number can always be traced to its inputs."""

    model_config = ConfigDict(extra="forbid")

    model: str
    prompt_hash: str
    params: dict
    code_version: str
    created_at: datetime

    def hashable_params(self) -> dict:
        """``params`` minus ``nonce``.

        Floor runs differ only by nonce; they must be distinguishable in the
        artifact record without changing the artifact's content identity
        (PLAN.md M5).
        """
        return {k: v for k, v in self.params.items() if k != "nonce"}


# --------------------------------------------------------------------------- #
# Truth sheets
# --------------------------------------------------------------------------- #


class Fact(BaseModel):
    """One atomic truth: one field, one value."""

    model_config = ConfigDict(extra="forbid")

    fact_id: str
    field: str
    value: str | bool | int
    specificity: Specificity
    disclosure: Disclosure
    importance: Importance
    #: ``contradict`` only: stated first, corrected later. Unused in v1 (the
    #: ``contradict`` disclosure type is deferred to v1.1) but kept in the
    #: contract so enabling it later is not a breaking change.
    decoy_value: str | bool | int | None = None

    @model_validator(mode="after")
    def _decoy_requires_contradict(self) -> Fact:
        if self.decoy_value is not None and self.disclosure != "contradict":
            raise ValueError(
                f"fact {self.fact_id}: decoy_value is only valid for disclosure='contradict', "
                f"got {self.disclosure!r}"
            )
        return self


class TruthSheet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persona_id: str
    seed: int
    style: Style
    demographics: dict[str, str | int]
    facts: list[Fact]
    bio: str
    provenance: ProvenanceStamp

    @model_validator(mode="after")
    def _check_facts(self) -> TruthSheet:
        ids = [f.fact_id for f in self.facts]
        if len(ids) != len(set(ids)):
            raise ValueError(f"persona {self.persona_id}: duplicate fact_ids")
        for field in EXACT_FIELDS:
            n = sum(1 for f in self.facts if f.field == field)
            if n != 1:
                raise ValueError(
                    f"persona {self.persona_id}: exact-scored field {field!r} must appear "
                    f"exactly once, found {n}"
                )
        unknown = {f.field for f in self.facts} - set(SCORED_FIELDS)
        if unknown:
            raise ValueError(
                f"persona {self.persona_id}: facts for unknown fields {sorted(unknown)}"
            )
        return self

    def fact(self, fact_id: str) -> Fact:
        for f in self.facts:
            if f.fact_id == fact_id:
                return f
        raise KeyError(f"no such fact: {fact_id}")


# --------------------------------------------------------------------------- #
# Question script
# --------------------------------------------------------------------------- #


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    text: str
    targets: list[str]
    max_followups: int = 1


class QuestionScript(BaseModel):
    model_config = ConfigDict(extra="forbid")

    script_id: str
    questions: list[Question]
    followup_trigger_words: int = 40
    #: ``None`` = as written; an int shuffles the order (stability runs).
    order_seed: int | None = None

    @model_validator(mode="after")
    def _check_questions(self) -> QuestionScript:
        ids = [q.question_id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate question_ids")
        for q in self.questions:
            unknown = set(q.targets) - set(ALL_FIELDS)
            if unknown:
                raise ValueError(f"question {q.question_id}: unknown targets {sorted(unknown)}")
        return self


# --------------------------------------------------------------------------- #
# Transcripts
# --------------------------------------------------------------------------- #


def word_count(text: str) -> int:
    """The lab's one definition of a word, used for every time-based metric."""
    return len(text.split())


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_id: str
    speaker: Literal["interviewer", "user"]
    #: Set on interviewer turns and on the user turn answering them. Follow-up
    #: turns carry the **parent** question_id; there is no separate id scheme
    #: (PLAN.md M15).
    question_id: str | None
    is_followup: bool = False
    text: str
    word_count: int
    #: User turns only, from the user agent's side channel. This is the ground
    #: truth for every recall and yield number, so it is itself audited
    #: (PLAN.md M10).
    disclosed_fact_ids: list[str] = []

    @model_validator(mode="after")
    def _check_turn(self) -> Turn:
        expected = word_count(self.text)
        if self.word_count != expected:
            raise ValueError(
                f"turn {self.turn_id}: word_count={self.word_count} but text has {expected} words"
            )
        if self.speaker != "user" and self.disclosed_fact_ids:
            raise ValueError(f"turn {self.turn_id}: only user turns may disclose facts")
        if self.is_followup and self.question_id is None:
            raise ValueError(f"turn {self.turn_id}: a follow-up must carry its parent question_id")
        if len(set(self.disclosed_fact_ids)) != len(self.disclosed_fact_ids):
            raise ValueError(f"turn {self.turn_id}: duplicate disclosed_fact_ids")
        return self


class Transcript(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript_id: str
    persona_id: str
    script_hash: str
    sim_seed: int
    noise_rate: float = 0.0
    noise_seed: int | None = None
    #: The clean transcript this was derived from. Time-based metrics read word
    #: counts from the source, because noise adds fillers and drops words and
    #: would otherwise move yield-per-minute for reasons unrelated to
    #: extraction (PLAN.md M8).
    source_transcript_id: str | None = None
    turns: list[Turn]
    simulated_minutes: float
    provenance: ProvenanceStamp

    @model_validator(mode="after")
    def _check_transcript(self) -> Transcript:
        ids = [t.turn_id for t in self.turns]
        if len(ids) != len(set(ids)):
            raise ValueError(f"transcript {self.transcript_id}: duplicate turn_ids")
        if self.noise_rate > 0:
            if self.noise_seed is None:
                raise ValueError(f"transcript {self.transcript_id}: noised copy needs noise_seed")
            if self.source_transcript_id is None:
                raise ValueError(
                    f"transcript {self.transcript_id}: noised copy needs source_transcript_id"
                )
        elif self.source_transcript_id is not None:
            raise ValueError(
                f"transcript {self.transcript_id}: source_transcript_id is for noised copies only"
            )
        expected = round(sum(t.word_count for t in self.turns) / 150.0, 4)
        if abs(self.simulated_minutes - expected) > 1e-4:
            raise ValueError(
                f"transcript {self.transcript_id}: simulated_minutes={self.simulated_minutes} "
                f"but turns total {expected}"
            )
        return self

    def turn(self, turn_id: str) -> Turn:
        for t in self.turns:
            if t.turn_id == turn_id:
                return t
        raise KeyError(f"no such turn: {turn_id}")


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    field: str
    value: str | bool | int


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extraction_id: str
    transcript_id: str
    tag: str
    prompt_hash: str
    schema_hash: str
    model: str
    status: ExtractionStatus
    profile: dict | None
    claims: list[Claim] = []
    raw: str
    provenance: ProvenanceStamp

    @model_validator(mode="after")
    def _check_status(self) -> Extraction:
        if (self.status == "ok") != (self.profile is not None):
            raise ValueError(
                f"extraction {self.extraction_id}: status={self.status!r} is inconsistent with "
                f"profile={'set' if self.profile is not None else 'None'}"
            )
        if self.status != "ok" and self.claims:
            raise ValueError(
                f"extraction {self.extraction_id}: claims are only produced when status == 'ok'"
            )
        ids = [c.claim_id for c in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError(f"extraction {self.extraction_id}: duplicate claim_ids")
        return self


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


class Alignment(BaseModel):
    """One claim's verdict, with the evidence that produced it."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    verdict: Verdict
    method: Literal["exact", "judge"]
    matched_fact_id: str | None = None
    span_turn_id: str | None = None
    #: Must occur verbatim in the named user turn after normalisation, else the
    #: verdict is ``span_invalid``.
    span_text: str | None = None
    specificity: Specificity | None = None

    @model_validator(mode="after")
    def _check_alignment(self) -> Alignment:
        where = f"alignment for claim {self.claim_id}"
        if self.span_text is not None and self.span_turn_id is None:
            raise ValueError(f"{where}: span_text without span_turn_id")
        if self.method == "exact":
            if self.verdict in JUDGE_ONLY_VERDICTS:
                raise ValueError(
                    f"{where}: verdict {self.verdict!r} is judge-only; an exact comparison "
                    f"cannot produce it"
                )
            if self.span_turn_id is not None:
                raise ValueError(f"{where}: exact comparisons never cite a span")
        if self.verdict == "supported_offsheet":
            if self.matched_fact_id is not None:
                raise ValueError(f"{where}: supported_offsheet means no matched fact")
            if self.span_turn_id is None:
                raise ValueError(f"{where}: supported_offsheet requires a valid span")
        if self.verdict == "supported":
            if self.matched_fact_id is None:
                raise ValueError(f"{where}: supported requires matched_fact_id")
            if self.method == "judge" and self.span_turn_id is None:
                raise ValueError(f"{where}: a judged 'supported' requires a cited span")
        return self


class FactCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_id: str
    captured: bool
    by_claim_id: str | None = None
    #: First user turn whose ``disclosed_fact_ids`` contains this fact. ``None``
    #: means the interview never elicited it, which is what separates pipeline
    #: recall from extraction recall (PLAN.md M6).
    disclosed_turn_id: str | None = None
    question_id: str | None = None
    via_followup: bool | None = None

    @model_validator(mode="after")
    def _check_coverage(self) -> FactCoverage:
        if not self.captured and self.by_claim_id is not None:
            raise ValueError(f"fact {self.fact_id}: by_claim_id set on an uncaptured fact")
        return self


class JudgeHealth(BaseModel):
    """Judge reliability, pooled.

    ``error_rate`` is over **judged claims**, not all claims: a denominator that
    included exact-method claims the judge never saw would deflate the rate and
    weaken the 5% gate (PLAN.md M11).
    """

    model_config = ConfigDict(extra="forbid")

    calls: int
    judged_claims: int
    errors: int
    span_invalid: int
    span_invalid_first_pass: int = 0

    @computed_field
    @property
    def error_rate(self) -> float:
        if self.judged_claims == 0:
            return 0.0
        return (self.errors + self.span_invalid) / self.judged_claims

    @computed_field
    @property
    def degraded(self) -> bool:
        return self.error_rate > 0.05


class Scores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extraction_id: str
    alignments: list[Alignment]
    coverage: list[FactCoverage]
    judge: JudgeHealth
    metrics: dict[str, float]
    n: dict[str, int]

    @model_validator(mode="after")
    def _check_metrics(self) -> Scores:
        for key in self.metrics:
            if not metrics_schema.is_known(key):
                raise ValueError(f"unknown metric key {key!r} (not in metrics_schema)")
            if metrics_schema.needs_denominator(key) and key not in self.n:
                raise ValueError(f"metric {key!r} is a rate/ratio but has no denominator in n")
        for key in self.n:
            if not metrics_schema.is_known(key):
                raise ValueError(f"unknown denominator key {key!r} (not in metrics_schema)")
        return self

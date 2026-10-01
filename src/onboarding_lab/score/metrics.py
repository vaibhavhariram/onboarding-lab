"""Metric computation for one extraction, and pooled aggregation across personas.

Every number the report prints comes from here, so the denominators are the
whole point. Four rules, each correcting a bug in the original spec:

- **Errors are never verdicts.** ``judge_error`` and ``span_invalid`` are
  excluded from every rate denominator and counted in judge health instead
  (CLAUDE.md rule 3). A refusal or truncation never reaches this module: it
  leaves ``Extraction.status != "ok"`` with no claims at all.
- **``abstained`` is excluded from the precision denominator** (PLAN.md M4). A
  model that correctly declines to answer must not be scored as having
  hallucinated. It still counts as not-captured for recall.
- **A zero denominator omits the key** from ``Scores.metrics``, with its ``0``
  recorded in ``Scores.n`` (PLAN.md M12). ``0.0`` reads as "we got everything
  wrong", which is the silent misreporting this harness exists to prevent.
- **Aggregation pools counts** rather than averaging per-persona ratios
  (PLAN.md M13), and Wilson intervals are printed only for proportions.

Joins are by id throughout. An alignment naming a claim the extraction does not
contain, or a coverage row naming a fact the truth sheet does not contain, is an
error rather than a dropped row: silently dropping either changes every number.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .. import metrics_schema
from ..metrics_schema import MetricKind
from ..models import (
    DEALBREAKER_FIELDS,
    EXACT_FIELDS,
    SCORED_FIELDS,
    Alignment,
    Extraction,
    Fact,
    FactCoverage,
    JudgeHealth,
    Scores,
    Transcript,
    TruthSheet,
)
from .disclosure_check import DisclosureReport, check_disclosures
from .stats import Tally, count, pool, rate, ratio, wilson_interval

#: Mirrors ``models.Transcript.simulated_minutes``: the lab's one definition of
#: how long a transcript would have taken to speak.
WORDS_PER_MINUTE = 150.0

#: Not verdicts. Excluded from every rate denominator except the judge-error
#: rate itself, which exists to measure them.
ERROR_VERDICTS: frozenset[str] = frozenset({"judge_error", "span_invalid"})

#: A claim that earns precision credit. ``supported_offsheet`` counts: extraction
#: was faithful to the transcript and the simulator leaked an off-sheet fact, which
#: is a simulator-quality problem tracked by ``offsheet_rate``.
CREDITED_VERDICTS: frozenset[str] = frozenset({"supported", "supported_offsheet"})


# --------------------------------------------------------------------------- #
# Per-extraction
# --------------------------------------------------------------------------- #


def compute_tallies(
    *,
    truth: TruthSheet,
    transcript: Transcript,
    extraction: Extraction,
    alignments: Sequence[Alignment],
    coverage: Sequence[FactCoverage],
    judge: JudgeHealth,
    clean_transcript: Transcript | None = None,
    claims_deduped: int = 0,
    disclosure: DisclosureReport | None = None,
    stability: Tally | None = None,
) -> dict[str, Tally]:
    """Every metric for one extraction, as numerator/denominator pairs.

    ``clean_transcript`` is the noise-0 source. Time-based metrics read their
    word counts from it, never from the scored copy: noise drops words and
    inserts fillers, which would move ``yield_per_minute`` across the noise axis
    for reasons that have nothing to do with extraction (PLAN.md M8). Scoring a
    noised transcript without supplying its source is an error, not a fallback.

    ``disclosure`` is computed here when not supplied, since both checks are
    pure Python over artifacts this function already has.
    """
    source = _time_source(transcript, clean_transcript)
    by_claim = _alignments_by_claim(alignments, extraction)
    field_of = {c.claim_id: c.field for c in extraction.claims}
    if disclosure is None:
        disclosure = check_disclosures(
            truth=truth,
            clean_transcript=source,
            coverage=coverage,
            alignments=alignments,
            span_transcript=transcript,
        )
    captured = _captured_fact_ids(truth, coverage, by_claim)
    credited = captured - disclosure.untagged_fact_ids

    tallies: dict[str, Tally] = {}
    tallies.update(_precision_tallies(alignments, field_of))
    tallies.update(_recall_tallies(truth=truth, coverage=coverage, captured=captured))
    tallies.update(_judge_tallies(judge))
    tallies.update(_yield_tallies(coverage=coverage, source=source, captured=credited))
    tallies.update(_followup_tallies(style=truth.style, coverage=coverage, captured=credited))
    tallies["claims_deduped"] = count(claims_deduped)
    tallies["untagged_capture"] = count(disclosure.untagged_count)
    tallies["disclosure_mismatch"] = disclosure.mismatch
    if stability is not None:
        tallies["stability_extraction"] = stability

    unknown = sorted(k for k in tallies if not metrics_schema.is_known(k))
    if unknown:
        raise ValueError(f"metric keys not in metrics_schema: {unknown}")
    return tallies


def score_extraction(
    *,
    truth: TruthSheet,
    transcript: Transcript,
    extraction: Extraction,
    alignments: Sequence[Alignment],
    coverage: Sequence[FactCoverage],
    judge: JudgeHealth,
    clean_transcript: Transcript | None = None,
    claims_deduped: int = 0,
    disclosure: DisclosureReport | None = None,
    stability: Tally | None = None,
) -> Scores:
    """``compute_tallies`` rendered into the ``Scores`` contract.

    A rate or ratio with a zero denominator is absent from ``metrics`` and
    present in ``n`` as ``0`` (PLAN.md M12). Counts carry no denominator.
    """
    tallies = compute_tallies(
        truth=truth,
        transcript=transcript,
        extraction=extraction,
        alignments=alignments,
        coverage=coverage,
        judge=judge,
        clean_transcript=clean_transcript,
        claims_deduped=claims_deduped,
        disclosure=disclosure,
        stability=stability,
    )
    metrics: dict[str, float] = {}
    n: dict[str, int] = {}
    for key, tally in sorted(tallies.items()):
        if tally.kind != "count":
            n[key] = tally.denominator
        value = tally.value
        if value is not None:
            metrics[key] = value
    return Scores(
        extraction_id=extraction.extraction_id,
        alignments=list(alignments),
        coverage=list(coverage),
        judge=judge,
        metrics=metrics,
        n=n,
    )


def tallies_from_scores(scores: Scores) -> dict[str, Tally]:
    """Recover numerator/denominator pairs from a score file, for pooling.

    The run aggregates from ``Scores`` on disk, and ``Scores`` stores the
    quotient rather than its parts. Every numerator here is an integer count (or
    an integer times ``WORDS_PER_MINUTE``) well under 2**53, so ``value * n``
    rounds back to it exactly.
    """
    out: dict[str, Tally] = {}
    for key, value in scores.metrics.items():
        kind = metrics_schema.classify(key)
        if kind == "count":
            out[key] = count(round(value))
            continue
        denominator = scores.n[key]
        if kind == "rate":
            out[key] = rate(round(value * denominator), denominator)
        else:
            facts = round(value * denominator / WORDS_PER_MINUTE)
            out[key] = ratio(facts * WORDS_PER_MINUTE, denominator)
    for key, denominator in scores.n.items():
        if key not in out:
            kind = metrics_schema.classify(key)
            out[key] = Tally(kind, 0.0, denominator)
    return out


# --------------------------------------------------------------------------- #
# Metric families
# --------------------------------------------------------------------------- #


def _precision_tallies(
    alignments: Sequence[Alignment], field_of: Mapping[str, str]
) -> dict[str, Tally]:
    """Precision, hallucination, off-sheet and abstention rates.

    ``scorable`` drops the two error states; ``judged`` is the precision
    population for ``offsheet_rate`` so that it is the same cut as ``precision``.
    ``abstention_rate`` keeps abstentions in its own denominator — they are the
    numerator, so excluding them could put the rate above 1.
    """
    scorable = [a for a in alignments if a.verdict not in ERROR_VERDICTS]
    scored = [a for a in scorable if a.verdict != "abstained"]
    judged = [a for a in scored if a.method == "judge"]

    out: dict[str, Tally] = {
        "precision": rate(_credited(scored), len(scored)),
        "hallucination_rate": rate(_unsupported(scored), len(scored)),
        "offsheet_rate": rate(
            sum(1 for a in judged if a.verdict == "supported_offsheet"), len(judged)
        ),
        "abstention_rate": rate(
            sum(1 for a in scorable if a.verdict == "abstained"), len(scorable)
        ),
    }

    dealbreakers = [a for a in scored if field_of[a.claim_id] in DEALBREAKER_FIELDS]
    out["precision_dealbreakers"] = rate(_credited(dealbreakers), len(dealbreakers))

    for field in SCORED_FIELDS:
        rows = [a for a in scored if field_of[a.claim_id] == field]
        out[f"precision_by_field.{field}"] = rate(_credited(rows), len(rows))
    return out


def _credited(alignments: Iterable[Alignment]) -> int:
    return sum(1 for a in alignments if a.verdict in CREDITED_VERDICTS)


def _unsupported(alignments: Iterable[Alignment]) -> int:
    return sum(1 for a in alignments if a.verdict == "unsupported")


def _captured_fact_ids(
    truth: TruthSheet,
    coverage: Sequence[FactCoverage],
    by_claim: Mapping[str, Alignment],
) -> frozenset[str]:
    """Facts that count as captured, re-checked against the citing verdict.

    ``FactCoverage.captured`` is set upstream; this re-reads the verdict of the
    claim it cites so that an ``abstained`` or ``unsupported`` claim can never
    count as a capture (PLAN.md M4). Only ``supported`` matches a fact —
    ``supported_offsheet`` carries no ``matched_fact_id`` by contract.
    """
    known = {f.fact_id for f in truth.facts}
    captured: set[str] = set()
    for cov in coverage:
        if cov.fact_id not in known:
            raise ValueError(
                f"coverage names fact {cov.fact_id!r}, absent from truth sheet {truth.persona_id}"
            )
        if not cov.captured:
            continue
        if cov.by_claim_id is not None:
            alignment = by_claim.get(cov.by_claim_id)
            if alignment is None:
                raise ValueError(
                    f"coverage for fact {cov.fact_id} cites claim "
                    f"{cov.by_claim_id!r}, which has no alignment"
                )
            if alignment.verdict != "supported":
                continue
        captured.add(cov.fact_id)
    return frozenset(captured)


def _recall_tallies(
    *,
    truth: TruthSheet,
    coverage: Sequence[FactCoverage],
    captured: frozenset[str],
) -> dict[str, Tally]:
    """Pipeline recall, elicited recall, and the cuts by field and disclosure.

    ``recall`` and ``recall_elicited`` differ by exactly the facts the interview
    never surfaced. Reporting only one would mix "nothing asked" with
    "extraction missed it", and that stage attribution is the whole pitch
    (PLAN.md M6). ``recall_elicited`` is the gated one.
    """
    elicited = {cov.fact_id for cov in coverage if cov.disclosed_turn_id is not None}

    def over(facts: Sequence[Fact]) -> Tally:
        return rate(sum(1 for f in facts if f.fact_id in captured), len(facts))

    out: dict[str, Tally] = {
        "recall": over(truth.facts),
        "recall_dealbreakers": over([f for f in truth.facts if f.field in DEALBREAKER_FIELDS]),
        "recall_exact": over([f for f in truth.facts if f.field in EXACT_FIELDS]),
    }
    elicited_facts = [f for f in truth.facts if f.fact_id in elicited]
    out["recall_elicited"] = over(elicited_facts)

    for field in SCORED_FIELDS:
        out[f"recall_by_field.{field}"] = over([f for f in truth.facts if f.field == field])
    for field in EXACT_FIELDS:
        out[f"recall_exact_by_field.{field}"] = over([f for f in truth.facts if f.field == field])
    for disclosure in sorted({f.disclosure for f in truth.facts}):
        out[f"recall_by_disclosure.{disclosure}"] = over(
            [f for f in truth.facts if f.disclosure == disclosure]
        )
    return out


def _judge_tallies(judge: JudgeHealth) -> dict[str, Tally]:
    """Judge health, over **judged claims**.

    A denominator that included exact-method claims the judge never saw would
    deflate the rate and weaken the 5% degraded gate (PLAN.md M11). The aligner
    retries once on ``span_invalid`` (PLAN.md amendment 5), so first-pass and
    final rates are both reported — a high first pass with a low final means the
    retry is carrying the result.
    """
    judged = judge.judged_claims
    return {
        "judge_error_rate": rate(judge.errors + judge.span_invalid, judged),
        "span_invalid_rate_first_pass": rate(judge.span_invalid_first_pass, judged),
        "span_invalid_rate_final": rate(judge.span_invalid, judged),
    }


def _yield_tallies(
    *,
    coverage: Sequence[FactCoverage],
    source: Transcript,
    captured: frozenset[str],
) -> dict[str, Tally]:
    """Facts per question and per interview minute.

    Word counts come from the clean source and **include the interviewer's
    words**: the metric measures interview time, and a verbose question does
    cost minutes (PLAN.md M15). Follow-up turns carry the parent ``question_id``,
    so grouping on it folds them into their parent question with no extra field.

    ``captured`` has ``untagged_capture`` already removed: a fact the
    self-report never claimed cannot be credited to a question (PLAN.md M10).
    """
    by_question: dict[str, int] = {}
    words: dict[str, int] = {}
    for turn in source.turns:
        if turn.question_id is None:
            continue
        words[turn.question_id] = words.get(turn.question_id, 0) + turn.word_count
        by_question.setdefault(turn.question_id, 0)
    for cov in coverage:
        if cov.fact_id not in captured or cov.question_id is None:
            continue
        by_question[cov.question_id] = by_question.get(cov.question_id, 0) + 1
        words.setdefault(cov.question_id, 0)

    out: dict[str, Tally] = {}
    for question_id, n in sorted(by_question.items()):
        out[f"yield.{question_id}"] = count(n)
        out[f"yield_per_minute.{question_id}"] = ratio(
            n * WORDS_PER_MINUTE, words.get(question_id, 0)
        )
    return out


def _followup_tallies(
    *,
    style: str,
    coverage: Sequence[FactCoverage],
    captured: frozenset[str],
) -> dict[str, Tally]:
    """Share of captured facts that a follow-up had to surface.

    Broken down by persona style because the headline number is largely an
    artifact of the sampled style mix (PLAN.md M15): a terse persona answers
    below the follow-up trigger on every question while a rambling one never
    trips it on length, so the aggregate moves with the mix rather than with the
    interviewer. Denominator = captured facts with a non-null ``via_followup``;
    ``captured`` has ``untagged_capture`` already removed.
    """
    rows = [cov for cov in coverage if cov.fact_id in captured and cov.via_followup is not None]
    tally = rate(sum(1 for cov in rows if cov.via_followup), len(rows))
    return {
        "followup_share": tally,
        f"followup_share_by_style.{style}": tally,
    }


# --------------------------------------------------------------------------- #
# The exact-field majority-class floor (PLAN.md M9)
# --------------------------------------------------------------------------- #


def exact_field_floor(truth_sheets: Sequence[TruthSheet]) -> dict[str, Tally]:
    """What a constant guesser scores per exact field, from the run's own sheets.

    ``has_kids`` is boolean and the other categorical exact fields are small
    enums, so a model that answers the same thing every time and reads nothing
    scores well above zero. Exact-field recall is only interpretable beside this
    floor, and the report shows the lift over it.

    Computed from the run's sheets rather than from ``EXACT_CLASSES`` because the
    floor that matters is the one in the data actually scored — a sampler bug
    that skews the class balance should move the floor, not hide behind a
    nominal 1/k.
    """
    out: dict[str, Tally] = {}
    for field in EXACT_FIELDS:
        counts: dict[str, int] = {}
        for sheet in truth_sheets:
            for fact in sheet.facts:
                if fact.field == field:
                    key = repr(fact.value)
                    counts[key] = counts.get(key, 0) + 1
        total = sum(counts.values())
        out[f"exact_floor.{field}"] = rate(max(counts.values()) if counts else 0, total)
    return out


def lift_over_floor(
    metrics: Mapping[str, float | None],
) -> dict[str, float | None]:
    """``recall_exact_by_field`` minus ``exact_floor``, per exact field.

    Derived rather than stored: ``metrics_schema`` has no key for it, and a
    difference of two rates is neither a proportion nor a ratio, so it has no
    denominator and no interval. ``None`` where either side is undefined.
    """
    out: dict[str, float | None] = {}
    for field in EXACT_FIELDS:
        recall = metrics.get(f"recall_exact_by_field.{field}")
        floor = metrics.get(f"exact_floor.{field}")
        out[field] = None if recall is None or floor is None else recall - floor
    return out


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Aggregate:
    """Pooled metrics for one tag, plus the per-persona vectors behind them.

    ``metrics`` carries **every** expected key, with ``None`` where the pooled
    denominator is zero — the report must never be missing a row, and ``None``
    is distinguishable from ``0.0`` in JSON where ``0.0`` would be a lie
    (PLAN.md M12). ``per_persona`` keeps the distribution so the report can show
    it beside the pooled headline (PLAN.md M13).
    """

    extraction_ids: tuple[str, ...]
    persona_ids: tuple[str, ...]
    metrics: dict[str, float | None]
    n: dict[str, int]
    ci: dict[str, tuple[float, float] | None]
    kinds: dict[str, MetricKind]
    per_persona: dict[str, list[float | None]]
    per_persona_n: dict[str, list[int | None]]
    judge: JudgeHealth
    excluded: dict[str, int]
    lift_over_floor: dict[str, float | None]
    floor: dict[str, float]

    @property
    def n_personas(self) -> int:
        return len(self.extraction_ids)

    @property
    def n_excluded(self) -> int:
        return sum(self.excluded.values())

    def persona_records(self) -> list[dict]:
        """The distribution as one record per persona, which is how it renders.

        The in-memory form is a vector per key, because that is what pooling and
        charting want; the file form is a record per persona, because that is
        what a table row is.
        """
        records = []
        for i, extraction_id in enumerate(self.extraction_ids):
            values = {k: v[i] for k, v in self.per_persona.items() if v[i] is not None}
            denominators = {k: v[i] for k, v in self.per_persona_n.items() if v[i] is not None}
            records.append(
                {
                    "persona_id": self.persona_ids[i],
                    "extraction_id": extraction_id,
                    "metrics": dict(sorted(values.items())),
                    "n": dict(sorted(denominators.items())),
                }
            )
        return records

    def to_dict(self) -> dict:
        """JSON-ready. ``None`` means undefined, never zero."""
        return {
            "n_personas": self.n_personas,
            "n_excluded": self.n_excluded,
            "excluded": dict(sorted(self.excluded.items())),
            "extraction_ids": list(self.extraction_ids),
            "judge": self.judge.model_dump(),
            "metrics": dict(sorted(self.metrics.items())),
            "n": dict(sorted(self.n.items())),
            "floor": dict(sorted(self.floor.items())),
            "ci": {k: list(v) if v else None for k, v in sorted(self.ci.items())},
            "kinds": dict(sorted(self.kinds.items())),
            "per_persona": self.persona_records(),
            "lift_over_floor": dict(sorted(self.lift_over_floor.items())),
        }


def aggregate(
    scores: Sequence[Scores],
    *,
    truth_sheets: Sequence[TruthSheet] = (),
    statuses: Mapping[str, str] | None = None,
    expected: Iterable[str] | None = None,
    persona_ids: Mapping[str, str] | None = None,
    floors: Mapping[str, object] | None = None,
) -> Aggregate:
    """Pool per-persona scores into one row set for the report.

    Pooled counts, not a mean of per-persona ratios (PLAN.md M13): a persona
    with two claims must not weigh the same as one with thirty. Judge health is
    pooled the same way and ``degraded`` is derived from the pooled rate — never
    an average of per-persona booleans (PLAN.md M11).

    ``statuses`` maps ``extraction_id -> Extraction.status`` so the report can
    print "n personas excluded"; a persona whose extraction was not ``ok``
    contributes no claims and therefore nothing to any precision denominator,
    and zero recall (PLAN.md M15).

    ``floors`` is carried through from ``floor.compute_floor`` so the report can
    print the measured floor beside every gated number rather than leaving the
    reader to take the gate on trust (PLAN.md P5).
    """
    per: list[dict[str, Tally]] = [tallies_from_scores(s) for s in scores]
    floor = exact_field_floor(truth_sheets) if truth_sheets else {}

    keys: set[str] = set(expected) if expected is not None else set()
    for tallies in per:
        keys |= set(tallies)
    keys |= set(floor)
    unknown = sorted(k for k in keys if not metrics_schema.is_known(k))
    if unknown:
        raise ValueError(f"metric keys not in metrics_schema: {unknown}")

    metrics: dict[str, float | None] = {}
    n: dict[str, int] = {}
    ci: dict[str, tuple[float, float] | None] = {}
    kinds: dict[str, MetricKind] = {}
    per_persona: dict[str, list[float | None]] = {}
    per_persona_n: dict[str, list[int | None]] = {}
    for key in sorted(keys):
        kind = metrics_schema.classify(key)
        kinds[key] = kind
        present = [t[key] for t in per if key in t]
        if key in floor:
            present = [floor[key]]
        pooled = pool(present) if present else Tally(kind, 0.0, 0)
        metrics[key] = pooled.value
        if kind != "count":
            n[key] = pooled.denominator
        ci[key] = (
            wilson_interval(pooled.numerator, pooled.denominator)
            if metrics_schema.supports_wilson(key) and pooled.denominator
            else None
        )
        per_persona[key] = [t[key].value if key in t else None for t in per]
        per_persona_n[key] = [
            t[key].denominator if key in t and kind != "count" else None for t in per
        ]

    judge = JudgeHealth(
        calls=sum(s.judge.calls for s in scores),
        judged_claims=sum(s.judge.judged_claims for s in scores),
        errors=sum(s.judge.errors for s in scores),
        span_invalid=sum(s.judge.span_invalid for s in scores),
        span_invalid_first_pass=sum(s.judge.span_invalid_first_pass for s in scores),
    )
    excluded: dict[str, int] = {}
    for s in scores:
        status = (statuses or {}).get(s.extraction_id, "ok")
        if status != "ok":
            excluded[status] = excluded.get(status, 0) + 1

    ids = tuple(s.extraction_id for s in scores)
    return Aggregate(
        extraction_ids=ids,
        persona_ids=tuple((persona_ids or {}).get(i, i) for i in ids),
        metrics=metrics,
        n=n,
        ci=ci,
        kinds=kinds,
        per_persona=per_persona,
        per_persona_n=per_persona_n,
        judge=judge,
        excluded=excluded,
        lift_over_floor=lift_over_floor(metrics),
        floor=_floor_values(floors),
    )


def _floor_values(floors: Mapping[str, object] | None) -> dict[str, float]:
    """Accept either plain numbers or ``floor.MetricFloor`` objects.

    Imported structurally rather than by type, so ``metrics`` does not depend on
    ``floor`` while ``floor`` depends on ``stats``.
    """
    out: dict[str, float] = {}
    for key, value in (floors or {}).items():
        resolved = getattr(value, "floor", value)
        if resolved is not None:
            out[key] = float(resolved)  # type: ignore[arg-type]
    return out


# --------------------------------------------------------------------------- #
# Joins
# --------------------------------------------------------------------------- #


def _alignments_by_claim(
    alignments: Sequence[Alignment], extraction: Extraction
) -> dict[str, Alignment]:
    """One alignment per claim, keyed by id, with both sides checked.

    A claim with no alignment, or an alignment for a claim the extraction does
    not contain, means the two artifacts were joined by position somewhere
    upstream (CLAUDE.md rule 4).
    """
    out: dict[str, Alignment] = {}
    for a in alignments:
        if a.claim_id in out:
            raise ValueError(f"two alignments for claim {a.claim_id!r}")
        out[a.claim_id] = a
    claim_ids = {c.claim_id for c in extraction.claims}
    orphans = sorted(set(out) - claim_ids)
    if orphans:
        raise ValueError(
            f"extraction {extraction.extraction_id}: alignments for unknown claims {orphans}"
        )
    unaligned = sorted(claim_ids - set(out))
    if unaligned:
        raise ValueError(
            f"extraction {extraction.extraction_id}: claims with no alignment {unaligned}"
        )
    return out


def _time_source(transcript: Transcript, clean: Transcript | None) -> Transcript:
    """The transcript whose word counts time-based metrics may read (M8)."""
    if clean is not None:
        if clean.noise_rate or clean.source_transcript_id is not None:
            raise ValueError(
                f"transcript {clean.transcript_id} was passed as the clean source but is "
                f"itself a noised copy"
            )
        if clean.persona_id != transcript.persona_id:
            raise ValueError(
                f"clean source {clean.transcript_id} belongs to persona "
                f"{clean.persona_id}, not {transcript.persona_id}"
            )
        return clean
    if transcript.source_transcript_id is not None or transcript.noise_rate:
        raise ValueError(
            f"transcript {transcript.transcript_id} is noised: pass its clean source, or "
            f"yield-per-minute moves with the noise rate rather than with extraction (M8)"
        )
    return transcript

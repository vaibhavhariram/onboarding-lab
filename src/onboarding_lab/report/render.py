"""Static HTML report, Markdown summary, and diff rendering.

The report *is* the demo (SPEC §5), so it is built to run from already-loaded
artifacts: every entry point here takes dicts, and the ``load_*`` helpers are the
only things that touch the filesystem. That is what makes ``make demo`` cost $0
and need no provider, no key, and no network (PLAN.md D1).

Two rules shape nearly every decision in this module.

**No placeholder numbers, ever.** A rate with a zero denominator is omitted
upstream with its ``0`` recorded in ``Scores.n`` (CLAUDE.md rule 6), so a key
missing from the aggregate means "nobody computed this", not "this is zero".
Rendering it as ``0.0`` would read as *we got everything wrong* — the exact class
of silent misreporting this project exists to prevent. Missing keys render as
``n/a``. With no audit labels the report says "judge not yet audited" in red
rather than inventing an agreement rate (SPEC §3.13), which is also why the cut
list is happy to drop ``lab audit`` last.

**Honest determinism.** The harness does not claim deterministic model output; it
measures run-to-run variance and gates only beyond it (PLAN.md P5). The measured
floor is therefore printed beside every gated number rather than buried in the
diff, because that measurement is the only thing making the gate sound.

Interval policy follows ``metrics_schema``: a 95% Wilson interval is printed
where ``supports_wilson`` is true and nowhere else. ``yield_per_minute`` is a
ratio, not a proportion, so it gets a unit instead of an interval (PLAN.md M13).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .. import metrics_schema
from ..models import (
    ALL_FIELDS,
    DEALBREAKER_FIELDS,
    EXACT_FIELDS,
    SCORED_FIELDS,
    Style,
)
from . import charts

TEMPLATE_DIR = Path(__file__).parent / "templates"

#: Two-sided 95%.
_Z = 1.959963984540054

#: Above this pooled judge error rate the run is marked ``DEGRADED`` in every
#: report (SPEC §1 rule 5, pooled per PLAN.md M11).
DEGRADED_THRESHOLD = 0.05

#: §3.14 section 6. Enforced here, not left to the caller.
MAX_HALLUCINATION_EXAMPLES = 5

#: The literal string the report prints instead of an agreement number.
NOT_AUDITED = "judge not yet audited"

#: Committed fallback, because ``make clean`` is ``rm -rf runs`` and would
#: otherwise destroy 30-40 minutes of hand labelling (PLAN.md D2).
COMMITTED_AUDIT_LABELS = Path("fixtures/audit_labels.yaml")

#: Order the disclosure table is printed in. ``contradict`` is deferred to v1.1
#: and so is normally absent; it is listed last rather than dropped, because the
#: literal stays in the contract and a v1.1 aggregate must not silently lose a
#: row.
DISCLOSURE_ORDER = ("volunteer", "needs_followup", "hedge", "contradict")

#: Styles in the contract's own order, not alphabetical: terse to rambling is the
#: axis the follow-up share moves along, so the table reads as a trend.
STYLE_ORDER: tuple[str, ...] = get_args(Style)


def _ordered(present: list[str] | set[str], canonical: tuple[str, ...]) -> list[str]:
    """Canonical order first, then anything unexpected, so a new value is visible
    rather than silently dropped."""
    known = [k for k in canonical if k in present]
    return known + sorted(k for k in present if k not in canonical)


# --------------------------------------------------------------------------- #
# Intervals
# --------------------------------------------------------------------------- #


def wilson(p: float, n: int, *, z: float = _Z) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion.

    Wilson rather than normal-approximation because the interesting rates sit
    near 0 or 1 (hallucination rate, judge error rate) where the normal interval
    runs outside [0, 1]. Returns ``None`` for a zero denominator: there is no
    interval to report, and a fabricated one would be worse than none.

    The interval ignores persona clustering; the README says so.
    """
    if n <= 0:
        return None
    p = min(max(p, 0.0), 1.0)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - margin), min(1.0, centre + margin)


# --------------------------------------------------------------------------- #
# One rendered metric
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Metric:
    """A metric as the report shows it, including the case where it is absent.

    ``value is None`` is a first-class state, not an error and not zero. The
    templates render it as ``n/a`` with the reason in a tooltip.
    """

    key: str
    label: str
    value: float | None
    n: int | None = None
    interval: tuple[float, float] | None = None
    floor: float | None = None
    kind: str | None = None
    note: str = ""

    @property
    def present(self) -> bool:
        return self.value is not None

    @property
    def gated(self) -> bool:
        return self.key in metrics_schema.GATED_KEYS

    @property
    def value_text(self) -> str:
        return format_value(self.value, self.kind)

    @property
    def n_text(self) -> str:
        if self.n is not None:
            return f"n = {self.n:,}"
        # A count has no denominator by design (metrics_schema.needs_denominator),
        # so it gets a dash. "n/a" is reserved for a denominator that should
        # exist and does not.
        return "\u2014" if self.kind == "count" else "n/a"

    @property
    def n_cell(self) -> str:
        """``n_text`` without the ``n =`` prefix, for a column already headed n."""
        if self.n is not None:
            return f"{self.n:,}"
        return "\u2014" if self.kind == "count" else "n/a"

    @property
    def interval_text(self) -> str:
        if self.interval is None:
            return ""
        lo, hi = self.interval
        return f"95% CI [{lo * 100:.1f}, {hi * 100:.1f}]"

    @property
    def floor_text(self) -> str:
        return "" if self.floor is None else f"±{self.floor * 100:.2f} pp"


def format_value(value: float | None, kind: str | None) -> str:
    """One formatter, so HTML and Markdown cannot drift apart."""
    if value is None:
        return "n/a"
    if kind == "rate":
        return f"{value * 100:.1f}%"
    if kind == "count":
        return f"{value:,.0f}"
    if kind == "ratio":
        return f"{value:.2f}"
    return f"{value:.4f}"


def _kind(key: str) -> str | None:
    try:
        return metrics_schema.classify(key)
    except KeyError:
        return None


# --------------------------------------------------------------------------- #
# Aggregate access
# --------------------------------------------------------------------------- #


class Aggregate:
    """Read-only view over ``aggregate.json``, tolerant of absent keys.

    Track C owns the aggregate's production; this wrapper is the report's only
    way in, so "absent" is handled in exactly one place. Unknown extra keys are
    ignored rather than rejected, so the report does not become a second schema
    the scorer has to satisfy.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self.metrics: dict[str, Any] = data.get("metrics") or {}
        self.n: dict[str, Any] = data.get("n") or {}
        self.floor: dict[str, Any] = data.get("floor") or {}

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def raw_value(self, key: str) -> float | None:
        """The metric, or ``None`` when absent or not a number."""
        value = self.metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        return float(value)

    def denominator(self, key: str) -> int | None:
        value = self.n.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        return int(value)

    def metric(self, key: str, label: str, *, note: str = "") -> Metric:
        kind = _kind(key)
        value = self.raw_value(key)
        n = self.denominator(key)
        interval = None
        if value is not None and n and kind is not None and metrics_schema.supports_wilson(key):
            interval = wilson(value, n)
        floor = self.floor.get(key)
        return Metric(
            key=key,
            label=label,
            value=value,
            n=n,
            interval=interval,
            floor=float(floor) if isinstance(floor, int | float) else None,
            kind=kind,
            note=note,
        )

    def keys_with_prefix(self, prefix: str) -> list[str]:
        want = f"{prefix}."
        return sorted(k for k in self.metrics if k.startswith(want))

    def suffixes(self, prefix: str) -> list[str]:
        """Suffixes present under a dynamic prefix, in either metrics or n.

        Reads ``n`` too, so a question or field whose rate was omitted for a zero
        denominator still gets a row saying so instead of vanishing.
        """
        want = f"{prefix}."
        found = {k[len(want) :] for k in self.metrics if k.startswith(want)}
        found |= {k[len(want) :] for k in self.n if k.startswith(want)}
        return sorted(found)


# --------------------------------------------------------------------------- #
# Loaders — the only filesystem access in this module
# --------------------------------------------------------------------------- #


def load_aggregate(path: str | Path) -> dict[str, Any]:
    import json

    return json.loads(Path(path).read_text())


def _normalise_labels(raw: Any) -> dict[str, bool | None]:
    """Accept both label spellings, keyed by content hash either way.

    ``lab audit`` writes ``<hash>: {agree: null}`` per SPEC §3.13; a human
    editing the file by hand naturally shortens that to ``<hash>: true``. Both
    are read, because a report that silently ignored half the labels would print
    a wrong agreement rate, which is worse than printing none.
    """
    if isinstance(raw, dict) and "labels" in raw:
        raw = raw["labels"]
    if not isinstance(raw, dict):
        return {}
    out: dict[str, bool | None] = {}
    for key, entry in raw.items():
        value = entry.get("agree") if isinstance(entry, dict) else entry
        out[str(key)] = value if isinstance(value, bool) else None
    return out


def load_audit_labels(
    *,
    run_local: str | Path | None = None,
    committed: str | Path | None = COMMITTED_AUDIT_LABELS,
) -> tuple[dict[str, bool | None], str | None]:
    """Hand-filled judge labels, run-local first, committed fixture second.

    ``make clean`` is ``rm -rf runs .cache``, so labels written inside the
    gitignored run directory do not survive it. They are committed at
    ``fixtures/audit_labels.yaml`` and keyed by **content hash, not list
    position**, so re-running ``lab audit`` with a different sample size or seed
    does not silently re-point every label at a different alignment
    (PLAN.md D2).

    Returns ``({}, None)`` when neither file exists, which is what puts the red
    banner in the report.
    """
    for path, source in ((run_local, "run-local"), (committed, "committed fixture")):
        if path is None:
            continue
        p = Path(path)
        if not p.is_file():
            continue
        labels = _normalise_labels(yaml.safe_load(p.read_text()))
        if labels:
            return labels, f"{source} ({p})"
    return {}, None


def audit_agreement(
    labels: dict[str, bool | None] | None, *, source: str | None = None
) -> dict[str, Any] | None:
    """``agreed / labeled`` with n, or ``None`` when nothing is labelled.

    ``agree: null`` entries are unlabelled, not disagreements: ``lab audit``
    writes one null per sampled item and the author fills them in, so counting
    nulls as false would turn an unfinished label pass into a bad judge.
    """
    if not labels:
        return None
    decided = [v for v in labels.values() if isinstance(v, bool)]
    if not decided:
        return None
    agreed = sum(1 for v in decided if v)
    rate = agreed / len(decided)
    return {
        "metric": Metric(
            key="audit_agreement",
            label="Audit agreement",
            value=rate,
            n=len(decided),
            interval=wilson(rate, len(decided)),
            kind="rate",
            note="hand-labelled sample of judged alignments",
        ),
        "agreed": agreed,
        "labeled": len(decided),
        "sampled": len(labels),
        "source": source,
    }


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #


def _judge_block(agg: Aggregate) -> dict[str, Any]:
    """Pooled judge health.

    ``error_rate`` is over **judged claims**: a denominator that included
    exact-method claims the judge never saw would deflate the rate and weaken
    the 5% gate (PLAN.md M11). Pooled across personas, never an average of
    per-persona booleans.
    """
    judge = agg.get("judge") or {}
    judged = judge.get("judged_claims")
    errors = judge.get("errors")
    span_invalid = judge.get("span_invalid")
    rate: float | None = None
    if isinstance(judged, int) and judged > 0:
        rate = ((errors or 0) + (span_invalid or 0)) / judged
    return {
        "raw": judge,
        "judged_claims": judged,
        "metric": Metric(
            key="judge_error_rate",
            label="Judge error rate",
            value=rate,
            n=judged if isinstance(judged, int) else None,
            interval=wilson(rate, judged) if rate is not None and judged else None,
            floor=None,
            kind="rate",
            note="(judge_error + span_invalid) / judged claims, pooled",
        ),
        "degraded": rate is not None and rate > DEGRADED_THRESHOLD,
        "threshold_text": f"{DEGRADED_THRESHOLD * 100:.0f}%",
    }


def _headline(agg: Aggregate, judge: dict[str, Any], audit: dict[str, Any] | None) -> list[Metric]:
    """§3.14 section 1, plus the rows the original spec lacked.

    ``recall`` and ``recall_elicited`` are both here and labelled by the stage
    they measure. One number would mix "the interview never elicited the fact"
    with "extraction missed it", and that stage attribution is the whole pitch
    (PLAN.md M6).
    """
    rows = [
        agg.metric("precision", "Precision", note="supported + offsheet / judged-eligible claims"),
        agg.metric(
            "hallucination_rate", "Hallucination rate", note="unsupported / same denominator"
        ),
        agg.metric(
            "precision_dealbreakers",
            "Precision, dealbreaker fields",
            note=f"fields: {', '.join(DEALBREAKER_FIELDS)}",
        ),
        agg.metric(
            "recall",
            "Recall (pipeline)",
            note="captured / all facts — includes facts the interview never elicited",
        ),
        agg.metric(
            "recall_elicited",
            "Recall (extraction only)",
            note="captured / facts actually disclosed — the metric the diff gate uses",
        ),
        agg.metric(
            "recall_dealbreakers",
            "Recall, dealbreaker fields",
            note=f"fields: {', '.join(DEALBREAKER_FIELDS)}",
        ),
        agg.metric(
            "recall_exact",
            "Recall, exact-scored fields",
            note="no model in the loop; majority-class floor per field below",
        ),
        agg.metric(
            "abstention_rate",
            "Abstention rate",
            note="declining to answer is its own verdict, not a hallucination",
        ),
        agg.metric(
            "stability_extraction",
            "Stability (extraction)",
            note="pairwise Jaccard of captured fact-id sets across the floor runs",
        ),
        judge["metric"],
    ]
    if audit is not None:
        rows.append(audit["metric"])
    return rows


def _health(agg: Aggregate) -> list[Metric]:
    """Health statistics, **not gates** (PLAN.md M10).

    ``disclosed_fact_ids`` is the simulator's own side channel and is ground
    truth for every recall and yield number, so it is checked rather than taken
    on trust. A mismatch is a simulation-quality signal; failing a build on it
    would be gating the wrong thing.
    """
    return [
        agg.metric(
            "disclosure_mismatch",
            "Disclosure mismatch",
            note="judged specific facts only, anchor-token match, "
            "on the clean transcript rather than the noised copy",
        ),
        agg.metric(
            "untagged_capture",
            "Untagged capture",
            note="captured with a valid span but tagged in no disclosed_fact_ids; "
            "excluded from yield and from the follow-up share denominator",
        ),
        agg.metric(
            "offsheet_rate",
            "Offsheet rate",
            note="faithful to the transcript but not on the truth sheet — "
            "a simulation leak, not an extraction hallucination",
        ),
        agg.metric(
            "span_invalid_rate_first_pass",
            "Span invalid (first pass)",
            note="before the single targeted re-ask",
        ),
        agg.metric(
            "span_invalid_rate_final",
            "Span invalid (final)",
            note="after the re-ask; still an error state, never a verdict",
        ),
        agg.metric(
            "claims_deduped", "Claims deduped", note="duplicate (field, value) claims dropped"
        ),
    ]


def _exact_floor_rows(agg: Aggregate) -> list[dict[str, Any]]:
    """Exact-field recall beside its constant-majority-class floor (PLAN.md M9).

    Exact-field recall is meaningless on its own: a model that always answered
    the most common class would score the floor, which under the Python sampler
    is 1/k by construction. The lift is the only part that is evidence.
    """
    rows = []
    for field in EXACT_FIELDS:
        recall = agg.metric(f"recall_exact_by_field.{field}", field)
        floor = agg.metric(f"exact_floor.{field}", f"{field} floor")
        lift = None
        if recall.value is not None and floor.value is not None:
            lift = recall.value - floor.value
        rows.append(
            {
                "field": field,
                "recall": recall,
                "floor": floor,
                "lift": lift,
                "lift_text": "n/a" if lift is None else f"{lift * 100:+.1f} pp",
            }
        )
    return rows


def _field_rows(agg: Aggregate) -> list[dict[str, Any]]:
    """§3.14 section 2. Every scored field gets a row even when both rates are
    absent, so a field the extraction never produced is visible rather than
    missing."""
    return [
        {
            "field": field,
            "exact": field in EXACT_FIELDS,
            "dealbreaker": field in DEALBREAKER_FIELDS,
            "precision": agg.metric(f"precision_by_field.{field}", field),
            "recall": agg.metric(f"recall_by_field.{field}", field),
        }
        for field in SCORED_FIELDS
    ]


def _yield_section(agg: Aggregate) -> dict[str, Any]:
    """§3.14 section 3.

    ``yield_per_minute`` counts interviewer words too: it measures interview
    time, not speaking time. Word counts come from the **source** (noise-0)
    transcript, because noise adds fillers and drops words and would otherwise
    move the ratio for reasons unrelated to extraction (PLAN.md M8).
    """
    question_ids = agg.suffixes("yield_per_minute") or agg.suffixes("yield")
    rows = [
        {
            "question_id": qid,
            "count": agg.metric(f"yield.{qid}", qid),
            "per_minute": agg.metric(f"yield_per_minute.{qid}", qid),
        }
        for qid in question_ids
    ]
    aux = agg.get("followup_share_by_question") or {}
    overlay = {
        str(q): float(v)
        for q, v in aux.items()
        if isinstance(v, int | float) and not isinstance(v, bool)
    }
    points = [
        (r["question_id"], r["per_minute"].value) for r in rows if r["per_minute"].value is not None
    ]
    svg = None
    if points:
        svg = charts.yield_per_minute_chart(points, followup_share_by_question=overlay)

    styles = _ordered(agg.suffixes("followup_share_by_style"), STYLE_ORDER)
    return {
        "rows": rows,
        "chart": svg,
        "followup_share": agg.metric(
            "followup_share",
            "Follow-up share",
            note="captured facts with a non-null via_followup / captured facts",
        ),
        "followup_marked": bool(overlay),
        "by_style": [agg.metric(f"followup_share_by_style.{style}", style) for style in styles],
    }


def _noise_section(agg: Aggregate) -> dict[str, Any]:
    """§3.14 section 4. Points come from the aggregate's own noise sweep.

    A level whose metric is absent is rendered ``n/a`` in the table and skipped
    in the chart, never plotted at zero.
    """
    rows = []
    for entry in agg.get("noise_curve") or []:
        rate = entry.get("noise_rate")
        if not isinstance(rate, int | float):
            continue
        level = Aggregate(entry)
        rows.append(
            {
                "noise_rate": float(rate),
                "rate_text": f"{float(rate) * 100:.0f}%",
                "gated": float(rate) == 0.0,
                "precision": level.metric("precision", "precision"),
                "recall": level.metric("recall", "recall"),
                "recall_elicited": level.metric("recall_elicited", "recall_elicited"),
            }
        )
    rows.sort(key=lambda r: r["noise_rate"])
    svg = None
    if rows:
        svg = charts.noise_curve_chart(
            [(r["noise_rate"], r["precision"].value, r["recall"].value) for r in rows]
        )
    return {"rows": rows, "chart": svg}


def _disclosure_rows(agg: Aggregate) -> list[dict[str, Any]]:
    """§3.14 section 5.

    ``hedge`` reads near zero unless its semantics are written down, so the
    definition travels with the table: a hedge fact counts as captured when it
    is matched at ``generic`` specificity (PLAN.md M15).
    """
    ordered = _ordered(agg.suffixes("recall_by_disclosure"), DISCLOSURE_ORDER)
    return [
        {"disclosure": d, "recall": agg.metric(f"recall_by_disclosure.{d}", d)} for d in ordered
    ]


def _hallucinations(agg: Aggregate) -> list[dict[str, Any]]:
    """§3.14 section 6, capped at five here rather than upstream.

    Each one carries the field, the claimed value, and the closest user turn,
    because "this is a hallucination" is only checkable with the transcript text
    beside it — and hand-checking these is a Phase 4 task.
    """
    out = []
    for item in (agg.get("hallucinations") or [])[:MAX_HALLUCINATION_EXAMPLES]:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "persona_id": item.get("persona_id"),
                "claim_id": item.get("claim_id"),
                "field": item.get("field"),
                "claimed_value": item.get("claimed_value"),
                "turn_id": item.get("turn_id"),
                "question_id": item.get("question_id"),
                "turn_text": item.get("turn_text"),
            }
        )
    return out


def _style_mix(mix: dict[str, Any]) -> list[tuple[str, Any]]:
    """Style mix in the contract's order; the follow-up share depends on it."""
    return [(style, mix[style]) for style in _ordered(list(mix), STYLE_ORDER)]


def _flatten(value: Any) -> str:
    """``{"effort": "low"} -> 'effort=low'``, for the manifest's nested records."""
    if isinstance(value, dict):
        return "  ".join(f"{k}={value[k]}" for k in sorted(value))
    return str(value)


def _manifest_section(agg: Aggregate) -> dict[str, Any]:
    """§3.14 section 7, plus the persona counters and the exclusion line.

    Silently dropping a persona changes every number in the report, so the
    requested / ok / failed counters are printed next to the metrics they
    affect, not buried in a log (PLAN.md A2, M15).
    """
    manifest = agg.get("manifest") or {}
    personas = manifest.get("personas") or {}
    schema_invalid = personas.get("schema_invalid") or 0
    return {
        "raw": manifest,
        "personas": personas,
        "excluded_text": (
            f"{schema_invalid} personas excluded: schema_invalid"
            if isinstance(schema_invalid, int) and schema_invalid > 0
            else None
        ),
        "usage": manifest.get("usage") or {},
        "style_mix": _style_mix(manifest.get("style_mix") or {}),
        "hashes": manifest.get("hashes") or {},
        "seeds": manifest.get("seeds") or {},
        "models": manifest.get("models") or {},
        # Flattened to strings here rather than in the template: the values are
        # heterogeneous (ints, strings, bools) and formatting them in Jinja is
        # how a template ends up with a stray repr in it.
        "role_params": {
            role: _flatten(params) for role, params in (manifest.get("role_params") or {}).items()
        },
        "tags": {tag: _flatten(ident) for tag, ident in (manifest.get("tags") or {}).items()},
    }


def build_context(
    aggregate: dict[str, Any],
    *,
    audit_labels: dict[str, bool | None] | None = None,
    audit_source: str | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Everything both templates need, computed once.

    Formatting lives on ``Metric`` rather than in the templates so the HTML and
    the Markdown cannot drift into disagreeing about the same number.
    """
    agg = Aggregate(aggregate)
    judge = _judge_block(agg)
    audit = audit_agreement(audit_labels, source=audit_source)
    return {
        "tag": agg.get("tag"),
        "run_id": agg.get("run_id"),
        "headline_noise_rate": agg.get("headline_noise_rate"),
        "generated_at": (generated_at or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC"),
        "headline": _headline(agg, judge, audit),
        "judge": judge,
        "audit": audit,
        "not_audited": NOT_AUDITED,
        "health": _health(agg),
        "exact_floor_rows": _exact_floor_rows(agg),
        "field_rows": _field_rows(agg),
        "yield_section": _yield_section(agg),
        "noise": _noise_section(agg),
        "disclosure_rows": _disclosure_rows(agg),
        "hallucinations": _hallucinations(agg),
        "manifest": _manifest_section(agg),
        "gated_keys": metrics_schema.GATED_KEYS,
        "unscored_fields": tuple(f for f in ALL_FIELDS if f not in SCORED_FIELDS),
        "max_examples": MAX_HALLUCINATION_EXAMPLES,
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _autoescape(name: str | None) -> bool:
    """HTML escapes; Markdown must not, or every ``&`` and ``<`` is mangled."""
    return bool(name) and ".html" in str(name)


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=_autoescape,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def render_html(aggregate: dict[str, Any], **kwargs: Any) -> str:
    """Self-contained ``index.html``: inline CSS, inline SVG, no JavaScript."""
    return _env().get_template("report.html.j2").render(**build_context(aggregate, **kwargs))


def render_summary(aggregate: dict[str, Any], **kwargs: Any) -> str:
    """``summary.md``, for the README and as the CI pull-request comment."""
    return _env().get_template("summary.md.j2").render(**build_context(aggregate, **kwargs))


def write_report(
    aggregate: dict[str, Any],
    out_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Path]:
    """Write ``index.html`` and ``summary.md``. Both come from one context."""
    context = build_context(aggregate, **kwargs)
    env = _env()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, template in (("index.html", "report.html.j2"), ("summary.md", "summary.md.j2")):
        path = out / name
        path.write_text(env.get_template(template).render(**context))
        written[name] = path
    return written


# --------------------------------------------------------------------------- #
# Diff (§3.15)
# --------------------------------------------------------------------------- #

#: Verdicts the renderer understands. Anything else is printed as given rather
#: than coerced, because the scorer owns the decision and exit codes are the
#: CLI's job.
_VERDICT_LABELS = {
    "pass": "pass",
    "regression": "REGRESSION",
    "improved": "improved",
    "ungated": "not gated",
    "unknown": "n/a",
}


def _derive_verdict(row: dict[str, Any]) -> str:
    """Fallback only, for a row that arrives without a verdict.

    The gate is ``candidate < baseline_mean - floor`` on gated keys only
    (SPEC §3.12); everything else is reported and not gated.
    """
    key = row.get("metric")
    base, cand, floor = row.get("baseline_mean"), row.get("candidate"), row.get("floor")
    if not all(isinstance(v, int | float) for v in (base, cand, floor)):
        return "unknown"
    if key not in metrics_schema.GATED_KEYS:
        return "ungated"
    return "regression" if cand < base - floor else "pass"  # type: ignore[operator]


def diff_rows(diff: dict[str, Any]) -> list[dict[str, Any]]:
    """The six columns of §3.15: metric, baseline mean, candidate, delta, floor, verdict."""
    rows = []
    for row in diff.get("rows") or []:
        key = str(row.get("metric", ""))
        kind = _kind(key)
        base = row.get("baseline_mean")
        cand = row.get("candidate")
        delta = row.get("delta")
        if delta is None and isinstance(base, int | float) and isinstance(cand, int | float):
            delta = cand - base
        floor = row.get("floor")
        verdict = row.get("verdict") or _derive_verdict(row)
        rows.append(
            {
                "metric": key,
                "gated": key in metrics_schema.GATED_KEYS,
                "baseline_mean": format_value(
                    base if isinstance(base, int | float) else None, kind
                ),
                "candidate": format_value(cand if isinstance(cand, int | float) else None, kind),
                "delta": (
                    f"{delta * 100:+.1f} pp"
                    if isinstance(delta, int | float) and kind == "rate"
                    else format_value(delta if isinstance(delta, int | float) else None, kind)
                ),
                "floor": (
                    f"±{floor * 100:.2f} pp"
                    if isinstance(floor, int | float) and kind == "rate"
                    else format_value(floor if isinstance(floor, int | float) else None, kind)
                ),
                "verdict": _VERDICT_LABELS.get(str(verdict), str(verdict)),
            }
        )
    return rows


def render_diff(diff: dict[str, Any], *, generated_at: datetime | None = None) -> str:
    """Markdown diff table.

    Exit codes are the CLI's job; this only renders. The floor column is the
    measured run-to-run variance, which is what makes a "regression" verdict
    mean more than noise (PLAN.md P5).
    """
    baseline = diff.get("baseline", "baseline")
    candidate = diff.get("candidate", "candidate")
    rows = diff_rows(diff)
    stamp = (generated_at or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    floor_runs = diff.get("floor_runs")

    lines = [
        f"# Diff: `{baseline}` → `{candidate}`",
        "",
        f"Generated {stamp}."
        + (f" Floor measured over {floor_runs} extraction runs." if floor_runs else ""),
        "",
        # Sigma below is the spec's own notation for the variance floor.
        "The floor is the **measured** run-to-run variance (`max(2σ, 0.005)`), not an",  # noqa: RUF001
        "assumed zero. Only the gated metrics fail a build; the rest are reported.",
        "",
        "| metric | baseline mean | candidate | delta | floor | verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        name = f"**{row['metric']}**" if row["gated"] else row["metric"]
        lines.append(
            f"| {name} | {row['baseline_mean']} | {row['candidate']} "
            f"| {row['delta']} | {row['floor']} | {row['verdict']} |"
        )
    if not rows:
        lines.append("| _no metrics compared_ | n/a | n/a | n/a | n/a | n/a |")
    lines += [
        "",
        f"Gated metrics (**bold**): {', '.join(f'`{k}`' for k in metrics_schema.GATED_KEYS)}.",
        "",
    ]
    return "\n".join(lines)


def write_diff(diff: dict[str, Any], out_dir: str | Path, **kwargs: Any) -> Path:
    """``report/diff-<baseline>-<candidate>.md`` (§3.15)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"diff-{diff.get('baseline', 'baseline')}-{diff.get('candidate', 'candidate')}.md"
    path.write_text(render_diff(diff, **kwargs))
    return path

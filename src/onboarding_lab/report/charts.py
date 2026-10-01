"""Inline SVG charts for the HTML report.

Matplotlib's SVG backend only — no JavaScript, no CDN, no build step, so the
report is one self-contained file a reviewer can open from a clone or email to
someone (PLAN.md D1: ``make demo`` costs $0 and needs no key).

Three things are done to the backend's output before it is embedded:

- **Ids are namespaced.** Matplotlib numbers its ids from 1 per figure
  (``axes_1``, ``patch_3``, and hash-named ``clipPath`` ids), so two inline
  figures in one document collide and the second one's clip paths can resolve
  against the first one's. Every ``id`` and every ``url(#...)`` reference gets a
  per-chart prefix.
- **``width``/``height`` are dropped, ``viewBox`` kept**, so the chart scales to
  its container on a laptop and to the page when printed.
- **The XML prolog and DOCTYPE are stripped**, because they are illegal inside an
  HTML document.

Output is deterministic: no timestamp metadata and a fixed ``svg.hashsalt``, so
re-rendering the same aggregate produces a byte-identical report and the diff of
a committed report shows only real changes.

Only the two charts §3.14 specifies are drawn. Everything else is a table,
because a table is clearer and prints better.
"""

from __future__ import annotations

import io
import re
from html import escape

import matplotlib

matplotlib.use("svg")

# The backend must be selected before pyplot is imported.
import matplotlib.pyplot as plt

#: Two series at most, so two colours. Chosen to stay distinguishable in
#: greyscale print (clearly different lightness) and to pass contrast against
#: the report's off-white background; marker and dash style carry the same
#: information again for anyone who cannot use colour.
INK = "#1f1d1b"
MUTED = "#6b6560"
GRID = "#ded9d2"
SERIES_A = "#2f5d50"
SERIES_B = "#9c5424"

_BASE_RC: dict[str, object] = {
    # Real text, not outlines: selectable, searchable, and a fraction of the
    # size. The font stack degrades to a generic serif anywhere.
    "svg.fonttype": "none",
    "svg.hashsalt": "onboarding-lab",
    "font.family": "serif",
    "font.serif": ["Georgia", "Times New Roman", "serif"],
    "font.size": 9.0,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 10.0,
    "axes.titlelocation": "left",
    "axes.titlepad": 10.0,
    "axes.labelsize": 9.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.frameon": False,
    "legend.fontsize": 8.0,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.transparent": True,
}


def _namespace_ids(svg: str, prefix: str) -> str:
    """Prefix every id and internal reference so two figures cannot collide."""
    svg = re.sub(r'id="([^"]+)"', lambda m: f'id="{prefix}{m.group(1)}"', svg)
    svg = re.sub(r"url\(#([^)]+)\)", lambda m: f"url(#{prefix}{m.group(1)})", svg)
    return re.sub(
        r'(xlink:href|href)="#([^"]+)"',
        lambda m: f'{m.group(1)}="#{prefix}{m.group(2)}"',
        svg,
    )


def _to_inline_svg(fig: plt.Figure, *, prefix: str, label: str) -> str:
    buf = io.StringIO()
    # metadata Date=None keeps the output byte-stable across renders.
    fig.savefig(buf, format="svg", metadata={"Date": None}, bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()

    # Drop the RDF metadata block. Nothing reads it, and its purl.org and
    # creativecommons.org URIs are the only http(s) strings in the document that
    # are not XML namespace declarations — removing them lets "the report
    # contains no external URL" be checked mechanically.
    svg = re.sub(r"<metadata>.*?</metadata>", "", svg, flags=re.DOTALL)

    start = svg.index("<svg")
    end = svg.index(">", start)
    tag = re.sub(r'\s(width|height)="[^"]*"', "", svg[start : end + 1])
    tag = f'{tag[:-1]} role="img" aria-label="{escape(label, quote=True)}">'
    return _namespace_ids(tag + svg[end + 1 :], prefix)


def yield_per_minute_chart(
    points: list[tuple[str, float]],
    *,
    followup_share_by_question: dict[str, float] | None = None,
) -> str:
    """Captured facts per interview minute, by question.

    No interval: ``yield_per_minute`` is a ratio, not a proportion, and Wilson
    does not apply to it (PLAN.md M13). The axis label carries the unit instead.

    The follow-up share is marked as a hatched lower portion of each bar **only
    where the aggregate supplies a per-question share**. Nothing is estimated: a
    question with no share recorded gets a plain bar, and the pooled share is
    stated in the report's caption instead.
    """
    if not points:
        raise ValueError("yield_per_minute_chart needs at least one question")

    shares = followup_share_by_question or {}
    labels = [q for q, _ in points]
    values = [v for _, v in points]

    with plt.rc_context(_BASE_RC):
        fig, ax = plt.subplots(figsize=(7.4, 2.9))
        ax.bar(labels, values, color=SERIES_A, width=0.62, label="all captured facts")
        marked = [(i, v * shares[q]) for i, (q, v) in enumerate(points) if q in shares]
        if marked:
            ax.bar(
                [i for i, _ in marked],
                [h for _, h in marked],
                color=SERIES_A,
                width=0.62,
                hatch="////",
                edgecolor="#ffffff",
                linewidth=0.0,
                label="arrived via a follow-up",
            )
            # Above the plot area, on the title's row: inside the axes it sits on
            # top of the tallest bar, which is the one worth reading.
            ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2)
        ax.set_xlabel("question id")
        ax.set_ylabel("captured facts\nper interview minute")
        # Short, because the legend shares this row and the section's own lede
        # already carries the definition.
        ax.set_title("Per-question yield")
        ax.margins(x=0.01)
        return _to_inline_svg(
            fig,
            prefix="yq-",
            label=(
                "Bar chart of captured facts per interview minute for each question in the "
                "script. The same values are listed in the table below the chart."
            ),
        )


def noise_curve_chart(points: list[tuple[float, float | None, float | None]]) -> str:
    """Precision and recall against ASR noise rate.

    A level whose metric is absent from the aggregate is *skipped*, not plotted
    at zero: a zero-denominator rate is deliberately omitted upstream and a
    point at 0.0 would read as a collapse in quality (CLAUDE.md rule 6).
    """
    if not points:
        raise ValueError("noise_curve_chart needs at least one noise level")

    ordered = sorted(points, key=lambda p: p[0])
    with plt.rc_context(_BASE_RC):
        fig, ax = plt.subplots(figsize=(5.2, 3.0))
        plotted = 0
        for idx, (name, colour, marker, dashes) in enumerate(
            (
                ("precision", SERIES_A, "o", (None, None)),
                ("recall", SERIES_B, "s", (4, 2)),
            )
        ):
            xs = [p[0] * 100 for p in ordered if p[idx + 1] is not None]
            ys = [p[idx + 1] * 100 for p in ordered if p[idx + 1] is not None]  # type: ignore[operator]
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                color=colour,
                marker=marker,
                markersize=4.5,
                linewidth=1.6,
                dashes=dashes,
                label=name,
            )
            plotted += 1
        ax.set_xlabel("ASR noise rate (% of user words edited)")
        ax.set_ylabel("score (%)")
        # Ticks at the levels that were actually measured. Matplotlib's automatic
        # ticks invent intermediate rates (2.5%, 7.5%) that no run produced.
        ax.set_xticks([p[0] * 100 for p in ordered])
        ax.set_xticklabels([f"{p[0] * 100:g}" for p in ordered])
        # Zero-based: a truncated axis would exaggerate the decline, and the
        # decline is the argument being made.
        ax.set_ylim(0, 100)
        ax.set_title("Extraction quality against transcription noise")
        if plotted:
            # Matplotlib warns on an empty legend, and pytest treats UserWarning
            # as an error — an axes with both series absent is a real case.
            ax.legend(loc="lower left")
        return _to_inline_svg(
            fig,
            prefix="nc-",
            label=(
                "Line chart of precision and recall against ASR noise rate. The same values "
                "are listed in the table below the chart."
            ),
        )

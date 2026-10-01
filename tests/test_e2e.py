"""End-to-end scaffold.

Phase 0 freezes the CLI contract and the three swappable files; the pipeline
stages land in Phase 1. What is asserted here now is what Phase 0 actually
delivers. The skipped tests below are the gate every later merge has to pass,
and they are written out rather than left implicit so the gate is visible.
"""

from __future__ import annotations

import json
import string
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from onboarding_lab.cli import app
from onboarding_lab.models import ALL_FIELDS, QuestionScript

runner = CliRunner()

#: The frozen CLI surface. Adding or renaming one is a contract change.
EXPECTED_COMMANDS = (
    "gen",
    "sim",
    "noise",
    "extract",
    "score",
    "audit",
    "report",
    "diff",
    "run",
    "smoke",
    "info",
)

#: Stages not yet built. Each must fail loudly rather than print a number.
PENDING_STAGES = ("gen", "sim", "noise", "extract", "score", "audit", "report", "diff", "run")


def test_every_command_is_registered() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in EXPECTED_COMMANDS:
        assert name in result.output, f"`lab {name}` missing from the CLI"


@pytest.mark.parametrize("stage", PENDING_STAGES)
def test_unbuilt_stages_fail_loudly(stage: str) -> None:
    """Never a silent success and never a fabricated number."""
    result = runner.invoke(app, [stage])
    assert result.exit_code == 2
    assert "not implemented yet" in result.output


def test_info_reports_resolved_config() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["personas"] == 12
    assert payload["noise_rates"] == [0.0, 0.05, 0.1, 0.2]


def test_commands_need_no_api_key(monkeypatch) -> None:
    """`make test` must pass with no environment variables set."""
    monkeypatch.delenv("LAB_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert runner.invoke(app, ["--help"]).exit_code == 0
    assert runner.invoke(app, ["info"]).exit_code == 0


# -- the three swappable files ------------------------------------------------


def test_question_script_parses_against_the_contract() -> None:
    script = QuestionScript.model_validate(yaml.safe_load(Path("questions.yaml").read_text()))
    assert len(script.questions) == 10
    assert script.followup_trigger_words == 40


def test_every_scored_field_is_targeted_by_some_question() -> None:
    """A field no question aims at can only ever be recalled by accident."""
    script = QuestionScript.model_validate(yaml.safe_load(Path("questions.yaml").read_text()))
    targeted = {t for q in script.questions for t in q.targets}
    untargeted = set(ALL_FIELDS) - targeted - {"summary", "age"}
    assert not untargeted, f"no question targets {sorted(untargeted)}"


PROMPT_FILES = (
    "prompts/persona_gen.txt",
    "prompts/user_agent.txt",
    "prompts/interviewer_followup.txt",
    "prompts/extract.txt",
    "prompts/extract_v2_worse.txt",
    "prompts/aligner.txt",
)


@pytest.mark.parametrize(
    ("prompt", "placeholders"),
    [
        (
            "prompts/persona_gen.txt",
            {"skeleton", "n_specific", "n_volunteer", "n_followup", "n_hedge"},
        ),
        (
            "prompts/user_agent.txt",
            {"bio", "fact_sheet", "style", "min_words", "max_words", "style_note"},
        ),
        ("prompts/interviewer_followup.txt", {"question", "targets", "answer"}),
        ("prompts/extract.txt", {"transcript", "schema"}),
        ("prompts/extract_v2_worse.txt", {"transcript", "schema"}),
        ("prompts/aligner.txt", {"facts", "claims", "transcript"}),
    ],
)
def test_prompt_placeholders_are_exactly_as_expected(prompt: str, placeholders: set[str]) -> None:
    """A renamed placeholder silently produces a prompt with a literal brace in it."""
    text = Path(prompt).read_text()
    found = {name for _, name, _, _ in string.Formatter().parse(text) if name}
    assert found == placeholders, (
        f"{prompt}: expected {sorted(placeholders)}, found {sorted(found)}"
    )


def test_every_prompt_forbids_internal_tags() -> None:
    """Carried into every system prompt as the disabled-thinking mitigation.

    Collapses whitespace first: the prompts are hard-wrapped, so the phrase
    straddles a newline in most of them.
    """
    for path in Path("prompts").glob("*.txt"):
        collapsed = " ".join(path.read_text().split())
        assert "internal or system XML tags" in collapsed, path


def test_placeholder_check_covers_every_prompt() -> None:
    """A new prompt file must not slip past the placeholder test."""
    on_disk = {str(p) for p in Path("prompts").glob("*.txt")}
    assert on_disk == set(PROMPT_FILES)


def test_worse_prompt_is_actually_worse() -> None:
    """`lab diff` needs a real regression to catch during the demo."""
    good = Path("prompts/extract.txt").read_text()
    worse = Path("prompts/extract_v2_worse.txt").read_text()
    assert "Do not infer" in good
    assert "only information the user stated" in good
    assert "Infer likely values" in worse
    assert "Do not infer" not in worse


# -- the Phase 1+ gate --------------------------------------------------------


def phase_gate(waiting_on: str):
    """Skip with a reason that names exactly what the test waits on."""
    return pytest.mark.skip(reason=f"waits on {waiting_on}; this is its merge gate")


@phase_gate("Phase 1 track A (personas, sim, extract, score)")
def test_two_personas_through_lab_run_on_the_fake_provider() -> None:
    """Every artifact exists, validates against the models, and joins by id."""


@phase_gate("Phase 1 track C (score/metrics.py aggregation)")
def test_aggregate_carries_every_metric_key() -> None:
    """Against metrics_schema.expected_keys, not a hardcoded list."""


@phase_gate("Phase 1 track D (report/render.py)")
def test_report_and_summary_render() -> None:
    """index.html and summary.md, from committed fixtures, with no API key."""


@phase_gate("Phase 1 tracks C and D (score/floor.py plus diff rendering)")
def test_diff_exits_one_when_the_candidate_is_worse() -> None:
    """Fake variance is zero, so the floor is its 0.005 minimum; the fake
    candidate must be worse by a wide margin so this cannot flake."""


@phase_gate("Phase 1 track C (score/metrics.py denominators)")
def test_judge_errors_are_excluded_from_every_denominator() -> None:
    """The regression test for the bug class this project is defined against."""


@phase_gate("Phase 1 track A (run manifest and failures.jsonl)")
def test_dropped_persona_appears_in_failures_and_the_manifest() -> None:
    """A silent drop changes every number in the report."""

"""Run identity and artifact paths."""

from __future__ import annotations

from pathlib import Path

from onboarding_lab.paths import (
    RunPaths,
    append_failure,
    floor_tag,
    latest_run,
    noise_token,
    run_id,
    transcript_stem,
)


def test_noise_token_has_no_dots() -> None:
    """A dotted float in a filename is ambiguous against the `.` separator."""
    assert noise_token(0.0) == "n000"
    assert noise_token(0.05) == "n005"
    assert noise_token(0.1) == "n010"
    assert noise_token(0.2) == "n020"
    for rate in (0.0, 0.05, 0.1, 0.2):
        assert "." not in noise_token(rate)


def test_noise_tokens_sort_in_rate_order() -> None:
    rates = [0.2, 0.0, 0.1, 0.05]
    assert [noise_token(r) for r in sorted(rates)] == sorted(noise_token(r) for r in rates)


def test_transcript_stems_are_distinct_across_the_axes() -> None:
    stems = {
        transcript_stem("p001", 1, 0.0),
        transcript_stem("p001", 1, 0.1),
        transcript_stem("p001", 2, 0.0),
        transcript_stem("p002", 1, 0.0),
        transcript_stem("p001", 1, 0.0, shuffled=True),
    }
    assert len(stems) == 5


def test_floor_runs_get_distinct_tags() -> None:
    """Without this, three floor runs overwrite one file and sigma is computed
    over a single survivor."""
    tags = {floor_tag("v1", k) for k in range(5)}
    assert len(tags) == 5
    assert "v1" not in tags


def test_run_id_is_stable_for_a_config(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    now = datetime(2026, 9, 30, 22, 15, tzinfo=UTC)
    a = run_id({"personas": 12, "seed": 7}, now=now)
    b = run_id({"seed": 7, "personas": 12}, now=now)
    c = run_id({"personas": 8, "seed": 7}, now=now)
    assert a == b, "key order must not change run identity"
    assert a != c
    assert a.startswith("20260930-2215-")


def test_run_paths_layout(tmp_path: Path) -> None:
    p = RunPaths.for_run("r1", runs_root=tmp_path)
    assert p.aggregate("v1") == tmp_path / "r1" / "scores" / "v1" / "aggregate.json"
    assert p.extractions(floor_tag("v1", 2)).name == "v1.floor2"
    p.mkdirs()
    assert p.personas.is_dir() and p.transcripts.is_dir()


def test_failures_are_appended_not_overwritten(tmp_path: Path) -> None:
    """A dropped persona changes every number, so it can never vanish."""
    p = RunPaths.for_run("r1", runs_root=tmp_path)
    append_failure(p, {"persona_id": "p001", "stage": "gen", "reason": "schema_invalid"})
    append_failure(p, {"persona_id": "p002", "stage": "sim", "reason": "refusal"})
    lines = p.failures.read_text().strip().splitlines()
    assert len(lines) == 2
    assert "p001" in lines[0] and "p002" in lines[1]


def test_latest_run_picks_the_newest(tmp_path: Path) -> None:
    assert latest_run(tmp_path) is None
    for rid in ("20260930-1000-aaaaaa", "20260930-2200-bbbbbb"):
        (tmp_path / rid).mkdir()
    assert latest_run(tmp_path) == "20260930-2200-bbbbbb"

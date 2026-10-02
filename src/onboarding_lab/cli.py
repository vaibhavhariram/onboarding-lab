"""Typer CLI. Console script ``lab``.

Every command prints a one-line summary and the path it wrote. ``--run`` defaults
to the newest run under ``runs/``.

Pipeline stages land in Phase 1 and later; their signatures are frozen here so
modules can be built against them in parallel. ``smoke`` is implemented now
because it is what validates the model parameters before any full run spends
money.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from . import paths, role_schemas
from .config import Config, api_key
from .models import Role
from .paths import RunPaths
from .providers.base import Message

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Synthetic evaluation harness for conversational intake pipelines.",
)

RunOpt = Annotated[str | None, typer.Option("--run", help="Run id; defaults to the newest run.")]
TagOpt = Annotated[str, typer.Option("--tag", help="Extraction tag, e.g. v1.")]


def _pending(stage: str, phase: int) -> None:
    """Fail loudly rather than printing a number nobody computed."""
    typer.secho(
        f"`lab {stage}` is not implemented yet (lands in Phase {phase}). "
        f"The CLI contract is frozen; the stage is not built.",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(code=2)


def _resolve_run(run: str | None) -> str:
    rid = run or paths.latest_run()
    if rid is None:
        typer.secho("No runs found under runs/. Start one with `lab run`.", fg="red", err=True)
        raise typer.Exit(code=2)
    return rid


@app.command()
def gen(
    run: RunOpt = None,
    n: Annotated[int, typer.Option("--n", help="Number of personas.")] = 12,
    seed: Annotated[int, typer.Option("--seed")] = 7,
) -> None:
    """Generate truth sheets."""
    _pending("gen", 1)


@app.command()
def sim(
    run: RunOpt = None,
    script: Annotated[str, typer.Option("--script")] = "questions.yaml",
    seed: Annotated[int, typer.Option("--seed")] = 1,
    shuffle_seed: Annotated[int | None, typer.Option("--shuffle-seed")] = None,
) -> None:
    """Run the simulated interview."""
    _pending("sim", 1)


@app.command()
def noise(
    run: RunOpt = None,
    rate: Annotated[float, typer.Option("--rate")] = 0.1,
    seed: Annotated[int, typer.Option("--seed")] = 1,
) -> None:
    """Rewrite user turns to imitate ASR error."""
    _pending("noise", 2)


@app.command()
def extract(
    run: RunOpt = None,
    prompt: Annotated[str, typer.Option("--prompt")] = "prompts/extract.txt",
    schema: Annotated[str, typer.Option("--schema")] = "schema.json",
    tag: TagOpt = "v1",
    model: Annotated[str | None, typer.Option("--model")] = None,
    nonce: Annotated[str | None, typer.Option("--nonce")] = None,
) -> None:
    """Extract a profile from each transcript."""
    _pending("extract", 1)


@app.command()
def score(run: RunOpt = None, tag: TagOpt = "v1") -> None:
    """Align claims to truth facts and compute metrics."""
    _pending("score", 1)


@app.command()
def audit(
    run: RunOpt = None,
    tag: TagOpt = "v1",
    sample: Annotated[int, typer.Option("--sample")] = 20,
) -> None:
    """Draw a judged-alignment sample for hand review."""
    _pending("audit", 2)


@app.command()
def report(
    run: RunOpt = None,
    tag: TagOpt = "v1",
    from_fixtures: Annotated[
        bool, typer.Option("--from-fixtures", help="Render from committed fixtures; no API key.")
    ] = False,
) -> None:
    """Render the HTML report and Markdown summary."""
    from .report.render import load_aggregate, write_report

    if from_fixtures:
        source = Path("fixtures/scores") / tag / "aggregate.json"
        out_dir = Path("report") / tag
    else:
        paths = RunPaths.for_run(_resolve_run(run))
        source = paths.aggregate(tag)
        out_dir = paths.report(tag)

    if not source.is_file():
        typer.secho(f"no aggregate at {source}", fg="red", err=True)
        raise typer.Exit(code=2)

    written = write_report(load_aggregate(source), out_dir)
    typer.echo(f"report for {tag}: {written['index.html']}")
    typer.echo(f"summary for {tag}: {written['summary.md']}")


@app.command()
def diff(
    run: RunOpt = None,
    baseline: Annotated[str, typer.Option("--baseline")] = "v1",
    candidate: Annotated[str, typer.Option("--candidate")] = "v2",
    floor_runs: Annotated[int, typer.Option("--floor-runs")] = 5,
    from_fixtures: Annotated[bool, typer.Option("--from-fixtures")] = False,
) -> None:
    """Compare a candidate against a baseline. Exit 1 on regression beyond the floor."""
    from .report.render import load_aggregate, write_diff
    from .score.floor import diff as compute_diff

    root = Path("fixtures/scores") if from_fixtures else None
    if root is None:
        paths = RunPaths.for_run(_resolve_run(run))
        base_path, cand_path = paths.aggregate(baseline), paths.aggregate(candidate)
        out_dir = paths.report(candidate)
    else:
        base_path = root / baseline / "aggregate.json"
        cand_path = root / candidate / "aggregate.json"
        out_dir = Path("report") / candidate

    for path in (base_path, cand_path):
        if not path.is_file():
            typer.secho(f"no aggregate at {path}", fg="red", err=True)
            raise typer.Exit(code=2)

    base, cand = load_aggregate(base_path), load_aggregate(cand_path)
    result = compute_diff(
        baseline=base.get("metrics", {}),
        candidate=cand.get("metrics", {}),
        floors=base.get("floor", {}),
    )
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result.__dict__)
    written = write_diff(payload, out_dir, baseline=baseline, candidate=candidate)
    typer.echo(f"diff {baseline} -> {candidate}: exit {result.exit_code}; wrote {written}")
    if result.invalid_reason:
        typer.secho(result.invalid_reason, fg="red", err=True)
    raise typer.Exit(code=result.exit_code)


@app.command()
def run(
    config: Annotated[str, typer.Option("--config")] = "lab.yaml",
) -> None:
    """Run the whole pipeline: gen, sim, noise, extract, score, report."""
    from .pipeline import run_pipeline
    from .providers.anthropic import AnthropicProvider

    cfg = Config.load(config)
    model = cfg.models.default
    provider = AnthropicProvider(api_key=api_key(), model=model)
    outcome = asyncio.run(run_pipeline(cfg, provider=provider, model=model))

    typer.echo(
        f"run {outcome.run_id}: {len(outcome.sheets)}/{cfg.personas} personas, "
        f"{len(outcome.aggregates)} tag(s), {outcome.failures} failure record(s)"
    )
    for tag in sorted(outcome.aggregates):
        typer.echo(f"  wrote {outcome.paths.aggregate(tag)}")
    if outcome.failures:
        typer.secho(f"  failures logged: {outcome.paths.failures}", fg="yellow")


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #

#: One minimal request per role, with that role's real schema and parameters.
_SMOKE_ROLES: tuple[tuple[Role, dict | None, str, str], ...] = (
    (
        "persona_gen",
        role_schemas.PERSONA_GEN_SCHEMA,
        "You write truth sheets for a synthetic evaluation harness.",
        "Skeleton: age 34, city Portland, occupation nurse, style balanced. "
        "Produce 8 judged-field facts and a 120-word first-person bio.",
    ),
    (
        "user_agent",
        role_schemas.USER_AGENT_SCHEMA,
        "You are playing a person in a recorded onboarding interview. "
        "Your style is balanced, so answer in 40 to 120 words. "
        "Facts: f01 occupation=nurse (volunteer).",
        "Interviewer: What do you do for work, and how do you feel about it?",
    ),
    (
        "interviewer_followup",
        role_schemas.INTERVIEWER_FOLLOWUP_SCHEMA,
        "You are an interviewer running an onboarding script.",
        'Question: "What do you do for work?" Targets: [occupation, values]. '
        'Answer: "I am a nurse."',
    ),
    (
        "extract",
        None,  # replaced with the real profile schema below
        "Extract a profile from this onboarding transcript into the JSON Schema.",
        "[t001 interviewer] What do you do?\n[t002 user] I am a nurse in Portland.",
    ),
    (
        "aligner",
        role_schemas.ALIGNER_SCHEMA,
        "You are a judge. Quote verbatim from user turns only, at most 12 words.",
        'Facts: [{"fact_id":"f01","field":"occupation","value":"nurse"}]\n'
        'Claims: [{"claim_id":"c01","field":"occupation","value":"nurse"}]\n'
        "Transcript:\n[t002 user] I am a nurse in Portland.",
    ),
)


async def _run_smoke(cfg: Config, model: str) -> int:
    from .llm import call
    from .providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(api_key=api_key(), model=model)
    refusals = 0
    failures = 0

    for role, schema, system, user in _SMOKE_ROLES:
        json_schema = role_schemas.load_profile_schema() if role == "extract" else schema
        reps = 2 if role == "user_agent" else 1
        for rep in range(reps):
            result = await call(
                role=role,
                provider=provider,
                system=system,
                messages=[Message(role="user", content=user)],
                model=model,
                role_cfg=cfg.role(role),
                json_schema=json_schema,
            )
            label = f"{role}" + (f" (call {rep + 1})" if reps > 1 else "")
            if result.ok:
                extra = ""
                if role == "user_agent":
                    words = len((result.parsed or {}).get("utterance", "").split())
                    extra = f" words={words}"
                    if rep == 1:
                        extra += f" cache_read={result.usage.cache_read_input_tokens}"
                typer.secho(
                    f"  ok    {label}: {result.latency_ms}ms "
                    f"in={result.usage.input_tokens} out={result.usage.output_tokens}{extra}",
                    fg="green",
                )
            else:
                failures += 1
                if result.reason == "refusal":
                    refusals += 1
                typer.secho(
                    f"  FAIL  {label}: reason={result.reason} {result.error}",
                    fg="red",
                )

    total = sum(2 if r[0] == "user_agent" else 1 for r in _SMOKE_ROLES)
    typer.echo(f"\n{total - failures}/{total} calls ok; {refusals} refusal(s)")
    if refusals / total > 0.02:
        typer.secho(
            f"Refusal rate {refusals / total:.0%} exceeds 2%: fall back to "
            f"{cfg.models.fallback} with thinking={cfg.models.fallback_thinking!r}.",
            fg="yellow",
        )
    return 1 if failures else 0


@app.command()
def smoke(
    config: Annotated[str, typer.Option("--config")] = "lab.yaml",
    model: Annotated[str | None, typer.Option("--model")] = None,
) -> None:
    """One real request per role, to validate parameters before a full run.

    Checks that each role's exact parameters and schema are accepted, that the
    user-agent prefix caches, that style word ranges hold, and what the refusal
    rate is.
    """
    cfg = Config.load(config)
    chosen = model or cfg.models.default
    typer.echo(f"smoke test: model={chosen}")
    code = asyncio.run(_run_smoke(cfg, chosen))
    raise typer.Exit(code=code)


@app.command()
def info(config: Annotated[str, typer.Option("--config")] = "lab.yaml") -> None:
    """Print the resolved configuration and the newest run."""
    cfg = Config.load(config)
    typer.echo(
        json.dumps(
            {
                "run_name": cfg.run_name,
                "personas": cfg.personas,
                "noise_rates": cfg.noise_rates,
                "model": cfg.models.default,
                "floor_runs": cfg.floor_runs,
                "latest_run": paths.latest_run(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()

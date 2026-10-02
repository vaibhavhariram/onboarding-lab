"""Stage orchestration and artifact persistence.

Every stage writes a JSON artifact, so any stage can be rerun alone and a run can
be resumed. Stages are idempotent: an artifact whose inputs are unchanged is
reused rather than regenerated, which is what makes a cached rerun free.

Failures are recorded in ``failures.jsonl`` and counted in the manifest. A
persona that drops out silently would change every number in the report.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .config import Config
from .extract.run import extract_one
from .hashing import code_version as current_code_version
from .models import (
    JudgeHealth,
    QuestionScript,
    Scores,
    Transcript,
    TruthSheet,
)
from .noise.inject import inject_noise, write_edit_log
from .paths import RunPaths, append_failure, noise_token, run_id
from .personas.generate import generate_all
from .personas.sample_skeleton import sample_skeletons
from .providers.base import Provider
from .score import metrics as metrics_mod
from .score.align import align_claims
from .score.coverage import build_coverage
from .score.disclosure_check import check_disclosures
from .score.exact import score_exact
from .sim.run import simulate_all


def save_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def save_model(path: Path, model) -> Path:
    return save_json(path, json.loads(model.model_dump_json()))


def load_model(path: Path, cls):
    return cls.model_validate_json(path.read_text())


def load_script(path: str | Path) -> QuestionScript:
    return QuestionScript.model_validate(yaml.safe_load(Path(path).read_text()))


@dataclass
class RunOutcome:
    paths: RunPaths
    run_id: str
    sheets: list[TruthSheet]
    transcripts: dict[str, list[Transcript]]
    scores: dict[str, list[Scores]]
    aggregates: dict[str, dict]
    failures: int


async def stage_gen(
    cfg: Config, paths: RunPaths, *, provider: Provider, model: str, code_version: str
) -> list[TruthSheet]:
    skeletons = sample_skeletons(cfg.personas, cfg.seed)
    existing: list[TruthSheet] = []
    todo = []
    for sk in skeletons:
        target = paths.personas / f"{sk.persona_id}.json"
        if target.is_file():
            existing.append(load_model(target, TruthSheet))
        else:
            todo.append(sk)

    results = (
        await generate_all(
            todo,
            provider=provider,
            model=model,
            role_cfg=cfg.role("persona_gen"),
            concurrency=cfg.concurrency,
            code_version=code_version,
        )
        if todo
        else []
    )
    for result in results:
        if result.sheet is None:
            append_failure(paths, result.as_failure_record())
            continue
        save_model(paths.personas / f"{result.persona_id}.json", result.sheet)
        existing.append(result.sheet)
    return sorted(existing, key=lambda s: s.persona_id)


async def stage_sim(
    cfg: Config,
    paths: RunPaths,
    sheets: list[TruthSheet],
    script: QuestionScript,
    *,
    provider: Provider,
    model: str,
    code_version: str,
) -> list[Transcript]:
    results = await simulate_all(
        sheets,
        script,
        provider=provider,
        model=model,
        config=cfg,
        code_version=code_version,
    )
    clean: list[Transcript] = []
    for result in results:
        if result.transcript is None:
            append_failure(paths, result.as_failure_record())
            continue
        if result.invented_fact_ids:
            append_failure(
                paths,
                {
                    "stage": "sim",
                    "persona_id": result.persona_id,
                    "status": "invented_fact_ids",
                    "detail": result.invented_fact_ids,
                },
            )
        save_model(paths.transcripts / f"{result.transcript.transcript_id}.json", result.transcript)
        clean.append(result.transcript)
    return clean


def stage_noise(
    cfg: Config, paths: RunPaths, clean: list[Transcript], *, seed: int = 1
) -> dict[float, list[Transcript]]:
    """One transcript set per noise rate. Rate 0.0 reuses the clean transcript."""
    by_rate: dict[float, list[Transcript]] = {}
    for rate in cfg.noise_rates:
        out: list[Transcript] = []
        for source in clean:
            result = inject_noise(source, rate=rate, seed=seed)
            if rate > 0:
                save_model(
                    paths.transcripts / f"{result.transcript.transcript_id}.json",
                    result.transcript,
                )
                write_edit_log(result.edit_log, paths.transcripts)
            out.append(result.transcript)
        by_rate[rate] = out
    return by_rate


async def stage_extract_and_score(
    cfg: Config,
    paths: RunPaths,
    sheets: list[TruthSheet],
    clean: list[Transcript],
    noised: list[Transcript],
    *,
    tag: str,
    provider: Provider,
    model: str,
    code_version: str,
    prompt_path: str,
    schema_path: str,
    nonce: str | None = None,
) -> list[Scores]:
    sheet_by_id = {s.persona_id: s for s in sheets}
    clean_by_id = {t.transcript_id: t for t in clean}
    gate = asyncio.Semaphore(cfg.concurrency)

    async def one(transcript: Transcript) -> Scores | None:
        async with gate:
            truth = sheet_by_id[transcript.persona_id]
            source_id = transcript.source_transcript_id or transcript.transcript_id
            clean_t = clean_by_id.get(source_id, transcript)

            extraction, decomposition = await extract_one(
                transcript,
                provider=provider,
                model=model,
                tag=tag,
                role_cfg=cfg.role("extract"),
                prompt_path=prompt_path,
                schema_path=schema_path,
                code_version=code_version,
                nonce=nonce,
            )
            save_model(paths.extractions(tag) / f"{transcript.transcript_id}.json", extraction)

            if extraction.status != "ok":
                append_failure(
                    paths,
                    {
                        "stage": "extract",
                        "persona_id": transcript.persona_id,
                        "transcript_id": transcript.transcript_id,
                        "tag": tag,
                        "status": extraction.status,
                    },
                )

            exact_alignments = score_exact(extraction.claims, truth.facts)
            align_result = await align_claims(
                truth,
                transcript,
                extraction.claims,
                provider=provider,
                model=model,
                role_cfg=cfg.role("aligner"),
                nonce=nonce,
            )
            alignments = [*exact_alignments, *align_result.alignments]
            coverage = build_coverage(truth.facts, alignments, transcript)
            disclosure = check_disclosures(
                truth=truth,
                clean_transcript=clean_t,
                coverage=coverage,
                alignments=alignments,
                span_transcript=transcript,
            )
            judge = JudgeHealth(
                calls=align_result.calls,
                judged_claims=align_result.judged_claims,
                errors=align_result.errors,
                span_invalid=align_result.span_invalid,
                span_invalid_first_pass=align_result.span_invalid_first_pass,
            )
            scores = metrics_mod.score_extraction(
                truth=truth,
                transcript=transcript,
                extraction=extraction,
                alignments=alignments,
                coverage=coverage,
                judge=judge,
                clean_transcript=clean_t,
                claims_deduped=decomposition.deduped if decomposition else 0,
                disclosure=disclosure,
            )
            save_model(paths.scores(tag) / f"{transcript.transcript_id}.json", scores)
            return scores

    results = await asyncio.gather(*(one(t) for t in noised))
    return [r for r in results if r is not None]


def write_aggregate(
    paths: RunPaths,
    tag: str,
    scores: list[Scores],
    sheets: list[TruthSheet],
    *,
    statuses: dict[str, str] | None = None,
    persona_ids: dict[str, str] | None = None,
) -> dict:
    agg = metrics_mod.aggregate(
        scores,
        truth_sheets=sheets,
        statuses=statuses,
        persona_ids=persona_ids,
    ).to_dict()
    save_json(paths.aggregate(tag), agg)
    return agg


async def run_pipeline(
    cfg: Config,
    *,
    provider: Provider,
    model: str,
    runs_root: Path | None = None,
    code_version: str | None = None,
    tag: str | None = None,
    prompt_path: str | None = None,
) -> RunOutcome:
    version = code_version or current_code_version()
    rid = run_id(cfg.model_dump(mode="json"))
    paths = RunPaths.for_run(rid, runs_root=runs_root)
    paths.mkdirs()
    script = load_script(cfg.script)
    chosen_tag = tag or cfg.extraction.tag

    sheets = await stage_gen(cfg, paths, provider=provider, model=model, code_version=version)
    clean = await stage_sim(
        cfg, paths, sheets, script, provider=provider, model=model, code_version=version
    )
    by_rate = stage_noise(cfg, paths, clean, seed=cfg.sim.seed)

    scores_by_tag: dict[str, list[Scores]] = {}
    aggregates: dict[str, dict] = {}
    transcripts_by_tag: dict[str, list[Transcript]] = {}

    for rate, transcripts in by_rate.items():
        rate_tag = chosen_tag if rate == 0.0 else f"{chosen_tag}.{noise_token(rate)}"
        scores = await stage_extract_and_score(
            cfg,
            paths,
            sheets,
            clean,
            transcripts,
            tag=rate_tag,
            provider=provider,
            model=model,
            code_version=version,
            prompt_path=prompt_path or cfg.extraction.prompt,
            schema_path=cfg.extraction.schema_path,
        )
        scores_by_tag[rate_tag] = scores
        transcripts_by_tag[rate_tag] = transcripts
        persona_ids = {f"{t.transcript_id}:{rate_tag}": t.persona_id for t in transcripts}
        aggregates[rate_tag] = write_aggregate(
            paths, rate_tag, scores, sheets, persona_ids=persona_ids
        )

    failures = (
        len(paths.failures.read_text().strip().splitlines()) if paths.failures.is_file() else 0
    )
    save_json(
        paths.manifest,
        {
            "run_id": rid,
            "created_at": datetime.now(UTC).isoformat(),
            "code_version": version,
            "model": model,
            "config": cfg.model_dump(mode="json"),
            "script_id": script.script_id,
            "n_personas_requested": cfg.personas,
            "n_ok": len(sheets),
            "n_failed": cfg.personas - len(sheets),
            "failure_records": failures,
            "tags": sorted(aggregates),
            "style_mix": {
                style: sum(1 for s in sheets if s.style == style)
                for style in sorted({s.style for s in sheets})
            },
        },
    )
    return RunOutcome(
        paths=paths,
        run_id=rid,
        sheets=sheets,
        transcripts=transcripts_by_tag,
        scores=scores_by_tag,
        aggregates=aggregates,
        failures=failures,
    )

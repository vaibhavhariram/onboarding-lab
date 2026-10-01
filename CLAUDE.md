# onboarding-lab — working agreement

A synthetic evaluation harness for conversational intake / voice onboarding pipelines.

- **`PLAN.md`** — the build plan and its amendments. Authoritative.
- **`docs/SPEC.md`** — the reference design. Every `§` reference in `PLAN.md` resolves here.
  Where the two disagree, `PLAN.md` wins.

## Hard rules

1. **Never read `ANTHROPIC_API_KEY`.** Claude Code consumes that variable in preference to a
   subscription, so reading it would bill per token. The lab reads **`LAB_ANTHROPIC_API_KEY`**
   from a gitignored `.env`, loaded by the CLI only, passed to the SDK explicitly. Never use the
   zero-arg `Anthropic()` constructor.
2. **Tests run without API keys.** `env -u ANTHROPIC_API_KEY make test` must pass on a clean
   checkout. A deterministic `FakeProvider` backs every test. No network in tests.
3. **Errors are states, not verdicts.** Provider error, refusal, truncation, invalid schema, and
   a non-verbatim cited span each get their own status. They never become a scoring verdict, are
   excluded from every metric denominator, and are counted in judge health and `failures.jsonl`.
   A `refusal` arrives as HTTP 200 — guard `stop_reason` before reading content.
4. **Join by id, never by position.** Every artifact carries `persona_id` / `transcript_id` /
   `extraction_id` and input hashes. Scoring joins claims → facts → turns by id.
5. **No number without a script.** README figures are written only by `scripts/numbers.py`
   between `<!-- numbers:start -->` / `<!-- numbers:end -->`. Never hand-type a metric into any
   document. Never invent example numbers in docstrings or the README.
6. **A rate with a zero denominator is omitted**, with its `0` recorded in `Scores.n`. Never
   `0.0`, never `NaN` — `0.0` reads as "we got everything wrong".
7. **Product-agnostic naming.** No customer, product, or individual is named in any committed
   file, prompt, comment, or fixture.
8. **Contract changes go through the main thread**, with an explicit note in the status update.
   Phase 0 files are main-thread only: `models.py`, `metrics_schema.py`, `paths.py`, `llm.py`,
   `cache.py`, `cli.py`, `pyproject.toml`, `Makefile`.
9. **`test_e2e` gates every merge.** `make demo` stays green from the end of Phase 1; a change
   that breaks it gets fixed before the next feature.
10. **Status after every phase gate**, three lines: done / next / blocked.
11. **Commits** are small, after every gate, conventional-commit messages.
12. **Dependencies** limited to `pydantic>=2`, `typer`, `jinja2`, `pyyaml`, `anthropic`,
    `httpx`, `matplotlib` (SVG backend only), `jsonschema`, `pytest`, `ruff`. Ask before adding.
13. **Persona content guardrails.** Adults 25–45, PG-13, no sexual content, no real people, no
    stereotyping by ethnicity, religion, or nationality. Diversity comes from seed-sampled
    demographics, not caricature.

## Model and parameters

`claude-sonnet-5-5` for every role. **Consult the bundled `claude-api` skill before touching
`providers/anthropic.py`.**

- `thinking: {"type": "between_tools"}` — `disabled` returns 400 on 5.5. Valid at
  `low`/`medium`/`high` effort only.
- No sampling params (`temperature`/`top_p`/`top_k` → 400). No assistant prefill (→ 400).
- Structured outputs via `output_config.format`. Cache minimum 512 tokens.
- Read response content **by block type**, never `content[0]`.
- Fallback to `claude-sonnet-5` with `thinking: {"type": "disabled"}` if the smoke test shows
  refusals above ~2% or the `claude-sonnet-5-5` id is rejected.

"Seed" means a Python-side `random.Random(seed)` only — skeleton sampling, noise, question
order, audit sampling. The Messages API has no seed parameter; never send one.

## Never cut

Exact + judged scoring · span verification · the error states including refusal ·
`make numbers` · the diff · the noise floor · failure accounting.

## Cut order if time runs short

1. OpenAI adapter — delete, don't stub (Protocol stays).
2. GitHub Action — write the YAML, never debug it.
3. Drop to 8 personas (`personas` is config; the report prints intervals).
4. `lab audit` last — the red "judge not yet audited" banner is more credible than a rushed
   label pass.

Already deferred to v1.1: `contradict` facts and `decoy_rate`, `specificity_match`, the
dollar-cost table. The `Disclosure` literal and `Fact.decoy_value` stay in the contract.

## Commands

```
make setup    # uv sync
make test     # ruff check + pytest          (no API key needed)
make dev      # 3 personas, 1 noise level    (~90s, ~$0.05)
make demo     # report from committed fixtures (no API key, $0)
make numbers  # full run, then rewrite README numbers
make audit    # lab audit --sample 20
make clean    # rm -rf runs .cache
```

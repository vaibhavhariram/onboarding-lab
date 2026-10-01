# onboarding-lab — build plan

A synthetic evaluation harness for conversational intake / voice onboarding pipelines.
`docs/SPEC.md` is the reference design; every `§` reference below resolves against it.
Where this plan and the spec disagree, **this plan wins**.

## Context

When a long voice conversation is turned into a structured profile, every downstream stage
inherits whatever extraction got right or wrong — and there is no number for how good that
step is. Hand-labelling real transcripts is slow (~20 min each), goes stale the moment a
question changes, and a real user's true facts are never fully knowable. So question and
prompt changes ship on intuition.

This harness inverts the order: write the person first as a structured truth sheet, run that
person through a simulated interview, extract a profile from the transcript, score the profile
against the truth. Labels are exact and free by construction, and the fixture set becomes a
regression suite that runs on every question or prompt change.

Deliverable: precision/recall per field with hallucinations named, per-question yield, a
quality-vs-ASR-noise curve, a stability number, and a CI diff gate — every figure regenerated
from a committed script.

Three rules drive the build:

1. **Errors are states, not verdicts.** A failed judge call, a failed provider call, a refusal,
   a truncation, an invalid schema, or a cited span that doesn't literally exist each get their
   own status. They never become a scoring verdict, they're excluded from metric denominators,
   and they're counted in judge health and `failures.jsonl`.
2. **Join by id, never by position.** Every artifact carries `persona_id` / `transcript_id` /
   `extraction_id` and input hashes. Scoring joins claims → facts → turns by id.
3. **No number without a script.** README figures are written only by `scripts/numbers.py`
   between markers, and regenerate on a clean checkout.

## Settled decisions

| | |
| --- | --- |
| Model | `claude-sonnet-5-5` for every role (sim, extraction, judge). Same price as Sonnet 5. |
| Toolchain | uv, Python 3.12 pinned via `.python-version`. Adoption stays `uv run lab run --config lab.yaml`. |
| Credentials | `LAB_ANTHROPIC_API_KEY` from a gitignored `.env`, read by the CLI only (see amendment 2). |
| Headline run | 12 personas, four noise levels `[0.0, 0.05, 0.1, 0.2]`. ~15 min, ~$5, precision CI ≈ ±3pp. |
| Stability | `stability_extraction` derived from the floor runs (pairwise Jaccard). No extra pass. |
| Integrity extras | Exact-field majority-class floor (M9) and the disclosure self-report check (M10). |
| Naming | Product-agnostic in every committed file. |
| License / OpenAI / fixtures | MIT. No OpenAI adapter (Protocol only). Commit the 12 personas. |

---

# Amendments (authoritative)

## 1. Model and parameters

`claude-sonnet-5-5` for every role. **Before writing `providers/anthropic.py`, consult the
bundled `claude-api` skill for current parameter shapes.**

- `thinking: {"type": "disabled"}` returns **400** on 5.5. Send `{"type": "between_tools"}` —
  the lowest setting, text-only when no tools are sent, valid at `low`/`medium`/`high` effort
  only (not `xhigh`/`max`).
- **P2/P5/P6/P7 stand:** no sampling params (`temperature`/`top_p`/`top_k` → 400), no assistant
  prefill (→ 400), structured outputs via `output_config.format`, 512-token cache minimum.
- Read response content **by block type**, never `content[0]`.
- **First real call is a smoke test** — one request per role with its exact params and schema
  (truth sheet, `schema.json`, aligner output). Confirm: 200s; schema acceptance; nonzero
  `cache_read_input_tokens` on the second `user_agent` call; user-agent word counts inside each
  style's range at the chosen effort; and the refusal rate. **Fall back to `claude-sonnet-5`
  with `thinking: {"type": "disabled"}` if refusals exceed ~2%, or if the `claude-sonnet-5-5`
  id is rejected as unknown** (the bundled skill's model table lists `claude-sonnet-5` but not
  `claude-sonnet-5-5`, so the smoke test is the arbiter).

## 2. API key handling

**Never read `ANTHROPIC_API_KEY`** — Claude Code consumes that variable in preference to a
subscription. The lab reads `LAB_ANTHROPIC_API_KEY` from a gitignored `.env`, loaded by the CLI
only, and passes it to the SDK explicitly (`Anthropic(api_key=...)`, never the zero-arg
constructor). `env -u ANTHROPIC_API_KEY make test` stays the Phase 0 gate.

## 3. Refusal and truncation are error states

`stop_reason: "refusal"` arrives as **HTTP 200**, so an unguarded reader parses refusal prose as
output or records it as `schema_invalid`. Personas in this domain are exactly the content that
trips `general_harms`.

`Completion` carries `stop_reason` and `stop_details`. `status="error"` with
`reason ∈ {refusal, max_tokens, model_context_window_exceeded}`. Never parsed as output,
excluded from every denominator, counted in judge health and `failures.jsonl`.

## 4. Exact-field sampler — break the field correlation

`class = i mod k` makes `relationship_goal` and `religion_importance` (both k=4) perfectly
correlated and `has_kids` a function of both. Keep exact per-field balance **and** break the
correlation: per field, repeat the classes to length N, then shuffle with
`random.Random(f"{seed}:{field}")`.

## 5. Aligner

Quotes capped at **12 words**. On `span_invalid`, re-ask **once** with only the failing claims
and the exact text of each cited turn; a second failure stays `span_invalid`. Report **first-pass
and final** `span_invalid` rates. This is the plan's own #1 time risk — models paraphrase.

## 6. Durable context (first step of Phase 0)

Committed: `PLAN.md` (this file), `docs/SPEC.md` (spec §1–6 only, §0 omitted entirely,
"~27-minute" replaced with "long"), `CLAUDE.md` (working agreement, never-cut list, cut order,
pointers). No committed file references the private context.

## 7. Time — 17.5h serial against a ~10h budget

After Phase 0, four parallel tracks. **Phase 0 files are main-thread only**
(`models.py`, `metrics_schema.py`, `paths.py`, `llm.py`, `cache.py`, `cli.py`,
`pyproject.toml`, `Makefile`). **Contract changes go through the main thread.** `test_e2e`
gates every merge.

| track | scope |
| --- | --- |
| **A** (main, needs key) | personas → sim → extract → judge + `verify_span` → aligner iteration |
| **B** | `noise/` and its tests |
| **C** | `metrics.py`, `floor.py`, `disclosure_check.py`, Wilson intervals, tests on hand-built fixtures |
| **D** | `report/` and diff rendering against `tests/fixtures/aggregate_synthetic.json` (test fixture only — never rendered into the README or the demo) |

**Cut to v1.1:** `contradict` facts and `decoy_rate` (M7) — disclosure mix becomes
60/30/10 volunteer/needs_followup/hedge; `specificity_match`; the dollar-cost table (token
usage stays in the manifest). The `Disclosure` literal and `Fact.decoy_value` stay in the
contract so v1.1 doesn't break it; the sampler and prompts just don't emit `contradict`.

**Schedule from Phase 0 start:** h0–2 Phase 0 · h2–6.5 tracks A–D · h6.5–8 integrate, one
`make numbers`, commit fixtures/baselines/report · h8–10 README, 20 audit labels, `make demo`
rehearsal. **Hard stop at h10**; at the stop, commit and `git tag pre-trial-2026-09-30`.

## 8. `questions.yaml` freezes in Phase 0

It is upstream of every transcript, so it freezes alongside `user_agent.txt` and
`interviewer_followup.txt`. The starter text is replaced during Phase 0. **No real simulation
until it is confirmed.**

---

# Phase 0 contracts

## Provider and call path

**P1 — `Provider.complete` is `async`.** §3.3 declares it `def` while also mandating asyncio
with a semaphore at `concurrency: 8`. `async def complete(...)`, `llm.call` async, CLI wraps
with `asyncio.run`.

**P2 — no sampling params.** `temperature`/`top_p`/`top_k` are removed and return 400; the
Messages API has no `seed`. Signature:

```python
async def complete(self, *, system: str, messages: list[Message],
                   json_schema: dict | None = None, max_tokens: int = 4000,
                   thinking: str = "between_tools",
                   effort: Literal["low","medium","high"] = "low",
                   nonce: str | None = None) -> Completion: ...
```

**P3 — two different things were called "seed".** Python-side seeds (`--seed 7`, `sim_seed`,
`noise_seed`, `order_seed`, audit-sample seed) are `random.Random(seed)` and fully
deterministic — they do all the reproducibility work that matters. A model-side seed does not
exist and leaves the call path entirely. One comment at each site.

**P4 — what replaces temperature 0.** `thinking: {"type": "between_tools"}` plus
`output_config: {effort: ...}` — `low` for `persona_gen`/`user_agent`/`interviewer_followup`,
`medium` for `aligner` (span selection is accuracy-critical), and for `extract` **effort is a
pinned declared parameter**, part of the system under test, byte-identical between baseline and
candidate. Both live per-role in `lab.yaml`, go into `ProvenanceStamp.params`, the cache key,
and the report manifest. Every system prompt carries: *"Do not include internal or system XML
tags in your response."*

**P5 — honest determinism claim.** Reproducibility rests on the content-hash cache plus the
§3.12 **measured** variance floor, not on temperature 0. The floor is on the never-cut list —
it is the only thing making the diff gate sound. `floor_runs: 5` for the committed baseline.
The README says this plainly.

**P6 — structured outputs** via `output_config: {format: {type: "json_schema", ...}}` for
persona_gen, extraction, and the judge. Keep §3.3's parse → `jsonschema` → retry-with-error →
`error` ladder as fallback. Consequence stated honestly: `schema_invalid` will rarely fire
against the real API and is exercised by `FakeProvider` tests. Keep the state.

**P7 — prompt caching on the user-agent system prompt.** Bio + fact sheet (~1,400 tokens) is
byte-stable across ~25 calls per persona and is the bulk of all input tokens. Cache minimum is
512 tokens, so it qualifies. Requires a byte-stable prefix: no interpolated timestamps, sorted
serialization of the fact sheet. Verify `cache_read_input_tokens != 0` in the smoke test.

**P8 — cache key.** `sha256(canonical_json({role, system, messages, json_schema, model,
max_tokens, thinking, effort, nonce}))`, keys sorted. No temperature, no seed.

**P9 — small freezes.** `Completion.json` → **`parsed`** (Pydantic 2.13.4 warns
`Field name "json" ... shadows an attribute in parent "BaseModel"`, and it shadows the stdlib
import inside provider modules). `Completion.usage` typed for `input_tokens` / `output_tokens` /
`cache_read_input_tokens`; token usage goes in the manifest (dollar table deferred to v1.1).
`Role = Literal["persona_gen","user_agent","interviewer_followup","extract","aligner"]`, since
`llm.call(role,…)`, FakeProvider keying, and per-role config would otherwise drift as three
independent strings. `code_version()` returns `"dev"` when `git rev-parse HEAD` fails and
`<sha>-dirty` on a dirty tree. `max_tokens` default 4000, 8000 for persona_gen.

## Models

**M1 — `Alignment` validators** (§3.16 asserts one that §3.2 doesn't define):

- `span_text` ⟹ `span_turn_id`
- `method == "exact"` ⟹ `verdict ∈ {supported, unsupported, abstained}` and `span_turn_id is None`
- `verdict ∈ {judge_error, span_invalid, supported_offsheet}` ⟹ `method == "judge"`
- `supported_offsheet` ⟹ `matched_fact_id is None` and span valid
- `supported` ⟹ `matched_fact_id is not None`, and span present when `method == "judge"`

**M2 — exact-field values move from the LLM into the Python sampler.** `relationship_goal`,
`wants_kids`, `has_kids`, `religion_importance` are sampled in `personas/sample_skeleton.py`
(as `age` already is) and passed to `persona_gen` as *constraints*; the LLM writes a consistent
person and bio **around** those anchors and generates only judged-field facts. Assignment per
amendment 4: repeat classes to length N, shuffle with `random.Random(f"{seed}:{field}")` —
exact per-field balance, no cross-field correlation. The majority-class floor then sits at 1/k
by construction, and one more LLM-generated oracle disappears. The sampler emits those `Fact`s
complete (value, `disclosure`, `importance`, `specificity`), so the exact-scored path is
deterministic end to end.

**M3 — type comparison.** `Fact.value: str | bool | int` is fine as written (verified on
Pydantic 2.13.4: smart union resolves `true`→bool, `35`→int, `"35"`→str; reordering changes
nothing). The JSON Schema is the type gate — a string in an integer field is `schema_invalid`,
never silently coerced into agreement or into a spurious `unsupported`. `exact.py` normalizes
case and whitespace for string enums only, never coerces types. `test_exact.py` covers `"35"`
vs `35` and `"no"` vs `False`.

**M4 — a fifth verdict, `abstained`.** `extract.txt` says *"Use `unsure` when the user did not
say"* while §3.8 says a missing scalar is not a claim — so a model following the prompt is
scored `unsupported` against a known truth value, recording a declined answer as a
hallucination. And `unsure` can't just be dropped: it's a legitimate truth class for
`wants_kids` and `relationship_goal`, and under M2 a guaranteed one for 1/3 and 1/4 of
personas. Rule: an exact claim equal to the enum's unsure member is `abstained` when the truth
value is not `unsure`, `supported` when it is. Excluded from the precision denominator; counts
as not-captured for recall. Report `abstention_rate`.

**M5 — other model freezes.** `JudgeHealth.degraded` derived in a validator.
`Extraction`: `status == "ok" ⟺ profile is not None`; `claims` non-empty only when `ok`.
`FactCoverage`: `captured is False ⟹ by_claim_id is None`. `Transcript` gains
**`source_transcript_id: str | None`** (M8). `ProvenanceStamp.params` records the nonce but
excludes it from hashed identity.

## Metrics

**M6 — `recall` conflates two stages, and stage attribution is the whole pitch.** A
`needs_followup` fact no follow-up probed was never disclosed, so extraction *cannot* capture
it — yet `captured / facts` mixes "the interview didn't elicit it" with "extraction missed it".
Report both (free, `disclosed_turn_id` already exists):

- `recall_elicited` = captured / facts with non-null `disclosed_turn_id` — extraction's own
  recall, and **the one the diff gate uses**
- `recall` = pipeline recall, as spec'd

**M8 — noise changes word counts.** Dropping words and inserting fillers changes `word_count` →
`simulated_minutes` → `yield_per_minute` drifts across the noise axis for reasons unrelated to
extraction. Time-based metrics use the **source (noise-0)** counts via `source_transcript_id`.

**M9 — exact-field recall baseline.** Compute the constant-majority-class floor **per exact
field from the run's own truth sheets**, report `exact_floor.<field>`, and show **lift over
floor** beside exact-field recall. Under M2 the floor is 1/k by construction. Unit-tested
against hand-built truth sheets with known class counts.

**M10 — disclosure self-report check.** `disclosed_fact_ids` is the user agent's own side
channel and is ground truth for every recall and yield number, yet the spec applies verbatim-span
skepticism to the judge while taking the simulator's word for it. Two pure-Python checks, no new
API calls:

- `disclosure_mismatch` — **judged-field `specific` facts only** (enums and booleans can't be
  overlap-checked), run on the **clean** transcript, not the noised copy (ASR noise would
  inflate it). Matches on **anchor tokens** (numbers, proper nouns, rare content words), not
  full-value overlap, so paraphrase isn't read as mismatch. Reported with `n`.
- `untagged_capture` — reverse check from existing artifacts: facts captured with a valid judge
  span whose `fact_id` appears in no `disclosed_fact_ids`, or whose cited span precedes the first
  tagged turn. As spec'd these drop out of `yield` but stay in the `followup_share` denominator;
  **exclude them from both** and report the count.

Both are **health stats printed beside judge health, not gates.** Tests: one match, one
paraphrase, one true mismatch.

**M11 — one definition of judge error rate.** §3.11's `(judge_error + span_invalid) / claims`
has a denominator including exact-method claims the judge never saw, deflating the rate and
weakening the 5% gate. Use **judged claims** for both it and `JudgeHealth.error_rate`. Aggregate
by **pooling** calls and errors across personas; `degraded = pooled_error_rate > 0.05`. Never
average per-persona booleans.

**M12 — metric keys and undefined rates.** Flat dotted string keys in `dict[str, float]`,
exactly as §3.11 spells them. `metrics_schema.py` owns `METRIC_KEYS` plus documented dynamic
key *patterns*, because `test_e2e`'s "aggregate.json has every metric key" is otherwise
undefined. A rate with a zero denominator is **omitted from per-persona `metrics` with its `0`
recorded in `n`** — never `0.0`, never `NaN`. `0.0` there reads as "we got everything wrong",
exactly the silent-misreporting class this project exists to prevent.

**M13 — aggregation and intervals.** §3.11 says "per persona first, then mean" and then asks for
a Wilson interval — but Wilson needs a proportion, not a mean of ratios, and `yield_per_minute`
isn't a proportion. So: **pooled counts + Wilson as the headline**, per-persona distribution
shown separately. Mark each key count-or-rate; `Scores.n` only for rates. README states the
interval ignores persona clustering.

**M14 — `precision_dealbreakers` and `recall_dealbreakers` cut differently.** One by schema
*field*, the other by persona-assigned `importance`. Adjacent rows, near-identical names,
different populations. Define `DEALBREAKER_FIELDS` and use the field-based cut for both.

**M15 — remaining freezes.**
- Dedupe claims by (field, normalized value) in `decompose.py` — otherwise a model listing
  "travel" three times earns three `supported` claims, because coverage dedupes via singular
  `by_claim_id` but precision doesn't. Log the dedupe count; enforce fact-match injectivity.
- `schema_invalid` personas: excluded from precision denominators, counted zero recall, printed
  as "n personas excluded: schema_invalid". **`schema.json` must have no `required` array** (or
  only `summary`) — otherwise every incomplete profile is `schema_invalid` and the eval
  collapses.
- `recall_by_disclosure.hedge` reads near zero unless semantics are written down: hedge facts
  count as captured when matched at `generic` specificity. Say so in `aligner.txt` *and* the
  metric doc.
- `followup_share` is largely a function of the style mix — terse personas are 15–40 words
  against a 40-word trigger, so they follow up on every question; rambling personas never
  trigger on length. Break it down by style; style mix goes in the manifest. Denominator =
  captured facts with non-null `via_followup`.
- Follow-up turns carry the **parent** `question_id` with `is_followup=True`. `yield_per_minute`
  counts interviewer words too — it measures interview time. Per-question word counts come from
  grouping `Turn`s on `question_id`; no new field.
- `lab diff` gates on noise **0.0**; other levels reported un-gated.

## Artifacts and paths

**A1 — floor runs currently overwrite each other.** §3.12 runs extraction `floor_runs` times
with distinct nonces, but §3.4's paths don't encode the nonce, so three runs write one file and
σ is computed over a single survivor. Use `extractions/<tag>.floor<k>/` and
`scores/<tag>.floor<k>/`, from one `artifact_key()` / `artifact_path()` in `paths.py` that every
stage calls. Drop dotted-float filenames: `p001__s1__n010.json`.

**A2 — no status for failed personas or simulations.** §3.9 says "fail the persona with status
logged" but only `Extraction` has a `status`. Silently dropping personas changes every number.
`runs/<run_id>/failures.jsonl` plus manifest counters (`n_personas_requested` / `n_ok` /
`n_failed`); the report prints both.

**A3 — tag collision silently corrupts the diff.** `lab extract --tag v1` with a different
prompt overwrites `extractions/v1/` without complaint; the worst available demo failure is
comparing v1 against a "v1" quietly regenerated with v2's prompt. The manifest records
`tag → (prompt_hash, schema_hash, model, effort, thinking)`; a mismatch on rerun is an error.

**A4 — `run_id` collides within a minute.** Acceptable (idempotent by design) but documented;
`lab run` prints whether it resumed or created.

## Demo and infra

**D1 — `make demo` cannot work as spec'd.** §3.17 wants it on committed fixture personas
"(cached)" while also saying never commit `.cache/` or `runs/`, and §5 opens with `make demo`
from a fresh terminal. Fix: commit `fixtures/extractions/` and `fixtures/scores/`;
**`make demo` = score + report + diff from committed artifacts, $0 and no key.**
`make demo-live` for the real thing. A reviewer cloning the repo must not spend $5 to see the
report.

**D2 — `make clean` destroys the hand-filled audit labels.** §3.13 writes them inside the
gitignored run dir; `make clean` is `rm -rf runs .cache`. Phase 4 spends 30–40 min labelling and
the headline agreement number evaporates, leaving the red "judge not yet audited" banner in the
demo. Labels live at `fixtures/audit_labels.yaml`, **keyed by content hash, not list position**;
the report prefers run-local and falls back to committed.

**D3 — `FakeProvider` keying is a frozen contract.** Dispatch on
`(role, prompt_hash, messages_hash)`; canned-response format frozen in Phase 0. Under
FakeProvider variance is exactly 0, so `floor = max(2σ, 0.005) = 0.005` — make the fake
candidate worse by ~20pp so `test_e2e`'s `lab diff v1 v2` exit-1 assertion can't flake.

**D4 — noise edit accounting.** The punctuation op is sentence-level but triggered per word and
is idempotent, making the ±15% test ill-defined. `edit_count` = number of *words* at which an
operation fired, including no-op strips — then exactly Binomial(words, rate); at n=10,000,
p=0.1 the ±15% band is ~5σ. Seed the test anyway.

---

# Run tiers and prompt freeze order

Per persona: 1 persona_gen + ~15.5 `user_agent` + 10 `interviewer_followup` + extraction +
judge ≈ **28 calls**, ~90% simulation. Simulation is **serial per persona** (turn N+1 depends on
turn N), so parallelism only spans personas. Dollars aren't the constraint; wall clock is.

| tier | config | wall clock | cost |
| --- | --- | --- | --- |
| `make dev` | 3 personas, 1 noise level, 1 tag, no floor | ~90 s | ~$0.05 |
| integration | 12 personas, 4 noise levels, 2 tags, floor 3 | ~15 min | ~$5 |
| `make numbers` | 12 personas, 4 noise levels, floor 5 | ~20 min | ~$6 |

`make numbers` runs **once**; commit fixtures, baselines, report; don't run it again before the
demo. Measured token usage comes from `Completion.usage`, not an estimate.

**Prompt freeze order** — cache invalidation cascades downstream and simulation sits upstream of
everything: `questions.yaml` + `user_agent.txt` + `interviewer_followup.txt` freeze **in
Phase 0** (amendment 8), then `aligner.txt`, and `extract.txt` last — the system under test is
*meant* to change and it's the cheap axis (extraction+judge only). Any later tweak to a sim
prompt throws away every transcript and re-pays the expensive axis.

# Cut order if time runs short

1. **OpenAI adapter — deleted, not stubbed.** Protocol stays; one README line is the swappable story.
2. **GitHub Action — write the YAML, never debug it.** It's a demo artifact; the README says so.
3. **Drop to 8 personas.** `personas` is config; intervals widen and the report prints them.
4. **`lab audit` last.** Printing the red "judge not yet audited" banner and saying so in the
   demo is more credible than a rushed label pass.

**Never cut:** exact + judged scoring, span verification, the error states (including refusal),
`make numbers`, the diff, **the noise floor** (P5), **failure accounting** (A2).

# Verification

- **Phase 0 gate, the important one:** `git clean -xdf && uv sync && env -u ANTHROPIC_API_KEY make test`.
- **Per phase:** `make test` (ruff + pytest) plus the gate. `make demo` stays green from end of
  Phase 1; a change that breaks it gets fixed before the next feature.
- **`test_e2e.py` gates every merge:** two fixture personas through `lab run` on `FakeProvider`;
  every artifact exists and validates; `aggregate.json` carries every key in `metrics_schema.py`;
  `index.html` and `summary.md` render; `lab diff v1 v2` exits 1 on the worse fake candidate.
- **Error-state regressions:** `judge_error`, `span_invalid`, `refusal`, and `max_tokens`
  excluded from every rate denominator; degraded trips above 5% on *pooled* counts; a
  zero-denominator rate absent rather than `0.0`; a dropped persona appears in `failures.jsonl`
  and the manifest counters.
- **New metrics:** `exact_floor` against hand-built truth sheets with known class counts; the
  disclosure overlap check against one match, one paraphrase, one true mismatch; `abstained`
  against an `unsure` claim with both an `unsure` and a non-`unsure` truth value.
- **Sampler:** per-field class balance exact, and `relationship_goal` vs `religion_importance`
  not correlated across personas (amendment 4).
- **Noise:** edit count within ±15% of `rate × words` over 10,000 words (D4); determinism under
  seed; interviewer turns untouched; `disclosed_fact_ids` preserved.
- **Caching:** identical inputs hit, changed parameter misses, nonce bypasses,
  `cache_read_input_tokens != 0` on the second `user_agent` call.
- **`make numbers`** on a clean checkout rewrites README between the markers and fails loudly if
  they're missing. No figure anywhere is hand-typed.
- **Demo rehearsal:** fresh terminal, `make demo`, open `report/index.html`, walk §5's eight
  steps under five minutes.

# Open items

1. `questions.yaml` starter text is replaced in Phase 0; no real simulation until confirmed.
2. `.env` with `LAB_ANTHROPIC_API_KEY` before the first real call.
3. §2 architecture decision 6 needs its README wording per P5.
4. Judge and system-under-test share a model. One honest line in the README's limits section: a
   judge sharing the SUT's failure modes can't catch them.

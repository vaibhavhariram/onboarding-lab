# onboarding-lab — reference spec

This is the frozen design reference. `PLAN.md` is the build plan and its amendments
**supersede** this document wherever the two disagree (notably: model id, the
`thinking` parameter, the API-key variable, refusal handling, the exact-field
sampler, the aligner retry, run size, and the cut list). Section numbers here are
the targets for every `§` reference in `PLAN.md`.

---

## 1. Working agreement

1. **Plan first.** Write PLAN.md and stop. No code until approved.
2. **Contracts before modules.** Phase 0 freezes the Pydantic models, CLI signatures, artifact layout, and provider interface. Modules are built against those contracts; contract changes require an explicit note in the status update.
3. **Tests run without API keys.** A deterministic `FakeProvider` backs every test. `make test` must pass on a clean checkout with no environment variables set.
4. **No number without a script.** README figures are written only by `scripts/numbers.py` between `<!-- numbers:start -->` and `<!-- numbers:end -->` markers. Never type a metric into any document by hand. Never invent example numbers in docstrings or the README.
5. **Errors are states, not verdicts.** A failed judge call, a failed provider call, an invalid schema, or a cited span that does not literally exist in the transcript each get their own status. They are excluded from metric denominators and counted in a "judge health" section. If the judge error rate exceeds 5%, the run is marked `degraded` in every report.
6. **Join by id, never by position.** Every artifact carries `persona_id`, `transcript_id`, `extraction_id`, and input hashes. Scoring joins claims to facts to turns by id.
7. **Product-agnostic naming.** The repo, README, code comments, prompts, and fixtures describe a generic "synthetic evaluation harness for conversational intake / voice onboarding pipelines". No customer, product, or individual is named in any committed file.
8. **Status after every phase gate**, three lines: done / next / blocked.
9. **Subagents** only after Phase 0 is merged, one module per subagent, working against the frozen contracts. The two-persona end-to-end test gates every merge.
10. **Dependencies** are limited to: `pydantic>=2`, `typer`, `jinja2`, `pyyaml`, `anthropic`, `openai`, `httpx`, `matplotlib` (SVG backend only), `jsonschema`, `pytest`, `ruff`. Ask before adding anything else.
11. **`make demo` stays green** from the end of Phase 1 onward. If a change breaks it, fixing it comes before the next feature.
12. **Content guardrails for generated personas.** Adults only (ages 25–45), PG-13, no sexual content, no real people, no stereotyping by ethnicity, religion, or nationality. Diversity comes from seed-sampled demographics, not from caricature.
13. **Commits:** small, after every gate, conventional-commit messages.

---

## 2. The writeup (reference design)

### Onboarding Lab: Synthetic Evaluation Harness for Voice Onboarding

#### Why the team needs this

The product rests on one step: a long voice conversation becomes a structured profile, and every downstream stage inherits what extraction got right or wrong. Today there is no number for how good that step is. Onboarding has no agreed success metric (speed through the questions versus richness of extracted detail), prompt-output quality is checked by hand, and a failed match cannot be traced to the stage that broke it.

The blocker is labels, not compute. Hand-judging one real transcript takes about 20 minutes, a labeled set goes stale the moment a question changes, and a real user's true facts are never fully knowable. So question and prompt changes ship on intuition.

The lab inverts the order. It writes the person first as a structured truth sheet, runs that person through a simulated onboarding interview, extracts a profile from the transcript, and scores the profile against the truth. Labels are exact and free by construction, and the fixture set becomes a regression suite that runs on every question or prompt change.

What the team gets: precision and recall per profile field, with hallucinations named; per-question yield, which settles speed versus richness with data; a CI gate on prompt and question changes; a curve of extraction quality against transcription (ASR) noise, which is the concrete argument in the voice-native migration; and a test-retest number for whether one person yields one profile. Synthetic users are not real users, so this complements audits of real transcripts rather than replacing them. It is, however, the only source of exact ground truth, and it runs without touching production or user data.

#### What it does

One command runs six stages. Each stage writes a JSON artifact, so any stage can be rerun alone.

1. Persona factory — generates N truth sheets: demographics, values, life events, preferences, dealbreakers, plus disclosure rules (which facts are volunteered, which need a follow-up, which are hedged, which are contradicted) and a conversational style (terse, rambling, tangential).
2. Interview simulator — an interviewer agent runs a question script with a follow-up policy against a user agent playing the persona. Output is a turn-by-turn transcript, with every disclosed fact tagged to the turn that elicited it.
3. Noise injector — rewrites the transcript at a configurable rate to imitate ASR errors: dropped words, homophone swaps, fillers, lost punctuation.
4. Extraction runner — applies a pluggable extraction prompt and JSON Schema to the transcript and returns a profile.
5. Scorer — aligns each extracted claim to a truth fact, with a cited transcript span. Enum and boolean fields are checked exactly, with no model in the loop.
6. Report and diff — a static HTML report plus a Markdown summary. Diff mode reruns only extraction on a candidate prompt and compares it against a baseline.

Metrics reported:

- Precision by field. An unsupported claim counts as a hallucination; dealbreaker fields are reported separately because they are the costliest to get wrong.
- Recall by field and by disclosure type (volunteered vs. needed a follow-up), which prices the follow-ups in numbers.
- Specificity: generic claims ("likes travel") vs. specific ones ("spent six months in Oaxaca").
- Per-question yield: facts captured per question and per simulated minute (words ÷ 150).
- Test-retest stability: same persona, new seed, shuffled question order; profile agreement. If one person yields different profiles depending on order, matching inherits that noise.
- Every metric above plotted against noise rate.
- Stretch: match impact. Truth compatibility (shared dealbreakers and values, known by construction) vs. compatibility computed from extracted profiles, reported as top-k overlap.

#### How it is built

Python 3.12 with uv, a Typer CLI, Pydantic models as the contracts between stages, Jinja2 for the static report, and thin provider adapters (Anthropic default, OpenAI supported). No database, no server, nothing to deploy.

CLI: `lab gen`, `lab sim`, `lab noise`, `lab extract`, `lab score`, `lab report`, `lab diff`, and `lab run` for the whole pipeline. Every artifact is keyed by a content hash of its inputs (persona, question script, prompt, schema, model, parameters), so reruns are cache hits. `lab diff --baseline v1 --candidate v2` reuses the transcripts and reruns only extraction.

Build order: Pydantic models and the CLI contract first, then a two-persona end-to-end test that must pass before any module is filled in. Modules are then built in parallel against the fixed contracts, and the integration test gates every merge. `make numbers` regenerates every figure in the README from a clean checkout, so no number in the writeup exists without a script that produces it.

#### Key architecture decisions

1. Truth-first personas instead of hand labels. Ground truth exists before the conversation does, so scoring is exact and labels cost nothing. This is the whole reason the lab is cheap enough to run on every change.
2. Disclosure rules and styles live in the persona. Without them, simulated users are cooperative and the eval is too easy. With them, recall by disclosure type measures the value of follow-ups directly, and terse or tangential users stress the interview policy.
3. Everything product-specific is a file. Questions (YAML), extraction prompt (text), profile schema (JSON Schema), model and provider (config). The shipped versions are stand-ins; a team swaps in the real ones and the numbers become theirs.
4. Claims must cite spans. Each extracted claim is aligned to a truth fact and must cite the transcript span that supports it. No span means unsupported, which counts as a hallucination. Enum and boolean fields never pass through the model judge.
5. The judge is audited. A fixed sample of alignments is reviewed by hand and the agreement rate is printed in the report. An unaudited judge is a number nobody should trust.
6. Determinism plus a measured noise floor. Fixed Python-side seeds, pinned and recorded sampling configuration, content-hash caching. The baseline runs repeatedly, and the CI diff fails only on regressions larger than the observed run-to-run variance, so the gate never flakes. (Superseded in detail by PLAN.md P2/P5 — model sampling is not deterministic and is not assumed to be.)
7. Transcription noise is a first-class axis. The voice-native migration changes the input distribution; plotting every metric against noise rate is the cheapest available proxy for that change.
8. Facts are attributed to the eliciting turn. This is what makes per-question yield and follow-up value computable rather than estimated.
9. Static outputs only. JSON, HTML, Markdown. It runs on a laptop or in a GitHub Action with no infrastructure.

#### v1 scope, cuts, and adoption

v1 ships personas at multiple noise levels, the stand-in question script and schema, every metric except match impact, the HTML report, diff mode, a GitHub Actions example, and a README whose figures regenerate from `make numbers`.

Cut from v1: audio input, real transcripts, match impact (stretch), and any UI beyond the static report.

Adoption is one command with a team's own files:

```
uv run lab run --config lab.yaml
```

From there, `lab diff` on the next prompt change, and the Action on every pull request that touches a question or a prompt.

---

## 3. Frozen contracts

### 3.1 Repo layout

```
onboarding-lab/
  pyproject.toml            # uv-managed; package name onboarding_lab; console script `lab`
  Makefile
  README.md                 # numbers only between <!-- numbers:start --> / <!-- numbers:end -->
  lab.yaml                  # default run config (Section 3.6)
  questions.yaml            # question script (Section 3.7)
  schema.json               # profile JSON Schema (Section 3.8)
  prompts/
    persona_gen.txt
    user_agent.txt
    interviewer_followup.txt
    extract.txt             # the system under test (stand-in)
    extract_v2_worse.txt    # deliberately degraded variant for the diff demo
    aligner.txt             # judge
  src/onboarding_lab/
    models.py               # all Pydantic contracts (Section 3.2)
    providers/              # base.py, anthropic.py, openai.py, fake.py
    llm.py                  # call(role, ...) with cache + retries + status
    cache.py
    personas/               # sample_skeleton.py, generate.py
    sim/                    # interviewer.py, user_agent.py, run.py
    noise/                  # inject.py, homophones.py
    extract/                # decompose.py (profile -> claims), run.py
    score/                  # exact.py, align.py, verify_span.py, metrics.py, stability.py, floor.py
    report/                 # render.py, charts.py, templates/report.html.j2, summary.md.j2
    cli.py
  scripts/numbers.py        # regenerates README numbers
  fixtures/                 # committed synthetic personas + transcripts
  baselines/                # committed baseline scores for CI diff
  tests/
  .github/workflows/eval.yml
  runs/                     # gitignored
  .cache/                   # gitignored
```

### 3.2 Models (Pydantic v2)

```python
Disclosure = Literal["volunteer", "needs_followup", "hedge", "contradict"]
Importance = Literal["dealbreaker", "core", "color"]
Specificity = Literal["generic", "specific"]
Style = Literal["terse", "balanced", "rambling", "tangential"]

class ProvenanceStamp(BaseModel):
    model: str
    prompt_hash: str
    params: dict
    code_version: str          # git sha or "dev"
    created_at: datetime

class Fact(BaseModel):
    fact_id: str               # "f01"
    field: str                 # key in schema.json; array fields -> one Fact per item
    value: str | bool | int
    specificity: Specificity
    disclosure: Disclosure
    importance: Importance
    decoy_value: str | bool | int | None = None   # contradict only: stated first, corrected later

class TruthSheet(BaseModel):
    persona_id: str            # "p001"
    seed: int
    style: Style
    demographics: dict[str, str | int]             # age, city, occupation, education
    facts: list[Fact]
    bio: str                   # 120-200 words in the persona's voice, derived from facts
    provenance: ProvenanceStamp

class Question(BaseModel):
    question_id: str           # "q01"
    text: str
    targets: list[str]         # schema fields this question aims at
    max_followups: int = 1

class QuestionScript(BaseModel):
    script_id: str
    questions: list[Question]
    followup_trigger_words: int = 40   # user answer shorter than this -> follow-up
    order_seed: int | None = None      # None = as written; int = shuffled order (stability runs)

class Turn(BaseModel):
    turn_id: str               # "t007"
    speaker: Literal["interviewer", "user"]
    question_id: str | None    # interviewer turns, and the user turn answering them
    is_followup: bool = False
    text: str
    word_count: int
    disclosed_fact_ids: list[str] = []   # user turns only; from the user agent's side channel

class Transcript(BaseModel):
    transcript_id: str
    persona_id: str
    script_hash: str
    sim_seed: int
    noise_rate: float = 0.0
    noise_seed: int | None = None
    turns: list[Turn]
    simulated_minutes: float   # total words across all turns / 150
    provenance: ProvenanceStamp

class Claim(BaseModel):
    claim_id: str              # "c01"
    field: str
    value: str | bool | int

class Extraction(BaseModel):
    extraction_id: str
    transcript_id: str
    tag: str                   # e.g. "v1"
    prompt_hash: str
    schema_hash: str
    model: str
    status: Literal["ok", "schema_invalid", "provider_error"]
    profile: dict | None
    claims: list[Claim]        # decomposed from profile; empty unless status == "ok"
    raw: str
    provenance: ProvenanceStamp

Verdict = Literal["supported", "supported_offsheet", "unsupported", "judge_error", "span_invalid"]

class Alignment(BaseModel):
    claim_id: str
    verdict: Verdict
    method: Literal["exact", "judge"]
    matched_fact_id: str | None
    span_turn_id: str | None
    span_text: str | None      # must literally occur in that user turn (normalized), else span_invalid
    specificity: Specificity | None

class FactCoverage(BaseModel):
    fact_id: str
    captured: bool
    by_claim_id: str | None
    disclosed_turn_id: str | None    # first user turn whose disclosed_fact_ids includes this fact
    question_id: str | None
    via_followup: bool | None

class JudgeHealth(BaseModel):
    calls: int
    errors: int
    span_invalid: int
    error_rate: float
    degraded: bool             # error_rate > 0.05

class Scores(BaseModel):
    extraction_id: str
    alignments: list[Alignment]
    coverage: list[FactCoverage]
    judge: JudgeHealth
    metrics: dict[str, float]  # keys defined in Section 3.11
    n: dict[str, int]          # denominators for every rate in metrics
```

### 3.3 Provider adapter

```python
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class Completion(BaseModel):
    status: Literal["ok", "error"]
    text: str | None
    json: dict | list | None   # parsed and schema-validated when json_schema was given
    model: str
    usage: dict
    latency_ms: int
    error: str | None = None
    cached: bool = False

class Provider(Protocol):
    name: str
    def complete(self, *, system: str, messages: list[Message], json_schema: dict | None = None,
                 temperature: float = 0.0, max_tokens: int = 2000, seed: int | None = None) -> Completion: ...
```

- `AnthropicProvider` (default), `OpenAIProvider` (model configurable; stub-level in v1 unless time permits), `FakeProvider` (deterministic canned responses keyed by a `role` tag in the system prompt plus a hash of the messages; records every call; used by all tests).
- Default model IDs live in `lab.yaml` only. Verify current model IDs against Anthropic's docs before pinning; do not hardcode IDs in code.
- All calls go through `llm.call(role, provider, ...)`, which applies caching, then up to 3 retries with exponential backoff and jitter, and returns a `Completion` with `status="error"` rather than raising. Structured output: pass `json_schema` where the provider supports it; otherwise parse, validate with `jsonschema`, retry once with the validation error appended, then return `error`.
- Concurrency: `asyncio` with a semaphore (`concurrency: 8` in config).

(Superseded in detail by PLAN.md P1/P2/P6/P9: `complete` is async, sampling params are removed, `Completion.json` is renamed `parsed`, and refusal/truncation are explicit error states.)

### 3.4 Caching and artifacts

- Cache key = `sha256(canonical_json({...}))`. Stored at `.cache/llm/<key>.json`. Disabled by `--no-cache` or `LAB_CACHE=0`. Noise-floor runs pass a `nonce` in params to bypass the cache deliberately.
- Run layout:

```
runs/<run_id>/
  manifest.json             # config snapshot, hashes of script/prompt/schema, code_version, timestamps
  personas/p001.json ...
  transcripts/p001.s1.json  # sim_seed 1
  transcripts/p001.s1.n0.10.json   # noised copy
  transcripts/p001.s2.shuffled.json  # stability run
  extractions/<tag>/p001.s1.json
  scores/<tag>/p001.s1.json
  scores/<tag>/aggregate.json
  audit/<tag>/sample.md, labels.yaml
  report/<tag>/index.html, summary.md, charts/*.svg
```

- `run_id = <UTC yyyymmdd-HHMM>-<6-char hash of config>`. Every command is idempotent: a stage is skipped when its output exists and its input hashes match.

(Superseded in detail by PLAN.md A1/A2/A3: floor runs get their own tag directories, filenames avoid dotted floats, failures are recorded, and tag reuse with changed hashes is an error.)

### 3.5 CLI (Typer, console script `lab`)

```
lab gen      --run RUN --n 30 --seed 7
lab sim      --run RUN --script questions.yaml --seed 1 [--shuffle-seed 3]
lab noise    --run RUN --rate 0.1 --seed 1
lab extract  --run RUN --prompt prompts/extract.txt --schema schema.json --tag v1 [--model ID] [--nonce N]
lab score    --run RUN --tag v1
lab audit    --run RUN --tag v1 --sample 20
lab report   --run RUN --tag v1
lab diff     --run RUN --baseline v1 --candidate v2 [--floor-runs 3]     # exit 1 on regression beyond floor
lab run      --config lab.yaml                                            # gen -> sim -> noise -> extract -> score -> report
```

Every command prints a one-line summary and the path it wrote. `--run` defaults to the newest run under `runs/`.

### 3.6 Config (`lab.yaml`)

```yaml
run_name: default
personas: 30
seed: 7
script: questions.yaml
noise_rates: [0.0, 0.1]
concurrency: 8
sim:
  model: claude-sonnet-5-5
  seed: 1
extraction:
  prompt: prompts/extract.txt
  schema: schema.json
  model: claude-sonnet-5-5
  tag: v1
judge:
  model: claude-sonnet-5-5
stability:
  enabled: true
  shuffle_seed: 3
  sim_seed: 1001
floor_runs: 3
```

(Run size superseded by PLAN.md: 12 personas, four noise levels.)

### 3.7 Question script

Format:

```yaml
script_id: intake-v1
followup_trigger_words: 40
questions:
  - question_id: q01
    text: "Tell me about where you grew up and what your family was like."
    targets: [location, life_events, values]
    max_followups: 1
```

Starter script (generic; the question text is to be replaced with notes from a real
onboarding walkthrough — keep the ids and structure):

| id | text | targets |
| --- | --- | --- |
| q01 | Tell me about where you grew up and what your family was like. | location, life_events, values |
| q02 | What does a typical week look like for you right now? | occupation, interests, communication_style |
| q03 | What do you do for work, and how do you feel about it? | occupation, values |
| q04 | What are you most passionate about outside of work? | interests, values |
| q05 | Tell me about a past relationship that shaped how you think about love. | life_events, partner_preferences, dealbreakers |
| q06 | What are you looking for in a partner right now? | relationship_goal, partner_preferences |
| q07 | What are your non-negotiables? | dealbreakers, wants_kids, religion_importance |
| q08 | How do you like to spend a free Sunday? | interests, communication_style |
| q09 | What role do family, faith, or tradition play in your life? | religion_importance, values, has_kids |
| q10 | Where do you see your life in five years? | relationship_goal, wants_kids, life_events |

Follow-up policy: after a user answer, ask one follow-up (up to `max_followups`) if the answer has fewer than `followup_trigger_words` words OR a cheap judge call reports that at least one target field was not addressed. The interviewer never sees the truth sheet or `disclosed_fact_ids`; it decides from the answer text only, like a real agent would.

### 3.8 Profile schema (stand-in, JSON Schema 2020-12)

Fields chosen so every scoring path is exercised (exact-match scalars, judged free text, arrays):

| field | type | scoring |
| --- | --- | --- |
| age | integer | exact |
| location | string | judge |
| occupation | string | judge |
| relationship_goal | enum: casual, long_term, marriage, unsure | exact |
| wants_kids | enum: yes, no, unsure | exact (dealbreaker) |
| has_kids | boolean | exact (dealbreaker) |
| religion_importance | enum: none, low, medium, high | exact (dealbreaker) |
| values | array of string | judge, one claim per item |
| interests | array of string | judge, one claim per item |
| life_events | array of string | judge, one claim per item |
| dealbreakers | array of string | judge, one claim per item (dealbreaker) |
| partner_preferences | array of string | judge, one claim per item |
| communication_style | string | judge |
| summary | string | not scored (excluded from precision/recall; shown in report) |

`additionalProperties: false`. Missing scalar → not a claim (counts against recall only). Every persona's truth sheet always contains a fact for each exact-scored field.

### 3.9 Prompts

**persona_gen.txt.** Input: a seed-sampled skeleton computed in Python (age, city, occupation, education, style, and target counts: 3 dealbreaker facts, 8–10 core, 6–10 color; disclosure mix, and roughly 60% of facts specific). Output: a `TruthSheet` (minus provenance) as JSON. Rules: facts are atomic (one field, one value), mutually consistent, and each exact-scored field appears exactly once; `contradict` facts carry a plausible `decoy_value`; specific facts contain concrete places, names of things, numbers, or durations; the bio is 120–200 words in the persona's own voice and mentions only facts on the sheet; adults 25–45; PG-13; no real people. Validate with Pydantic; on failure, regenerate once with the error appended, then fail the persona with status logged.

**user_agent.txt (system prompt).** "You are {persona}. Here is your bio and your fact sheet with disclosure rules." Rules: answer only the question asked; disclose `volunteer` facts when relevant to the question; disclose `needs_followup` facts only when a follow-up probes that topic; state `hedge` facts vaguely without concrete details; for `contradict` facts, state the decoy value the first time the topic arises and correct it if probed again or when it comes up later; never invent profile-relevant facts that are not on the sheet (small talk and filler are fine); obey the style word range (terse 15–40, balanced 40–120, rambling 120–250, tangential 120–250 with at least one digression unrelated to the question). Output JSON only: `{"utterance": "...", "disclosed_fact_ids": ["f03", "f07"]}`. A fact is "disclosed" when its true value is stated; decoy statements are not disclosures.

**interviewer_followup.txt.** Input: the question, its targets, the user's answer. Output JSON: `{"targets_addressed": [...], "followup": "..." | null}`. The follow-up must be one sentence, grounded in something the user said, and aimed at an unaddressed target.

**extract.txt (system under test).** "Extract a profile from this onboarding transcript into the JSON Schema below. Include only information the user stated. Use `unsure` when the user did not say. Omit anything not stated. Output JSON only." Transcript is rendered as `[t007 user] ...` lines so span citation has stable turn ids.

**extract_v2_worse.txt.** Same as `extract.txt` but with the "include only information the user stated" and "omit anything not stated" lines removed and an added instruction to "infer likely values from context when not stated." This exists so `lab diff` has a regression to catch during the demo.

**aligner.txt (judge).** Input: truth facts `(fact_id, field, value)` for judged fields only, extracted claims `(claim_id, field, value)` for judged fields only, and the transcript as `[turn_id speaker] text`. Output JSON array, one object per claim: `matched_fact_id | null` (same field, same meaning; paraphrase allowed), `span_turn_id` and `span_text` (a verbatim substring from a **user** turn that supports the claim, or null), `specificity`. Never cite interviewer turns. Never cite text that is not verbatim.

### 3.10 Noise injector

- Applies only to **user** turns (the agent's own text is known; ASR error lives on the user side).
- Per word, with probability `rate`, apply one operation drawn from: drop word (40%), homophone / near-phonetic swap from a small dictionary with a character-level fallback (30%), insert filler before the word — um, uh, like, you know (20%), strip sentence punctuation in the surrounding sentence (10%).
- Deterministic under `noise_seed`. Records an edit log per turn. Writes a new transcript file with `noise_rate` and `noise_seed` set; `disclosed_fact_ids` are copied unchanged (the truth of what was said does not change, only its rendering).
- Unit test: over 10,000 words the edit count is within ±15% of `rate × words`.

### 3.11 Scoring rules and metric definitions

Claim decomposition (`extract/decompose.py`): scalar fields → one claim; array fields → one claim per item; `summary` → no claim.

Exact fields (`score/exact.py`): claim value == truth value → `supported`; differs → `unsupported`; never sent to the judge; `method="exact"`.

Judged fields (`score/align.py` + `verify_span.py`): judge output per claim, then deterministic verification: `span_text` must occur in the named user turn after normalizing whitespace, case, and punctuation. Valid span + `matched_fact_id` → `supported`. Valid span + no fact match → `supported_offsheet` (extraction was faithful to the transcript; the simulator leaked a fact not on the sheet — a sim-quality problem, not an extraction hallucination). No span → `unsupported`. Judge call failed → `judge_error`. Cited span not found verbatim → `span_invalid`.

Coverage (`FactCoverage`): a fact is captured if some claim with `matched_fact_id == fact_id` is `supported`, or an exact-field claim equals it. `disclosed_turn_id` = the first user turn whose `disclosed_fact_ids` contains the fact; `question_id` and `via_followup` come from that turn.

Metric keys in `Scores.metrics` (every rate also has its denominator in `Scores.n`, and the report prints a 95% Wilson interval for each):

| key | definition |
| --- | --- |
| `precision` | (supported + supported_offsheet) / (claims − judge_error − span_invalid) |
| `hallucination_rate` | unsupported / same denominator |
| `precision_dealbreakers` | precision over claims in dealbreaker-scored fields |
| `recall` | captured facts / facts |
| `recall_dealbreakers` | captured dealbreaker facts / dealbreaker facts |
| `recall_by_disclosure.<type>` | captured / facts, per disclosure type |
| `recall_by_field.<field>` and `precision_by_field.<field>` | per schema field |
| `offsheet_rate` | supported_offsheet / judged claims (sim leakage; target < 3%) |
| `specificity_match` | captured `specific` facts whose claim was judged `specific` / captured `specific` facts |
| `yield.<question_id>` | captured facts whose `disclosed_turn_id` belongs to that question (main or follow-up) |
| `yield_per_minute.<question_id>` | `yield.<q>` / (words in that question's turns / 150) |
| `followup_share` | captured facts with `via_followup` / captured facts |
| `judge_error_rate` | (judge_error + span_invalid) / judged claims |
| `stability` | agreement of captured fact-id sets between stability runs |

Aggregation: the aggregate file stores the per-persona vector so the report can show distributions.

(Metric corrections and additions are in PLAN.md M4, M6, M9–M15.)

### 3.12 Stability and noise-floor procedures

Stability: for each persona, a second run; same extraction prompt; compare `FactCoverage` sets.

Noise floor (`score/floor.py`): run extraction for the baseline tag `floor_runs` times with distinct nonces (cache bypassed), score each, compute per-metric standard deviation across runs; `floor[metric] = max(2 × σ, 0.005)` (0.5 percentage points). `lab diff` fails (exit 1) if `candidate < baseline_mean − floor` for any gated metric. All other metric deltas are reported but not gated.

### 3.13 Judge audit flow

`lab audit --sample 20` draws a seeded sample of judged alignments, stratified by verdict, and writes `audit/sample.md` (claim, matched fact, cited span with the surrounding turn, verdict) plus a labels file with one `agree: null` entry per item. The author fills in `agree: true|false`. `lab report` reads the labels and prints `audit_agreement = agreed / labeled` with n. If no labels exist, the report prints "judge not yet audited" in red — never a placeholder number.

### 3.14 Report

`report/index.html` (Jinja2, inline CSS, inline SVG charts from matplotlib's SVG backend, no JavaScript) with these sections in order:

1. Headline numbers: precision, hallucination rate, recall, recall on dealbreakers, stability, judge health, audit agreement — each with n and interval. `DEGRADED` banner if the judge error rate exceeds 5%.
2. Per-field precision and recall table.
3. Per-question yield: bar chart of `yield_per_minute` by question, with the follow-up share marked.
4. Noise curve: precision and recall vs. noise rate (line chart).
5. Recall by disclosure type (small table).
6. Hallucination examples: up to 5 `unsupported` claims, each with the field, the claimed value, and the closest user turn for context.
7. Run manifest: models, prompt/schema/script hashes, seeds, code version, timestamps.

`summary.md` carries the same numbers in Markdown (used for the README and as the CI PR comment).

### 3.15 Diff

`lab diff --baseline v1 --candidate v2` prints a table: metric, baseline mean, candidate, delta, floor, verdict. Exit code 0 = pass, 1 = regression beyond floor, 2 = invalid input. Also writes `report/diff-<baseline>-<candidate>.md`.

### 3.16 Tests (pytest, all under `FakeProvider`, no network)

- `test_models.py`: round-trip every model; reject an `Alignment` with `span_text` but no `span_turn_id`.
- `test_decompose.py`: profile → claims, including arrays and missing scalars.
- `test_exact.py`: exact-field verdicts.
- `test_verify_span.py`: normalization rules; a paraphrased span is `span_invalid`.
- `test_metrics.py`: hand-built alignments and coverage with known answers for every metric key; judge errors excluded from denominators; degraded flag at > 5%.
- `test_noise.py`: rate accounting within ±15%; determinism under seed; interviewer turns untouched; `disclosed_fact_ids` preserved.
- `test_cache.py`: identical inputs hit; a changed parameter misses; nonce bypasses.
- `test_floor.py`: floor math; diff exit codes.
- `test_e2e.py`: two fixture personas through `lab run` with `FakeProvider`; asserts every artifact exists, validates against the models, `aggregate.json` has every metric key, `index.html` and `summary.md` render, and `lab diff v1 v2` exits 1 when the fake candidate is worse.

### 3.17 Makefile and CI

```
make setup     # uv sync
make test      # ruff check + pytest
make demo      # report from committed fixture artifacts, then opens report/index.html
make numbers   # full lab run per lab.yaml, then scripts/numbers.py rewrites README numbers
make audit     # lab audit --sample 20
make clean     # rm -rf runs .cache
```

`scripts/numbers.py`: runs the pipeline per `lab.yaml`, reads `aggregate.json`, and rewrites README.md between the markers with a table of the headline numbers, the noise curve table, and the run id. Fails loudly if the markers are missing.

`.github/workflows/eval.yml`: on pull requests touching `questions.yaml`, `prompts/**`, or `schema.json`: sync deps, `lab extract` + `lab score` on the committed fixture transcripts, `lab diff` against `baselines/aggregate.json`, upload `summary.md` as an artifact, and fail the job on exit 1.

Fixtures: commit `fixtures/personas/*.json` and `fixtures/transcripts/*.json` (synthetic, no privacy concern). Never commit `.cache/` or `runs/`.

---

## 4. Build phases and gates

| phase | builds | gate (must pass before the next phase) |
| --- | --- | --- |
| 0 skeleton | repo, uv, package layout, `models.py`, CLI stubs, `FakeProvider`, `llm.call` with cache and status, `test_e2e.py` scaffold | `make test` green on a clean checkout with no env vars |
| 1 spine | persona skeleton sampling + generation, interviewer + user agent, extraction + decomposition, exact scoring + aligner + span verification, `summary.md` | `lab run` with a small persona count against the real provider writes `aggregate.json` and `summary.md`; judge error rate reported |
| 2 metrics | noise injector, yield attribution, recall by disclosure, specificity, stability, judge audit flow, Wilson intervals | metric unit tests green; `lab audit` produces a sample; noise curve numbers exist |
| 3 report, diff, CI | HTML report with charts, `lab diff` + noise floor, `extract_v2_worse.txt`, fixtures, `baselines/`, GitHub Action, `scripts/numbers.py` | `make numbers` regenerates README on a clean checkout; `lab diff v1 v2` exits 1 on the worse prompt |
| 4 demo | README, `make demo` path, hallucination examples verified by hand, audit labels filled | five-minute demo runs end to end from a fresh terminal |

If time runs short, cut in the order given in PLAN.md. Never cut: exact + judged scoring, span verification, error states, `make numbers`, the diff, the noise floor, failure accounting.

---

## 5. Demo path

1. `make demo` from a fresh terminal; open `report/index.html`.
2. Show one hallucination example first (claim, cited-span rule, why it counts).
3. Headline numbers with n and intervals; judge health and audit agreement.
4. Per-question yield chart — the speed-vs-richness answer.
5. Noise curve — the voice-native argument.
6. Stability number.
7. `lab diff --baseline v1 --candidate v2` live on the deliberately worse prompt; show the exit code and the floor.
8. Close with adoption: the three files a team swaps in (`questions.yaml`, `prompts/extract.txt`, `schema.json`), and the limits (synthetic users, stand-in schema, judge error rate).

Keep it under five minutes. Have `report/index.html` from the last good run committed under `docs/demo-report/` as a fallback if the live run fails.

---

## 6. Open questions

1. Replace the starter question text in `questions.yaml` with notes from a real onboarding walkthrough (keep ids and targets structure). Freezes in Phase 0, before the first real simulation.
2. Confirm model ids to pin in `lab.yaml`. (Resolved in PLAN.md amendment 1, with a smoke test as arbiter.)
3. Commit fixtures for every persona — yes.
4. OpenAI adapter in v1 — no; Protocol only.
5. License — MIT.

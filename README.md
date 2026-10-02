# onboarding-lab

A synthetic evaluation harness for conversational intake / voice onboarding
pipelines.

When a long voice conversation is turned into a structured profile, every
downstream stage inherits whatever extraction got right or wrong. There is
usually no number for how good that step is: hand-labelling real transcripts is
slow, a labelled set goes stale the moment a question changes, and a real user's
true facts are never fully knowable. So question and prompt changes ship on
intuition.

This harness inverts the order. It writes the person first as a structured truth
sheet, runs that person through a simulated interview, extracts a profile from
the transcript, and scores the profile against the truth. Labels are exact and
free by construction, and the fixture set becomes a regression suite that runs on
every question or prompt change.

Synthetic users are not real users, so this complements audits of real
transcripts rather than replacing them. It is, however, the only source of exact
ground truth, and it runs without touching production or user data.

## Status

**Code complete, no committed run yet.** All six stages, the orchestrator, and
every CLI command are implemented, and the full suite (`ruff` + `pytest`) passes
on a cold clone with no API key and no network.

What is *not* here is a run. No fixture set has been generated and committed, so
there are no numbers in the Results section below, `make demo` fails, and
`baselines/` is empty. Reproducing the measurements needs an API key; reading and
replicating the code does not.

## Results

<!-- numbers:start -->
No numbers yet. Every figure in this section is written by `scripts/numbers.py`
from a committed run; none is typed by hand. Run `make numbers` to generate them.
<!-- numbers:end -->

## What it does

Six stages, each writing a JSON artifact so any stage can be rerun alone.

1. **Persona factory** — truth sheets: demographics, values, life events,
   preferences, dealbreakers, plus disclosure rules (which facts are volunteered,
   which need a follow-up, which are hedged) and a conversational style.
2. **Interview simulator** — an interviewer agent runs a question script with a
   follow-up policy against a user agent playing the persona. Every disclosed
   fact is tagged to the turn that elicited it.
3. **Noise injector** — rewrites user turns to imitate ASR error: dropped words,
   homophone swaps, fillers, lost punctuation.
4. **Extraction runner** — applies a pluggable prompt and JSON Schema.
5. **Scorer** — aligns each claim to a truth fact with a cited transcript span.
   Enum and boolean fields are checked exactly, with no model in the loop.
6. **Report and diff** — static HTML plus a Markdown summary. Diff mode reruns
   only extraction on a candidate prompt and compares against a baseline.

## Quickstart

```bash
make setup                      # uv sync
make test                       # ruff + pytest; no API key, no network
cp .env.example .env            # then add LAB_ANTHROPIC_API_KEY
make smoke                      # one real request per role, validates parameters
make dev                        # 3 personas, one noise level
```

`make demo` is intended to render the report from committed fixtures with no API
key and no network, so a reviewer never has to spend money to see the output. It
does not work on a fresh clone yet: no fixture set has been generated, so the
command exits with `no aggregate at fixtures/scores/v1/aggregate.json`. Generate
one with `make numbers` (needs a key) before relying on it.

## Replicating this repo on another machine

Everything needed to rebuild and run the code is committed; `uv.lock` pins every
dependency. On a clean machine:

```bash
git clone https://github.com/vaibhavhariram/onboarding-lab.git
cd onboarding-lab
make setup                      # uv sync, from the committed lock file
env -u ANTHROPIC_API_KEY make test
```

That is the whole cold-start path, and it is the gate: the suite passes with no
API key and no network, backed by a deterministic `FakeProvider`. Read
`CLAUDE.md` first for the working agreement, then `PLAN.md` for the build order,
then `docs/SPEC.md` for the reference design — every `§` in `PLAN.md` resolves
there, and where the two disagree, `PLAN.md` wins.

Three things are deliberately absent and are not recoverable from the repo:

- **A key.** `cp .env.example .env` and set `LAB_ANTHROPIC_API_KEY`. The lab never
  reads `ANTHROPIC_API_KEY`. Only `make smoke`, `make dev`, `make numbers`, and
  `make audit` need it.
- **Fixtures and baselines.** `fixtures/` and `baselines/` are empty, so
  `make demo` and `lab diff` have nothing to read until a run is committed.
- **CI.** `.github/` is empty; the workflow is item 2 in the cut order.

## Adoption

Three files make the numbers yours:

| file | what it is |
| --- | --- |
| `questions.yaml` | your question script, with the fields each question targets |
| `prompts/extract.txt` | your extraction prompt — the system under test |
| `schema.json` | your profile schema |

```bash
uv run lab run --config lab.yaml
```

Then `lab diff --baseline v1 --candidate v2` on the next prompt change.

## Design notes

**Errors are states, not verdicts.** A provider error, a refusal, a truncation,
an invalid schema, and a cited span that does not literally exist each get their
own status. None becomes a scoring verdict, all are excluded from metric
denominators, and all are counted in a judge-health section. A refusal arrives as
HTTP 200, so it is detected explicitly rather than parsed as output.

**Join by id, never by position.** Every artifact carries `persona_id`,
`transcript_id`, and `extraction_id`. Scoring joins claims to facts to turns by
id.

**Reproducibility is measured, not assumed.** Sampling parameters no longer exist
on current models, so the harness does not claim deterministic model output. What
it guarantees is that identical inputs regenerate identical artifacts, via a
content-hash cache; and what it does about the rest is *measure* it. The baseline
runs several times with the cache deliberately bypassed, and the diff gate fails
only on regressions larger than the observed run-to-run variance. The measured
floor is printed beside every gated number.

**The judge is audited.** A seeded sample of alignments is reviewed by hand and
the agreement rate is printed. With no labels, the report says "judge not yet
audited" — never a placeholder number.

**Pipeline recall and extraction recall are reported separately.** A fact that no
follow-up ever probed was never disclosed, so extraction could not have captured
it. Reporting one number would mix "the interview did not elicit it" with
"extraction missed it", which is exactly the stage attribution the harness exists
to provide.

**No number without a script.** Every figure above is written by
`scripts/numbers.py` between the markers, and regenerates on a clean checkout.

## Limits

- Synthetic users are not real users. This complements transcript audits.
- The shipped schema and question script are stand-ins.
- The judge and the system under test are currently the same model, so the judge
  cannot catch failure modes it shares.
- Reported intervals are pooled Wilson intervals and ignore persona clustering.

## License

MIT.

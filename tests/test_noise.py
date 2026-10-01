"""The ASR noise injector: rate accounting, determinism, and what it may not touch.

Every assertion here exists because the noise axis is a reported curve: if noise
moves for any reason other than ``(rate, noise_seed)``, the curve stops meaning
what the report says it means.
"""

from __future__ import annotations

import json
import random

import pytest

from conftest import make_turn
from onboarding_lab.models import ProvenanceStamp, Transcript, Turn, word_count
from onboarding_lab.noise import homophones
from onboarding_lab.noise.inject import (
    EDIT_LOG_SUFFIX,
    FILLERS,
    OPERATION_WEIGHTS,
    edit_log_path,
    inject_noise,
    noise_text,
    operation_for,
    write_edit_log,
)

# -- helpers ------------------------------------------------------------------


class ScriptedRandom(random.Random):
    """A ``Random`` whose ``random()`` returns scripted values, then real ones.

    ``getrandbits`` is re-declared so that ``Random.__init_subclass__`` keeps the
    getrandbits-based ``_randbelow``: without it, overriding ``random()`` alone
    makes ``choice()`` consume the script, and the operation a test meant to pin
    would be chosen by the filler or homophone draw instead.
    """

    def __init__(self, *draws: float, seed: int = 0) -> None:
        super().__init__(seed)
        self._draws = list(draws)

    def random(self) -> float:
        if self._draws:
            return self._draws.pop(0)
        return super().random()

    def getrandbits(self, k: int) -> int:
        return super().getrandbits(k)


#: Any draw below the rate fires an operation; this one fires at every rate > 0.
FIRE = 0.0
#: Above every rate used in these tests, so the word is left alone.
NO_FIRE = 0.999


def draw_for(operation: str) -> float:
    """The midpoint of ``operation``'s slice of ``OPERATION_WEIGHTS``."""
    lower = 0.0
    for name, weight in OPERATION_WEIGHTS:
        if name == operation:
            return lower + weight / 2
        lower += weight
    raise AssertionError(f"no such operation: {operation}")


#: Single words only (the injector splits on whitespace), with a homophone, a
#: dictionary miss, and sentence punctuation all represented.
WORD_POOL: tuple[str, ...] = (
    "there",
    "their",
    "week",
    "weekend",
    "kayaking",
    "mornings",
    "quietly",
    "coordinating",
    "schedule",
    "rebuilt",
    "two",
    "plans",
)


def long_user_text(rng: random.Random, words: int) -> str:
    """``words`` words of filler prose with a sentence break every so often."""
    tokens = []
    for i in range(words):
        token = rng.choice(WORD_POOL)
        if i % 11 == 10:
            token += "."
        tokens.append(token)
    return " ".join(tokens)


def build_transcript(
    stamp: ProvenanceStamp,
    turns: list[Turn],
    *,
    transcript_id: str = "p001__s1__n000",
    persona_id: str = "p001",
    sim_seed: int = 1,
) -> Transcript:
    return Transcript(
        transcript_id=transcript_id,
        persona_id=persona_id,
        script_hash="abc123",
        sim_seed=sim_seed,
        turns=turns,
        simulated_minutes=round(sum(t.word_count for t in turns) / 150.0, 4),
        provenance=stamp,
    )


def sample_turns() -> list[Turn]:
    return [
        make_turn("t001", "interviewer", "What do you do for work, and where?"),
        make_turn(
            "t002",
            "user",
            "I run scheduling for a hospital group. There are two sites. I have been "
            "there about a week over four years.",
            disclosed=["f06", "f01"],
        ),
        make_turn("t003", "interviewer", "What do your weekends look like?"),
        make_turn(
            "t004",
            "user",
            "Mostly kayaking, and I rebuilt a motorcycle over the winter. Quiet mornings!",
            disclosed=["f07"],
        ),
    ]


@pytest.fixture
def clean(stamp: ProvenanceStamp) -> Transcript:
    return build_transcript(stamp, sample_turns())


@pytest.fixture
def long_clean(stamp: ProvenanceStamp) -> Transcript:
    """Ten thousand user words, in twenty turns, so per-turn seeding is exercised."""
    rng = random.Random(13)
    turns: list[Turn] = []
    for i in range(20):
        turns.append(make_turn(f"t{2 * i + 1:03d}", "interviewer", f"Question number {i}?"))
        turns.append(make_turn(f"t{2 * i + 2:03d}", "user", long_user_text(rng, 500)))
    return build_transcript(stamp, turns)


# -- rate accounting (PLAN.md D4) ---------------------------------------------


@pytest.mark.parametrize("rate", [0.05, 0.1, 0.2])
def test_edit_count_is_within_15_percent_of_rate_times_words(
    long_clean: Transcript, rate: float
) -> None:
    """``edit_count`` is fire sites, so it is exactly Binomial(words, rate).

    The seed is pinned as well as the band being wide, because a flaky accounting
    test would get muted rather than read.
    """
    log = inject_noise(long_clean, rate=rate, seed=11).edit_log
    assert log["user_words"] == 10_000
    expected = rate * log["user_words"]
    assert abs(log["edit_count"] - expected) <= 0.15 * expected


def test_edit_count_is_the_sum_of_the_per_turn_counts(long_clean: Transcript) -> None:
    log = inject_noise(long_clean, rate=0.1, seed=11).edit_log
    assert log["edit_count"] == sum(t["edit_count"] for t in log["turns"])
    assert log["user_words"] == sum(t["words"] for t in log["turns"])


def test_punctuation_no_ops_are_still_counted() -> None:
    """A strip on a sentence with no punctuation changes nothing and still counts.

    This is the whole reason ``edit_count`` is defined as fire sites (PLAN.md D4).
    """
    rng = ScriptedRandom(FIRE, draw_for("punctuation"), NO_FIRE)
    text, edits = noise_text("alpha beta", rate=0.5, rng=rng)
    assert text == "alpha beta"
    assert len(edits) == 1
    assert edits[0]["operation"] == "punctuation"
    assert edits[0]["changed"] is False


# -- determinism --------------------------------------------------------------


def test_same_seed_gives_byte_identical_output(clean: Transcript) -> None:
    first = inject_noise(clean, rate=0.2, seed=5).transcript
    second = inject_noise(clean, rate=0.2, seed=5).transcript
    assert first.model_dump_json() == second.model_dump_json()


def test_different_seeds_give_different_output(clean: Transcript) -> None:
    first = inject_noise(clean, rate=0.2, seed=5).transcript
    second = inject_noise(clean, rate=0.2, seed=6).transcript
    assert first.model_dump_json() != second.model_dump_json()


def test_adding_a_turn_does_not_reshuffle_the_other_turns(stamp: ProvenanceStamp) -> None:
    """The point of per-turn seeding: an extra follow-up must not move earlier noise.

    Under a single per-transcript RNG the draw stream shifts and every later turn
    is renoised, so the noised copy would change for a reason unrelated to noise.
    """
    shorter = build_transcript(stamp, sample_turns())
    longer_turns = [
        *sample_turns(),
        make_turn("t005", "interviewer", "Anything else?"),
        make_turn("t006", "user", "Not really, that is most of it.", disclosed=["f02"]),
    ]
    longer = build_transcript(stamp, longer_turns)

    a = inject_noise(shorter, rate=0.2, seed=5).transcript
    b = inject_noise(longer, rate=0.2, seed=5).transcript
    for turn in a.turns:
        assert b.turn(turn.turn_id).text == turn.text


def test_homophone_swap_is_deterministic_under_a_seed() -> None:
    first = homophones.swap("coordinating", random.Random("k"))
    second = homophones.swap("coordinating", random.Random("k"))
    assert first == second


# -- what noise may not touch -------------------------------------------------


def test_interviewer_turns_are_untouched(clean: Transcript) -> None:
    """The interviewer's text is generated by the harness; no recogniser saw it."""
    noised = inject_noise(clean, rate=1.0, seed=5).transcript
    for turn in clean.turns:
        if turn.speaker == "interviewer":
            assert noised.turn(turn.turn_id).text == turn.text


def test_disclosed_fact_ids_are_preserved_exactly(clean: Transcript) -> None:
    """Noise corrupts the rendering of what was said, not what was said."""
    noised = inject_noise(clean, rate=1.0, seed=5).transcript
    for turn in clean.turns:
        assert noised.turn(turn.turn_id).disclosed_fact_ids == turn.disclosed_fact_ids
    assert any(t.disclosed_fact_ids for t in noised.turns), "fixture must disclose something"


def test_user_text_actually_changes(clean: Transcript) -> None:
    noised = inject_noise(clean, rate=0.2, seed=5).transcript
    assert any(noised.turn(t.turn_id).text != t.text for t in clean.turns if t.speaker == "user")


# -- rate zero ----------------------------------------------------------------


def test_rate_zero_returns_the_clean_transcript_unchanged(clean: Transcript) -> None:
    """No new artifact: the model forbids ``source_transcript_id`` at rate zero."""
    result = inject_noise(clean, rate=0.0, seed=5)
    assert result.transcript is clean
    assert result.transcript.source_transcript_id is None
    assert result.transcript.noise_rate == 0.0
    assert result.edit_log["edit_count"] == 0
    assert result.edit_log["source_transcript_id"] is None
    assert result.edit_log["transcript_id"] == clean.transcript_id


def test_rate_outside_the_unit_interval_is_rejected(clean: Transcript) -> None:
    with pytest.raises(ValueError, match="noise rate"):
        inject_noise(clean, rate=1.5, seed=5)


def test_a_noised_transcript_cannot_be_noised_again(clean: Transcript) -> None:
    """``source_transcript_id`` must name the noise-0 source (PLAN.md M8)."""
    once = inject_noise(clean, rate=0.1, seed=5).transcript
    with pytest.raises(ValueError, match="clean transcripts only"):
        inject_noise(once, rate=0.1, seed=5)


# -- the four operations, individually ----------------------------------------


def test_operation_weights_cover_the_unit_interval() -> None:
    assert sum(weight for _, weight in OPERATION_WEIGHTS) == pytest.approx(1.0)
    for name, _ in OPERATION_WEIGHTS:
        assert operation_for(draw_for(name)) == name
    assert operation_for(0.0) == "drop"
    assert operation_for(0.999) == "punctuation"


def test_drop_removes_the_word() -> None:
    rng = ScriptedRandom(FIRE, draw_for("drop"), NO_FIRE)
    text, edits = noise_text("alpha beta", rate=0.5, rng=rng)
    assert text == "beta"
    assert edits == [{"word_index": 0, "word": "alpha", "operation": "drop"}]


def test_homophone_swaps_the_word_and_keeps_case_and_punctuation() -> None:
    rng = ScriptedRandom(FIRE, draw_for("homophone"), NO_FIRE, NO_FIRE)
    text, edits = noise_text("There, really?", rate=0.5, rng=rng)
    first = text.split()[0]
    assert first != "There,"
    assert first[0].isupper() and first.endswith(",")
    assert first.strip(",").lower() in homophones.HOMOPHONES["there"]
    assert edits[0]["source"] == "dictionary"
    assert text.endswith("really?"), "only the firing word is touched"


def test_filler_is_inserted_before_the_word() -> None:
    rng = ScriptedRandom(FIRE, draw_for("filler"))
    text, edits = noise_text("beta", rate=0.5, rng=rng)
    assert text.endswith("beta")
    filler = text[: -len(" beta")]
    assert filler in FILLERS
    assert edits[0]["filler"] == filler
    assert word_count(text) > word_count("beta")


def test_punctuation_strip_covers_the_whole_surrounding_sentence() -> None:
    """Sentence-level: the strip reaches words that never fired themselves."""
    rng = ScriptedRandom(FIRE, draw_for("punctuation"), NO_FIRE, NO_FIRE, NO_FIRE)
    text, edits = noise_text("alpha beta, gamma. delta!", rate=0.5, rng=rng)
    assert text == "alpha beta gamma delta!"
    assert edits[0]["sentence_index"] == 0
    assert edits[0]["changed"] is True


def test_a_second_strip_of_the_same_sentence_is_idempotent() -> None:
    rng = ScriptedRandom(FIRE, draw_for("punctuation"), FIRE, draw_for("punctuation"))
    text, edits = noise_text("alpha beta.", rate=0.5, rng=rng)
    assert text == "alpha beta"
    assert [e["changed"] for e in edits] == [True, False]


# -- homophones ---------------------------------------------------------------


def test_dictionary_swap_uses_a_curated_candidate() -> None:
    swapped = homophones.swap("week", random.Random(1))
    assert swapped.source == "dictionary"
    assert swapped.token in homophones.HOMOPHONES["week"]


def test_character_level_fallback_fires_for_a_word_not_in_the_dictionary() -> None:
    """Most conversational vocabulary misses the dictionary; the swap still happens."""
    word = "kayaking"
    assert word not in homophones.HOMOPHONES
    swapped = homophones.swap(word, random.Random(3))
    assert swapped.source == "character"
    assert swapped.token != word


def test_character_fallback_clips_a_word_it_cannot_substitute() -> None:
    """A recogniser's number token has no plausible phonetic neighbour here."""
    swapped = homophones.character_variant("1972", random.Random(3))
    assert swapped.source == "character"
    assert swapped.token == "197"


def test_character_fallback_leaves_an_unsubstitutable_short_token_alone() -> None:
    swapped = homophones.character_variant("77", random.Random(3))
    assert swapped == ("77", "unchanged")


def test_split_token_isolates_the_word_core() -> None:
    assert homophones.split_token('"there."') == ('"', "there", '."')
    assert homophones.split_token("it's") == ("", "it's", "")


# -- the output artifact ------------------------------------------------------


def test_noised_transcript_validates_and_records_its_source(clean: Transcript) -> None:
    noised = inject_noise(clean, rate=0.2, seed=5).transcript
    reloaded = Transcript.model_validate(json.loads(noised.model_dump_json()))

    assert reloaded.transcript_id == "p001__s1__n020"
    assert reloaded.transcript_id != clean.transcript_id
    assert reloaded.source_transcript_id == clean.transcript_id
    assert reloaded.noise_rate == 0.2
    assert reloaded.noise_seed == 5
    assert reloaded.persona_id == clean.persona_id
    assert reloaded.sim_seed == clean.sim_seed
    assert [t.turn_id for t in reloaded.turns] == [t.turn_id for t in clean.turns]


def test_word_counts_and_minutes_are_recomputed(clean: Transcript) -> None:
    noised = inject_noise(clean, rate=0.2, seed=5).transcript
    for turn in noised.turns:
        assert turn.word_count == word_count(turn.text)
    expected = round(sum(t.word_count for t in noised.turns) / 150.0, 4)
    assert noised.simulated_minutes == expected
    assert noised.simulated_minutes != clean.simulated_minutes


def test_a_shuffled_source_keeps_its_marker(stamp: ProvenanceStamp) -> None:
    """Otherwise the shuffled run's noised copy collides with the as-written one."""
    shuffled = build_transcript(stamp, sample_turns(), transcript_id="p001__s1__shuf__n000")
    noised = inject_noise(shuffled, rate=0.1, seed=5).transcript
    assert noised.transcript_id == "p001__s1__shuf__n010"


def test_provenance_carries_the_noise_parameters(clean: Transcript) -> None:
    noised = inject_noise(clean, rate=0.1, seed=5).transcript
    assert noised.provenance.params["noise_rate"] == 0.1
    assert noised.provenance.params["noise_seed"] == 5
    assert noised.provenance.created_at == clean.provenance.created_at


# -- the edit-log sidecar -----------------------------------------------------


def test_edit_log_records_every_user_turn_and_only_user_turns(clean: Transcript) -> None:
    log = inject_noise(clean, rate=0.2, seed=5).edit_log
    user_ids = [t.turn_id for t in clean.turns if t.speaker == "user"]
    assert [t["turn_id"] for t in log["turns"]] == user_ids
    for entry in log["turns"]:
        assert entry["words"] == clean.turn(entry["turn_id"]).word_count
        for edit in entry["edits"]:
            assert edit["operation"] in {name for name, _ in OPERATION_WEIGHTS}
            assert 0 <= edit["word_index"] < entry["words"]


def test_edit_log_is_written_as_a_json_sidecar(clean: Transcript, tmp_path) -> None:
    """``Transcript`` forbids extra fields, so the log is an artifact of its own."""
    result = inject_noise(clean, rate=0.2, seed=5)
    path = write_edit_log(result.edit_log, tmp_path)
    assert path == edit_log_path(tmp_path, result.transcript.transcript_id)
    assert path.name == result.transcript.transcript_id + EDIT_LOG_SUFFIX
    assert json.loads(path.read_text()) == result.edit_log

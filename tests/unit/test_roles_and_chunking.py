import itertools

import pytest

from audiosearch.domain import Transcript, Utterance
from audiosearch.pipeline.alignment import build_turns
from audiosearch.pipeline.chunking import build_chunks, find_context_question
from audiosearch.pipeline.roles import infer_roles, speaker_profiles


def utt(i: int, spk: str, text: str, t: float) -> Utterance:
    return Utterance(f"f_u{i:04d}", i, spk, t, t + 2.0, text, 0, 0)


def interview() -> list[Utterance]:
    lines = [
        ("SPEAKER_00", "Thanks for joining us today."),
        ("SPEAKER_00", "How did you get started in aviation?"),
        ("SPEAKER_01", "My father was a pilot in the Air Force and I loved airplanes from the start."),
        ("SPEAKER_01", "I watched an F-15 go straight up at an air show and knew I wanted to fly."),
        ("SPEAKER_01", "So I studied engineering and then joined the Air Force as a pilot myself."),
        ("SPEAKER_00", "Which aircraft is your favorite?"),
        ("SPEAKER_01", "The T-38, it flies like a little sports car and it is a joy every time."),
    ]
    return [utt(i, s, t, i * 3.0) for i, (s, t) in enumerate(lines)]


def test_roles_host_asks_questions_and_opens():
    utts = interview()
    profiles = infer_roles(speaker_profiles(utts, build_turns(utts)), utts, "Courtney", "DJ")
    roles = {p.label: (p.role, p.display_name) for p in profiles}
    assert roles == {"SPEAKER_00": ("host", "Courtney"), "SPEAKER_01": ("guest", "DJ")}
    assert profiles[0].question_rate == pytest.approx(2 / 3, abs=1e-3)


def test_context_question_is_the_other_speakers_last_question():
    utts = interview()
    assert find_context_question(utts, 4).text == "How did you get started in aviation?"
    assert find_context_question(utts, 6).text == "Which aircraft is your favorite?"
    assert find_context_question(utts, 0) is None
    assert find_context_question(utts, 1) is None  # window starts with the host: no preceding question


def make_transcript() -> Transcript:
    utts = interview()
    return Transcript("f", 25.0, [], utts, build_turns(utts), [])


def test_chunks_cover_all_utterances_without_splitting_sentences():
    t = make_transcript()
    chunks = build_chunks(t, target_words=20, stride_words=10, context="none")
    covered = set()
    for c in chunks:
        covered.update(range(c.utterance_start, c.utterance_end))
        assert c.text == " ".join(u.text for u in t.utterances[c.utterance_start : c.utterance_end])
        assert c.start == t.utterances[c.utterance_start].start
    assert covered == set(range(len(t.utterances)))
    assert all(a.utterance_start < b.utterance_start for a, b in itertools.pairwise(chunks))


def test_dialogue_context_goes_into_embedding_text_only():
    t = make_transcript()
    chunks = build_chunks(t, target_words=15, stride_words=15, context="question")
    mid = next(c for c in chunks if c.utterance_start == 3)
    assert mid.embed_text.startswith("How did you get started in aviation?\n")
    assert "How did you get started" not in mid.text  # lexical index keeps only what was said here
    titled = build_chunks(t, target_words=15, stride_words=15, context="question+title", title="Runway")
    assert titled[0].embed_text.startswith("Runway.\n")


def test_chunking_rejects_bad_parameters():
    with pytest.raises(ValueError):
        build_chunks(make_transcript(), target_words=0)
    with pytest.raises(ValueError):
        build_chunks(make_transcript(), context="bogus")

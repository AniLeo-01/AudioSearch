from audiosearch.domain import Word
from audiosearch.pipeline.alignment import build_turns, build_utterances, ends_sentence
from audiosearch.pipeline.asr import AsrSegment, clean_words, merge_continuations


def w(text: str, start: float, spk: str = "SPEAKER_00", dur: float = 0.3) -> Word:
    return Word(text, start, start + dur, 0.9, spk)


def test_ends_sentence_handles_abbreviations():
    assert ends_sentence("today.")
    assert ends_sentence("really?")
    assert ends_sentence('"yes!"')
    assert not ends_sentence("Dr.")
    assert not ends_sentence("U.S.")
    assert not ends_sentence("hello,")


def test_utterances_split_on_speaker_change_punctuation_and_pause():
    words = [
        w("Hello", 0.0),
        w("there.", 0.4),  # sentence end
        w("How", 1.0),
        w("are", 1.3),
        w("you?", 1.6),  # question
        w("Fine", 2.2, "SPEAKER_01"),
        w("thanks", 2.5, "SPEAKER_01"),  # speaker change, no punctuation
        w("and", 5.0, "SPEAKER_01"),
        w("you", 5.3, "SPEAKER_01"),  # long pause
    ]
    utts = build_utterances(words, "f")
    assert [u.text for u in utts] == ["Hello there.", "How are you?", "Fine thanks", "and you"]
    assert [u.speaker for u in utts] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_01"]
    assert utts[1].is_question
    assert utts[0].id == "f_u0000" and utts[3].idx == 3
    assert (utts[2].word_start, utts[2].word_end) == (5, 7)


def test_long_utterances_split_at_comma():
    words = [w(f"word{i}" + ("," if i == 35 else ""), i * 0.3) for i in range(60)]
    utts = build_utterances(words, "f", max_words=50)
    assert len(utts) == 2
    assert utts[0].text.endswith("word35,")
    assert sum(u.word_end - u.word_start for u in utts) == 60


def test_turns_group_consecutive_speaker_utterances():
    words = [w("A.", 0), w("B.", 1), w("C.", 2, "SPEAKER_01"), w("D.", 3)]
    turns = build_turns(build_utterances(words, "f"))
    assert [(t.speaker, t.utterance_start, t.utterance_end) for t in turns] == [
        ("SPEAKER_00", 0, 2),
        ("SPEAKER_01", 2, 3),
        ("SPEAKER_00", 3, 4),
    ]


def test_clean_words_enforces_monotonic_times_and_strips():
    out = clean_words([Word(" a", 1.0, 1.2), Word("  ", 1.2, 1.3), Word("b", 0.9, 0.9)])
    assert [x.text for x in out] == ["a", "b"]
    assert out[1].start >= out[0].start and out[1].end > out[1].start


def test_merge_continuations_rejoins_hyphenated_tokens():
    words = [Word("an", 0, 0.2), Word("F", 0.2, 0.4), Word("-15", 0.4, 0.8), Word("jet.", 0.8, 1.0)]
    seg = AsrSegment(0, 1, "an F-15 jet.", -0.1, 0.01, 0, 4)
    merged, ends = merge_continuations(words, [seg])
    assert [x.text for x in merged] == ["an", "F-15", "jet."]
    assert merged[1].start == 0.2 and merged[1].end == 0.8
    assert ends == {2}

"""Fast unit tests: no models, no database."""

import numpy as np

from audiosearch.asr import merge_continuations
from audiosearch.diarize import viterbi
from audiosearch.evaluate import query_metrics
from audiosearch.models import Word
from audiosearch.search import Hit, idf_coverage, rrf
from audiosearch.transcript import build_utterances, ends_sentence


def w(text, start, spk="SPEAKER_00"):
    return Word(text, start, start + 0.3, 1.0, spk)


def test_merge_continuations_rejoins_subword_pieces():
    words = merge_continuations(
        " the F-15 jet", [(" the", 0, 0.2, 1), (" F", 0.2, 0.3, 1), ("-15", 0.3, 0.5, 1), (" jet", 0.5, 0.8, 1)]
    )
    assert [x.text for x in words] == ["the", "F-15", "jet"]


def test_viterbi_switches_only_with_evidence_and_prefers_boundaries():
    ll = np.log(np.array([[0.9, 0.1]] * 3 + [[0.3, 0.7]] + [[0.9, 0.1]] * 3))
    assert viterbi(ll, np.zeros(7, bool)).tolist() == [0] * 7  # one weak word does not flip the speaker
    ll = np.log(np.array([[0.9, 0.1]] * 3 + [[0.1, 0.9]] * 3))
    bound = np.array([False, False, False, True, False, False])
    assert viterbi(ll, bound).tolist() == [0, 0, 0, 1, 1, 1]


def test_utterances_split_on_sentence_end_and_speaker_change():
    words = [w("Hello", 0), w("Dr.", 0.5), w("Smith.", 1), w("How", 2), w("are", 2.3), w("you?", 2.6),
             w("Fine.", 3.5, "SPEAKER_01")]  # fmt: skip
    utts = build_utterances(words)
    assert [u.text for u in utts] == ["Hello Dr. Smith.", "How are you?", "Fine."]
    assert utts[1].is_question and utts[2].speaker == "SPEAKER_01"
    assert not ends_sentence("Dr.") and not ends_sentence("U.S.")


def test_coverage_rewards_informative_terms():
    lexemes = [("clock", 1.0, "clock"), ("tick", 1.0, "tick"), ("mar", 1.0, "mar")]
    hits = [("a", 9.0, ["mar"]), ("b", 5.0, ["clock", "tick"])]
    cov = idf_coverage({"clock": 3.0, "tick": 3.0, "mar": 1.0}, 100, lexemes, hits)
    assert cov["b"] > cov["a"]  # two rare terms beat one common term, whatever the BM25 score
    order, _, _ = rrf({"lexical": ["a", "b"], "dense": ["b", "a"]}, 10, {"lexical": cov})
    assert order[0] == "b"


def test_metrics_credit_each_labelled_moment_once():
    gold = [{"file": "f", "start": 10, "end": 20, "grade": 2}, {"file": "f", "start": 100, "end": 110, "grade": 2}]

    def hit(start):
        return Hit(0, "f", "", start, start + 5, start, "S", "guest", None, "", [], 0.0, {})

    m = query_metrics([hit(12), hit(14), hit(102)], gold, tol=5.0)
    assert m["R@1"] == 0.5 and m["R@3"] == 1.0 and m["MRR"] == 1.0
    assert query_metrics([hit(50)], gold, 5.0)["R@10"] == 0.0

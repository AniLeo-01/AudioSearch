import numpy as np
import pytest

from audiosearch.domain import Word
from audiosearch.pipeline.diarization import (
    frame_posteriors,
    is_boundary,
    make_windows,
    relabel_by_first_appearance,
    spectral_two_way,
    speech_regions,
    viterbi_smooth,
    word_log_likelihoods,
)


def test_speech_regions_merge_small_gaps():
    words = [Word("a", 0.0, 0.5), Word("b", 0.7, 1.0), Word("c", 3.0, 3.5)]
    assert speech_regions(words, max_gap=0.5) == [(0.0, 1.0), (3.0, 3.5)]


def test_make_windows_cover_region_and_skip_tiny_regions():
    wins = make_windows([(0.0, 4.0), (5.0, 5.2), (6.0, 7.0)], win=1.5, hop=0.75)
    assert wins[0] == (0.0, 1.5)
    assert wins[-2][1] == 4.0  # tail flushed so the whole region is covered
    assert (6.0, 7.0) in wins  # short region -> single window
    assert all(not (s >= 5.0 and e <= 5.2) for s, e in wins)  # 0.2 s region dropped


@pytest.mark.filterwarnings("ignore:Graph is not fully connected")
def test_spectral_two_way_separates_two_clear_clusters():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.05, (40, 16)) + np.eye(16)[0]
    b = rng.normal(0, 0.05, (30, 16)) + np.eye(16)[1]
    labels = spectral_two_way(np.vstack([a, b]))
    assert len(set(labels[:40])) == 1 and len(set(labels[40:])) == 1
    assert labels[0] != labels[-1]


def test_viterbi_removes_mid_sentence_flicker_but_allows_boundary_switch():
    # 10 words of speaker 0, one noisy word voting speaker 1 mid-sentence, then a real turn change.
    ll = np.log(np.array([[0.9, 0.1]] * 5 + [[0.3, 0.7]] + [[0.9, 0.1]] * 4 + [[0.1, 0.9]] * 5))
    bounds = np.zeros(15, dtype=bool)
    bounds[10] = True  # sentence boundary before word 10
    path = viterbi_smooth(ll, bounds, switch_penalty=4.0, boundary_discount=0.15)
    assert list(path) == [0] * 10 + [1] * 5
    raw = np.argmax(ll, axis=1)
    assert raw[5] == 1  # the flicker existed before smoothing


def test_is_boundary_uses_punctuation_segments_and_pauses():
    words = [Word("one.", 0, 0.3), Word("two", 0.4, 0.6), Word("three", 0.65, 0.9), Word("four", 2.0, 2.3)]
    assert is_boundary(words, 0, set())
    assert is_boundary(words, 1, set())  # after "one."
    assert not is_boundary(words, 2, set())
    assert is_boundary(words, 2, {1})  # ASR segment ended at word 1
    assert is_boundary(words, 3, set())  # 1.1 s pause


def test_frame_posteriors_and_word_likelihoods():
    post, covered = frame_posteriors([(0.0, 1.0), (1.0, 2.0)], np.array([[1.0, 0.0], [0.0, 1.0]]), duration=3.0)
    assert covered[:20].all() and not covered[25:].any()
    ll = word_log_likelihoods([Word("x", 0.1, 0.5), Word("y", 1.2, 1.6), Word("z", 2.5, 2.8)], post, covered)
    assert np.argmax(ll[0]) == 0 and np.argmax(ll[1]) == 1
    assert np.argmax(ll[2]) == 1  # uncovered -> nearest covered frame


def test_relabel_by_first_appearance():
    path, mapping = relabel_by_first_appearance(np.array([1, 1, 0, 1]))
    assert list(path) == [0, 0, 1, 0] and mapping == {1: 0, 0: 1}

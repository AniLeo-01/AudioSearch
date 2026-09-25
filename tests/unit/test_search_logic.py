import pytest

from audiosearch.search.filters import SearchFilters
from audiosearch.search.fusion import convex_combination, weighted_rrf
from audiosearch.search.moments import (
    ChunkRow,
    UtteranceRow,
    first_match_time,
    overlap_ratio,
    parse_headline,
    snap,
    temporal_nms,
    tsquery_text,
)
from audiosearch.search.phonetic import score_candidate, substring_ratio
from audiosearch.search.query import QueryError, classify_intent, parse_query
from audiosearch.textutil import content_tokens, tokens, vocabulary_terms


# ---------------------------------------------------------------- query parsing
def test_parse_query_extracts_phrases_and_filters():
    q = parse_query('"firing room" role:guest file:artemis_launch launch')
    assert q.phrases == ["firing room"]
    assert q.role == "guest" and q.file_ids == ["artemis_launch"]
    assert q.text == "firing room launch"
    assert q.intent == "phrase"


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("cesium", "keyword"),
        ("El Paso", "keyword"),
        ("Why do clocks on Mars run faster?", "question"),
        ("how the brain adapts in space", "question"),
        ("people who get lost even in their own neighborhood", "topic"),
    ],
)
def test_intent(text, intent):
    assert classify_intent(text, []) == intent


@pytest.mark.parametrize("bad", ["", "   ", "role:guest", "role:pilot cesium", "x" * 600])
def test_parse_query_rejects_bad_input(bad):
    with pytest.raises(QueryError):
        parse_query(bad)


def test_textutil_tokens():
    assert tokens("NASA's T-38s") == ["nasa", "t", "38s"]
    assert content_tokens("How do the atomic clocks agree?") == ["atomic", "clocks", "agree"]
    assert vocabulary_terms("the WB-57 and 2009 Guppy") == {"guppy"}


# ---------------------------------------------------------------- filters compile to parameterised SQL
def test_filters_are_parameterised():
    where, params = SearchFilters(file_ids=("a",), role="guest", phrases=("x'; drop table",)).chunk_clause()
    rendered = where.as_string(None)
    assert "drop table" not in rendered
    assert params["f_phrase_0"] == "x'; drop table" and params["f_role"] == "guest"


# ---------------------------------------------------------------- fusion
def test_rrf_rewards_agreement_and_respects_weights():
    ch = {"lexical": [("a", 9.0), ("b", 5.0)], "dense": [("b", 0.9), ("c", 0.8)]}
    fused = weighted_rrf(ch, {"lexical": 1.0, "dense": 1.0}, k=10)
    assert fused[0].id == "b"
    assert fused[0].ranks == {"lexical": 2, "dense": 1}
    only_dense = weighted_rrf(ch, {"lexical": 0.0, "dense": 1.0}, k=10)
    assert [f.id for f in only_dense] == ["b", "c"]


def test_rrf_multipliers_mute_low_coverage_lexical_hits():
    ch = {"lexical": [("noise", 3.0)], "dense": [("gold", 0.8), ("noise", 0.5)]}
    plain = weighted_rrf(ch, {"lexical": 1.0, "dense": 1.0}, k=10)
    assert plain[0].id == "noise"  # agreement across channels wins by default...
    muted = weighted_rrf(ch, {"lexical": 1.0, "dense": 1.0}, k=10, doc_multipliers={"lexical": {"noise": 0.05}})
    assert muted[0].id == "gold"  # ...unless the lexical match only covered uninformative words


def test_convex_combination_normalises_scores():
    ch = {"lexical": [("a", 10.0), ("b", 0.0)], "dense": [("b", 0.9), ("a", 0.1)]}
    fused = convex_combination(ch, {"lexical": 1.0, "dense": 1.0})
    assert {f.id: round(f.score, 3) for f in fused} == {"a": 1.0, "b": 1.0}


# ---------------------------------------------------------------- phonetic scoring
def test_phonetic_scoring():
    assert score_candidate("sells", "cells", 0.33, 1, True, True)[0] >= 0.8
    score, kind = score_candidate("pizzamiglio", "pizzamilio", 0.62, 1, False, False)
    assert kind == "spelling" and score >= 0.62
    assert score_candidate("inside", "insights", 0.3, 4, False, False)[0] < 0.62
    assert substring_ratio("zubaire", "abizubair") >= 0.85
    assert score_candidate("zubaire", "abizubair", 0.4, 4, False, False)[0] >= 0.62


# ---------------------------------------------------------------- moments
def urow(idx, text, start, dsim=None, matched=(), headline=None, speaker="SPEAKER_01"):
    words = []
    t = start
    for tok in text.split():
        words.append([tok, t, t + 0.3])
        t += 0.4
    return UtteranceRow(f"f_u{idx}", "f", idx, speaker, start, t, text, words, dsim, list(matched), headline)


def test_tsquery_text_escapes_and_dedupes():
    assert tsquery_text(["a", "b'c", "a"]) == "'a' | 'b''c'"


def test_parse_headline_and_match_time():
    plain, spans = parse_headline("we saw the \x02Guppy\x03 fly")
    assert plain == "we saw the Guppy fly" and spans == [(11, 16)]
    u = urow(0, plain, 10.0)
    assert first_match_time(u, spans) == pytest.approx(11.2)


def test_snap_prefers_lexical_evidence_then_semantic_and_honours_speaker_filter():
    chunk = ChunkRow("c", "f", 0, 0.0, 30.0, 0, 3, ["SPEAKER_00", "SPEAKER_01"], "")
    utts = {
        ("f", 0): urow(0, "What is your favorite plane?", 0.0, dsim=0.6, speaker="SPEAKER_00"),
        ("f", 1): urow(
            1,
            "The T-38 is a joy to fly every day",
            5.0,
            dsim=0.5,
            matched=["t-38"],
            headline="The \x02T-38\x03 is a joy to fly every day",
        ),
        ("f", 2): urow(2, "It flies like a little sports car", 10.0, dsim=0.9),
    }
    lexical = snap(chunk, utts, {"t-38": 3.0}, lexical_weight=0.8, allowed_speakers=None)
    assert lexical is not None and lexical.utterance.idx == 1 and lexical.highlights == [(4, 8)]
    semantic = snap(chunk, utts, {}, lexical_weight=0.8, allowed_speakers=None)
    assert semantic is not None and semantic.utterance.idx == 2
    host_only = snap(chunk, utts, {}, lexical_weight=0.8, allowed_speakers={"SPEAKER_00"})
    assert host_only is not None and host_only.utterance.speaker == "SPEAKER_00"
    assert snap(chunk, utts, {}, 0.8, allowed_speakers=set()) is None


def test_temporal_nms_and_overlap():
    assert overlap_ratio(0, 10, 5, 20) == pytest.approx(0.5)
    c1 = ChunkRow("c1", "f", 0, 0.0, 30.0, 0, 3, [], "")
    c2 = ChunkRow("c2", "f", 1, 10.0, 40.0, 2, 5, [], "")
    c3 = ChunkRow("c3", "f", 2, 100.0, 130.0, 8, 10, [], "")
    from audiosearch.search.moments import Moment

    ms = [
        Moment(c1, urow(1, "a b", 5.0), 5.0),
        Moment(c2, urow(3, "c d", 20.0), 20.0),
        Moment(c3, urow(9, "e", 110.0), 110.0),
    ]
    kept = temporal_nms(ms, gap_sec=10.0)
    assert [m.chunk.id for m in kept] == ["c1", "c3"]  # c2 overlaps c1 by 50%+ -> suppressed

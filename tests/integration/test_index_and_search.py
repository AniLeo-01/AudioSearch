"""End-to-end SQL paths against a real PostgreSQL + pgvector (throw-away schema, hashing embedder)."""

from __future__ import annotations

import psycopg
import pytest

from audiosearch.dataset import load_manifest
from audiosearch.db import SchemaMismatchError, connect, create_pool, migrate
from audiosearch.domain import Transcript
from audiosearch.indexing import Indexer
from audiosearch.search.engine import SearchEngine, SearchOptions
from audiosearch.search.query import QueryError

pytestmark = pytest.mark.db
FILES = ("runway", "telling_time")


@pytest.fixture(scope="module")
def indexed(db_settings, hash_embedder):
    migrate(db_settings, hash_embedder.model_name, hash_embedder.dim)
    entries = {e.file_id: e for e in load_manifest(db_settings.manifest_path, db_settings.audio_dir)}
    indexer = Indexer(db_settings, hash_embedder)  # type: ignore[arg-type]
    with connect(db_settings) as conn:
        for fid in FILES:
            t = Transcript.load(db_settings.transcripts_dir / f"{fid}.json")
            indexer.index(conn, t, entries[fid])
    return db_settings, indexer, entries


@pytest.fixture(scope="module")
def engine(indexed, hash_embedder):
    settings = indexed[0]
    pool = create_pool(settings)
    yield SearchEngine(settings, pool, hash_embedder)  # type: ignore[arg-type]
    pool.close()


def assert_stats_consistent(conn: psycopg.Connection) -> None:
    """Incrementally maintained BM25 statistics must equal a from-scratch recomputation."""
    n_docs, sum_dl = conn.execute("SELECT n_docs, sum_doc_len FROM corpus_stats WHERE id = 1").fetchone()
    assert n_docs == conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    assert sum_dl == conn.execute("SELECT coalesce(sum(doc_len), 0) FROM chunks").fetchone()[0]
    drift = conn.execute(
        """SELECT count(*) FROM (
               SELECT lexeme, count(*) AS df FROM chunk_terms GROUP BY lexeme
           ) x FULL JOIN term_stats ts USING (lexeme)
           WHERE x.df IS DISTINCT FROM ts.df"""
    ).fetchone()[0]
    assert drift == 0
    assert conn.execute("SELECT count(*) FROM vocabulary WHERE df <= 0").fetchone()[0] == 0


def test_index_is_idempotent_and_stats_stay_consistent(indexed, hash_embedder):
    settings, indexer, entries = indexed
    with connect(settings) as conn:
        assert_stats_consistent(conn)
        before = conn.execute("SELECT count(*), sum(doc_len) FROM chunks").fetchone()
        t = Transcript.load(settings.transcripts_dir / "runway.json")
        assert indexer.index(conn, t, entries["runway"]).skipped  # unchanged signature -> no-op
        indexer.index(conn, t, entries["runway"], force=True)  # replace in place
        assert conn.execute("SELECT count(*), sum(doc_len) FROM chunks").fetchone() == before
        assert_stats_consistent(conn)
        assert indexer.delete(conn, "runway")
        assert_stats_consistent(conn)
        assert conn.execute("SELECT count(*) FROM chunks WHERE file_id = 'runway'").fetchone()[0] == 0
        indexer.index(conn, t, entries["runway"])
        assert_stats_consistent(conn)


def test_migration_refuses_a_different_embedding_dimension(indexed):
    with pytest.raises(SchemaMismatchError):
        migrate(indexed[0], "some/other-model", 768)


def test_keyword_search_returns_moment_speaker_and_highlight(engine):
    resp = engine.search("Guppy", k=5, options=SearchOptions(mode="lexical"))
    assert resp.hits, "expected results"
    top = resp.hits[0]
    assert top.file_id == "runway"
    assert "guppy" in top.text.lower()
    a, b = top.highlights[0]
    assert top.text[a:b].lower().startswith("guppy")
    assert top.start <= top.match_time <= top.end
    assert top.speaker_role in ("host", "guest") and top.speaker.startswith("SPEAKER_")
    assert resp.timings_ms["total"] > 0


def test_phrase_query_is_strict(engine):
    resp = engine.search('"pilot\'s license"', k=5)
    assert resp.intent == "phrase" and resp.hits
    assert all("pilot" in h.passage_text.lower() and "license" in h.passage_text.lower() for h in resp.hits)


def test_role_filter_only_returns_that_role(engine):
    for role in ("host", "guest"):
        resp = engine.search("Guppy", k=5, role=role)
        assert resp.hits and all(h.speaker_role == role for h in resp.hits)


def test_file_filter_and_inline_filter_syntax(engine):
    resp = engine.search("time file:telling_time", k=5)
    assert resp.hits and all(h.file_id == "telling_time" for h in resp.hits)


def test_sounds_like_expansion_recovers_misspelling(engine):
    # "Guppie" would already match: Snowball stems guppie and guppy to the same lexeme.
    off = engine.search("Guppee", k=3, options=SearchOptions(mode="lexical", phonetic=False))
    on = engine.search("Guppee", k=3, options=SearchOptions(mode="lexical", phonetic=True))
    assert not off.hits
    assert any(e.term == "guppy" for e in on.expansions)
    assert on.hits and on.hits[0].file_id == "runway"


def test_common_words_are_not_expanded(engine):
    resp = engine.search("inside the hangar", k=3, options=SearchOptions(mode="lexical"))
    assert not any(e.source == "inside" for e in resp.expansions)


def test_nms_returns_distinct_moments(engine):
    resp = engine.search("T-38", k=10)
    starts = [(h.file_id, h.start) for h in resp.hits]
    for i, (f1, s1) in enumerate(starts):
        for f2, s2 in starts[i + 1 :]:
            assert f1 != f2 or abs(s1 - s2) >= engine.settings.nms_gap_sec


def test_invalid_requests_raise_query_error(engine):
    with pytest.raises(QueryError):
        engine.search("   ")
    with pytest.raises(QueryError):
        engine.search("cesium", role="pilot")

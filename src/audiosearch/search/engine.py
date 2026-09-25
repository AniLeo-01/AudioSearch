"""Search engine: query -> {BM25 (+ sounds-like expansion), dense ANN} -> fusion -> (rerank) -> moments."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import psycopg
from psycopg_pool import ConnectionPool

from audiosearch.config import Settings
from audiosearch.db import pgvector_version
from audiosearch.embeddings import CrossEncoderReranker, Embedder
from audiosearch.search.dense import dense_search
from audiosearch.search.filters import SearchFilters
from audiosearch.search.fusion import Fused, convex_combination, weighted_rrf
from audiosearch.search.lexical import WeightedLexeme, analyze, analyze_terms, bm25_search
from audiosearch.search.moments import Moment, fetch_chunks, fetch_utterances, snap, temporal_nms
from audiosearch.search.phonetic import EXPANSION_WEIGHT, Expansion, expand_terms
from audiosearch.search.query import QueryError, fusion_weights, parse_query

log = logging.getLogger(__name__)

Mode = Literal["hybrid", "lexical", "semantic"]
MODES: tuple[str, ...] = ("hybrid", "lexical", "semantic")


@dataclass
class SearchHit:
    rank: int
    file_id: str
    file_title: str
    start: float  # start of the moment (utterance) - where playback should begin
    end: float
    match_time: float  # time of the first matching word (== start for purely semantic matches)
    speaker: str
    speaker_role: str
    speaker_name: str | None
    text: str
    highlights: list[tuple[int, int]]
    passage_text: str
    passage_start: float
    passage_end: float
    score: float
    channels: dict[str, int]  # channel -> 1-based rank in that channel
    chunk_id: str
    utterance_id: str


@dataclass
class SearchResponse:
    query: str
    mode: str
    intent: str
    weights: dict[str, float]
    expansions: list[Expansion]
    hits: list[SearchHit]
    timings_ms: dict[str, float] = field(default_factory=dict)
    total_candidates: int = 0


@dataclass(frozen=True)
class SearchOptions:
    """Per-request overrides of the configured defaults (used by the API and the ablation study)."""

    mode: Mode = "hybrid"
    adaptive: bool | None = None
    phonetic: bool | None = None
    rerank: bool | None = None
    fusion: Literal["rrf", "cc"] = "rrf"
    nms: bool = True
    snap: bool = True


class _Timer:
    def __init__(self) -> None:
        self.t: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[name] = round(self.t.get(name, 0.0) + (time.perf_counter() - t0) * 1000, 2)


class SearchEngine:
    def __init__(
        self,
        settings: Settings,
        pool: ConnectionPool,
        embedder: Embedder,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.settings = settings
        self.pool = pool
        self.embedder = embedder
        self.reranker = reranker
        with pool.connection() as conn:
            self.iterative_scan = pgvector_version(conn) >= (0, 8, 0)

    # ---------------------------------------------------------------------------------------------------
    def search(
        self,
        query: str,
        k: int | None = None,
        *,
        role: str | None = None,
        speaker: str | None = None,
        file_ids: list[str] | None = None,
        options: SearchOptions | None = None,
    ) -> SearchResponse:
        s = self.settings
        opt = options or SearchOptions()
        if opt.mode not in MODES:
            raise QueryError(f"unknown mode '{opt.mode}'")
        k = max(1, min(k or s.default_k, s.max_k))
        adaptive = s.adaptive_fusion if opt.adaptive is None else opt.adaptive
        phonetic = s.phonetic_expansion if opt.phonetic is None else opt.phonetic
        use_rerank = (s.rerank if opt.rerank is None else opt.rerank) and self.reranker is not None
        timer = _Timer()
        t_start = time.perf_counter()

        parsed = parse_query(query, s.max_query_chars)
        if role and role not in ("host", "guest"):
            raise QueryError(f"unknown role '{role}'")
        filters = SearchFilters(
            file_ids=tuple(file_ids or parsed.file_ids),
            role=role or parsed.role,
            speaker=(speaker or parsed.speaker),
            phrases=tuple(parsed.phrases),
        )
        n = s.candidates_per_channel
        channels: dict[str, list[tuple[str, float]]] = {}
        expansions: list[Expansion] = []
        lexemes: list[WeightedLexeme] = []
        qvec: np.ndarray | None = None

        with self.pool.connection() as conn:
            if opt.mode in ("hybrid", "lexical"):
                with timer("lexical"):
                    lexemes = [WeightedLexeme(lx, 1.0, "query") for lx in analyze(conn, parsed.text)]
                    if phonetic:
                        expansions = expand_terms(conn, parsed.content_tokens)
                        lexemes += self._expansion_lexemes(conn, expansions, {w.lexeme for w in lexemes})
                    channels["lexical"] = bm25_search(conn, lexemes, filters, n)
            if opt.mode in ("hybrid", "semantic"):
                with timer("embed"):
                    qvec = self.embedder.embed_query(parsed.text)
                with timer("dense"):
                    channels["dense"] = dense_search(
                        conn, qvec, filters, n, s.hnsw_ef_search, self.iterative_scan
                    )
            weights = (
                fusion_weights(parsed.intent, adaptive) if opt.mode == "hybrid" else {opt.mode: 1.0}
            )
            if opt.mode == "semantic":
                weights = {"dense": 1.0}
            with timer("fusion"):
                if opt.fusion == "cc":
                    fused = convex_combination(channels, weights)
                else:
                    fused = weighted_rrf(channels, weights, k=s.rrf_k)
            if use_rerank and fused:
                with timer("rerank"):
                    fused = self._rerank(conn, parsed.text, fused)

            pool_size = min(len(fused), max(3 * k, k + 20))
            candidates = fused[:pool_size]
            with timer("moments"):
                moments = self._moments(conn, parsed.intent, opt, candidates, qvec, lexemes, filters)
            if opt.nms:
                moments = temporal_nms(moments, s.nms_gap_sec)
            moments = moments[:k]
            with timer("hydrate"):
                hits = self._hydrate(conn, moments, {f.id: f for f in candidates})
            conn.rollback()  # read-only work; end the implicit transaction promptly

        timer.t["total"] = round((time.perf_counter() - t_start) * 1000, 2)
        return SearchResponse(
            query=query,
            mode=opt.mode,
            intent=parsed.intent,
            weights=weights,
            expansions=expansions,
            hits=hits,
            timings_ms=timer.t,
            total_candidates=len(fused),
        )

    # ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _expansion_lexemes(
        conn: psycopg.Connection, expansions: list[Expansion], existing: set[str]
    ) -> list[WeightedLexeme]:
        term_lex = analyze_terms(conn, sorted({e.term for e in expansions}))
        out: list[WeightedLexeme] = []
        seen = set(existing)
        for e in expansions:
            for lx in term_lex.get(e.term, []):
                if lx not in seen:
                    seen.add(lx)
                    out.append(WeightedLexeme(lx, EXPANSION_WEIGHT * e.score, e.term))
        return out

    def _rerank(self, conn: psycopg.Connection, text: str, fused: list[Fused]) -> list[Fused]:
        assert self.reranker is not None
        top = fused[: self.settings.rerank_top_n]
        rows = fetch_chunks(conn, [f.id for f in top])
        scores = self.reranker.score(text, [rows[f.id].text for f in top])
        order = np.argsort(-scores, kind="stable")
        reranked = weighted_rrf(
            {"fused": [(f.id, f.score) for f in top], "rerank": [(top[i].id, float(scores[i])) for i in order]},
            {"fused": 1.0, "rerank": 1.0},
            k=self.settings.rrf_k,
        )
        by_id = {f.id: f for f in top}
        for r in reranked:  # keep per-channel provenance for explanations
            r.ranks = {**by_id[r.id].ranks, "rerank": r.ranks["rerank"]}
        return reranked + fused[len(top) :]

    def _moments(
        self,
        conn: psycopg.Connection,
        intent: str,
        opt: SearchOptions,
        candidates: list[Fused],
        qvec: np.ndarray | None,
        lexemes: list[WeightedLexeme],
        filters: SearchFilters,
    ) -> list[Moment]:
        chunks = fetch_chunks(conn, [f.id for f in candidates])
        ordered = [chunks[f.id] for f in candidates if f.id in chunks]
        lex_list = sorted({w.lexeme for w in lexemes})
        utts = fetch_utterances(conn, ordered, qvec, lex_list)
        idf = self._idf(conn, lexemes)
        allowed = self._allowed_speakers(conn, filters, {c.file_id for c in ordered})
        lexical_weight = 0.8 if intent in ("keyword", "phrase") else 0.5
        if opt.mode == "semantic":
            lexical_weight = 0.3
        moments: list[Moment] = []
        for c in ordered:
            if not opt.snap:  # ablation: report the passage start instead of the best utterance
                first = utts.get((c.file_id, c.utt_start))
                if first is not None:
                    moments.append(Moment(c, first, c.start))
                continue
            m = snap(c, utts, idf, lexical_weight, allowed.get(c.file_id, set()) if allowed is not None else None)
            if m is not None:
                moments.append(m)
        return moments

    @staticmethod
    def _idf(conn: psycopg.Connection, lexemes: list[WeightedLexeme]) -> dict[str, float]:
        if not lexemes:
            return {}
        weight = {}
        for w in lexemes:
            weight[w.lexeme] = max(weight.get(w.lexeme, 0.0), w.weight)
        rows = conn.execute(
            """SELECT ts.lexeme, ln(1 + (cs.n_docs - ts.df + 0.5) / (ts.df + 0.5))
               FROM term_stats ts CROSS JOIN corpus_stats cs WHERE ts.lexeme = ANY(%s)""",
            (list(weight),),
        ).fetchall()
        return {lx: float(v) * weight[lx] for lx, v in rows}

    @staticmethod
    def _allowed_speakers(
        conn: psycopg.Connection, filters: SearchFilters, file_ids: set[str]
    ) -> dict[str, set[str]] | None:
        """Per-file speaker labels that satisfy the role/speaker filter (None = unrestricted)."""
        if not filters.restricts_speaker:
            return None
        allowed: dict[str, set[str]] = {fid: set() for fid in file_ids}
        if filters.role:
            for fid, label in conn.execute(
                "SELECT file_id, label FROM speakers WHERE role = %s AND file_id = ANY(%s)",
                (filters.role, list(file_ids)),
            ).fetchall():
                allowed[fid].add(label)
            if filters.speaker:
                allowed = {fid: labels & {filters.speaker} for fid, labels in allowed.items()}
        else:
            allowed = {fid: {filters.speaker} for fid in file_ids if filters.speaker}
        return allowed

    @staticmethod
    def _hydrate(conn: psycopg.Connection, moments: list[Moment], fused: dict[str, Fused]) -> list[SearchHit]:
        if not moments:
            return []
        fids = sorted({m.chunk.file_id for m in moments})
        titles = dict(conn.execute("SELECT file_id, title FROM audio_files WHERE file_id = ANY(%s)", (fids,)).fetchall())
        spk = {
            (r[0], r[1]): (r[2], r[3])
            for r in conn.execute(
                "SELECT file_id, label, role, display_name FROM speakers WHERE file_id = ANY(%s)", (fids,)
            ).fetchall()
        }
        hits = []
        for rank, m in enumerate(moments, start=1):
            u = m.utterance
            role, name = spk.get((u.file_id, u.speaker), ("unknown", None))
            f = fused.get(m.chunk.id)
            hits.append(
                SearchHit(
                    rank=rank,
                    file_id=u.file_id,
                    file_title=titles.get(u.file_id, u.file_id),
                    start=round(u.start, 2),
                    end=round(u.end, 2),
                    match_time=round(m.match_time, 2),
                    speaker=u.speaker,
                    speaker_role=role,
                    speaker_name=name,
                    text=u.text,
                    highlights=m.highlights,
                    passage_text=m.chunk.text,
                    passage_start=round(m.chunk.start, 2),
                    passage_end=round(m.chunk.end, 2),
                    score=round(f.score, 6) if f else 0.0,
                    channels=dict(f.ranks) if f else {},
                    chunk_id=m.chunk.id,
                    utterance_id=u.id,
                )
            )
        return hits

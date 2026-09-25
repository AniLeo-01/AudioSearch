"""Moment localisation: turn a passage hit into *the* moment - one utterance, one speaker, one time.

Passages (~30 s) are the right unit for recall, but a user wants to press play at the right second
and see who said it.  For every candidate passage we score each of its utterances by

    lexical evidence  - IDF-weighted query lexemes present in the utterance, and
    semantic evidence - cosine similarity between the query and the utterance embedding,

mix them according to which channel surfaced the passage, and return the best utterance together
with the exact time of the first matching word and character-level highlight spans.  Temporal
non-maximum suppression then removes near-duplicate moments produced by overlapping windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import psycopg

HL_START, HL_STOP = "\x02", "\x03"
MIN_SEMANTIC_WORDS = 4


@dataclass
class ChunkRow:
    id: str
    file_id: str
    idx: int
    start: float
    end: float
    utt_start: int
    utt_end: int
    speakers: list[str]
    text: str


@dataclass
class UtteranceRow:
    id: str
    file_id: str
    idx: int
    speaker: str
    start: float
    end: float
    text: str
    words: list[list[Any]]
    dsim: float | None
    matched: list[str]
    headline: str | None


@dataclass
class Moment:
    chunk: ChunkRow
    utterance: UtteranceRow
    match_time: float
    highlights: list[tuple[int, int]] = field(default_factory=list)
    lexical_evidence: float = 0.0
    semantic_evidence: float | None = None


def tsquery_text(lexemes: list[str]) -> str:
    """OR-query over already-normalised lexemes, in tsquery input syntax (no re-stemming)."""
    quoted = []
    for lex in dict.fromkeys(lexemes):
        esc = lex.replace("\\", "\\\\").replace("'", "''")
        quoted.append(f"'{esc}'")
    return " | ".join(quoted)


def fetch_chunks(conn: psycopg.Connection, chunk_ids: list[str]) -> dict[str, ChunkRow]:
    if not chunk_ids:
        return {}
    rows = conn.execute(
        """SELECT id, file_id, idx, start_sec, end_sec, utt_start, utt_end, speakers, text
           FROM chunks WHERE id = ANY(%s)""",
        (chunk_ids,),
    ).fetchall()
    return {r[0]: ChunkRow(*r) for r in rows}


def fetch_utterances(
    conn: psycopg.Connection,
    chunks: list[ChunkRow],
    query_vec: np.ndarray | None,
    lexemes: list[str],
) -> dict[tuple[str, int], UtteranceRow]:
    if not chunks:
        return {}
    tsq = tsquery_text(lexemes) if lexemes else ""
    rows = conn.execute(
        """
        WITH r AS (SELECT * FROM unnest(%(fids)s::text[], %(a)s::int[], %(b)s::int[]) AS r(fid, a, b)),
             u AS (SELECT DISTINCT u.* FROM r JOIN utterances u
                     ON u.file_id = r.fid AND u.idx >= r.a AND u.idx < r.b)
        SELECT u.id, u.file_id, u.idx, u.speaker, u.start_sec, u.end_sec, u.text, u.words,
               CASE WHEN %(has_q)s THEN 1 - (u.embedding <=> %(q)s::vector) END AS dsim,
               CASE WHEN %(tsq)s <> '' THEN ARRAY(SELECT unnest(tsvector_to_array(u.tsv))
                                                   INTERSECT SELECT unnest(%(lexemes)s::text[]))
                    ELSE ARRAY[]::text[] END AS matched,
               CASE WHEN %(tsq)s <> '' AND u.tsv @@ %(tsq)s::tsquery
                    THEN ts_headline('english', u.text, %(tsq)s::tsquery,
                                     'HighlightAll=true, StartSel=' || chr(2) || ', StopSel=' || chr(3))
               END AS headline
        FROM u
        """,
        {
            "fids": [c.file_id for c in chunks],
            "a": [c.utt_start for c in chunks],
            "b": [c.utt_end for c in chunks],
            "has_q": query_vec is not None,
            "q": query_vec if query_vec is not None else np.zeros(1, dtype=np.float32),
            "tsq": tsq,
            "lexemes": lexemes or [],
        },
    ).fetchall()
    return {
        (r[1], r[2]): UtteranceRow(
            id=r[0],
            file_id=r[1],
            idx=r[2],
            speaker=r[3],
            start=r[4],
            end=r[5],
            text=r[6],
            words=r[7],
            dsim=None if r[8] is None else float(r[8]),
            matched=list(r[9] or []),
            headline=r[10],
        )
        for r in rows
    }


def parse_headline(headline: str) -> tuple[str, list[tuple[int, int]]]:
    """Strip highlight markers and return (plain_text, [(start, end), ...]) character spans."""
    spans: list[tuple[int, int]] = []
    out: list[str] = []
    pos = 0
    start: int | None = None
    for ch in headline:
        if ch == HL_START:
            start = pos
        elif ch == HL_STOP:
            if start is not None and pos > start:
                spans.append((start, pos))
            start = None
        else:
            out.append(ch)
            pos += 1
    return "".join(out), spans


def first_match_time(u: UtteranceRow, spans: list[tuple[int, int]]) -> float:
    """Timestamp of the first highlighted word (words are space-joined to form the text)."""
    if not spans or not u.words:
        return u.start
    target = spans[0][0]
    offset = 0
    for token, start, _end in u.words:
        if offset + len(token) > target:
            return float(start)
        offset += len(token) + 1
    return u.start


def snap(
    chunk: ChunkRow,
    utts: dict[tuple[str, int], UtteranceRow],
    idf: dict[str, float],
    lexical_weight: float,
    allowed_speakers: set[str] | None,
) -> Moment | None:
    """Pick the most relevant utterance of ``chunk``; None if no utterance passes the speaker filter."""
    cands = [
        u
        for i in range(chunk.utt_start, chunk.utt_end)
        if (u := utts.get((chunk.file_id, i))) is not None
        and (allowed_speakers is None or u.speaker in allowed_speakers)
    ]
    if not cands:
        return None
    lex = np.array([sum(idf.get(m, 0.0) for m in u.matched) for u in cands])
    sem = np.array([u.dsim if u.dsim is not None else 0.0 for u in cands])
    short = np.array([len(u.text.split()) < MIN_SEMANTIC_WORDS for u in cands])
    lex_n = lex / lex.max() if lex.max() > 0 else lex
    if np.ptp(sem) > 0:
        sem_n = (sem - sem.min()) / np.ptp(sem)
        sem_n[short] *= 0.5  # "Yeah." can look similar to anything; prefer contentful sentences
    else:
        sem_n = np.zeros_like(sem)
    w = lexical_weight if lex.max() > 0 else 0.0
    score = w * lex_n + (1.0 - w) * sem_n
    best = cands[int(np.argmax(score))]  # argmax returns the earliest utterance on ties
    spans: list[tuple[int, int]] = []
    if best.headline:
        _, spans = parse_headline(best.headline)
    return Moment(
        chunk=chunk,
        utterance=best,
        match_time=first_match_time(best, spans),
        highlights=spans,
        lexical_evidence=float(lex[cands.index(best)]),
        semantic_evidence=best.dsim,
    )


def overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    shorter = max(min(a1 - a0, b1 - b0), 1e-6)
    return inter / shorter


def temporal_nms(moments: list[Moment], gap_sec: float, max_overlap: float = 0.5) -> list[Moment]:
    """Keep the best moment per region: drop same-file moments that are too close to a better one."""
    kept: list[Moment] = []
    for m in moments:
        dup = any(
            k.chunk.file_id == m.chunk.file_id
            and (
                k.utterance.id == m.utterance.id
                or abs(k.utterance.start - m.utterance.start) < gap_sec
                or overlap_ratio(k.chunk.start, k.chunk.end, m.chunk.start, m.chunk.end) > max_overlap
            )
            for k in kept
        )
        if not dup:
            kept.append(m)
    return kept

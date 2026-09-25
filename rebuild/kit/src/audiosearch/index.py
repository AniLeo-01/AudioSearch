"""Transcripts -> Postgres. Rebuild-all in one transaction: idempotent, and ~1 min for 6 files."""

from __future__ import annotations

from collections import Counter

import psycopg
from psycopg.types.json import Jsonb

from .chunking import build_chunks
from .embed import Embedder
from .models import Transcript
from .text import content_tokens


def index_all(conn: psycopg.Connection, transcripts: list[Transcript], titles: dict[str, str], emb: Embedder) -> None:
    vocab: Counter[str] = Counter()  # term -> number of chunks containing it
    with conn.transaction():
        conn.execute("TRUNCATE audio_files, vocabulary CASCADE")
        for t in transcripts:
            chunks = build_chunks(t)
            c_vecs = emb.docs([c.text for c in chunks])
            u_vecs = emb.docs([u.text for u in t.utterances])
            conn.execute("INSERT INTO audio_files VALUES (%s, %s, %s)", (t.file_id, titles[t.file_id], t.duration))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO speakers VALUES (%s, %s, %s, %s)",
                    [(t.file_id, s.label, s.role, s.name) for s in t.speakers],
                )
                cur.executemany(
                    """INSERT INTO utterances (file_id, idx, speaker, start_sec, end_sec, text, words, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    [
                        (t.file_id, u.idx, u.speaker, u.start, u.end, u.text, Jsonb(u.words), v)
                        for u, v in zip(t.utterances, u_vecs, strict=True)
                    ],
                )
                cur.executemany(
                    """INSERT INTO chunks (id, file_id, utt_start, utt_end, start_sec, end_sec, speakers, text,
                                           embedding)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    [
                        (c.id, c.file_id, c.utt_start, c.utt_end, c.start, c.end, c.speakers, c.text, v)
                        for c, v in zip(chunks, c_vecs, strict=True)
                    ],
                )
            for c in chunks:
                vocab.update(set(content_tokens(c.text)))
        # BM25 postings straight from the generated tsvector: one row per (lexeme, chunk) with its tf
        conn.execute(
            """INSERT INTO chunk_terms (lexeme, chunk_id, tf)
               SELECT t.lexeme, c.id, coalesce(array_length(t.positions, 1), 1) FROM chunks c, unnest(c.tsv) t"""
        )
        conn.execute(
            """UPDATE chunks c SET doc_len = x.dl
               FROM (SELECT chunk_id, sum(tf) AS dl FROM chunk_terms GROUP BY chunk_id) x WHERE c.id = x.chunk_id"""
        )
        terms = sorted(vocab)
        conn.execute(
            """INSERT INTO vocabulary
               SELECT x.t, x.d, metaphone(x.t, 12), dmetaphone(x.t), dmetaphone_alt(x.t)
               FROM unnest(%s::text[], %s::int[]) AS x(t, d)""",
            (terms, [vocab[t] for t in terms]),
        )

"""Write transcripts into PostgreSQL: utterances, chunks, embeddings, BM25 postings and vocabulary.

Indexing a file is a single transaction guarded by an advisory lock, so readers never observe a
half-indexed file and concurrent writers cannot corrupt the global BM25 statistics.  Statistics are
maintained *incrementally* (df/N/sum(dl) deltas) - adding or replacing one recording costs O(file),
not O(corpus).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from audiosearch.config import Settings
from audiosearch.dataset import ManifestEntry
from audiosearch.domain import Transcript
from audiosearch.embeddings import Embedder
from audiosearch.pipeline.chunking import build_chunks
from audiosearch.textutil import vocabulary_terms

log = logging.getLogger(__name__)

INDEX_LOCK_KEY = 0x0A0D10  # arbitrary constant for pg_advisory_xact_lock
INDEX_FORMAT_VERSION = 1


@dataclass
class IndexReport:
    file_id: str
    skipped: bool
    n_utterances: int = 0
    n_chunks: int = 0
    seconds: float = 0.0


def index_signature(transcript: Transcript, settings: Settings, embedding_model: str) -> str:
    payload = {
        "v": INDEX_FORMAT_VERSION,
        "transcript": transcript.meta.get("signature"),
        "n_utts": len(transcript.utterances),
        "chunking": [settings.chunk_target_words, settings.chunk_stride_words, settings.chunk_context,
                     settings.chunk_context_max_words],
        "embedding_model": embedding_model,
    }  # fmt: skip
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


class Indexer:
    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        self.settings = settings
        self.embedder = embedder

    def index(
        self, conn: psycopg.Connection, transcript: Transcript, entry: ManifestEntry, force: bool = False
    ) -> IndexReport:
        t0 = time.perf_counter()
        fid = transcript.file_id
        sig = index_signature(transcript, self.settings, self.embedder.model_name)
        row = conn.execute("SELECT index_signature FROM audio_files WHERE file_id = %s", (fid,)).fetchone()
        conn.commit()
        if row and row[0] == sig and not force:
            return IndexReport(fid, skipped=True)

        s = self.settings
        chunks = build_chunks(
            transcript,
            target_words=s.chunk_target_words,
            stride_words=s.chunk_stride_words,
            context=s.chunk_context,
            context_max_words=s.chunk_context_max_words,
            title=entry.title,
        )
        # Embed outside the transaction (slow, CPU-bound) -------------------------------------------
        chunk_vecs = self.embedder.embed_documents([c.embed_text for c in chunks])
        utt_vecs = self.embedder.embed_documents([u.text for u in transcript.utterances])

        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (INDEX_LOCK_KEY,))
            _delete_file(conn, fid)
            conn.execute(
                """INSERT INTO audio_files (file_id, title, audio_path, sha256, duration_sec, sample_rate,
                       channels, metadata, transcript_meta, index_signature)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    fid, entry.title, entry.audio_path.name, transcript.meta.get("audio", {}).get("sha256", ""),
                    transcript.duration, transcript.meta.get("audio", {}).get("sample_rate"),
                    transcript.meta.get("audio", {}).get("channels"),
                    Jsonb(_entry_metadata(entry)), Jsonb(_transcript_meta(transcript)), sig,
                ),
            )  # fmt: skip
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO speakers (file_id, label, role, role_confidence, display_name, talk_time,
                           n_words, question_rate) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    [
                        (fid, p.label, p.role, p.role_confidence, p.display_name, p.talk_time, p.n_words,
                         p.question_rate)
                        for p in transcript.speakers
                    ],
                )  # fmt: skip
                cur.executemany(
                    """INSERT INTO utterances (id, file_id, idx, speaker, start_sec, end_sec, text, words, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    [
                        (
                            u.id, fid, u.idx, u.speaker, u.start, u.end, u.text,
                            Jsonb([[w.text, w.start, w.end] for w in transcript.words[u.word_start : u.word_end]]),
                            utt_vecs[i],
                        )
                        for i, u in enumerate(transcript.utterances)
                    ],
                )  # fmt: skip
                cur.executemany(
                    """INSERT INTO chunks (id, file_id, idx, start_sec, end_sec, utt_start, utt_end, speakers,
                           text, embed_text, n_words, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    [
                        (c.id, fid, c.idx, c.start, c.end, c.utterance_start, c.utterance_end, c.speakers,
                         c.text, c.embed_text, c.n_words, chunk_vecs[i])
                        for i, c in enumerate(chunks)
                    ],
                )  # fmt: skip
            _add_postings(conn, fid)
            _update_vocabulary(conn, _vocab_counts([c.text for c in chunks]), sign=+1)
        report = IndexReport(fid, False, len(transcript.utterances), len(chunks), time.perf_counter() - t0)
        log.info("indexed %s: %d utterances, %d chunks in %.1fs", fid, report.n_utterances, report.n_chunks,
                 report.seconds)  # fmt: skip
        return report

    def delete(self, conn: psycopg.Connection, file_id: str) -> bool:
        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (INDEX_LOCK_KEY,))
            return _delete_file(conn, file_id)


# --------------------------------------------------------------------------------------------------------
def _entry_metadata(entry: ManifestEntry) -> dict[str, Any]:
    return {
        "page_url": entry.page_url,
        "source_audio_url": entry.source_audio_url,
        "excerpt": {"start": entry.excerpt_start, "end": entry.excerpt_end},
        "topic": entry.topic,
        **entry.extra,
    }


def _transcript_meta(t: Transcript) -> dict[str, Any]:
    asr = t.meta.get("asr", {})
    diar = t.meta.get("diarization", {})
    return {
        "signature": t.meta.get("signature"),
        "asr_model": asr.get("model"),
        "asr_engine": asr.get("engine"),
        "diarization_engine": diar.get("engine"),
        "inter_speaker_cosine": diar.get("inter_speaker_cosine"),
    }


def _vocab_counts(chunk_texts: list[str]) -> Counter[str]:
    df: Counter[str] = Counter()
    for text in chunk_texts:
        df.update(vocabulary_terms(text))
    return df


def _add_postings(conn: psycopg.Connection, fid: str) -> None:
    conn.execute(
        """INSERT INTO chunk_terms (lexeme, chunk_id, tf)
           SELECT t.lexeme, c.id, coalesce(array_length(t.positions, 1), 1)
           FROM chunks c, unnest(c.tsv) AS t
           WHERE c.file_id = %s""",
        (fid,),
    )
    conn.execute(
        """UPDATE chunks c SET doc_len = coalesce(x.dl, 0)
           FROM (SELECT ct.chunk_id, sum(ct.tf) AS dl FROM chunk_terms ct JOIN chunks c2 ON c2.id = ct.chunk_id
                 WHERE c2.file_id = %s GROUP BY ct.chunk_id) x
           WHERE c.id = x.chunk_id""",
        (fid,),
    )
    conn.execute(
        """INSERT INTO term_stats (lexeme, df)
           SELECT ct.lexeme, count(*) FROM chunk_terms ct JOIN chunks c ON c.id = ct.chunk_id
           WHERE c.file_id = %s GROUP BY ct.lexeme
           ON CONFLICT (lexeme) DO UPDATE SET df = term_stats.df + EXCLUDED.df""",
        (fid,),
    )
    conn.execute(
        """UPDATE corpus_stats SET
               n_docs = n_docs + (SELECT count(*) FROM chunks WHERE file_id = %(f)s),
               sum_doc_len = sum_doc_len + (SELECT coalesce(sum(doc_len), 0) FROM chunks WHERE file_id = %(f)s),
               updated_at = now()
           WHERE id = 1""",
        {"f": fid},
    )


def _update_vocabulary(conn: psycopg.Connection, counts: Counter[str], sign: int) -> None:
    if not counts:
        return
    terms = sorted(counts)
    dfs = [counts[t] for t in terms]
    if sign > 0:
        conn.execute(
            """INSERT INTO vocabulary (term, df, metaphone, dmetaphone, dmetaphone_alt)
               SELECT x.t, x.d, metaphone(x.t, 12), dmetaphone(x.t), dmetaphone_alt(x.t)
               FROM unnest(%s::text[], %s::int[]) AS x(t, d)
               ON CONFLICT (term) DO UPDATE SET df = vocabulary.df + EXCLUDED.df""",
            (terms, dfs),
        )
    else:
        conn.execute(
            """UPDATE vocabulary v SET df = v.df - x.d
               FROM unnest(%s::text[], %s::int[]) AS x(t, d) WHERE v.term = x.t""",
            (terms, dfs),
        )
        conn.execute("DELETE FROM vocabulary WHERE df <= 0")


def _delete_file(conn: psycopg.Connection, fid: str) -> bool:
    exists = conn.execute("SELECT 1 FROM audio_files WHERE file_id = %s", (fid,)).fetchone()
    if not exists:
        return False
    conn.execute(
        """UPDATE term_stats ts SET df = ts.df - x.cnt
           FROM (SELECT ct.lexeme, count(*) AS cnt FROM chunk_terms ct JOIN chunks c ON c.id = ct.chunk_id
                 WHERE c.file_id = %s GROUP BY ct.lexeme) x
           WHERE ts.lexeme = x.lexeme""",
        (fid,),
    )
    conn.execute("DELETE FROM term_stats WHERE df <= 0")
    conn.execute(
        """UPDATE corpus_stats SET
               n_docs = n_docs - (SELECT count(*) FROM chunks WHERE file_id = %(f)s),
               sum_doc_len = sum_doc_len - (SELECT coalesce(sum(doc_len), 0) FROM chunks WHERE file_id = %(f)s),
               updated_at = now()
           WHERE id = 1""",
        {"f": fid},
    )
    texts = [r[0] for r in conn.execute("SELECT text FROM chunks WHERE file_id = %s", (fid,)).fetchall()]
    _update_vocabulary(conn, _vocab_counts(texts), sign=-1)
    conn.execute("DELETE FROM audio_files WHERE file_id = %s", (fid,))
    return True

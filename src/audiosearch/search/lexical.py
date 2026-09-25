"""Lexical channel: Okapi BM25 computed in SQL over an incrementally maintained inverted index.

Postgres' built-in ``ts_rank`` has no inverse-document-frequency term, so rare words ("Ellington",
"cesium") are not rewarded.  We keep real BM25 statistics (df, N, avgdl) in ordinary tables and score
with a single aggregate query - portable to any managed Postgres that offers pgvector (RDS, Cloud SQL,
Azure, Supabase, Neon), unlike BM25 extensions that need custom builds.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg
from psycopg import sql

from audiosearch.search.filters import SearchFilters

K1 = 1.2
B = 0.75


@dataclass(frozen=True)
class WeightedLexeme:
    lexeme: str
    weight: float
    source: str  # the query token (or expansion term) it came from


def analyze(conn: psycopg.Connection, text: str) -> list[str]:
    """Stemmed, stopword-free lexemes exactly as Postgres indexes them (one per occurrence)."""
    rows = conn.execute(
        "SELECT lexeme, coalesce(array_length(positions, 1), 1) FROM unnest(to_tsvector('english', %s))",
        (text,),
    ).fetchall()
    out: list[str] = []
    for lexeme, n in rows:
        out.extend([lexeme] * int(n))
    return out


def analyze_terms(conn: psycopg.Connection, terms: list[str]) -> dict[str, list[str]]:
    """Map several surface terms to their lexemes in one round trip."""
    if not terms:
        return {}
    rows = conn.execute(
        """SELECT t.term, array_remove(array_agg(x.lexeme), NULL)
           FROM unnest(%s::text[]) AS t(term)
           LEFT JOIN LATERAL unnest(to_tsvector('english', t.term)) AS x ON TRUE
           GROUP BY t.term""",
        (terms,),
    ).fetchall()
    return {term: list(lexemes or []) for term, lexemes in rows}


def bm25_search(
    conn: psycopg.Connection,
    lexemes: list[WeightedLexeme],
    filters: SearchFilters,
    limit: int,
    k1: float = K1,
    b: float = B,
) -> list[tuple[str, float]]:
    """Top chunks by BM25; each query lexeme contributes weight * idf * tf-saturation."""
    if not lexemes:
        return []
    merged: dict[str, float] = {}
    for wl in lexemes:  # repeated query terms add up (query term frequency)
        merged[wl.lexeme] = merged.get(wl.lexeme, 0.0) + wl.weight
    where, params = filters.chunk_clause("c")
    query = sql.SQL(
        """
        WITH q(lexeme, w) AS (SELECT * FROM unnest(%(lexemes)s::text[], %(weights)s::float8[])),
             cs AS (SELECT n_docs::float8 AS n, sum_doc_len::float8 / greatest(n_docs, 1) AS avgdl
                    FROM corpus_stats WHERE id = 1)
        SELECT c.id,
               sum(q.w * ln(1 + (cs.n - ts.df + 0.5) / (ts.df + 0.5))
                   * (ct.tf * (%(k1)s + 1)) / (ct.tf + %(k1)s * (1 - %(b)s + %(b)s * c.doc_len / cs.avgdl)))
                   AS score
        FROM q
        JOIN term_stats ts ON ts.lexeme = q.lexeme
        JOIN chunk_terms ct ON ct.lexeme = q.lexeme
        JOIN chunks c ON c.id = ct.chunk_id
        CROSS JOIN cs
        WHERE {where}
        GROUP BY c.id
        ORDER BY score DESC, c.id
        LIMIT %(limit)s
        """
    ).format(where=where)
    params.update(
        lexemes=list(merged), weights=list(merged.values()), k1=k1, b=b, limit=limit
    )
    return [(r[0], float(r[1])) for r in conn.execute(query, params).fetchall()]

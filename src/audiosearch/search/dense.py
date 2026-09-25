"""Dense channel: approximate nearest-neighbour search with pgvector HNSW (cosine distance)."""

from __future__ import annotations

import numpy as np
import psycopg
from psycopg import sql

from audiosearch.search.filters import SearchFilters


def dense_search(
    conn: psycopg.Connection,
    query_vec: np.ndarray,
    filters: SearchFilters,
    limit: int,
    ef_search: int = 100,
    iterative_scan: bool = False,
) -> list[tuple[str, float]]:
    """Top chunks by cosine similarity of their (dialogue-context-augmented) embeddings."""
    # set_config(..., is_local => true) scopes the knob to the current transaction.
    conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(ef_search, limit)),))
    if iterative_scan and (filters.file_ids or filters.role or filters.speaker or filters.phrases):
        # pgvector >= 0.8: keep scanning the graph until enough rows survive the filter.
        conn.execute("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)")
    where, params = filters.chunk_clause("c")
    query = sql.SQL(
        """
        SELECT c.id, 1 - (c.embedding <=> %(q)s) AS sim
        FROM chunks c
        WHERE c.embedding IS NOT NULL AND {where}
        ORDER BY c.embedding <=> %(q)s
        LIMIT %(limit)s
        """
    ).format(where=where)
    params.update(q=query_vec, limit=limit)
    rows = conn.execute(query, params).fetchall()
    # relaxed_order may return slightly out-of-order rows; restore exact order on the small result set
    return sorted(((r[0], float(r[1])) for r in rows), key=lambda x: (-x[1], x[0]))

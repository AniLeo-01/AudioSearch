"""Shared fixtures.

DB-backed tests run against ``AUDIOSEARCH_DATABASE_URL`` in a throw-away schema and are skipped when
no database is reachable.  A deterministic hashing embedder stands in for the neural model so SQL and
ranking logic can be tested quickly; the real model is exercised by the ``eval`` suite.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterator

import numpy as np
import psycopg
import pytest

from audiosearch.config import Settings, get_settings
from audiosearch.textutil import tokens


class HashEmbedder:
    """Bag-of-words feature hashing: texts sharing words get similar vectors (good enough for tests)."""

    model_name = "test/hash-embedder-64"
    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else np.full(self.dim, 1 / np.sqrt(self.dim), dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)

    def lexicon(self) -> frozenset[str]:
        return frozenset({"inside", "sells", "cord", "the", "airplane"})


@pytest.fixture(scope="session")
def hash_embedder() -> HashEmbedder:
    return HashEmbedder()


def _db_available(url: str) -> bool:
    try:
        with psycopg.connect(url, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def base_settings() -> Settings:
    return get_settings()


@pytest.fixture(scope="module")
def db_settings(base_settings: Settings) -> Iterator[Settings]:
    if not _db_available(base_settings.database_url):
        pytest.skip("PostgreSQL not reachable (set AUDIOSEARCH_DATABASE_URL)")
    schema = f"test_{uuid.uuid4().hex[:10]}"
    settings = base_settings.model_copy(update={"db_schema": schema, "db_pool_min": 1, "db_pool_max": 2})
    yield settings
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')

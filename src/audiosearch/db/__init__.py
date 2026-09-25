"""PostgreSQL access: connection pooling, schema migrations and index metadata checks."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg_pool import ConnectionPool

from audiosearch.config import Settings

log = logging.getLogger(__name__)

EXTENSIONS = ("vector", "pg_trgm", "fuzzystrmatch")


class SchemaMismatchError(RuntimeError):
    """The index was built with a different embedding model/dimension than the one configured."""


def _session_setup(conn: psycopg.Connection, settings: Settings) -> None:
    conn.execute(
        sql.SQL("SET search_path TO {}, public").format(sql.Identifier(settings.db_schema))
    )
    conn.execute(sql.SQL("SET statement_timeout = {}").format(sql.Literal(settings.db_statement_timeout_ms)))
    register_vector(conn)


def connect(settings: Settings, *, autocommit: bool = False) -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url, autocommit=True)
    _ensure_extensions_and_schema(conn, settings)
    _session_setup(conn, settings)
    conn.autocommit = autocommit
    return conn


def create_pool(settings: Settings) -> ConnectionPool:
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        _ensure_extensions_and_schema(conn, settings)

    def configure(conn: psycopg.Connection) -> None:
        conn.autocommit = True
        _session_setup(conn, settings)
        conn.autocommit = False

    pool = ConnectionPool(
        settings.database_url,
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
        configure=configure,
        open=True,
        name="audiosearch",
    )
    pool.wait(timeout=30)
    return pool


@contextmanager
def pooled(pool: ConnectionPool) -> Iterator[psycopg.Connection]:
    with pool.connection() as conn:
        yield conn


def _ensure_extensions_and_schema(conn: psycopg.Connection, settings: Settings) -> None:
    for ext in EXTENSIONS:
        try:
            conn.execute(sql.SQL("CREATE EXTENSION IF NOT EXISTS {} SCHEMA public").format(sql.Identifier(ext)))
        except psycopg.errors.InsufficientPrivilege as e:  # managed Postgres: ask an admin
            exists = conn.execute("SELECT 1 FROM pg_extension WHERE extname = %s", (ext,)).fetchone()
            if not exists:
                raise RuntimeError(
                    f"extension '{ext}' is missing and the current role cannot create it; "
                    f"run `CREATE EXTENSION {ext};` as an administrator"
                ) from e
    conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(settings.db_schema)))


# ------------------------------------------------------------------------------------------------------
# Migrations
# ------------------------------------------------------------------------------------------------------
def _migration_files() -> list[tuple[str, str]]:
    files = []
    pkg = resources.files("audiosearch.db") / "migrations"
    for entry in sorted(pkg.iterdir(), key=lambda p: p.name):
        if entry.name.endswith(".sql") and re.match(r"^\d{4}_", entry.name):
            files.append((entry.name.split("_", 1)[0], entry.read_text(encoding="utf-8")))
    return files


def migrate(settings: Settings, embedding_model: str, embedding_dim: int) -> list[str]:
    """Apply pending migrations; record and verify the embedding model/dimension."""
    applied: list[str] = []
    with connect(settings, autocommit=False) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version text PRIMARY KEY, "
            "applied_at timestamptz NOT NULL DEFAULT now())"
        )
        conn.commit()
        done = {r[0] for r in conn.execute("SELECT version FROM schema_migrations").fetchall()}
        for version, body in _migration_files():
            if version in done:
                continue
            rendered = body.replace("__EMBEDDING_DIM__", str(int(embedding_dim)))
            with conn.transaction():
                conn.execute(rendered)  # type: ignore[arg-type]  # trusted, packaged SQL
                conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            applied.append(version)
            log.info("applied migration %s", version)
        with conn.transaction():
            conn.execute(
                "INSERT INTO index_meta (key, value) VALUES ('embedding_model', %s), ('embedding_dim', %s) "
                "ON CONFLICT (key) DO NOTHING",
                (embedding_model, str(embedding_dim)),
            )
        verify_index_meta(conn, embedding_model, embedding_dim)
    return applied


def verify_index_meta(conn: psycopg.Connection, embedding_model: str, embedding_dim: int) -> None:
    rows = dict(conn.execute("SELECT key, value FROM index_meta").fetchall())
    if rows.get("embedding_model") not in (None, embedding_model) or rows.get("embedding_dim") not in (
        None,
        str(embedding_dim),
    ):
        raise SchemaMismatchError(
            f"index built with {rows.get('embedding_model')} (dim {rows.get('embedding_dim')}) but "
            f"{embedding_model} (dim {embedding_dim}) is configured; use a different AUDIOSEARCH_DB_SCHEMA "
            "for the new model (blue/green) or run `audiosearch db reset`"
        )


def reset_schema(settings: Settings) -> None:
    """Drop and recreate the configured schema (destructive; used by tests and `db reset`)."""
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        if settings.db_schema == "public":
            for table in (
                "vocabulary", "corpus_stats", "term_stats", "chunk_terms", "chunks", "utterances",
                "speakers", "audio_files", "index_meta", "schema_migrations",
            ):  # fmt: skip
                conn.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier("public", table)))
        else:
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(settings.db_schema)))


def pgvector_version(conn: psycopg.Connection) -> tuple[int, ...]:
    row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    return tuple(int(x) for x in row[0].split(".")) if row else (0,)

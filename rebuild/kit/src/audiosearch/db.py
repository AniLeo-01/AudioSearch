from __future__ import annotations

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from . import config


def init_db() -> None:
    with psycopg.connect(config.DATABASE_URL, autocommit=True) as conn:
        conn.execute(Path(__file__).with_name("schema.sql").read_text())


def connect() -> psycopg.Connection:
    conn = psycopg.connect(config.DATABASE_URL)
    register_vector(conn)  # numpy arrays <-> vector; needs the extension (init_db) first
    return conn

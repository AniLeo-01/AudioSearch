"""Process-wide service construction (connection pool, models, engine), shared by CLI, API and tests."""

from __future__ import annotations

import logging

from audiosearch.config import Settings
from audiosearch.db import create_pool, migrate
from audiosearch.embeddings import CrossEncoderReranker, Embedder
from audiosearch.search.engine import SearchEngine

log = logging.getLogger(__name__)


def build_engine(settings: Settings, with_reranker: bool | None = None, embedder: Embedder | None = None) -> SearchEngine:
    """Load models, verify the schema matches the embedding model, and return a ready engine."""
    embedder = embedder or Embedder(settings.embedding_model, settings.embedding_device, settings.embedding_batch_size)
    migrate(settings, settings.embedding_model, embedder.dim)  # idempotent; verifies index_meta
    reranker = None
    if with_reranker if with_reranker is not None else settings.rerank:
        reranker = CrossEncoderReranker(settings.reranker_model, settings.embedding_device)
    pool = create_pool(settings)
    engine = SearchEngine(settings, pool, embedder, reranker)
    embedder.embed_query("warm up")  # first call pays tokenizer/graph initialisation
    return engine

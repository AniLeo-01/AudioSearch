"""Local text embeddings (sentence-transformers) with per-model query/document conventions."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

BGE_QUERY = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class ModelProfile:
    query_prefix: str = ""
    doc_prefix: str = ""
    trust_remote_code: bool = False
    max_seq_length: int | None = 512


PROFILES: dict[str, ModelProfile] = {
    "BAAI/bge-small-en-v1.5": ModelProfile(query_prefix=BGE_QUERY),
    "BAAI/bge-base-en-v1.5": ModelProfile(query_prefix=BGE_QUERY),
    "BAAI/bge-large-en-v1.5": ModelProfile(query_prefix=BGE_QUERY),
    "Snowflake/snowflake-arctic-embed-s": ModelProfile(query_prefix=BGE_QUERY),
    "Snowflake/snowflake-arctic-embed-m-v1.5": ModelProfile(query_prefix=BGE_QUERY),
    "intfloat/e5-small-v2": ModelProfile(query_prefix="query: ", doc_prefix="passage: "),
    "intfloat/e5-base-v2": ModelProfile(query_prefix="query: ", doc_prefix="passage: "),
    "nomic-ai/nomic-embed-text-v1.5": ModelProfile(
        query_prefix="search_query: ", doc_prefix="search_document: ", trust_remote_code=True
    ),
    "sentence-transformers/all-MiniLM-L6-v2": ModelProfile(max_seq_length=256),
    "sentence-transformers/all-mpnet-base-v2": ModelProfile(max_seq_length=384),
}


class Embedder:
    """Thread-safe wrapper returning L2-normalised float32 vectors, with a small query LRU cache."""

    def __init__(self, model: str, device: str = "cpu", batch_size: int = 32, cache_size: int = 2048) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.profile = PROFILES.get(model, ModelProfile())
        self.batch_size = batch_size
        t0 = time.perf_counter()
        self._model = SentenceTransformer(model, device=device, trust_remote_code=self.profile.trust_remote_code)
        if self.profile.max_seq_length:
            self._model.max_seq_length = self.profile.max_seq_length
        self.dim = int(self._model.get_sentence_embedding_dimension() or 0)
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        log.info("loaded embedding model %s (dim=%d) in %.1fs", model, self.dim, time.perf_counter() - t0)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode([self.profile.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        key = text.strip()
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
        vec = self._encode([self.profile.query_prefix + key])[0]
        with self._lock:
            self._cache[key] = vec
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return vec


class CrossEncoderReranker:
    """Optional second-stage reranker (local cross-encoder)."""

    def __init__(self, model: str, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model
        t0 = time.perf_counter()
        self._model = CrossEncoder(model, device=device, max_length=384)
        log.info("loaded reranker %s in %.1fs", model, time.perf_counter() - t0)

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(self._model.predict([(query, p) for p in passages], batch_size=32), dtype=np.float32)

"""Evaluation runner: score system configurations on the golden query set.

Two kinds of variation are supported:

* **query-time systems** (mode, fusion, sounds-like expansion, reranking, moment snapping, NMS) run
  against one index via :class:`SearchOptions`;
* **index-time variants** (chunk size, dialogue context, embedding model, ASR model) are built into
  isolated Postgres schemas (``eval_<name>``), so experiments never disturb the serving index.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from audiosearch.config import Settings
from audiosearch.dataset import load_manifest
from audiosearch.domain import Transcript
from audiosearch.embeddings import CrossEncoderReranker, Embedder
from audiosearch.eval.golden import GoldenQuery, GoldenSet
from audiosearch.eval.metrics import KS, QueryMetrics, Returned, bootstrap_ci, mean, query_metrics
from audiosearch.search.engine import SearchEngine, SearchOptions

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SystemSpec:
    name: str
    options: SearchOptions
    description: str


FULL = SearchOptions(mode="hybrid", adaptive=True, phonetic=True, rerank=False)

QUERY_SYSTEMS: tuple[SystemSpec, ...] = (
    SystemSpec("bm25", SearchOptions(mode="lexical", phonetic=False), "Lexical only: BM25 over passages"),
    SystemSpec("bm25+soundslike", SearchOptions(mode="lexical", phonetic=True), "BM25 + sounds-like expansion"),
    SystemSpec("dense", SearchOptions(mode="semantic"), "Semantic only: pgvector HNSW"),
    SystemSpec("hybrid-cc", SearchOptions(mode="hybrid", adaptive=False, phonetic=False, fusion="cc"),
               "BM25 + dense, min-max convex combination"),
    SystemSpec("hybrid-rrf", SearchOptions(mode="hybrid", adaptive=False, phonetic=False),
               "BM25 + dense, reciprocal rank fusion"),
    SystemSpec("hybrid-rrf+adaptive", SearchOptions(mode="hybrid", adaptive=True, phonetic=False),
               "+ query-intent adaptive fusion weights"),
    SystemSpec("full", FULL, "+ sounds-like expansion (default system)"),
    SystemSpec("full+rerank", SearchOptions(mode="hybrid", adaptive=True, phonetic=True, rerank=True),
               "+ cross-encoder reranking"),
    SystemSpec("full-no-snap", SearchOptions(mode="hybrid", adaptive=True, phonetic=True, snap=False),
               "full, but report passage start instead of the snapped moment"),
    SystemSpec("full-no-nms", SearchOptions(mode="hybrid", adaptive=True, phonetic=True, nms=False),
               "full, without temporal non-maximum suppression"),
)  # fmt: skip


@dataclass
class QueryOutcome:
    query: GoldenQuery
    metrics: QueryMetrics
    latency_ms: float
    top: list[dict[str, Any]]


@dataclass
class SystemResult:
    name: str
    description: str
    outcomes: list[QueryOutcome] = field(default_factory=list)

    def values(self, key: str, k: int | None = None, category: str | None = None) -> list[float]:
        out = []
        for o in self.outcomes:
            if category and o.query.category != category:
                continue
            m = o.metrics
            v = getattr(m, key)
            out.append(float(v[k] if k is not None else v))
        return out

    def summary(self) -> dict[str, Any]:
        lat = [o.latency_ms for o in self.outcomes]
        offsets = [o.metrics.offset_sec for o in self.outcomes if o.metrics.offset_sec is not None]
        r5 = self.values("recall", 5)
        mrr = self.values("mrr")
        cats: dict[str, dict[str, float]] = {}
        for cat in sorted({o.query.category for o in self.outcomes}):
            cats[cat] = {
                "n": len(self.values("mrr", category=cat)),
                "recall@5": mean(self.values("recall", 5, cat)),
                "recall@10": mean(self.values("recall", 10, cat)),
                "mrr": mean(self.values("mrr", category=cat)),
            }
        return {
            "system": self.name,
            "description": self.description,
            "n_queries": len(self.outcomes),
            **{f"recall@{k}": mean(self.values("recall", k)) for k in KS},
            **{f"success@{k}": mean(self.values("success", k)) for k in KS},
            **{f"precision@{k}": mean(self.values("precision", k)) for k in KS},
            "mrr": mean(mrr),
            "ndcg@10": mean(self.values("ndcg10")),
            "recall@5_ci95": bootstrap_ci(r5),
            "mrr_ci95": bootstrap_ci(mrr),
            "median_offset_sec": statistics.median(offsets) if offsets else None,
            "latency_p50_ms": statistics.median(lat) if lat else None,
            "latency_p95_ms": sorted(lat)[max(0, int(round(0.95 * len(lat))) - 1)] if lat else None,
            "by_category": cats,
        }


def run_system(engine: SearchEngine, queries: list[GoldenQuery], spec: SystemSpec, tolerance: float) -> SystemResult:
    res = SystemResult(spec.name, spec.description)
    for q in queries:
        resp = engine.search(q.query, k=10, role=q.role, options=spec.options)
        returned = [Returned(h.file_id, h.start, h.end) for h in resp.hits]
        res.outcomes.append(
            QueryOutcome(
                query=q,
                metrics=query_metrics(returned, q.relevant, tolerance),
                latency_ms=resp.timings_ms.get("total", 0.0),
                top=[
                    {"file": h.file_id, "start": h.start, "end": h.end, "speaker": h.speaker, "text": h.text[:160],
                     "channels": h.channels}
                    for h in resp.hits[:5]
                ],  # fmt: skip
            )
        )
    return res


# ----------------------------------------------------------------------------------------------------------
# Index-time variants
# ----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class IndexVariant:
    name: str
    description: str
    overrides: tuple[tuple[str, Any], ...] = ()

    def settings(self, base: Settings) -> Settings:
        update = dict(self.overrides)
        update["db_schema"] = "eval_" + "".join(c if c.isalnum() else "_" for c in self.name.lower())[:50]
        return base.model_copy(update=update)


INDEX_VARIANTS: tuple[IndexVariant, ...] = (
    IndexVariant("chunk-35w", "35-word windows (stride 18)", (("chunk_target_words", 35), ("chunk_stride_words", 18))),
    IndexVariant("chunk-140w", "140-word windows (stride 70)", (("chunk_target_words", 140), ("chunk_stride_words", 70))),
    IndexVariant("context-none", "no dialogue-context augmentation", (("chunk_context", "none"),)),
    IndexVariant("context-question+title", "question + episode title context", (("chunk_context", "question+title"),)),
    IndexVariant("embed-minilm", "all-MiniLM-L6-v2 embeddings", (("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),)),
    IndexVariant("embed-bge-base", "bge-base-en-v1.5 embeddings", (("embedding_model", "BAAI/bge-base-en-v1.5"),)),
    IndexVariant("embed-e5-base", "e5-base-v2 embeddings", (("embedding_model", "intfloat/e5-base-v2"),)),
    IndexVariant("asr-base.en", "whisper base.en transcripts (higher WER)", (("transcript_set", "base.en"), ("asr_model", "base.en"))),
)  # fmt: skip


def build_index(settings: Settings, embedder: Embedder) -> None:
    """(Re)build the index for ``settings`` from the transcripts on disk (idempotent)."""
    from audiosearch.db import connect, migrate
    from audiosearch.indexing import Indexer

    migrate(settings, embedder.model_name, embedder.dim)
    indexer = Indexer(settings, embedder)
    with connect(settings) as conn:
        for e in load_manifest(settings.manifest_path, settings.audio_dir):
            path = settings.transcripts_dir / f"{e.file_id}.json"
            if not path.exists():
                raise FileNotFoundError(f"{path} missing - run `audiosearch ingest` for this transcript set")
            indexer.index(conn, Transcript.load(path), e)


class EngineFactory:
    """Caches models across variants so each embedding model is loaded once."""

    def __init__(self) -> None:
        self._embedders: dict[str, Embedder] = {}
        self._reranker: CrossEncoderReranker | None = None

    def embedder(self, settings: Settings) -> Embedder:
        if settings.embedding_model not in self._embedders:
            self._embedders[settings.embedding_model] = Embedder(
                settings.embedding_model, settings.embedding_device, settings.embedding_batch_size
            )
        return self._embedders[settings.embedding_model]

    def reranker(self, settings: Settings) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(settings.reranker_model, settings.embedding_device)
        return self._reranker

    def engine(self, settings: Settings, with_reranker: bool, build: bool = True) -> SearchEngine:
        from audiosearch.db import create_pool

        emb = self.embedder(settings)
        if build:
            build_index(settings, emb)
        return SearchEngine(settings, create_pool(settings), emb, self.reranker(settings) if with_reranker else None)


def run_evaluation(
    base: Settings,
    golden: GoldenSet,
    split: str | None,
    systems: tuple[SystemSpec, ...] = QUERY_SYSTEMS,
    variants: tuple[IndexVariant, ...] = (),
    factory: EngineFactory | None = None,
) -> dict[str, SystemResult]:
    factory = factory or EngineFactory()
    queries = golden.split(split)
    results: dict[str, SystemResult] = {}
    need_rerank = any(s.options.rerank for s in systems)
    engine = factory.engine(base, with_reranker=need_rerank)
    for spec in systems:
        log.info("evaluating system %s on %d queries", spec.name, len(queries))
        results[spec.name] = run_system(engine, queries, spec, golden.tolerance_sec)
    engine.pool.close()
    for variant in variants:
        vs = variant.settings(base)
        log.info("evaluating index variant %s (schema %s)", variant.name, vs.db_schema)
        eng = factory.engine(vs, with_reranker=False)
        spec = SystemSpec(f"variant:{variant.name}", SearchOptions(), variant.description)
        results[spec.name] = run_system(eng, queries, spec, golden.tolerance_sec)
        eng.pool.close()
    return results


def per_query_table(results: dict[str, SystemResult], metric: str = "recall", k: int = 5) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = defaultdict(dict)
    for name, res in results.items():
        for o in res.outcomes:
            v = getattr(o.metrics, metric)
            table[o.query.id][name] = float(v[k] if isinstance(v, dict) else v)
    return table

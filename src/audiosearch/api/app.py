"""HTTP API + web UI.

GET /api/search?q=...&k=10&mode=hybrid|lexical|semantic&role=host|guest&file_id=...&rerank=true
GET /api/files                       recordings with speakers/roles
GET /api/files/{file_id}/transcript  utterances (for the synchronized transcript view)
GET /media/{file_id}                 audio with HTTP Range support (seekable in the browser)
GET /healthz  /readyz  /metrics      liveness, readiness (DB + models), Prometheus
GET /                                single-page UI;  /api/docs  OpenAPI UI
"""

import logging
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.concurrency import run_in_threadpool

from audiosearch import __version__
from audiosearch.api import schemas
from audiosearch.config import Settings, get_settings
from audiosearch.logging_setup import setup_logging
from audiosearch.search.engine import SearchEngine, SearchOptions
from audiosearch.search.query import QueryError

log = logging.getLogger("audiosearch.api")
STATIC_DIR = Path(__file__).parent / "static"
FILE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
UI_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self'; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.http_requests = Counter(
            "audiosearch_http_requests_total",
            "HTTP requests",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.http_latency = Histogram(
            "audiosearch_http_request_duration_seconds",
            "HTTP latency",
            ["route"],
            registry=self.registry,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
        )
        self.search_stage = Histogram(
            "audiosearch_search_stage_seconds",
            "Search latency per pipeline stage",
            ["stage"],
            registry=self.registry,
            buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
        )
        self.searches = Counter(
            "audiosearch_searches_total",
            "Searches by mode and outcome",
            ["mode", "outcome"],
            registry=self.registry,
        )
        self.expansions = Counter(
            "audiosearch_soundslike_expansions_total",
            "Sounds-like expansions applied",
            registry=self.registry,
        )
        self.indexed_files = Gauge("audiosearch_indexed_files", "Recordings in the index", registry=self.registry)


def _fmt_ts(sec: float) -> str:
    m, s = divmod(max(sec, 0.0), 60)
    return f"{int(m):02d}:{s:04.1f}"


def create_app(settings: Settings | None = None, engine: SearchEngine | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level, settings.log_format)
    metrics = Metrics()
    reranker_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owns = app.state.engine is None
        if owns:
            from audiosearch.services import build_engine

            app.state.engine = await run_in_threadpool(build_engine, settings)
        log.info(
            "AudioSearch API %s ready (schema=%s, model=%s)",
            __version__,
            settings.db_schema,
            settings.embedding_model,
        )
        try:
            yield
        finally:
            if owns:
                app.state.engine.pool.close()

    app = FastAPI(
        title="AudioSearch",
        version=__version__,
        description="Conversation-aware hybrid search over two-speaker audio.",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    app.state.engine = engine
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.api_cors_origins, allow_methods=["GET"], allow_headers=["*"]
    )

    def get_engine() -> SearchEngine:
        eng = app.state.engine
        if eng is None:  # pragma: no cover - only during startup
            raise HTTPException(503, "engine not ready")
        return eng

    @app.middleware("http")
    async def observe(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error", extra={"rid": rid, "path": request.url.path})
            response = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
        route = getattr(request.scope.get("route"), "path", "unmatched")
        elapsed = time.perf_counter() - t0
        metrics.http_requests.labels(request.method, route, str(response.status_code)).inc()
        metrics.http_latency.labels(route).observe(elapsed)
        response.headers["x-request-id"] = rid
        response.headers["x-content-type-options"] = "nosniff"
        response.headers["referrer-policy"] = "no-referrer"
        if route in ("/", "/static") or request.url.path.startswith("/static/"):
            response.headers["content-security-policy"] = UI_CSP
        if route not in ("/healthz", "/readyz", "/metrics") and not request.url.path.startswith("/static/"):
            log.info(
                "%s %s -> %d (%.1f ms)",
                request.method,
                request.url.path,
                response.status_code,
                elapsed * 1000,
                extra={"rid": rid},
            )
        return response

    # ------------------------------------------------------------------------------------------------ API
    @app.get("/api/search", response_model=schemas.SearchResponse, tags=["search"])
    def api_search(
        q: Annotated[str, Query(min_length=1, max_length=settings.max_query_chars, description="query text")],
        k: Annotated[int, Query(ge=1, le=settings.max_k)] = settings.default_k,
        mode: Literal["hybrid", "lexical", "semantic"] = "hybrid",
        role: Literal["host", "guest"] | None = None,
        file_id: Annotated[list[str] | None, Query()] = None,
        rerank: bool | None = None,
    ) -> schemas.SearchResponse:
        eng = get_engine()
        if rerank and eng.reranker is None:
            with reranker_lock:
                if eng.reranker is None:
                    from audiosearch.embeddings import CrossEncoderReranker

                    eng.reranker = CrossEncoderReranker(settings.reranker_model, settings.embedding_device)
        try:
            resp = eng.search(q, k, role=role, file_ids=file_id, options=SearchOptions(mode=mode, rerank=rerank))
        except QueryError as e:
            metrics.searches.labels(mode, "invalid").inc()
            raise HTTPException(status_code=422, detail=str(e)) from e
        for stage, ms in resp.timings_ms.items():
            metrics.search_stage.labels(stage).observe(ms / 1000)
        metrics.searches.labels(mode, "ok" if resp.hits else "empty").inc()
        if resp.expansions:
            metrics.expansions.inc(len(resp.expansions))
        return schemas.SearchResponse(
            query=resp.query,
            mode=resp.mode,
            intent=resp.intent,
            weights=resp.weights,
            expansions=[schemas.Expansion(**asdict(e)) for e in resp.expansions],
            hits=[
                schemas.SearchHit(
                    **{k2: v for k2, v in asdict(h).items() if k2 not in ("chunk_id", "utterance_id")},
                    audio_url=f"/media/{h.file_id}",
                    timestamp=_fmt_ts(h.start),
                )
                for h in resp.hits
            ],
            timings_ms=resp.timings_ms,
            total_candidates=resp.total_candidates,
        )

    @app.get("/api/files", response_model=list[schemas.AudioFile], tags=["files"])
    def api_files() -> list[schemas.AudioFile]:
        eng = get_engine()
        with eng.pool.connection() as conn:
            files = conn.execute(
                "SELECT file_id, title, duration_sec, metadata FROM audio_files ORDER BY file_id"
            ).fetchall()
            spk = conn.execute(
                "SELECT file_id, label, role, role_confidence, display_name, talk_time, n_words "
                "FROM speakers ORDER BY file_id, label"
            ).fetchall()
            conn.rollback()
        by_file: dict[str, list[schemas.Speaker]] = {}
        for fid, label, role, conf, name, talk, nw in spk:
            by_file.setdefault(fid, []).append(
                schemas.Speaker(
                    label=label,
                    role=role,
                    role_confidence=conf,
                    display_name=name,
                    talk_time=talk,
                    n_words=nw,
                )
            )
        metrics.indexed_files.set(len(files))
        return [
            schemas.AudioFile(
                file_id=fid,
                title=title,
                duration_sec=dur,
                audio_url=f"/media/{fid}",
                topic=(meta or {}).get("topic"),
                page_url=(meta or {}).get("page_url"),
                speakers=by_file.get(fid, []),
            )
            for fid, title, dur, meta in files
        ]

    @app.get("/api/files/{file_id}/transcript", response_model=list[schemas.Utterance], tags=["files"])
    def api_transcript(file_id: str) -> list[schemas.Utterance]:
        if not FILE_ID_RE.match(file_id):
            raise HTTPException(404, "unknown file")
        with get_engine().pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, idx, speaker, start_sec, end_sec, text FROM utterances WHERE file_id = %s ORDER BY idx",
                (file_id,),
            ).fetchall()
            conn.rollback()
        if not rows:
            raise HTTPException(404, "unknown file")
        return [schemas.Utterance(id=r[0], idx=r[1], speaker=r[2], start=r[3], end=r[4], text=r[5]) for r in rows]

    @app.get("/media/{file_id}", tags=["files"], response_class=FileResponse)
    def media(file_id: str) -> FileResponse:
        if not FILE_ID_RE.match(file_id):
            raise HTTPException(404, "unknown file")
        with get_engine().pool.connection() as conn:
            row = conn.execute("SELECT audio_path FROM audio_files WHERE file_id = %s", (file_id,)).fetchone()
            conn.rollback()
        if not row:
            raise HTTPException(404, "unknown file")
        root = settings.audio_dir.resolve()
        path = (root / row[0]).resolve()
        if not path.is_relative_to(root) or not path.is_file():  # defence in depth against traversal
            raise HTTPException(404, "audio not available")
        return FileResponse(path, media_type="audio/mpeg", headers={"cache-control": "public, max-age=86400"})

    # ------------------------------------------------------------------------------------------ operations
    @app.get("/healthz", response_model=schemas.Health, tags=["ops"])
    def healthz() -> schemas.Health:
        return schemas.Health(status="ok")

    @app.get("/readyz", response_model=schemas.Health, tags=["ops"])
    def readyz() -> Response:
        eng = app.state.engine
        detail: dict[str, str] = {}
        ok = eng is not None
        detail["engine"] = "loaded" if ok else "loading"
        if eng is not None:
            try:
                with eng.pool.connection(timeout=2) as conn:
                    n = conn.execute("SELECT count(*) FROM audio_files").fetchone()
                    conn.rollback()
                detail["database"] = "ok"
                detail["indexed_files"] = str(n[0] if n else 0)
                ok = ok and bool(n and n[0] > 0)
            except Exception as e:  # pragma: no cover - exercised by ops, not unit tests
                ok = False
                detail["database"] = f"error: {type(e).__name__}"
        body = schemas.Health(status="ready" if ok else "not-ready", detail=detail)
        return JSONResponse(body.model_dump(), status_code=200 if ok else 503)

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    # ------------------------------------------------------------------------------------------------- UI
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"cache-control": "no-cache"})

    return app

"""HTTP API + one-page UI.  Run: audiosearch serve  (or uvicorn audiosearch.api:app)"""

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from . import config
from .search import Searcher

app = FastAPI(title="AudioSearch")
STATIC = Path(__file__).with_name("static")


@lru_cache(maxsize=1)
def searcher() -> Searcher:
    return Searcher.create()


@app.get("/api/search")
def api_search(
    q: str = Query(min_length=1, max_length=512),
    k: int = Query(10, ge=1, le=50),
    role: str | None = Query(None, pattern="^(host|guest)$"),
    mode: str = Query("hybrid", pattern="^(hybrid|lexical|semantic)$"),
) -> dict:
    hits = searcher().search(q, k=k, role=role, mode=mode)
    return {
        "query": q,
        "hits": [
            {
                **asdict(h),
                "timestamp": f"{int(h.start // 60):02d}:{h.start % 60:04.1f}",
                "audio_url": f"/media/{h.file_id}",
            }
            for h in hits
        ],
    }


@app.get("/api/files/{file_id}/transcript")
def transcript(file_id: str) -> list[dict]:
    with searcher().pool.connection() as conn:
        rows = conn.execute(
            "SELECT idx, speaker, start_sec, end_sec, text FROM utterances WHERE file_id = %s ORDER BY idx", (file_id,)
        ).fetchall()
    if not rows:
        raise HTTPException(404, "unknown file")
    return [dict(zip(("idx", "speaker", "start", "end", "text"), r, strict=True)) for r in rows]


@app.get("/media/{file_id}")
def media(file_id: str) -> FileResponse:
    audio_dir = (config.DATA / "audio").resolve()
    path = (audio_dir / f"{file_id}.mp3").resolve()
    if path.parent != audio_dir or not path.is_file():  # no path traversal
        raise HTTPException(404, "unknown file")
    return FileResponse(path, media_type="audio/mpeg")  # honours Range requests, so the player can seek


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")

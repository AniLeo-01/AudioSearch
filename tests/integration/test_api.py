"""HTTP contract tests (FastAPI TestClient, real Postgres, hashing embedder)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from audiosearch.api.app import create_app
from audiosearch.dataset import load_manifest
from audiosearch.db import connect, create_pool, migrate
from audiosearch.domain import Transcript
from audiosearch.indexing import Indexer
from audiosearch.search.engine import SearchEngine

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def client(db_settings, hash_embedder):
    migrate(db_settings, hash_embedder.model_name, hash_embedder.dim)
    entries = {e.file_id: e for e in load_manifest(db_settings.manifest_path, db_settings.audio_dir)}
    indexer = Indexer(db_settings, hash_embedder)  # type: ignore[arg-type]
    with connect(db_settings) as conn:
        t = Transcript.load(db_settings.transcripts_dir / "runway.json")
        indexer.index(conn, t, entries["runway"])
    pool = create_pool(db_settings)
    engine = SearchEngine(db_settings, pool, hash_embedder)  # type: ignore[arg-type]
    with TestClient(create_app(db_settings, engine)) as c:
        yield c
    pool.close()


def test_health_and_readiness(client):
    assert client.get("/healthz").json() == {"status": "ok", "detail": {}}
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json()["detail"]["indexed_files"] == "1"


def test_search_contract(client):
    r = client.get("/api/search", params={"q": "Guppy", "k": 3, "role": "guest"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "keyword" and 1 <= len(body["hits"]) <= 3
    hit = body["hits"][0]
    for key in (
        "file_id",
        "timestamp",
        "start",
        "end",
        "speaker",
        "speaker_role",
        "text",
        "highlights",
        "audio_url",
    ):
        assert key in hit
    assert hit["speaker_role"] == "guest" and hit["audio_url"] == "/media/runway"
    assert r.headers["x-request-id"]


@pytest.mark.parametrize(
    "params",
    [
        {"q": ""},
        {"q": "x", "k": 0},
        {"q": "x", "k": 999},
        {"q": "x", "mode": "fuzzy"},
        {"q": "x", "role": "pilot"},
        {"q": "role:guest"},
    ],
)
def test_search_rejects_invalid_input(client, params):
    assert client.get("/api/search", params=params).status_code == 422


def test_files_transcript_and_media(client):
    files = client.get("/api/files").json()
    assert [f["file_id"] for f in files] == ["runway"]
    assert {s["role"] for s in files[0]["speakers"]} == {"host", "guest"}
    utts = client.get("/api/files/runway/transcript").json()
    assert utts[0]["idx"] == 0 and utts[0]["start"] < utts[-1]["start"]
    r = client.get("/media/runway", headers={"Range": "bytes=0-1023"})
    assert r.status_code == 206 and len(r.content) == 1024
    assert r.headers["content-type"] == "audio/mpeg"


@pytest.mark.parametrize("path", ["/media/nope", "/media/..%2Fmanifest", "/api/files/nope/transcript"])
def test_unknown_or_malicious_paths_404(client, path):
    assert client.get(path).status_code == 404


def test_ui_and_security_headers(client):
    r = client.get("/")
    assert r.status_code == 200 and "AudioSearch" in r.text
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert client.get("/static/app.js").status_code == 200


def test_metrics_exposed(client):
    client.get("/api/search", params={"q": "T-38"})
    text = client.get("/metrics").text
    assert "audiosearch_searches_total" in text and "audiosearch_search_stage_seconds_bucket" in text

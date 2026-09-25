"""Recall@K quality gate on the held-out test split. Needs a running, indexed database."""

import statistics
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from audiosearch import config
from audiosearch.api import app
from audiosearch.evaluate import query_metrics
from audiosearch.search import Searcher

FLOORS = {"R@1": 0.60, "R@5": 0.87, "R@10": 0.92, "MRR": 0.86}  # just below the measured values


@pytest.fixture(scope="module")
def per_query():
    gold = yaml.safe_load(Path(config.DATA / "eval" / "queries.yaml").read_text())
    s = Searcher.create()
    test = [q for q in gold["queries"] if q["split"] == "test"]
    return [
        query_metrics(s.search(q["query"], k=10, role=q.get("role")), q["relevant"], gold["tolerance_sec"])
        for q in test
    ]


@pytest.mark.parametrize("metric", FLOORS)
def test_quality_floor(per_query, metric):
    assert statistics.mean(m[metric] for m in per_query) >= FLOORS[metric]


def test_api_returns_file_timestamp_speaker():
    client = TestClient(app)
    hit = client.get("/api/search", params={"q": "vestibular system", "role": "guest"}).json()["hits"][0]
    assert hit["file_id"] == "wayfinding" and hit["role"] == "guest" and hit["timestamp"].startswith("07:")
    audio = client.get(hit["audio_url"], headers={"Range": "bytes=0-99"})
    assert audio.status_code == 206 and len(audio.content) == 100
    assert client.get("/media/..%2F..%2Fetc%2Fpasswd").status_code == 404

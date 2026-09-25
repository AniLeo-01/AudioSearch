"""Retrieval-quality regression gate: Recall@K on the held-out golden TEST split.

This is the automated evaluation the problem statement asks for.  It runs the real embedding model
against the indexed corpus (``audiosearch index`` must have run - CI does this from the committed
transcripts) and asserts floors slightly below the numbers reported in docs/EVALUATION.md, so any
regression in ranking, fusion, expansion or moment localisation fails the build.
"""

from __future__ import annotations

import pytest

from audiosearch.eval.golden import load_golden
from audiosearch.eval.metrics import mean
from audiosearch.eval.runner import QUERY_SYSTEMS, EngineFactory, SystemResult, run_system

pytestmark = [pytest.mark.eval, pytest.mark.slow]

SYSTEMS = ("bm25", "dense", "hybrid-rrf", "full")

# Floors (test split).  Measured values are in docs/EVALUATION.md; floors leave ~0.03 of headroom for
# numerical noise across CPU types / library versions.
FLOOR = {
    "full": {"recall@1": 0.45, "recall@5": 0.80, "recall@10": 0.88, "mrr": 0.75},
}
KEYWORD_RECALL5_FLOOR = 0.90


@pytest.fixture(scope="module")
def results(base_settings) -> dict[str, SystemResult]:
    from audiosearch.db import connect

    try:
        with connect(base_settings) as conn:
            n = conn.execute("SELECT count(*) FROM audio_files").fetchone()[0]
    except Exception as e:  # pragma: no cover
        pytest.skip(f"database not available: {e}")
    if n == 0:
        pytest.skip("corpus not indexed; run `audiosearch index` first")
    golden = load_golden(base_settings.golden_queries_path)
    queries = golden.split("test")
    engine = EngineFactory().engine(base_settings, with_reranker=False, build=False)
    specs = {s.name: s for s in QUERY_SYSTEMS}
    out = {name: run_system(engine, queries, specs[name], golden.tolerance_sec) for name in SYSTEMS}
    engine.pool.close()
    print(f"\n\nRecall@K on the golden test split ({len(queries)} queries)")
    print(f"{'system':12} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6}")
    for name, res in out.items():
        s = res.summary()
        print(
            f"{name:12} {s['recall@1']:6.3f} {s['recall@3']:6.3f} {s['recall@5']:6.3f} "
            f"{s['recall@10']:6.3f} {s['mrr']:6.3f}"
        )
    return out


@pytest.mark.parametrize("metric", ["recall@1", "recall@5", "recall@10", "mrr"])
def test_full_system_meets_floor(results, metric):
    value = results["full"].summary()[metric]
    assert value >= FLOOR["full"][metric], f"{metric}={value:.3f} fell below {FLOOR['full'][metric]}"


def test_hybrid_is_at_least_as_good_as_each_single_channel(results):
    full = results["full"].summary()
    for single in ("bm25", "dense"):
        other = results[single].summary()
        assert full["recall@10"] >= other["recall@10"] - 1e-9, single
        assert full["mrr"] >= other["mrr"] - 0.02, single


def test_exact_terms_are_found(results):
    assert mean(results["full"].values("recall", 5, "keyword")) >= KEYWORD_RECALL5_FLOOR
    assert mean(results["full"].values("recall", 5, "phrase")) >= KEYWORD_RECALL5_FLOOR


def test_semantic_matches_beat_lexical_on_paraphrases(results):
    full = mean(results["full"].values("recall", 10, "paraphrase"))
    bm25 = mean(results["bm25"].values("recall", 10, "paraphrase"))
    assert full > bm25 + 0.2


def test_soundslike_channel_recovers_misspellings(results):
    full = mean(results["full"].values("recall", 5, "misspelled"))
    rrf = mean(results["hybrid-rrf"].values("recall", 5, "misspelled"))  # same fusion without expansion
    assert full >= rrf


def test_latency_budget(results):
    p95 = results["full"].summary()["latency_p95_ms"]
    assert p95 < 500, f"p95 latency {p95:.0f} ms exceeds the 500 ms CPU budget"

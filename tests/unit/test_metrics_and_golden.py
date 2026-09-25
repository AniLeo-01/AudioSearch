import subprocess

import pytest

from audiosearch.config import get_settings
from audiosearch.dataset import load_manifest
from audiosearch.eval.golden import CATEGORIES, RelevantMoment, load_golden
from audiosearch.eval.metrics import (
    Returned,
    bootstrap_ci,
    credit,
    hits,
    paired_permutation_pvalue,
    query_metrics,
)
from audiosearch.eval.stage_metrics import norm_tokens

GOLD = [
    RelevantMoment("f", 10.0, 20.0, 2),
    RelevantMoment("f", 100.0, 110.0, 1),
    RelevantMoment("g", 0.0, 5.0, 2),
]


def test_hit_requires_same_file_and_overlap_with_tolerance():
    g = GOLD[0]
    assert hits(Returned("f", 22.0, 30.0), g, tolerance=5.0)
    assert not hits(Returned("f", 26.0, 30.0), g, tolerance=5.0)
    assert not hits(Returned("g", 12.0, 14.0), g, tolerance=5.0)


def test_each_gold_moment_is_credited_once():
    res = [Returned("f", 12, 14), Returned("f", 15, 18), Returned("g", 1, 2)]
    assert credit(res, GOLD, 0.0) == [0, None, 2]


def test_query_metrics():
    res = [Returned("x", 0, 1), Returned("f", 12, 14), Returned("f", 101, 104), Returned("f", 13, 15)]
    m = query_metrics(res, GOLD, tolerance=0.0, ks=(1, 3, 5))
    assert m.recall == {1: 0.0, 3: pytest.approx(2 / 3), 5: pytest.approx(2 / 3)}
    assert m.success[1] == 0.0 and m.success[3] == 1.0
    assert m.precision[3] == pytest.approx(2 / 3)
    assert m.mrr == pytest.approx(0.5)
    assert m.first_hit_rank == 2 and m.offset_sec == pytest.approx(2.0)
    assert 0.0 < m.ndcg10 < 1.0
    perfect = query_metrics([Returned("f", 10, 12), Returned("g", 0, 1), Returned("f", 100, 101)], GOLD, 0.0)
    assert perfect.ndcg10 == pytest.approx(1.0) and perfect.recall[3] == 1.0


def test_bootstrap_and_permutation():
    lo, hi = bootstrap_ci([0.0, 1.0] * 20)
    assert lo < 0.5 < hi
    assert paired_permutation_pvalue([1.0] * 20, [0.0] * 20) < 0.001
    assert paired_permutation_pvalue([0.3, 0.5], [0.3, 0.5]) == 1.0


def test_stage_normalisation():
    assert norm_tokens("It's the T-38, um, two!") == ["it's", "the", "t", "38", "2"]


# -------------------------------------------------------------- dataset and golden-set contracts
@pytest.fixture(scope="module")
def manifest():
    s = get_settings()
    return {e.file_id: e for e in load_manifest(s.manifest_path, s.audio_dir)}


def test_dataset_contract(manifest):
    """Problem statement: 5-6 recordings, 8-10 minutes each, each with a unique pair of speakers."""
    assert 5 <= len(manifest) <= 6
    people = [p for e in manifest.values() for p in (e.host, e.guest)]
    assert len(people) == len(set(people)) == 2 * len(manifest)
    for e in manifest.values():
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(e.audio_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert 480.0 <= float(out.stdout) <= 600.0, e.file_id


def test_golden_set_is_valid(manifest):
    s = get_settings()
    golden = load_golden(s.golden_queries_path)
    assert len(golden.queries) >= 60
    assert {q.category for q in golden.queries} == set(CATEGORIES)
    assert len(golden.split("dev")) > 0 and len(golden.split("test")) > len(golden.split("dev"))
    for q in golden.queries:
        for r in q.relevant:
            assert r.file in manifest, q.id
            assert 0.0 <= r.start <= r.end <= 600.0
        if q.category == "speaker":
            assert q.role in ("host", "guest")

"""Retrieval metrics over time-interval relevance labels.

A returned moment *hits* a labelled relevant moment when it is in the same file and its time span
overlaps the labelled interval widened by ``tolerance`` seconds.  Each labelled moment can be credited
at most once (so returning the same region twice does not inflate recall or nDCG).

Definitions (per query; the report averages over queries):
    Recall@K     = |{labelled moments hit within the top K}| / |labelled moments|
    Success@K    = 1 if any labelled moment is hit within the top K
    Precision@K  = |{top-K results that hit some labelled moment}| / K
    MRR@K        = 1 / rank of the first hitting result (0 if none within K)
    nDCG@K       = DCG of graded gains (2 = relevant, 1 = partial) / ideal DCG
    Offset       = |result start - labelled start| for the first hit, in seconds (moment precision)
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from audiosearch.eval.golden import RelevantMoment

KS = (1, 3, 5, 10)


@dataclass(frozen=True)
class Returned:
    file: str
    start: float
    end: float


def hits(result: Returned, gold: RelevantMoment, tolerance: float) -> bool:
    return result.file == gold.file and result.start <= gold.end + tolerance and result.end >= gold.start - tolerance


def credit(results: Sequence[Returned], gold: Sequence[RelevantMoment], tolerance: float) -> list[int | None]:
    """For each result rank, the index of the (not yet credited) labelled moment it hits, else None.

    When a result overlaps several labelled moments, the highest-grade one is credited first.
    """
    credited: set[int] = set()
    out: list[int | None] = []
    for r in results:
        match = None
        for gi in sorted(range(len(gold)), key=lambda i: -gold[i].grade):
            if gi not in credited and hits(r, gold[gi], tolerance):
                match = gi
                break
        if match is not None:
            credited.add(match)
        out.append(match)
    return out


def any_hit(r: Returned, gold: Sequence[RelevantMoment], tolerance: float) -> bool:
    return any(hits(r, g, tolerance) for g in gold)


@dataclass
class QueryMetrics:
    recall: dict[int, float]
    success: dict[int, float]
    precision: dict[int, float]
    mrr: float
    ndcg10: float
    first_hit_rank: int | None
    offset_sec: float | None


def query_metrics(
    results: Sequence[Returned], gold: Sequence[RelevantMoment], tolerance: float, ks: Sequence[int] = KS
) -> QueryMetrics:
    assignment = credit(results, gold, tolerance)
    any_hits = [any_hit(r, gold, tolerance) for r in results]
    recall, success, precision = {}, {}, {}
    for k in ks:
        got = {g for g in assignment[:k] if g is not None}
        recall[k] = len(got) / len(gold)
        success[k] = 1.0 if any(any_hits[:k]) else 0.0
        precision[k] = sum(any_hits[:k]) / k
    first = next((i for i, h in enumerate(any_hits[:10]) if h), None)
    mrr = 1.0 / (first + 1) if first is not None else 0.0
    dcg = sum((2 ** gold[g].grade - 1) / math.log2(i + 2) for i, g in enumerate(assignment[:10]) if g is not None)
    ideal = sorted((g.grade for g in gold), reverse=True)[:10]
    idcg = sum((2**gr - 1) / math.log2(i + 2) for i, gr in enumerate(ideal))
    offset = None
    if first is not None:
        r = results[first]
        best = min((g for g in gold if hits(r, g, tolerance)), key=lambda g: abs(r.start - g.start))
        offset = abs(r.start - best.start)
    return QueryMetrics(
        recall=recall,
        success=success,
        precision=precision,
        mrr=mrr,
        ndcg10=dcg / idcg if idcg else 0.0,
        first_hit_rank=None if first is None else first + 1,
        offset_sec=offset,
    )


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(values: Sequence[float], n: int = 2000, alpha: float = 0.05, seed: int = 13) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean over queries."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(values[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return means[int(alpha / 2 * n)], means[min(n - 1, int((1 - alpha / 2) * n))]


def paired_permutation_pvalue(a: Sequence[float], b: Sequence[float], n: int = 5000, seed: int = 7) -> float:
    """Two-sided paired randomisation test on the mean difference (system A vs system B)."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be non-empty and of equal length")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    observed = abs(sum(diffs))
    if observed == 0:
        return 1.0
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(s) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (n + 1)

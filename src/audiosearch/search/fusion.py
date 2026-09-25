"""Rank fusion of retrieval channels.

Weighted Reciprocal Rank Fusion (Cormack et al., 2009) is the default: it needs no score calibration
between BM25 (unbounded) and cosine similarity (bounded), and is robust to outliers.  A convex
combination of min-max-normalised scores is provided for the ablation study.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Fused:
    id: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # channel -> 1-based rank
    raw: dict[str, float] = field(default_factory=dict)  # channel -> raw channel score


def weighted_rrf(
    channels: dict[str, list[tuple[str, float]]], weights: dict[str, float], k: int = 60
) -> list[Fused]:
    """score(d) = sum_c w_c / (k + rank_c(d)); ties broken by best single rank, then id."""
    fused: dict[str, Fused] = {}
    for name, ranking in channels.items():
        w = weights.get(name, 1.0)
        if w <= 0:
            continue
        for rank, (doc_id, raw) in enumerate(ranking, start=1):
            item = fused.setdefault(doc_id, Fused(doc_id, 0.0))
            item.score += w / (k + rank)
            item.ranks[name] = rank
            item.raw[name] = raw
    return sorted(fused.values(), key=lambda f: (-f.score, min(f.ranks.values()), f.id))


def convex_combination(
    channels: dict[str, list[tuple[str, float]]], weights: dict[str, float]
) -> list[Fused]:
    """score(d) = sum_c w_c * minmax_c(score_c(d)); a document missing from a channel scores 0 there."""
    fused: dict[str, Fused] = {}
    for name, ranking in channels.items():
        w = weights.get(name, 1.0)
        if w <= 0 or not ranking:
            continue
        scores = [s for _, s in ranking]
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        for rank, (doc_id, raw) in enumerate(ranking, start=1):
            item = fused.setdefault(doc_id, Fused(doc_id, 0.0))
            item.score += w * (raw - lo) / span
            item.ranks[name] = rank
            item.raw[name] = raw
    return sorted(fused.values(), key=lambda f: (-f.score, min(f.ranks.values()), f.id))

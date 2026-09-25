"""Recall@K & co. over time-interval labels: a result hits a labelled moment if same file and the
intervals overlap within +/- tolerance; each labelled moment is credited at most once."""

from __future__ import annotations

import math
import random
import statistics
import time
from pathlib import Path

import yaml

KS = (1, 3, 5, 10)
SYSTEMS = {
    "bm25": dict(mode="lexical", phonetic=False),
    "bm25+soundslike": dict(mode="lexical", phonetic=True),
    "dense": dict(mode="semantic"),
    "hybrid-rrf": dict(mode="hybrid", coverage=False, phonetic=False),
    "hybrid-rrf+coverage": dict(mode="hybrid", coverage=True, phonetic=False),
    "full": dict(mode="hybrid"),
    "full-no-snap": dict(mode="hybrid", snap=False),
    "full-no-nms": dict(mode="hybrid", nms=False),
}


def hits(r, g: dict, tol: float) -> bool:
    return r.file_id == g["file"] and r.start <= g["end"] + tol and r.end >= g["start"] - tol


def query_metrics(results: list, gold: list[dict], tol: float) -> dict[str, float]:
    credited: set[int] = set()
    assign: list[int | None] = []
    for r in results:  # credit the highest-grade uncredited labelled moment this result hits
        m = next(
            (i for i in sorted(range(len(gold)), key=lambda i: -gold[i]["grade"])
             if i not in credited and hits(r, gold[i], tol)),
            None,
        )  # fmt: skip
        if m is not None:
            credited.add(m)
        assign.append(m)
    any_hit = [any(hits(r, g, tol) for g in gold) for r in results]
    first = next((i for i, h in enumerate(any_hit[:10]) if h), None)
    grades = [g["grade"] for g in gold]
    dcg = sum((2 ** grades[a] - 1) / math.log2(i + 2) for i, a in enumerate(assign[:10]) if a is not None)
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(sorted(grades, reverse=True)[:10]))
    out = {f"R@{k}": len({a for a in assign[:k] if a is not None}) / len(gold) for k in KS}
    out["S@1"] = float(any(any_hit[:1]))
    out["S@5"] = float(any(any_hit[:5]))
    out["MRR"] = 0.0 if first is None else 1 / (first + 1)
    out["nDCG@10"] = dcg / idcg
    return out


def bootstrap_ci(xs: list[float], n: int = 2000, seed: int = 13) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def permutation_p(a: list[float], b: list[float], n: int = 5000, seed: int = 7) -> float:
    """Paired randomisation test: how often does a random sign flip give a difference this large?"""
    d = [x - y for x, y in zip(a, b, strict=True)]
    obs = abs(sum(d))
    if obs == 0:
        return 1.0
    rng = random.Random(seed)
    extreme = sum(abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= obs - 1e-12 for _ in range(n))
    return (extreme + 1) / (n + 1)


def run(searcher, golden: Path, split: str = "test", systems: dict = SYSTEMS) -> str:
    gold = yaml.safe_load(golden.read_text())
    tol = gold.get("tolerance_sec", 5.0)
    queries = [q for q in gold["queries"] if split == "all" or q["split"] == split]
    per: dict[str, list[dict]] = {}
    lat: dict[str, list[float]] = {}
    for name, opts in systems.items():
        per[name], lat[name] = [], []
        for q in queries:
            t0 = time.perf_counter()
            res = searcher.search(q["query"], k=10, role=q.get("role"), **opts)
            lat[name].append((time.perf_counter() - t0) * 1000)
            per[name].append(query_metrics(res, q["relevant"], tol))
    cols = ["R@1", "R@3", "R@5", "R@10", "MRR", "nDCG@10", "S@1", "S@5"]
    lines = [f"## Split `{split}` ({len(queries)} queries, tolerance {tol}s)", "",
             "| System | " + " | ".join(cols) + " | p50 ms |", "|---" * (len(cols) + 2) + "|"]  # fmt: skip
    for name, rows in per.items():
        vals = " | ".join(f"{statistics.mean(r[c] for r in rows):.3f}" for c in cols)
        lines.append(f"| `{name}` | {vals} | {statistics.median(lat[name]):.0f} |")
    if "full" in per:
        r5 = [r["R@5"] for r in per["full"]]
        lo, hi = bootstrap_ci(r5)
        lines += ["", f"`full` Recall@5 95% CI: [{lo:.3f}, {hi:.3f}]"]
        for base in ("bm25", "dense", "hybrid-rrf"):
            if base in per:
                p = permutation_p(r5, [r["R@5"] for r in per[base]])
                lines.append(f"`full` vs `{base}` Recall@5: p = {p:.3f}")
        cats = sorted({q["category"] for q in queries})
        lines += ["", "| Category | n | " + " | ".join(f"`{s}`" for s in per) + " |", "|---" * (len(per) + 2) + "|"]
        for cat in cats:
            idx = [i for i, q in enumerate(queries) if q["category"] == cat]
            vals = " | ".join(f"{statistics.mean(per[s][i]['R@5'] for i in idx):.3f}" for s in per)
            lines.append(f"| {cat} | {len(idx)} | {vals} |")
    return "\n".join(lines)

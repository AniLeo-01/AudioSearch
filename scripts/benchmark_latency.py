#!/usr/bin/env python
"""Search latency benchmark (run on an otherwise idle machine).

All 81 golden queries x ROUNDS, after a warm-up pass, for each mode. The query-embedding cache is
cleared before every round so embedding cost is always paid (worst case for repeated queries).
Writes reports/latency.md.

Usage: python scripts/benchmark_latency.py [--rounds 3] [--rerank]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from audiosearch.config import get_settings
from audiosearch.eval.golden import load_golden
from audiosearch.logging_setup import setup_logging
from audiosearch.search.engine import SearchOptions
from audiosearch.services import build_engine


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(p / 100 * len(xs)) - 1))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--rerank", action="store_true", help="also benchmark the cross-encoder stage")
    ap.add_argument("--out", default="reports/latency.md")
    args = ap.parse_args()
    setup_logging("WARNING")
    s = get_settings()
    engine = build_engine(s, with_reranker=args.rerank)
    queries = [(q.query, q.role) for q in load_golden(s.golden_queries_path).queries]
    configs = {
        "lexical (BM25 + sounds-like)": SearchOptions(mode="lexical"),
        "semantic (dense)": SearchOptions(mode="semantic"),
        "hybrid (default)": SearchOptions(mode="hybrid"),
    }
    if args.rerank:
        configs["hybrid + rerank"] = SearchOptions(mode="hybrid", rerank=True)
    for text, role in queries:  # warm-up: DB caches, HNSW pages, tokenizer
        engine.search(text, 10, role=role)
    rows = []
    for name, opt in configs.items():
        totals: list[float] = []
        stages: dict[str, list[float]] = defaultdict(list)
        for _ in range(args.rounds):
            engine.embedder.clear_cache()
            for text, role in queries:
                r = engine.search(text, 10, role=role, options=opt)
                totals.append(r.timings_ms["total"])
                for k, v in r.timings_ms.items():
                    if k != "total":
                        stages[k].append(v)
        stage_str = ", ".join(f"{k} {statistics.mean(v):.1f}" for k, v in stages.items())
        rows.append((name, len(totals), pct(totals, 50), pct(totals, 95), pct(totals, 99), stage_str))
    lines = [
        "# Search latency",
        "",
        f"Machine: {os.cpu_count()} vCPU, CPU-only, embedding model `{s.embedding_model}`; "
        f"{len(queries)} golden queries x {args.rounds} rounds after warm-up; k=10; query-embedding cache "
        "cleared every round.",
        "",
        "| Configuration | n | p50 ms | p95 ms | p99 ms | mean per stage (ms) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    lines += [f"| {n} | {c} | {p50:.1f} | {p95:.1f} | {p99:.1f} | {st} |" for n, c, p50, p95, p99, st in rows]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    engine.pool.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

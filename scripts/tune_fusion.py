#!/usr/bin/env python
"""Dev-split sweep over fusion hyper-parameters (never run against the test split).

Usage: python scripts/tune_fusion.py [--split dev]
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys

from audiosearch.config import get_settings
from audiosearch.eval.golden import load_golden
from audiosearch.eval.metrics import mean
from audiosearch.eval.runner import EngineFactory, SystemSpec, run_system
from audiosearch.logging_setup import setup_logging
from audiosearch.search.engine import SearchEngine, SearchOptions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev"])  # tuning is dev-only by construction
    args = ap.parse_args()
    setup_logging("WARNING")
    logging.getLogger("audiosearch").setLevel(logging.WARNING)
    base = get_settings()
    golden = load_golden(base.golden_queries_path)
    queries = golden.split(args.split)
    factory = EngineFactory()
    engine = factory.engine(base, with_reranker=False, build=False)
    cats = ["keyword", "phrase", "paraphrase", "question", "cross_file", "misspelled", "speaker"]
    print(f"{'config':44} R@5    R@10   MRR   | " + " ".join(f"{c[:5]:>5}" for c in cats))
    grid = itertools.product([60, 20, 10], [50, 20], [False, True], [False, True])
    for k, n, adaptive, coverage in grid:
        s = base.model_copy(update={"rrf_k": k, "candidates_per_channel": n})
        eng = SearchEngine(s, engine.pool, engine.embedder)
        spec = SystemSpec("x", SearchOptions(mode="hybrid", adaptive=adaptive, coverage=coverage, phonetic=True), "")
        res = run_system(eng, queries, spec, golden.tolerance_sec)
        name = f"rrf k={k:<2} n={n:<2} adaptive={int(adaptive)} coverage={int(coverage)}"
        by = " ".join(f"{mean(res.values('recall', 5, c)):5.2f}" for c in cats)
        print(
            f"{name:44} {mean(res.values('recall', 5)):.3f}  {mean(res.values('recall', 10)):.3f}  "
            f"{mean(res.values('mrr')):.3f} | {by}"
        )
    for extra in ("dense", "cc"):
        opts = SearchOptions(mode="semantic") if extra == "dense" else SearchOptions(mode="hybrid", fusion="cc")
        res = run_system(engine, queries, SystemSpec(extra, opts, ""), golden.tolerance_sec)
        by = " ".join(f"{mean(res.values('recall', 5, c)):5.2f}" for c in cats)
        print(
            f"{extra:44} {mean(res.values('recall', 5)):.3f}  {mean(res.values('recall', 10)):.3f}  "
            f"{mean(res.values('mrr')):.3f} | {by}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

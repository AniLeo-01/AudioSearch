#!/usr/bin/env python
"""Dev-split grid over index-time choices: embedding model x dialogue context x chunk size.

Each configuration is indexed into its own schema (eval_grid_*), so the serving index is untouched.
Usage: python scripts/tune_index.py
"""

from __future__ import annotations

import itertools
import logging
import sys

from audiosearch.config import get_settings
from audiosearch.eval.golden import load_golden
from audiosearch.eval.metrics import mean
from audiosearch.eval.runner import FULL, EngineFactory, SystemSpec, run_system
from audiosearch.logging_setup import setup_logging

MODELS = ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2"]
CONTEXTS = ["none", "question"]
CHUNKS = [(70, 35), (50, 25)]


def main() -> int:
    setup_logging("WARNING")
    logging.getLogger("audiosearch").setLevel(logging.WARNING)
    base = get_settings()
    golden = load_golden(base.golden_queries_path)
    queries = golden.split("dev")
    factory = EngineFactory()
    cats = ["keyword", "phrase", "paraphrase", "question", "cross_file", "misspelled", "speaker"]
    print(f"{'model':24} {'ctx':9} {'chunk':6} R@1   R@5   R@10  MRR   p50ms | " + " ".join(c[:5] for c in cats))
    for model, ctx, (tw, sw) in itertools.product(MODELS, CONTEXTS, CHUNKS):
        tag = f"grid_{model.split('/')[-1].replace('-', '_').replace('.', '_')}_{ctx.replace('+', '_')}_{tw}"
        s = base.model_copy(
            update={
                "embedding_model": model,
                "chunk_context": ctx,
                "chunk_target_words": tw,
                "chunk_stride_words": sw,
                "db_schema": f"eval_{tag}"[:60],
            }
        )
        eng = factory.engine(s, with_reranker=False)
        res = run_system(eng, queries, SystemSpec("full", FULL, ""), golden.tolerance_sec)
        eng.pool.close()
        sm = res.summary()
        by = " ".join(f"{mean(res.values('recall', 5, c)):5.2f}" for c in cats)
        print(
            f"{model.split('/')[-1]:24} {ctx:9} {tw:>3}/{sw:<2} {sm['recall@1']:.3f} {sm['recall@5']:.3f} "
            f"{sm['recall@10']:.3f} {sm['mrr']:.3f} {sm['latency_p50_ms']:5.0f} | {by}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

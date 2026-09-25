# Search latency

Machine: 4 vCPU, CPU-only, embedding model `BAAI/bge-base-en-v1.5`; 81 golden queries x 3 rounds after warm-up; k=10; query-embedding cache cleared every round.

| Configuration | n | p50 ms | p95 ms | p99 ms | mean per stage (ms) |
|---|---:|---:|---:|---:|---|
| lexical (BM25 + sounds-like) | 243 | 11.3 | 22.3 | 28.1 | lexical 5.4, fusion 0.0, moments 5.6, hydrate 0.8 |
| semantic (dense) | 243 | 46.7 | 66.2 | 74.0 | embed 36.1, dense 3.1, fusion 0.1, moments 7.5, hydrate 0.9 |
| hybrid (default) | 243 | 48.4 | 66.3 | 74.0 | lexical 4.9, embed 31.5, dense 2.7, fusion 1.3, moments 7.5, hydrate 0.8 |
| hybrid + rerank | 243 | 272.2 | 330.6 | 376.3 | lexical 5.1, embed 34.6, dense 2.9, fusion 1.3, rerank 198.7, moments 9.9, hydrate 0.9 |

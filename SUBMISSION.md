# AudioSearch: Submission

**HackerEarth 2026 · Problem Statement 1: Effective retrieval from audio transcripts**

AudioSearch finds *the moment* in two-speaker audio. A query, whether exact words, a paraphrase,
a question or a misspelled name, returns the recording, the second where the relevant sentence
starts, who said it (anonymous label, inferred host/guest role, name from metadata) and the
highlighted words. A player jumps straight there. Transcription, diarization, embeddings and
indexing all run locally on CPU, on PostgreSQL + pgvector.

| Deliverable | Where |
|---|---|
| Solution (code) | `src/audiosearch/`, run with `docker compose up` or `make setup && make index && make serve` |
| Golden dataset (6 × 9.3 min, two speakers each, unique pairs) | `data/audio/`, [`data/manifest.yaml`](data/manifest.yaml), [docs/DATASET.md](docs/DATASET.md) |
| Transcripts generated from each file | `data/transcripts/large-v3-turbo/` (and `base.en/` for the ablation) |
| Labelled query set and Recall@K tests | [`data/eval/queries.yaml`](data/eval/queries.yaml), [`tests/eval/test_retrieval_quality.py`](tests/eval/test_retrieval_quality.py) |
| Design, rationale, success criteria, achievement, limitations | this document, plus [PRD](docs/PRD.md), [TDD](docs/TDD.md), [EVALUATION](docs/EVALUATION.md) |
| Coding-agent disclosure | [docs/AGENT_COLLABORATION.md](docs/AGENT_COLLABORATION.md) (summary in §8) |

---

## 1. Results at a glance (held-out test split, 52 queries)

| System | Recall@1 | Recall@5 | Recall@10 | MRR | p95 latency |
|---|---:|---:|---:|---:|---:|
| BM25 only | 0.458 | 0.721 | 0.773 | 0.695 | 22 ms |
| Dense only | 0.620 | 0.887 | 0.929 | 0.889 | 66 ms |
| Hybrid, plain RRF (standard baseline, with our tuned k = 10) | 0.578 | 0.855 | 0.907 | 0.837 | — |
| **AudioSearch** | **0.660** | **0.915** | **0.966** | **0.908** | **66 ms** |

A relevant moment ranks **first for 87 % of queries** and in the top 5 for **96 %**. Upstream,
against NASA's human transcripts: **WER 4.3 %**, **99.9 %** word-level speaker attribution,
host/guest roles **6/6**.

## 2. Engineering design

```
audio ─► faster-whisper large-v3-turbo (int8 CPU, word timestamps, VAD) ─► re-join sub-word tokens
      ─► ECAPA speaker embeddings ─► spectral clustering (k = 2) ─► boundary-aware Viterbi (per word)
      ─► single-speaker utterances ─► turns ─► host/guest roles
      ─► 50-word sliding windows ─► bge-base embeddings (passages + utterances) + BM25 postings + spoken vocabulary
      ─► PostgreSQL 16: pgvector HNSW · GIN tsvector · pg_trgm · fuzzystrmatch

query ─► parse ("phrases", role:, file:) ─► BM25 in SQL + sounds-like expansion ║ HNSW ANN
      ─► IDF-coverage-weighted RRF ─► (optional cross-encoder) ─► moment snapping ─► temporal NMS
      ─► file · timestamp · speaker/role/name · highlighted text · per-channel ranks
```

Interfaces: a CLI, an HTTP API (OpenAPI, Range-seekable audio, health checks, Prometheus metrics) and
a web UI whose transcript panel follows playback. Operations: Docker (API and pipeline images),
docker-compose, CI (lint → types → unit → Postgres integration → Recall@K gate), idempotent
content-addressed ingestion, and migrations that refuse to reuse an index built with a different
embedding model. Details: [TDD](docs/TDD.md).

## 3. Answers to the brief's questions

### 3.1 An effective hybrid-search strategy

Run **BM25** (for exact words, names and numbers) and **dense retrieval** (for meaning) over the same
passages, and fuse them with **reciprocal rank fusion whose lexical votes are weighted by IDF
coverage**: the share of the query's informative content each passage actually matched. We added the
weighting after measuring that plain RRF was *worse than dense alone* (0.855 vs 0.887 Recall@5). BM25
hits on words like "people" and "world" outranked the right semantic match: plain RRF's paraphrase
Recall@5 was 0.667 vs 0.889 for dense. Coverage weighting fixes this (0.889) without losing exactness
(keyword and phrase Recall@5 = 1.0).

Two audio-specific additions complete it:

* **Sounds-like expansion** handles names that the ASR or the user got wrong ("apheresis" was
  transcribed as *aphoresis* and *ismoresis*; "Zubaire" for *Zubair*). It uses Metaphone,
  Double Metaphone, trigram and edit distance against the spoken vocabulary, under a lexicon-aware
  policy. Misspelled-query Recall@5: 0.52 → 0.95 for BM25; 0.81 → 0.95 in the full system.
* **Moment snapping** retrieves ~20 s passages for recall, then returns the single best
  *utterance*: one speaker, an exact start and word-level match times. It is worth +11 pts Recall@5
  (p = 0.03) over reporting the passage start.

### 3.2 Diarization strategy

Two-party conversations are a constrained problem: we **know k = 2**, and turns change at sentence
boundaries. So we fix k = 2 for spectral clustering of ECAPA window embeddings, and smooth per-word
posteriors with a Viterbi pass whose speaker-switch cost is **low after sentence ends, ASR segment
ends and pauses, and high mid-sentence**. It is fully local and needs no gated models.
Result: **0.09 % word diarization error**. For comparison, per-ASR-segment majority voting (WhisperX
style) gives 2.10 %, and uniform-cost smoothing 0.27 %. Host and guest roles are inferred from
question rate, talk share and who opens the conversation (6/6 correct). In production we would keep this
interface and swap in pyannote 3.x where a licensed token is available, adding overlap handling.

### 3.3 Database strategy

A **single PostgreSQL** holds everything: files, speakers, utterances, passages, embeddings (pgvector),
BM25 postings and a phonetic vocabulary. Every search is one connection and a handful of indexed
queries; transactional ingestion means readers never see a half-indexed file; there is only one
system to operate and back up. We implement **BM25 in plain SQL** over an incrementally maintained
postings table, because Postgres's `ts_rank` has no IDF, and BM25 extensions are missing on most
managed Postgres offerings. The design relies only on `vector`, `pg_trgm` and `fuzzystrmatch`, which
RDS, Cloud SQL, Azure, Supabase and Neon all provide.

### 3.4 Embedding strategy

Local sentence-transformers, chosen on the dev split: **bge-base-en-v1.5** (768-d). It beats bge-small
and MiniLM by 5–8 pts MRR on dev and ties e5-base. It is applied to ~50-word windows with 50 % overlap
(100-word windows lost 8 pts on dev), with each model's query/document instructions. Utterances
are also embedded, for moment snapping. We tested *dialogue-context augmentation* (prepending the
interviewer's question before embedding an answer) and **dropped it**: it did not help on dev or
test, and we report that negative result.

### 3.5 Indexing strategy, and how it scales

* **Now:** HNSW (m = 16, ef_construction = 64, ef_search ≥ 100) on passages and utterances; GIN on
  tsvectors; a lexeme → postings B-tree; trigram GIN plus Metaphone B-trees on the vocabulary.
  Per-file transactions under an advisory lock, with incremental df/N/Σdl updates (O(file), and
  tested to stay identical to a full recomputation).
* **~10k hours:** GPU ASR workers behind a queue; `halfvec` storage; partition by collection;
  pgvector ≥ 0.8 iterative scans for filtered ANN (already enabled in code); PgBouncer and read
  replicas.
* **~1M hours:** move BM25 to a dedicated engine (pg_search/Tantivy or OpenSearch with WAND);
  quantized vectors with re-scoring; shard by tenant; blue/green schemas for model upgrades.
  The transcript schema, time-interval evaluation and fusion/moment logic stay unchanged.
  ([TDD §10](docs/TDD.md#10-scaling-plan))

### 3.6 Production metrics and what makes an effective evaluation

**Evaluation (what we built):**
* Relevance labelled as **audio-time intervals**, so the same labels fairly score any ASR model,
  diarization or chunking. Segment-ID labels would break as soon as chunking changes.
* **Seven query categories**, validated by lexical overlap.
* Queries **frozen before tuning**, with a **dev/test split**.
* Bootstrap **CIs** and **paired significance tests**.
* **Ablation of every component**, and **stage-level metrics** (WER, word-level diarization error)
  against independent human transcripts.
* A **CI gate** on Recall@K.

**Production metrics we would add:** zero-result rate, CTR@k and **play-through rate** (does the
user keep listening after the jump? the best implicit relevance label for audio), reformulation
rate, time-to-first-play, per-stage p50/p95/p99 latency, ingestion lag and throughput, index
consistency checks, query OOV rate as a drift signal for new jargon, and cost per audio hour.
Ranking changes would ship through interleaved A/B tests. The golden set would grow from real
queries with independent annotators.

## 4. Why this design (rationale)

| Choice | Evidence |
|---|---|
| large-v3-turbo on CPU | Benchmarked 4 Whisper sizes on the target 4-vCPU box. turbo (RTF 0.24) beat medium (0.42) on both speed and quality; WER 4.3 %. |
| k = 2 + boundary-aware Viterbi diarization | Lowest WDER of 4 methods; no gated dependencies. |
| BM25 in SQL + pgvector in one Postgres | One system; exact IDF; portable to managed Postgres; sub-20 ms DB time. |
| IDF-coverage-weighted RRF, k = 10 | Dev sweep over k, depth, intent weights, coverage and convex combination (`reports/fusion_sweep_dev.txt`). Test: best of all fusions. |
| Sounds-like expansion, lexicon-aware | +43 pts misspelled Recall@5 for BM25. The unrestricted variant polluted ordinary words, which the policy fixes. |
| Moment snapping + NMS | Largest single ablation effect (+11 pts R@5); NMS +4 pts. |
| Reranker off by default | +0.008 MRR for +200 ms: not worth it on CPU. Available per request. |

## 5. Success criteria and level of achievement

Criteria from [PRD §9](docs/PRD.md#9-evaluation-strategy-and-success-criteria), measured on the
held-out test split unless noted.

| ID | Criterion | Target | Achieved | |
|---|---|---|---|:---:|
| SC-1 | Recall@5 / Recall@10 (default system) | ≥ 0.80 / ≥ 0.90 | **0.915 / 0.966** | ✅ |
| SC-2 | Hybrid ≥ best single channel on Recall@10 and MRR | ≥ both | R@10 0.966 vs 0.929 / 0.773; MRR 0.908 vs 0.889 / 0.695 | ✅ ¹ |
| SC-3 | Exact terms: keyword + phrase Recall@5 | ≥ 0.90 | **1.000** | ✅ |
| SC-4 | Paraphrase Recall@10; margin over BM25 | ≥ 0.80; ≥ +0.20 | **1.000**; +0.72 | ✅ |
| SC-5 | Sounds-like gain on misspelled Recall@5 | ≥ +0.10 | **+0.14** in the full system, +0.43 for BM25 | ✅ |
| SC-6 | Moment snapping gain; median start offset | > 0; ≤ 5 s | **+0.11** (p = 0.03); 0.0 s (3.5 s on base.en segmentation) | ✅ |
| SC-7 | Word-level speaker attribution; roles | ≥ 95 %; 6/6 | **99.9 %; 6/6** | ✅ |
| SC-8 | ASR WER vs human reference | ≤ 10 % | **4.3 %** | ✅ |
| SC-9 | p95 search latency, 4-core CPU, no reranker | ≤ 300 ms | **66 ms** | ✅ |
| SC-10 | CI: lint, types, unit, integration, Recall@K gate from a clean clone | green | All green on GitHub Actions ([run 36143600783](https://github.com/AniLeo-01/AudioSearch/actions/runs/36143600783)): 70 tests + 9-test quality gate | ✅ ² |

¹ The margin over dense-only is consistent across metrics but not statistically significant on 52
queries (R@5 +0.028, p = 0.38). The margins over BM25 are (p < 0.001).
² The hosted run uses a fresh Ubuntu runner, the `pgvector/pgvector:pg16` service container, and a
cold model cache. It also reproduced the Recall@K floors on different CPU hardware.

## 6. Limitations

* **Two-speaker assumption.** Diarization fixes k = 2. Overlapping speech gets a single label, and
  very short back-channels can be mis-attributed (one known case: "Thanks for having me." in `runway`
  is assigned to the host).
* **Errors neither phonetic nor semantic.** Sounds-like cannot fix "cod blood" for "cord blood".
  Only the dense channel can help, and only when the context is distinctive.
* **Dense neighbours always exist.** Name queries still pull weakly related passages at low
  ranks (e.g. "Zubaire" → "Guppy" at rank 2). A similarity floor would need per-model calibration.
* **Evaluation scale and independence.** 52 test queries give ±6-pt CIs. The labels were authored
  by the system builders (frozen before tuning, with verbatim quotes for auditing). One domain
  (NASA interviews), mostly studio audio.
* **Scale is designed, not benchmarked.** 56 minutes of audio exercises correctness, not HNSW or
  BM25 behaviour at millions of passages.
* **CPU-only ingestion** runs at ≈ 3 min per 9-minute file (ASR RTF 0.2 + diarization 0.13). That
  is fine for batch use; GPU workers are needed for volume.
* **Docker image.** The Dockerfile passes `docker build --check`, and the compose file validates.
  A full image build could not run inside the development sandbox, because its TLS-intercepting
  egress proxy is not trusted inside containers. The same steps run natively and in CI.

## 7. Novelty, in one paragraph

Most transcript-search systems stop at "BM25 + embeddings + RRF over chunks". We keep that
baseline and address what is specific to **two-person audio**:
- **Recognition errors** on exactly the rare terms people search for → a *sounds-like* channel
  over the spoken vocabulary.
- **Plain RRF rewarding filler-word matches** → *IDF-coverage* weighting.
- **Users needing a second to press play and a person to attribute** → *moment snapping* with
  word-level times, NMS and *role-aware* results and filters.
- **Turn-taking structure** → *punctuation-aware* diarization.
- **Labels that break when the pipeline changes** → *time-interval* ground truth.

Each idea is measured on held-out queries, and one idea that did not help is reported as such.

## 8. Coding-agent disclosure (summary)

The initial PRD was drafted with **ChatGPT**. **Claude Code** (Anthropic's coding agent) was then
directed with the problem statement, the draft PRD and the instruction to refine the PRD and TDD,
retain and strengthen the project's novelty, and build it production-ready. The agent was held to
measurable rules:
- Problem-statement requirements become tests.
- Every differentiator is ablated.
- The golden set is frozen before tuning; tuning happens only on dev and test is reported once.
- Upstream stages are scored against independent human transcripts.
- Deployment and CI must be real.

It worked in a cloud VM with PostgreSQL and 4 CPUs. It built the dataset, pipeline, search engine,
API/UI, tests and documents, and verified each step by running it (tests, benchmarks, evaluation
reports, browser screenshots). The full account, including a step-by-step trace and what a human
should review, is in [docs/AGENT_COLLABORATION.md](docs/AGENT_COLLABORATION.md).

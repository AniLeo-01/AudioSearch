# Evaluation

All numbers below come from the generated reports in [`reports/`](../reports) and are reproduced by
`make report` (retrieval), `audiosearch eval stages` (ASR/diarization),
`python scripts/diarization_ablation.py` and `python scripts/benchmark_latency.py`. The CI job re-checks
them on every push with floors set just below the measured values (`tests/eval/test_retrieval_quality.py`).

## 1. Summary

On the held-out **test split (52 queries, 96 labelled moments)**:

| System | R@1 | R@3 | R@5 | R@10 | MRR | nDCG@10 | Success@1 | Success@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 only | 0.458 | 0.628 | 0.721 | 0.773 | 0.695 | 0.694 | 0.635 | 0.788 |
| Dense only | 0.620 | 0.802 | 0.887 | 0.929 | 0.889 | 0.866 | 0.846 | 0.942 |
| Hybrid, plain RRF | 0.578 | 0.787 | 0.855 | 0.907 | 0.837 | 0.836 | 0.788 | 0.904 |
| **AudioSearch (default)** | **0.660** | **0.831** | **0.915** | **0.966** | **0.908** | **0.905** | **0.865** | **0.962** |

95 % bootstrap CI for the default system: Recall@5 [0.849, 0.971], MRR [0.839, 0.968].
Search latency on 4 vCPU without a GPU: **p50 48 ms, p95 66 ms**.

## 2. What makes an evaluation effective (our answer to the brief)

1. **Label what users need, in a representation that outlives the system.** A user needs a
   *place in the audio*, so relevance is a time interval in the recording, not a chunk ID. The same
   labels therefore score any ASR model, diarization, chunk size or index (§6 uses them for eight
   index variants).
2. **Stratify by query type.** A single average hides the trade-off that motivates hybrid search.
   Seven categories (keyword, phrase, paraphrase, question, cross-file, misspelled, speaker-scoped)
   show where each channel wins. A lexical-overlap check confirms each category tests what it
   claims to (paraphrase 0.07 vs keyword 0.94).
3. **Separate tuning from reporting.** The queries were frozen before tuning (commit `04a8f75`). All
   choices were made on 29 dev queries, and 52 test queries were reported once.
4. **Quantify uncertainty.** Bootstrap CIs and paired permutation tests. With ~50 queries, differences
   below ~5 points are usually not significant, and we say so.
5. **Evaluate every stage against independent ground truth.** Retrieval failures often start
   upstream, so WER and speaker attribution are measured against NASA's human transcripts, which the
   system never sees.
6. **Ablate every idea, including our own.** Each differentiator is switched off in turn. One idea,
   dialogue-context augmentation, did not help and is reported as a negative result.
7. **Gate regressions automatically.** The Recall@K floors run in CI.

## 3. Protocol

* **Corpus:** 6 recordings, 56 min, 242 passages (50-word windows, 25-word stride), 570 utterances.
* **Queries:** 81 (29 dev / 52 test), 154 labelled moments (graded 2 = relevant, 1 = partial),
  exhaustive per query. See [DATASET.md](DATASET.md).
* **Hit:** the returned moment (utterance span) is in the labelled file and overlaps the labelled
  interval ±5 s. Each labelled moment is credited once, so near-duplicate results earn nothing.
* **Metrics:** Recall@K (the brief's metric), Success@K, Precision@K, MRR@10, nDCG@10 with graded
  gains, median start offset of the first hit, and latency.
* **Statistics:** 2,000-sample percentile bootstrap over queries; two-sided paired permutation test
  (5,000 permutations) between systems.
* **Systems:** query-time options run on the serving index; index-time variants are built into
  isolated schemas named by a hash of their configuration.

## 4. Query-time ablations (test split)

| System | What changes vs the line above / default | R@5 | R@10 | MRR | Δ R@5 vs default (p) |
|---|---|---:|---:|---:|---:|
| `bm25` | lexical only | 0.721 | 0.773 | 0.695 | −0.194 (0.0006) |
| `bm25+soundslike` | + sounds-like expansion | 0.788 | 0.859 | 0.780 | −0.127 (0.011) |
| `dense` | semantic only | 0.887 | 0.929 | 0.889 | −0.028 (0.38) |
| `hybrid-cc` | BM25 + dense, min-max convex combination | 0.867 | 0.908 | 0.811 | −0.047 (0.18) |
| `hybrid-rrf` | BM25 + dense, plain RRF | 0.855 | 0.907 | 0.837 | −0.059 (0.12) |
| `hybrid-rrf+intent` | + surface-form intent weights | 0.873 | 0.904 | 0.872 | −0.042 (0.25) |
| `hybrid-rrf+coverage` | + **IDF-coverage weighting (N2)** | 0.896 | 0.946 | 0.889 | −0.019 (1.0) |
| **`full`** | + **sounds-like (N1)**: the default | **0.915** | **0.966** | **0.908** | — |
| `full+rerank` | + cross-encoder second stage | 0.899 | 0.946 | 0.916 | −0.016 (0.76) |
| `full-no-snap` | report the passage start instead of the snapped moment (N3) | 0.804 | 0.836 | 0.771 | **−0.111 (0.030)** |
| `full-no-nms` | no temporal NMS (N3) | 0.874 | 0.925 | 0.904 | **−0.041 (0.014)** |

Reading the table:

* **Hybrid beats each single channel on every metric.** vs BM25: +19 pts R@5 (p < 0.001). vs dense:
  +2.8 pts R@5, +3.7 pts R@10, +4 pts R@1. The dense margin is consistent but **not statistically
  significant** on 52 queries (p = 0.38). Dense retrieval with a good model is already strong on
  this corpus; hybrid adds exactness (keyword MRR 0.958 → 1.000) and robustness to misspellings.
* **Plain RRF is worse than dense alone** (0.855 vs 0.887 R@5). That is the failure mode that
  motivated N2: on paraphrases, plain RRF scores 0.667 R@5 vs 0.889 for dense. **Coverage weighting
  alone restores it (0.889) and lifts every metric**: +4.1 R@5, +3.9 R@10, +5.2 MRR over plain RRF.
  Adding sounds-like on top takes the total gain over plain RRF to +5.9 R@5 and +7.1 MRR (p = 0.07).
* **Moment snapping is the largest single contributor** (+11 pts R@5, +14 pts MRR, both p < 0.05).
  Reporting passage starts puts users up to ~20 s away from the relevant sentence.
* **NMS** turns duplicate hits into new moments (+4 pts R@5, p = 0.014). Without it, Precision@5
  looks higher (0.565) only because adjacent windows repeat the same moment.
* **The reranker is not worth it here**: +0.008 MRR, −0.016 R@5, for +200 ms. It stays optional.

### The fusion decision on dev (final configuration, `reports/fusion_sweep_dev.txt`)

| RRF k | depth | intent weights | coverage | R@5 | R@10 | MRR |
|---:|---:|:---:|:---:|---:|---:|---:|
| 60 | 50 | – | – | 0.802 | 0.934 | 0.795 |
| 60 | 50 | – | ✓ | 0.917 | 0.934 | 0.879 |
| 20 | 50 | – | ✓ | 0.928 | 0.957 | 0.897 |
| 10 | 50 | – | – | 0.876 | 0.934 | 0.823 |
| **10** | **50** | **–** | **✓** | **0.934** | **0.957** | **0.937** |
| 10 | 50 | ✓ | ✓ | 0.934 | 0.957 | 0.934 |
| dense only | | | | 0.899 | 0.922 | 0.874 |

Coverage weighting helps at every k (+6 to +12 pts R@5). Once it is on, intent weights add nothing.

## 5. Results by query category (test split)

Recall@5:

| Category | n | BM25 | BM25 + sounds-like | Dense | Plain RRF | **Default** | + rerank |
|---|---:|---:|---:|---:|---:|---:|---:|
| keyword | 12 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 |
| phrase | 6 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 |
| paraphrase | 9 | 0.167 | 0.222 | 0.889 | 0.667 | **0.889** | 0.778 |
| question | 10 | 0.825 | 0.825 | 0.800 | 0.825 | **0.800** | 0.800 |
| cross_file | 4 | 0.600 | 0.600 | 0.692 | 0.725 | **0.767** | 0.767 |
| misspelled | 7 | 0.524 | 0.952 | 0.810 | 0.810 | **0.952** | 1.000 |
| speaker | 4 | 0.917 | 0.917 | 0.917 | 0.917 | **0.958** | 0.917 |

MRR:

| Category | BM25 | BM25 + sounds-like | Dense | Plain RRF | **Default** | + rerank |
|---|---:|---:|---:|---:|---:|---:|
| keyword | 1.000 | 1.000 | 0.958 | 1.000 | **1.000** | 1.000 |
| phrase | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 |
| paraphrase | 0.079 | 0.174 | 0.747 | 0.407 | **0.745** | 0.794 |
| question | 0.800 | 0.800 | 0.850 | 0.900 | **0.850** | 0.850 |
| cross_file | 0.688 | 0.688 | 0.875 | 0.708 | **0.750** | 0.750 |
| misspelled | 0.381 | 0.893 | 0.857 | 0.857 | **1.000** | 1.000 |
| speaker | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 |

* **Exact terms and phrases are solved** by every system that includes BM25 (quoted phrases are strict).
* **Paraphrases need embeddings**: BM25 0.167 vs dense 0.889. The default matches dense, where
  plain RRF lost 22 pts.
* **Misspellings and ASR errors need sounds-like**: BM25 0.524 → 0.952. Example: *apheresis*,
  which the ASR wrote as *aphoresis* and *ismoresis*. In the full system, sounds-like adds +14 pts
  over the same fusion without it (0.810 → 0.952) and takes misspelled MRR to 1.000.
* **Cross-file topics are the hardest category** (0.77). They have several relevant moments per query
  and broad wording ("mentors who shaped their career").

## 6. Index-time study

**Test split** (default: bge-base, 50/25-word windows, no dialogue context, large-v3-turbo transcripts):

| Variant | R@1 | R@5 | R@10 | MRR | median offset (s) |
|---|---:|---:|---:|---:|---:|
| **default** | 0.660 | **0.915** | 0.966 | 0.908 | 0.0 |
| 25-word windows | 0.674 | 0.899 | 0.953 | 0.940 | 0.0 |
| 100-word windows | 0.646 | 0.912 | 0.976 | 0.932 | 0.0 |
| + dialogue context (question) | 0.660 | 0.912 | 0.962 | 0.909 | 0.0 |
| + question + episode title | 0.655 | 0.915 | 0.952 | 0.915 | 0.0 |
| all-MiniLM-L6-v2 (22M) | 0.615 | 0.883 | 0.969 | 0.883 | 0.0 |
| bge-small-en-v1.5 (33M) | 0.650 | 0.902 | 0.959 | 0.899 | 0.0 |
| e5-base-v2 (110M) | 0.670 | 0.909 | 0.966 | 0.935 | 1.1 |
| **base.en transcripts** (higher WER) | 0.637 | 0.883 | 0.959 | 0.893 | 3.5 |

None of the test differences is significant (all p > 0.18). The decisions were taken on dev
(`reports/evaluation_dev.md`, `reports/index_grid_dev.txt`), where the base-size models clearly
beat bge-small (MRR 0.937 vs 0.862) and 100-word windows clearly hurt (R@5 0.858 vs 0.934).
Takeaways:

* **Base-size embeddings pay off modestly.** bge-base and e5-base tie. Smaller models lose 1–3 pts
  on test and 5–8 pts on dev.
* **Dialogue-context augmentation is a negative result.** Prepending the interviewer's question to
  answer passages did not help on dev or test. In these interviews the guests restate the topic,
  and the prepended question often concerns the broad theme rather than the passage.
* **Retrieval is robust to ASR quality at this level.** Swapping in base.en transcripts costs
  ~3 pts R@5, but moment precision degrades (median offset 0 → 3.5 s) because sentence boundaries
  shift. The 0.0 s offsets of the default partly reflect that the labels were authored on the same
  segmentation.

## 7. Upstream stages (against NASA's human transcripts)

| File | ref words | WER | S / D / I | speaker attribution | host role | inter-speaker cos |
|---|---:|---:|---|---:|:---:|---:|
| ai_at_nasa | 1504 | 0.018 | 5 / 15 / 7 | 1.000 | ✓ | 0.176 |
| stem_cells | 1346 | 0.084 | 52 / 43 / 18 | 1.000 | ✓ | 0.051 |
| wayfinding | 1336 | 0.042 | 20 / 17 / 19 | 0.999 | ✓ | 0.026 |
| telling_time | 1494 | 0.044 | 13 / 33 / 20 | 1.000 | ✓ | 0.354 |
| runway | 1752 | 0.047 | 34 / 29 / 19 | 0.997 | ✓ | 0.133 |
| artemis_launch | 1674 | 0.026 | 17 / 25 / 1 | 1.000 | ✓ | 0.239 |
| **all** | 9106 | **0.043** | | **0.999** | **6/6** | |

The WER is an upper bound, since NASA's reference is clean verbatim. The highest WER, in
`stem_cells`, comes largely from medical jargon and names.

**Diarization smoothing ablation** (`reports/diarization_ablation.md`, word diarization error rate):

| Method | WDER |
|---|---:|
| raw window vote (no smoothing) | 0.11 % |
| ASR-segment majority vote (WhisperX-style) | **2.10 %** |
| Viterbi, uniform switch cost | 0.27 % |
| **Viterbi, boundary-aware switch cost (ours)** | **0.09 %** |

On clean studio interviews, even unsmoothed voting is accurate. Two points matter. First, naive
smoothing *hurts*, because it resists real turn changes. Second, the common per-segment assignment
is **23× worse**, because Whisper segments often straddle speaker changes. Making the switch cost
depend on punctuation and pauses gives the best of both.

## 8. Latency (quiet 4-vCPU VM, `reports/latency.md`)

| Configuration | p50 | p95 | p99 | Where the time goes (mean) |
|---|---:|---:|---:|---|
| lexical (BM25 + sounds-like) | 11 ms | 22 ms | 28 ms | lexical 5, moments 6 |
| semantic (dense) | 47 ms | 66 ms | 74 ms | embed 36, ANN 3, moments 8 |
| **hybrid (default)** | **48 ms** | **66 ms** | **74 ms** | embed 32, lexical 5, ANN 3, fusion 1, moments 8 |
| hybrid + rerank | 272 ms | 331 ms | 376 ms | rerank 199 |

The query embedding (bge-base on CPU) dominates. All database work together is ~15 ms.

## 9. Generalisation: dev → test

| Default system | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| dev (29 queries, tuned on) | 0.625 | 0.934 | 0.957 | 0.937 |
| test (52 queries, held out) | 0.660 | 0.915 | 0.966 | 0.908 |

The small R@5/MRR drop and slight R@1/R@10 gain are consistent with no material overfitting to dev.

## 10. Error analysis (test misses of the default system)

| Query | R@5 | What happened |
|---|---:|---|
| *How long has she worked at Kennedy Space Center?* | 0.00 | The answer, "I've been here for 30 years", never names Kennedy and needs inference across turns. The top hit is her *first visit* to Kennedy. |
| *a childhood classroom moment that inspired her* | 0.00 (first hit @8) | The host's retelling ("You always know at school…") outranks the guest's own story. Both are about it, but only the guest's is labelled. |
| *mentors who shaped their career* | 0.40 | Five labelled moments across 3 files; the top 5 cover 2. |
| *How do atomic clocks agree on what time it is?* | 0.50 | The first moment is found at rank 1; the second (the "gonculation" passage) is phrased very differently. |
| *Which aircraft does NASA fly at Ellington?* | 0.50 | The aircraft are listed in four places; two are in the top 5. |
| *why gravity matters for their research* | 0.67 | "So why does that matter?" is lexically attractive but only rhetorical. |
| *Guppie airplane* | 0.67 | Stemming already maps guppie → guppi; one of five Guppy moments missed. |
| *tell us about your background* (`role:host`) | 0.83 | Six host questions across six files; five found. |

Patterns: (1) answers that need **cross-turn inference** ("here" = Kennedy); (2) broad queries with
**many relevant moments**; (3) **rhetorical echoes** of query words. Multi-hop cues suggest turn-level
context for the reranker. Many-moment queries suggest result diversification by file.

## 11. Threats to validity

* **Small test set** (52 queries): CIs are ±6 pts. We only claim differences that are significant.
* **Builder-authored labels.** The queries and labels were written by the same party that built the
  system, from the transcripts, and frozen before tuning. An independent annotation pass would
  strengthen the claims (see [AGENT_COLLABORATION.md](AGENT_COLLABORATION.md) §4).
* **Label segmentation.** The labels are utterance-aligned on the turbo transcript, which flatters
  moment offsets of the default (0 s). The base.en variant (3.5 s) is a fairer estimate of offset
  across segmentations.
* **One domain, clean audio.** Single-domain (NASA interviews), mostly studio-quality audio.
  Diarization and WER will be worse on noisy or overlapping speech. The pipeline records an
  objective difficulty signal (inter-speaker cosine) to monitor this.
* **Corpus size.** With 242 passages, BM25 statistics are small-sample. The scale behaviour of HNSW
  and BM25 is argued in [TDD §10](TDD.md#10-scaling-plan), not measured here.

## 12. Production metrics

See [TDD §9.4](TDD.md#94-production-metrics-and-evaluation-answering-the-brief). In short: keep this
offline suite as a CI gate and grow the golden set from real queries. Online, add zero-result rate,
CTR@k, **play-through rate** (did the user keep listening after the jump?), reformulation rate,
per-stage latency percentiles, ingestion lag, and interleaved A/B tests for ranking changes.

## 13. Reproduce

```bash
make index                               # index committed transcripts
audiosearch eval run --split test --variants all --out reports
audiosearch eval run --split dev  --variants all --out reports --no-stages
python scripts/diarization_ablation.py   # needs the [diarization] extra; ~6 min on 4 vCPU
python scripts/benchmark_latency.py --rerank
python -m pytest tests/eval -q -s        # the CI quality gate
```

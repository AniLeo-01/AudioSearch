# Retrieval evaluation — split `test`

_Generated 2026-09-25 13:41 UTC by `audiosearch eval run`. A result counts as a hit when its moment overlaps a labelled interval (±5s) in the same file; each labelled moment is credited once._

## Systems

| System | R@1 | R@3 | R@5 | R@10 | MRR | nDCG@10 | P@5 | median offset (s) | p50 latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bm25` | 0.458 | 0.628 | **0.721** | 0.773 | 0.695 | 0.694 | 0.331 | 0.0 | 8 |
| `bm25+soundslike` | 0.525 | 0.685 | **0.788** | 0.859 | 0.780 | 0.776 | 0.354 | 0.0 | 10 |
| `dense` | 0.620 | 0.802 | **0.887** | 0.929 | 0.889 | 0.866 | 0.415 | 0.0 | 49 |
| `hybrid-cc` | 0.521 | 0.802 | **0.867** | 0.908 | 0.811 | 0.816 | 0.408 | 0.0 | 64 |
| `hybrid-rrf` | 0.578 | 0.787 | **0.855** | 0.907 | 0.837 | 0.836 | 0.392 | 0.0 | 66 |
| `hybrid-rrf+intent` | 0.620 | 0.787 | **0.873** | 0.904 | 0.872 | 0.857 | 0.404 | 0.0 | 58 |
| `hybrid-rrf+coverage` | 0.641 | 0.802 | **0.896** | 0.946 | 0.889 | 0.883 | 0.419 | 0.0 | 62 |
| `full` | 0.660 | 0.831 | **0.915** | 0.966 | 0.908 | 0.905 | 0.423 | 0.0 | 66 |
| `full+rerank` | 0.679 | 0.835 | **0.899** | 0.946 | 0.916 | 0.909 | 0.419 | 0.0 | 273 |
| `full-no-snap` | 0.490 | 0.696 | **0.804** | 0.836 | 0.771 | 0.753 | 0.385 | 0.0 | 52 |
| `full-no-nms` | 0.660 | 0.789 | **0.874** | 0.925 | 0.904 | 0.871 | 0.565 | 0.0 | 50 |

## Index-time variants (each vs the default `full` system)

| System | R@1 | R@3 | R@5 | R@10 | MRR | nDCG@10 | P@5 | median offset (s) | p50 latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full` | 0.660 | 0.831 | **0.915** | 0.966 | 0.908 | 0.905 | 0.423 | 0.0 | 66 |
| `variant:chunk-25w` | 0.674 | 0.831 | **0.899** | 0.953 | 0.940 | 0.911 | 0.415 | 0.0 | 46 |
| `variant:chunk-100w` | 0.646 | 0.865 | **0.912** | 0.976 | 0.932 | 0.915 | 0.373 | 0.0 | 55 |
| `variant:context-question` | 0.660 | 0.841 | **0.912** | 0.962 | 0.909 | 0.903 | 0.427 | 0.0 | 51 |
| `variant:context-question+title` | 0.655 | 0.841 | **0.915** | 0.952 | 0.915 | 0.901 | 0.427 | 0.0 | 54 |
| `variant:embed-minilm` | 0.615 | 0.829 | **0.883** | 0.969 | 0.883 | 0.884 | 0.373 | 0.0 | 25 |
| `variant:embed-bge-small` | 0.650 | 0.835 | **0.902** | 0.959 | 0.899 | 0.895 | 0.400 | 0.0 | 30 |
| `variant:embed-e5-base` | 0.670 | 0.851 | **0.909** | 0.966 | 0.935 | 0.920 | 0.412 | 1.1 | 44 |
| `variant:asr-base.en` | 0.637 | 0.824 | **0.883** | 0.959 | 0.893 | 0.895 | 0.408 | 3.5 | 47 |

## Recall@5 by query category

| Category | n | `bm25` | `bm25+soundslike` | `dense` | `hybrid-rrf` | `full` | `full+rerank` |
|---|---:|---:|---:|---:|---:|---:|---:|
| keyword | 12 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| phrase | 6 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| paraphrase | 9 | 0.167 | 0.222 | **0.889** | 0.667 | **0.889** | 0.778 |
| question | 10 | **0.825** | **0.825** | 0.800 | **0.825** | 0.800 | 0.800 |
| cross_file | 4 | 0.600 | 0.600 | 0.692 | 0.725 | **0.767** | **0.767** |
| misspelled | 7 | 0.524 | 0.952 | 0.810 | 0.810 | 0.952 | **1.000** |
| speaker | 4 | 0.917 | 0.917 | 0.917 | 0.917 | **0.958** | 0.917 |

## MRR by query category

| Category | n | `bm25` | `bm25+soundslike` | `dense` | `hybrid-rrf` | `full` | `full+rerank` |
|---|---:|---:|---:|---:|---:|---:|---:|
| keyword | 12 | **1.000** | **1.000** | 0.958 | **1.000** | **1.000** | **1.000** |
| phrase | 6 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| paraphrase | 9 | 0.079 | 0.174 | 0.747 | 0.407 | 0.745 | **0.794** |
| question | 10 | 0.800 | 0.800 | 0.850 | **0.900** | 0.850 | 0.850 |
| cross_file | 4 | 0.688 | 0.688 | **0.875** | 0.708 | 0.750 | 0.750 |
| misspelled | 7 | 0.381 | 0.893 | 0.857 | 0.857 | **1.000** | **1.000** |
| speaker | 4 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

## Uncertainty and significance (`full`)

95% bootstrap CI over queries — Recall@5: [0.849, 0.971], MRR: [0.839, 0.968].

| `full` vs | ΔR@5 | p (R@5) | ΔMRR | p (MRR) |
|---|---:|---:|---:|---:|
| `bm25` | +0.194 | 0.0006 | +0.213 | 0.0002 |
| `bm25+soundslike` | +0.127 | 0.0114 | +0.128 | 0.0020 |
| `dense` | +0.028 | 0.3773 | +0.019 | 0.7411 |
| `hybrid-cc` | +0.047 | 0.1812 | +0.097 | 0.0056 |
| `hybrid-rrf` | +0.059 | 0.1198 | +0.071 | 0.0692 |
| `hybrid-rrf+intent` | +0.042 | 0.2482 | +0.035 | 0.3471 |
| `hybrid-rrf+coverage` | +0.019 | 1.0000 | +0.019 | 1.0000 |
| `full+rerank` | +0.016 | 0.7558 | -0.008 | 0.7500 |
| `full-no-snap` | +0.111 | 0.0296 | +0.136 | 0.0050 |
| `full-no-nms` | +0.041 | 0.0140 | +0.003 | 0.5083 |
| `variant:chunk-25w` | +0.016 | 0.4965 | -0.032 | 0.1296 |
| `variant:chunk-100w` | +0.003 | 0.9354 | -0.024 | 0.4521 |
| `variant:context-question` | +0.003 | 1.0000 | -0.001 | 1.0000 |
| `variant:context-question+title` | +0.000 | 1.0000 | -0.007 | 0.7932 |
| `variant:embed-minilm` | +0.031 | 0.5249 | +0.025 | 0.4625 |
| `variant:embed-bge-small` | +0.013 | 0.7457 | +0.009 | 0.8838 |
| `variant:embed-e5-base` | +0.006 | 0.7477 | -0.027 | 0.3135 |
| `variant:asr-base.en` | +0.031 | 0.1842 | +0.015 | 0.6043 |

_p-values: two-sided paired permutation test over queries (5,000 permutations)._

## Misses of `full` (labelled moments not all found in the top 5)

| Query | Category | R@5 | first hit | top result |
|---|---|---:|---:|---|
| a childhood classroom moment that inspired her | paraphrase | 0.00 | 8 | artemis_launch 02:35.3 “You mentioned your teacher, but what was it like growing up knowing th…” |
| How do atomic clocks agree on what time it is? | question | 0.50 | 1 | telling_time 03:03.0 “On the Earth, we have all these atomic clocks, and they're all just a …” |
| Which aircraft does NASA fly at Ellington? | question | 0.50 | 2 | runway 03:38.9 “It's really cool to see, especially with our proximity to Ellington Fi…” |
| How long has she worked at Kennedy Space Center? | question | 0.00 | – | artemis_launch 04:51.5 “Well, I'll tell you, the very first time I came to Kennedy Space Cente…” |
| mentors who shaped their career | cross_file | 0.40 | 1 | runway 04:20.8 “And then have there been any mentors or key influences who helped shap…” |
| why gravity matters for their research | cross_file | 0.67 | 2 | telling_time 08:40.5 “So why does that matter?…” |
| Guppie airplane | misspelled | 0.67 | 1 | runway 08:25.3 “So if you think kind of Boeing 707, although that's a jet, it's about …” |
| tell us about your background | speaker | 0.83 | 1 | runway 00:14.1 “So can you start by telling us a little bit about your background and …” |

## Upstream stages vs NASA human transcripts

| File | ref words | WER | S / D / I | speaker attribution acc. | host role correct | inter-speaker cos |
|---|---:|---:|---|---:|:---:|---:|
| ai_at_nasa | 1504 | 0.018 | 5 / 15 / 7 | 1.000 | ✓ | 0.176 |
| stem_cells | 1346 | 0.084 | 52 / 43 / 18 | 1.000 | ✓ | 0.051 |
| wayfinding | 1336 | 0.042 | 20 / 17 / 19 | 0.999 | ✓ | 0.026 |
| telling_time | 1494 | 0.044 | 13 / 33 / 20 | 1.000 | ✓ | 0.354 |
| runway | 1752 | 0.047 | 34 / 29 / 19 | 0.997 | ✓ | 0.133 |
| artemis_launch | 1674 | 0.026 | 17 / 25 / 1 | 1.000 | ✓ | 0.239 |
| **all** | 9106 | **0.043** | | **0.999** | 6/6 | |


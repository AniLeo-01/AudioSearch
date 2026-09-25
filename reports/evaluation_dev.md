# Retrieval evaluation — split `dev`

_Generated 2026-09-25 13:42 UTC by `audiosearch eval run`. A result counts as a hit when its moment overlaps a labelled interval (±5s) in the same file; each labelled moment is credited once._

## Systems

| System | R@1 | R@3 | R@5 | R@10 | MRR | nDCG@10 | P@5 | median offset (s) | p50 latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bm25` | 0.449 | 0.624 | **0.661** | 0.713 | 0.696 | 0.665 | 0.303 | 0.0 | 8 |
| `bm25+soundslike` | 0.449 | 0.659 | **0.713** | 0.764 | 0.713 | 0.709 | 0.331 | 0.0 | 9 |
| `dense` | 0.550 | 0.797 | **0.899** | 0.922 | 0.874 | 0.845 | 0.400 | 0.0 | 44 |
| `hybrid-cc` | 0.521 | 0.768 | **0.899** | 0.928 | 0.865 | 0.832 | 0.379 | 0.0 | 45 |
| `hybrid-rrf` | 0.461 | 0.759 | **0.865** | 0.922 | 0.803 | 0.798 | 0.359 | 0.0 | 49 |
| `hybrid-rrf+intent` | 0.547 | 0.788 | **0.899** | 0.934 | 0.872 | 0.846 | 0.372 | 0.0 | 46 |
| `hybrid-rrf+coverage` | 0.616 | 0.822 | **0.922** | 0.945 | 0.920 | 0.886 | 0.407 | 0.0 | 49 |
| `full` | 0.625 | 0.843 | **0.934** | 0.957 | 0.937 | 0.906 | 0.414 | 0.0 | 46 |
| `full+rerank` | 0.579 | 0.871 | **0.908** | 0.945 | 0.908 | 0.887 | 0.407 | 0.0 | 294 |
| `full-no-snap` | 0.481 | 0.783 | **0.894** | 0.928 | 0.816 | 0.814 | 0.428 | 3.4 | 50 |
| `full-no-nms` | 0.625 | 0.732 | **0.890** | 0.945 | 0.931 | 0.872 | 0.559 | 0.0 | 49 |

## Index-time variants (each vs the default `full` system)

| System | R@1 | R@3 | R@5 | R@10 | MRR | nDCG@10 | P@5 | median offset (s) | p50 latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full` | 0.625 | 0.843 | **0.934** | 0.957 | 0.937 | 0.906 | 0.414 | 0.0 | 46 |
| `variant:chunk-25w` | 0.613 | 0.860 | **0.917** | 0.954 | 0.922 | 0.900 | 0.434 | 0.0 | 52 |
| `variant:chunk-100w` | 0.590 | 0.814 | **0.858** | 0.913 | 0.908 | 0.871 | 0.345 | 0.0 | 54 |
| `variant:context-question` | 0.613 | 0.831 | **0.922** | 0.945 | 0.915 | 0.891 | 0.414 | 0.0 | 58 |
| `variant:context-question+title` | 0.590 | 0.860 | **0.928** | 0.968 | 0.917 | 0.897 | 0.414 | 0.0 | 49 |
| `variant:embed-minilm` | 0.424 | 0.802 | **0.879** | 0.943 | 0.789 | 0.806 | 0.366 | 0.0 | 24 |
| `variant:embed-bge-small` | 0.533 | 0.825 | **0.874** | 0.937 | 0.862 | 0.857 | 0.379 | 0.0 | 31 |
| `variant:embed-e5-base` | 0.659 | 0.854 | **0.920** | 0.945 | 0.954 | 0.911 | 0.407 | 0.0 | 46 |
| `variant:asr-base.en` | 0.619 | 0.831 | **0.917** | 0.934 | 0.917 | 0.886 | 0.434 | 0.5 | 49 |

## Recall@5 by query category

| Category | n | `bm25` | `bm25+soundslike` | `dense` | `hybrid-rrf` | `full` | `full+rerank` |
|---|---:|---:|---:|---:|---:|---:|---:|
| keyword | 6 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| phrase | 4 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| paraphrase | 5 | 0.000 | 0.000 | **1.000** | 0.800 | **1.000** | **1.000** |
| question | 5 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| cross_file | 3 | 0.278 | 0.278 | 0.500 | 0.389 | **0.611** | 0.389 |
| misspelled | 4 | 0.417 | 0.792 | 0.854 | 0.854 | **0.938** | 0.875 |
| speaker | 2 | **0.833** | **0.833** | 0.583 | 0.750 | 0.750 | **0.833** |

## MRR by query category

| Category | n | `bm25` | `bm25+soundslike` | `dense` | `hybrid-rrf` | `full` | `full+rerank` |
|---|---:|---:|---:|---:|---:|---:|---:|
| keyword | 6 | **1.000** | **1.000** | 0.917 | **1.000** | **1.000** | **1.000** |
| phrase | 4 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| paraphrase | 5 | 0.033 | 0.033 | **0.900** | 0.350 | **0.900** | 0.767 |
| question | 5 | 0.867 | 0.867 | 0.750 | 0.867 | 0.867 | **0.900** |
| cross_file | 3 | 0.225 | 0.225 | **0.778** | 0.567 | **0.778** | 0.667 |
| misspelled | 4 | 0.750 | 0.875 | **1.000** | 0.875 | **1.000** | **1.000** |
| speaker | 2 | **1.000** | **1.000** | 0.625 | **1.000** | **1.000** | **1.000** |

## Uncertainty and significance (`full`)

95% bootstrap CI over queries — Recall@5: [0.876, 0.983], MRR: [0.862, 1.000].

| `full` vs | ΔR@5 | p (R@5) | ΔMRR | p (MRR) |
|---|---:|---:|---:|---:|
| `bm25` | +0.273 | 0.0020 | +0.241 | 0.0080 |
| `bm25+soundslike` | +0.221 | 0.0026 | +0.224 | 0.0080 |
| `dense` | +0.034 | 0.2517 | +0.063 | 0.1274 |
| `hybrid-cc` | +0.034 | 0.2521 | +0.072 | 0.1564 |
| `hybrid-rrf` | +0.069 | 0.1276 | +0.134 | 0.0450 |
| `hybrid-rrf+intent` | +0.034 | 0.2521 | +0.065 | 0.2791 |
| `hybrid-rrf+coverage` | +0.011 | 1.0000 | +0.017 | 1.0000 |
| `full+rerank` | +0.026 | 0.2490 | +0.029 | 0.4319 |
| `full-no-snap` | +0.040 | 0.2496 | +0.121 | 0.0298 |
| `full-no-nms` | +0.044 | 0.1344 | +0.006 | 0.4907 |
| `variant:chunk-25w` | +0.017 | 0.5003 | +0.014 | 0.8682 |
| `variant:chunk-100w` | +0.076 | 0.0618 | +0.029 | 0.6153 |
| `variant:context-question` | +0.011 | 1.0000 | +0.022 | 0.6231 |
| `variant:context-question+title` | +0.006 | 1.0000 | +0.020 | 0.6345 |
| `variant:embed-minilm` | +0.055 | 0.2753 | +0.148 | 0.0048 |
| `variant:embed-bge-small` | +0.060 | 0.1174 | +0.075 | 0.1280 |
| `variant:embed-e5-base` | +0.014 | 0.4985 | -0.017 | 1.0000 |
| `variant:asr-base.en` | +0.017 | 1.0000 | +0.020 | 0.4897 |

_p-values: two-sided paired permutation test over queries (5,000 permutations)._

## Misses of `full` (labelled moments not all found in the top 5)

| Query | Category | R@5 | first hit | top result |
|---|---|---:|---:|---|
| childhood inspiration for a career in aviation or space | cross_file | 0.67 | 3 | runway 01:18.0 “So you were that little boy who dreamed about being a pilot and grew u…” |
| military service before joining NASA | cross_file | 0.50 | 1 | telling_time 00:15.0 “Joined the Marine Corps at 17.…” |
| studying abroad for graduate school | cross_file | 0.67 | 1 | stem_cells 03:18.6 “So that journey took me through England where I did a PhD in cancer im…” |
| Gulfstrem jet | misspelled | 0.75 | 1 | runway 07:04.7 “We have one Gulfstream V aircraft that is used for our astronaut direc…” |
| thanks for having me | speaker | 0.50 | 1 | stem_cells 00:05.9 “Thank you.…” |


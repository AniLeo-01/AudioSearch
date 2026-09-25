# Diarization ablation — word-level speaker attribution accuracy vs NASA transcripts

| File | raw-window-vote | segment-majority | viterbi-uniform | viterbi-boundary |
|---|---:|---:|---:|---:|
| ai_at_nasa | 1.0000 | 0.9852 | 1.0000 | 1.0000 |
| stem_cells | 1.0000 | 1.0000 | 0.9985 | 1.0000 |
| wayfinding | 0.9985 | 0.9841 | 0.9985 | 0.9985 |
| telling_time | 0.9993 | 0.9747 | 0.9966 | 1.0000 |
| runway | 0.9971 | 0.9478 | 0.9954 | 0.9965 |
| artemis_launch | 0.9988 | 0.9891 | 0.9958 | 1.0000 |
| **all (word-weighted)** | **0.9989** | **0.9790** | **0.9973** | **0.9991** |

Word diarization error rate (1 - accuracy): raw-window-vote 0.11%, segment-majority 2.10%, viterbi-uniform 0.27%, viterbi-boundary 0.09%.

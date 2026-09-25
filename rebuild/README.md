# Rebuild kit (practice material, not part of the submitted system)

This folder supports the timed from-scratch rebuild described in [docs/REBUILD_PLAN.md](../docs/REBUILD_PLAN.md).

| Path | What it is |
|---|---|
| `preflight.py` | Run the night before: checks tools and Postgres extensions, downloads and loads every model, and benchmarks ASR speed on your machine. |
| `kit/` | A lean (~1,300-line) version of AudioSearch written from the plan. Every function was checked against the reference implementation in `src/`. With the committed golden set it gives the same test-split results (Recall@5 0.915, MRR 0.908). Use it as the answer key while you type, or to practise tonight. |

The kit keeps every problem-statement requirement plus the novelty features that carry the results
(sounds-like expansion, IDF-coverage-weighted RRF, moment snapping + temporal NMS, boundary-aware
diarization, host/guest roles, time-interval evaluation). It drops what the reference measured as
neutral or that is only operational polish: reranker, intent weights, dialogue-context
augmentation, index variants, incremental BM25 statistics, model/dimension guard and Prometheus
metrics. See the plan's section 2 for the full list.

## Practise with the kit tonight

```bash
cd rebuild/kit
cp -r ../../data .                         # manifest, audio, golden queries (transcripts get rebuilt)
rm -rf data/transcripts data/reference
docker compose up -d db                    # stop any other Postgres on port 5432 first
uv venv --python 3.11 && uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchaudio==2.5.1
uv pip install -e ".[pipeline,api,dev]"
.venv/bin/audiosearch init-db
.venv/bin/audiosearch ingest --asr-only    # ~11 min on 4 vCPU (large-v3-turbo)
.venv/bin/audiosearch ingest               # diarization + utterances + roles, ~7 min
.venv/bin/audiosearch index
.venv/bin/audiosearch search "why do clocks tick faster on Mars?"
.venv/bin/audiosearch eval                 # Recall@K table -> reports/evaluation.md
.venv/bin/pytest -q                        # unit + dataset contract + quality gate + API contract
.venv/bin/audiosearch serve                # http://localhost:8000
```

Delete this folder before submitting if you want to keep the repository focused on the system itself.

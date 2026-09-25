# 4-hour rebuild plan: AudioSearch from scratch

This is the plan for rebuilding AudioSearch by hand in a 4-hour window. It follows the same order as
the original build: environment, dataset, ASR, diarization, index, search, golden set, evaluation,
API/UI, packaging, write-up. The difference is that every decision is already made, so nothing is
tuned or explored on the day.

**The plan has been dress-rehearsed.** A lean version was written from it:
[`rebuild/kit/`](../rebuild/kit), about 1,350 lines against about 5,700 in the full repo. It was
checked against the reference implementation:

* One file (`runway`) was ingested live with the kit's own ASR and diarization. The result matched
  the reference transcript exactly: 1,718 words and start times, 100% of speaker labels, all 113
  utterances, and the host/guest names.
* Every pure function matches the reference on all six files: merging of Whisper sub-words,
  windows, boundaries, Viterbi, utterances, roles and chunks.
* On the committed golden set, the kit reproduces every test-split metric of the reference: full
  system Recall@5 **0.915**, MRR **0.908**; BM25 0.721; dense 0.887; plain RRF 0.855. This run used
  the kit's own `runway` transcript plus the other five reference transcripts converted to its format.
* Its 11 tests pass: units, dataset contract, Recall@K floors, and the API contract. The UI was
  checked in headless Chromium: clicking a result plays that moment.

Use the kit as your answer key while you type. Section 1 covers what you may bring in.

---

## 0. The whole plan on one screen

| Clock | Block | You finish with | Checkpoint |
|---|---|---|---|
| 0:00–0:15 | **B1** Skeleton + DB | repo, `pyproject`, `config`, `schema.sql`, `init-db` | `\dx` lists `vector`, `pg_trgm`, `fuzzystrmatch`; first push |
| 0:15–0:25 | **B2** Dataset | 6 excerpts + dataset contract test | `pytest tests/test_dataset.py` green |
| 0:25–0:45 | **B3** ASR | `models.py`, `asr.py`, `ingest --asr-only` | **0:45 ASR running in the background** (11–15 min) |
| 0:45–1:25 | **B4** Diarization, utterances, roles | `diarize.py`, `transcript.py`, unit tests | **1:25 `ingest` running** (7–12 min) |
| 1:25–1:40 | **B5a** Chunks, embeddings, indexer | `chunking.py`, `embed.py`, `index.py` | `index`: 242 chunks, 570 utterances |
| 1:40–2:30 | **B5b** Search | `search.py` (BM25, dense, RRF, moments, NMS, coverage, sounds-like) | 4 smoke queries return the expected moments |
| 2:30–2:55 | **B6** Golden set + evaluation | `queries.yaml`, `evaluate.py`, quality-gate test | Recall@K table (full R@5 ≈ 0.9) |
| 2:55–3:20 | **B7** API + UI | `api.py`, `static/index.html` | browser: click a result, hear the moment |
| 3:20–3:30 | **B8** Packaging | Dockerfile, compose, README quickstart (CI if time) | `docker compose config -q` |
| 3:30–3:50 | **B9** Submission | `SUBMISSION.md`, agent disclosure | every problem-statement item ticked (section 9) |
| 3:50–4:00 | Buffer | final test run, push, tag | `git push && git tag v1.0 && git push --tags` |

Three rules:

1. **Start the slow jobs early and code while they run.** ASR must start by 0:45 and diarization
   by 1:25. Everything else is fast: indexing takes about 1 minute and a full evaluation about 1 minute.
2. **Time boxes are hard.** When a box ends, drop scope using the cut ladder (section 7). Never skip
   a checkpoint.
3. **Commit at every checkpoint.** For example: `git add -A && git commit -m "B4: transcripts"`.

**Honesty rule:** the submission reports only numbers you measure tomorrow. The numbers in this
plan are there to sanity-check your build.

---

## 1. What you may prepare in advance (check the hackathon rules)

| Tier | What | Saves | When it's fine |
|---|---|---|---|
| A | Toolchain, Docker image, Python venv with the tested versions, downloaded model weights, this plan, the parameter table, reading your own design docs | ~30 min of installs and downloads | Always |
| B | `data/manifest.yaml` (episode choice, excerpt offsets, speaker names), `data/eval/queries.yaml` (the frozen golden set: 81 queries with time-interval labels), the PRD/TDD text | ~45–60 min | If pre-existing data and docs are allowed |
| C | Code: the kit or this repository | nearly everything | Only if pre-existing code is allowed |

About Tier B:

* **Golden labels survive a rebuild.** They are audio-time intervals, not segment IDs, so they stay
  valid for any transcript as long as the audio timeline is the same. `scripts/build_dataset.py`
  from the kit re-downloads and re-cuts an episode in about 7 s. The re-cut `runway` excerpt matched
  the committed one exactly: 0.0 ms lag and correlation 1.000 at three points.
* **If Tier B is not allowed**, B2 grows by about 20 min: pick 6 HWHAP episodes with 6 distinct
  hosts and guests and cut 9–9.5 min after the intro music. B6 grows by about 20 min: write about
  30 queries (section 4, B6). Take that time from the cut ladder.
* **If Tier C is allowed**, copy the kit (5 min) and spend the day on the P2 items instead:
  reranker, stage metrics (WER and speaker accuracy), CI, and the transcript panel in the UI.

---

## 2. Scope: the lean build vs the full repository

**Keep all of these. They are the requirements plus the features that carry the results:**

| | Feature | Evidence from the reference evaluation (test split) |
|---|---|---|
| P0 | 6 two-speaker files, local ASR, speaker labels, Postgres + pgvector, BM25 + dense + RRF, results with file/timestamp/speaker, Recall@K tests | the problem statement |
| P1 | **Moment snapping + temporal NMS** (N3) | +11 pts R@5 vs passage start (0.804 → 0.915); NMS +4 pts |
| P1 | **IDF-coverage-weighted RRF** (N2) | hybrid R@5 0.855 → 0.896; paraphrase 0.667 → 0.889 |
| P1 | **Sounds-like expansion** (N1) | misspelled R@5 0.810 → 0.952 in the full system (BM25 alone: 0.524 → 0.952) |
| P1 | **Host/guest roles + `role:` filter** (N4) | 6/6 roles correct |
| P1 | **Boundary-aware Viterbi diarization** (N5) | 99.91% word attribution. Plain window vote gets 99.89%, so it is a safe fallback |
| P1 | **Time-interval evaluation** with test split, CIs, permutation tests (N6) | chunking-invariant labels |

**Leave these out.** Each was neutral in the reference, or is operational polish:

| Dropped | Why it's safe |
|---|---|
| Cross-encoder reranker | Off by default in the reference (R@5 0.899 vs 0.915 without; +200 ms) |
| Intent-dependent fusion weights, convex combination | Not used in the final configuration |
| Dialogue-context augmentation of chunks | Negative result in the reference (R@5 0.912 vs 0.915) |
| Index variants (chunk size, other embedding models, base.en) | Ablations only; cite them as "prior prototype" if you mention them |
| Incremental BM25 statistics, advisory locks, per-file reindex | The kit rebuilds the index in one transaction (~1 min). Mention incremental stats as the scaling path |
| Model/dimension guard, Prometheus metrics, CSP headers | Operational polish. List them under limitations |
| WER / speaker accuracy against NASA transcripts | P2: the numbers exist from the reference build, but only report them if you re-measure |

---

## 3. Tonight: pre-flight (60–90 min)

```bash
# 1. Toolchain
uv --version && ffmpeg -version | head -1 && docker --version && git --version

# 2. Database (keep this container; tomorrow's compose file uses the same image)
docker run -d --name as-db -p 5432:5432 -e POSTGRES_USER=audiosearch -e POSTGRES_PASSWORD=audiosearch \
  -e POSTGRES_DB=audiosearch pgvector/pgvector:pg16

# 3. A ready-made venv with the tested versions (tomorrow: `uv pip install -e .` into it takes seconds)
uv venv ~/as-venv --python 3.11
uv pip install --python ~/as-venv/bin/python --index-url https://download.pytorch.org/whl/cpu \
  torch==2.5.1 torchaudio==2.5.1
uv pip install --python ~/as-venv/bin/python -c requirements.lock faster-whisper speechbrain scikit-learn \
  sentence-transformers "psycopg[binary,pool]" pgvector pyyaml typer fastapi uvicorn pytest httpx ruff

# 4. Models + ASR speed on YOUR machine (downloads ~2.3 GB the first time)
~/as-venv/bin/python rebuild/preflight.py          # from this repo; must end with READY
```

`preflight.py` checks the tools and the Postgres extensions. It loads the four models (Whisper
large-v3-turbo, Whisper base.en, ECAPA, bge-base) and times ASR on a 60 s clip. On the 4-vCPU
reference VM, turbo ran at RTF 0.19–0.26 (≈11–15 min for the 56-minute corpus) and base.en at 0.08.
**Pick the model tonight:** use turbo if its RTF is ≤ 0.35. Otherwise use `base.en`, which cost 3
points of R@5 in the reference (0.883 vs 0.915).

Then:

5. **Rehearse once with the kit** (optional, 30 min; `rebuild/README.md`). Run
   `ingest --only runway` for a quick end-to-end check (about 3–6 min for one file).
6. **Pack the Tier B folder** (if allowed): `manifest.yaml`, `eval/queries.yaml`, and optionally
   the 6 MP3s. Otherwise rely on `build_dataset.py` tomorrow, which takes about 1 min with a good
   connection.
7. **Create the empty GitHub repo** and check that `git push` authenticates. Set `git config user.name/email`.
8. **Decide in advance** whether you'll use a coding agent (section 8) and what you'll cut first (section 7).
9. Print or open sections 0, 4, 5 and 6 of this plan.

---

## 4. Blocks in detail

Line counts are the kit's (non-blank lines). They tell you how much typing each block holds.

### B1 · Skeleton + database (0:00–0:15, ~130 lines)

* `pyproject.toml` (32): hatchling; package in `src/audiosearch`; script `audiosearch = "audiosearch.cli:app"`.
  Extras: `pipeline` (faster-whisper, speechbrain, scikit-learn, torchaudio), `api` (fastapi, uvicorn),
  `dev` (pytest, httpx, ruff).
* `docker-compose.yml`: the `db` service (`pgvector/pgvector:pg16`, healthcheck). Add `api` in B8.
* `config.py` (15): copy the table in section 5.
* `schema.sql` (57): extensions `vector`, `pg_trgm`, `fuzzystrmatch`, and these tables:
  * `audio_files(file_id, title, duration_sec)`
  * `speakers(file_id, label, role, name)`
  * `utterances(file_id, idx, speaker, start_sec, end_sec, text, words jsonb, tsv GENERATED, embedding vector(768))`
  * `chunks(id, file_id, utt_start, utt_end, start_sec, end_sec, speakers text[], text, tsv GENERATED, doc_len, embedding vector(768))`,
    with an HNSW index (`vector_cosine_ops`) and a GIN index on `tsv`
  * `chunk_terms(lexeme, chunk_id, tf)`: the BM25 postings
  * `vocabulary(term, df, metaphone, dmetaphone, dmetaphone_alt)`, with a trigram GIN index
* `db.py` (12): `init_db()` runs `schema.sql` with autocommit. `connect()` calls `register_vector`.
* `cli.py`: add `init-db`.

**Checkpoint:** `audiosearch init-db`, then `psql ... -c '\dx'`. Commit, create the remote, push.
If tonight's `as-db` container is still running, either keep using it or `docker stop as-db` before
`docker compose up -d db`. Both want port 5432.

### B2 · Dataset (0:15–0:25, ~45 lines)

* Copy `data/manifest.yaml` (Tier B), then run `scripts/build_dataset.py` (29 lines). It downloads
  each episode and cuts it with `ffmpeg -ss S -t D -i ep.mp3 -ac 1 -ar 44100 -b:a 80k -af afade…`.
* `tests/test_dataset.py` (14): 5–6 files, 480–600 s each (ffprobe), and 12 distinct speaker names.
* The 6 files: `ai_at_nasa`, `stem_cells`, `wayfinding`, `telling_time`, `runway`, `artemis_launch`.
  Each is a ~9.3-minute NASA *Houston We Have a Podcast* interview with one host and one guest.

**Checkpoint:** 6 × ~5.6 MB MP3s; the contract test is green. Commit.

### B3 · ASR (0:25–0:45, ~155 lines)

* `models.py` (60): `Word(text, start, end, prob, speaker)`, `Utterance(idx, speaker, start, end, text, words)`
  with `is_question`, `Speaker(label, role, name, confidence, …)`, `Transcript` with JSON save/load,
  and `Chunk`.
* `asr.py` (49): faster-whisper with `language="en", beam_size=5, word_timestamps=True,
  vad_filter=True, vad_parameters={"min_silence_duration_ms": 500}, condition_on_previous_text=False`.
  * `merge_continuations`: Whisper splits "F-15" into "F" + "-15". Walk the segment text with a
    cursor. A token that starts exactly where the previous one ended (no space before it) is glued
    onto the previous word.
  * Keep word starts monotonic.
  * Cache `data/transcripts/<id>.asr.json` with the words and the indices of words that end an ASR segment.
* `cli.py ingest` with `--only` and `--asr-only`. Load `WhisperModel(ASR_MODEL, device="cpu",
  compute_type="int8")` once for all files.

**0:45: start** `nohup audiosearch ingest --asr-only > asr.log 2>&1 &`.
**Checkpoint:** `asr.log` shows `ai_at_nasa: 1493 words` after 2–3 min. Reference counts:
ai_at_nasa 1,493 · artemis_launch 1,648 · runway 1,718 · stem_cells 1,318 · telling_time 1,481 · wayfinding 1,338.

### B4 · Diarization, utterances, roles (0:45–1:25, ~220 lines)

`diarize.py` (120 lines), in order:

1. `speech_windows`: merge word spans into regions (gaps ≤ 0.5 s, drop regions < 0.4 s). Place
   1.5 s windows with a 0.75 s hop, plus a final window flush with each region's end.
2. `embed_windows`: `EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")`. Zero-pad
   to 1.5 s, pass relative lengths, batch 64. Audio comes from `faster_whisper.decode_audio(path, 16000)`.
3. `cluster_two`: L2-normalise and build the cosine matrix. Keep the top 30% of each row, clip at 0,
   symmetrise, set the diagonal to 1. Then `SpectralClustering(2, affinity="precomputed",
   assign_labels="cluster_qr", random_state=0)` followed by 5 spherical k-means steps.
4. Window posteriors are `softmax(10 · cos(window, centroid))`, averaged onto 0.1 s frames. A word's
   log-likelihood is the log of the mean over its frames (use the nearest covered frame if it has none).
5. `viterbi`: switch cost 4.0, multiplied by 0.15 when the previous word ends a sentence, ends an ASR
   segment, or is followed by a pause ≥ 0.6 s. *If late:* use `path = loglik.argmax(1)`. The
   reference measured 99.89% vs 99.91% word accuracy for the two.
6. The first speaker becomes `SPEAKER_00`.

`transcript.py` (56 lines):

* `ends_sentence` is abbreviation-aware (Dr., U.S., e.g., …).
* `build_utterances` starts a new utterance on a speaker change, a sentence end, a pause ≥ 1.5 s,
  or after 50 words (split at the last comma in the second half).
* `infer_roles`: host score = `3·question_rate + 1.5·(0.5 − word_share) + 0.5·[spoke first]`.
  Confidence = `logistic(4·margin)`. Attach the manifest names when confidence ≥ 0.75.

`tests/test_units.py`: Viterbi (one weak word doesn't flip; a switch lands on a boundary),
utterance splitting, and `merge_continuations`.

**1:25: start** `audiosearch ingest` (1–2 min per file, reusing the ASR cache).
**Checkpoint:** `audiosearch show runway | head` shows the host asking and the guest answering.
Utterances per file, in alphabetical order: ai_at_nasa 73 · artemis_launch 100 · runway 113 ·
stem_cells 90 · telling_time 118 · wayfinding 76 (570 in total). Roles are host/guest with the
right names. Commit, including `data/transcripts/*.json`.

### B5a · Chunks, embeddings, indexer (1:25–1:40, ~120 lines)

* `chunking.py` (34): sliding windows of whole utterances, about 50 words, advancing about 25
  words. Windows may span both speakers.
* `embed.py` (16): `SentenceTransformer("BAAI/bge-base-en-v1.5")`, `normalize_embeddings=True`.
  Queries only get the prefix `"Represent this sentence for searching relevant passages: "`.
  `lexicon()` returns the whole-word entries of the tokenizer vocabulary (~20k common English words,
  used by sounds-like).
* `index.py` (58): `TRUNCATE audio_files, vocabulary CASCADE`, then insert files, speakers,
  utterances (with embeddings) and chunks (with embeddings). Two SQL statements build BM25 straight
  from the generated tsvector:

  ```sql
  INSERT INTO chunk_terms (lexeme, chunk_id, tf)
  SELECT t.lexeme, c.id, coalesce(array_length(t.positions, 1), 1) FROM chunks c, unnest(c.tsv) t;
  UPDATE chunks c SET doc_len = x.dl
  FROM (SELECT chunk_id, sum(tf) AS dl FROM chunk_terms GROUP BY chunk_id) x WHERE c.id = x.chunk_id;
  ```

  Vocabulary = distinct content words per chunk, with `metaphone(t, 12), dmetaphone(t), dmetaphone_alt(t)`.

**Checkpoint (~1:40, once diarization finishes):** `audiosearch index` prints
`indexed 6 files: 242 chunks, 570 utterances, 5690 postings, 1323 vocabulary terms`.

### B5b · Search (1:40–2:30, ~390 lines; the heart of the build)

Build it in this order, and run `audiosearch search …` after each step:

| Step | Min | What | Test |
|---|---|---|---|
| 1 | 8 | `parse`: quoted phrases, `role:host\|guest`, intent (phrase / question / keyword / topic). `where_clause`: phrase filter via `phraseto_tsquery`, role filter via `speakers` | — |
| 2 | 8 | `analyze` (lexemes via `unnest(to_tsvector('english', q))`) + BM25 SQL (below) | `--mode lexical "cesium"` → telling_time |
| 3 | 4 | Dense: `ORDER BY embedding <=> q LIMIT 50` after `set_config('hnsw.ef_search','100',true)` | `--mode semantic` works |
| 4 | 6 | RRF (k = 10, depth 50) + hydrate hits (title, role, name). **Minimum viable search** | hybrid works |
| 5 | 12 | Moments: fetch the utterances of the candidate passages (cosine, matched lexemes, `ts_headline` with `chr(2)`/`chr(3)` markers), `snap`, `first_match_time`, NMS | results start at the right sentence, with highlights |
| 6 | 5 | IDF coverage as the lexical channel's RRF multiplier | paraphrase queries improve |
| 7 | 7 | Sounds-like `expand` + `expansion_lexemes` | `Zubaire` → stem_cells "Dr. Zubair" |

The BM25 SQL (k1 = 1.2, b = 0.75; df and N computed per query):

```sql
WITH q(lexeme, w) AS (SELECT * FROM unnest(%(lex)s::text[], %(w)s::float8[])),
     cs AS (SELECT count(*)::float8 AS n, avg(doc_len)::float8 AS avgdl FROM chunks),
     df AS (SELECT lexeme, count(*)::float8 AS df FROM chunk_terms WHERE lexeme = ANY(%(lex)s) GROUP BY lexeme)
SELECT c.id,
       sum(q.w * ln(1 + (cs.n - df.df + 0.5) / (df.df + 0.5))
           * ct.tf * 2.2 / (ct.tf + 1.2 * (0.25 + 0.75 * c.doc_len / cs.avgdl))) AS score,
       array_agg(q.lexeme) AS matched
FROM q JOIN df USING (lexeme) JOIN chunk_terms ct USING (lexeme) JOIN chunks c ON c.id = ct.chunk_id CROSS JOIN cs
WHERE {filters} GROUP BY c.id ORDER BY score DESC, c.id LIMIT %(n)s
```

IDF coverage (N2). Each lexical hit's RRF vote is multiplied by the share of the query's IDF mass it
covers:

```text
units    = the query's own lexemes;  w(u) = idf(u), or idf_max = ln(1 + (N + .5)/.5) if u is not in the corpus
credit_u = 1 if the hit contains u; 0.9·score if it contains only a sounds-like stand-in for u
cov(hit) = Σ w(u)·credit_u / Σ w(u)
rrf(d)   = Σ_channels  mult_c(d) / (10 + rank_c(d))
```

Moment snapping (N3). Candidate pool = the top `max(3k, k + 20)` passages. Within a passage:

```text
lex(u) = Σ idf(matched lexemes) · lexeme weight     sem(u) = cos(query, utterance)
score  = w·lex/max(lex) + (1−w)·minmax(sem)          (sem × 0.5 for utterances < 4 words)
w      = 0.8 for keyword/phrase queries, 0.5 otherwise, 0.3 in semantic-only mode, 0 if nothing matched
NMS    : drop a moment if an earlier one in the same file is the same utterance, starts < 10 s away,
         or its passage overlaps by more than 50 %
```

Sounds-like policy (N1). Candidates are query tokens with ≥ 4 letters:

| Token in corpus vocabulary? | Common English word? | Expansion |
|---|---|---|
| no | no (a name or typo) | spelling or phonetic neighbours, score ≥ 0.62 |
| no | yes ("inside") | only a Metaphone-identical term, score ≥ 0.80 |
| yes, rare (df ≤ 2) | no | other ASR spellings: Metaphone-identical, score ≥ 0.80 |
| otherwise | | none |

Scoring:

* `ortho = max(trigram, 1 − lev/maxlen, 0.70 if longest common substring ≥ 85% and len ≥ 5)`
* Metaphone-equal: `min(1, max(ortho, .55) + .25)`
* Double-Metaphone only (needs trigram ≥ .25): `ortho + .12`
* Keep at most 3 per token. Each expansion's lexemes get weight `0.9 × score`.

**Checkpoint (2:30).** These must hold:

| Query | Expected top hit |
|---|---|
| `vestibular system role:guest` | wayfinding **07:13.4**, Giuseppe Iaria (guest) |
| `Zubaire` | stem_cells 06:17.8, "Beautiful story, Dr. Zubair…" (sounds-like) |
| `"firing room one"` with `--role guest` | artemis_launch **07:02.5**, exactly 3 guest hits |
| `why do clocks tick faster on Mars?` | telling_time **08:19.9**, Kevin Coggins (guest) |

### B6 · Golden set + evaluation (2:30–2:55, ~130 lines)

* **Tier B:** copy `data/eval/queries.yaml` (81 queries: 52 test, 29 dev; 7 categories; ±5 s).
  **Commit it before the first eval run.**
* **No carry-over:** run `audiosearch show <file>` and write `data/eval/queries.src.yaml` with about
  30 queries, 5 per file. Cover keyword, quoted phrase, paraphrase, question, misspelled name and
  `role:` queries, and label every moment that answers each query (`runway:70-72`, `@1` = partial).
  `audiosearch label` turns the ranges into time intervals with quotes. Mark them all `split: test`;
  you won't tune anything.
* `evaluate.py` (92):
  * A hit = same file, and `r.start ≤ g.end + 5` and `r.end ≥ g.start − 5`.
  * Each labelled moment is credited once, highest grade first.
  * Metrics: R@1/3/5/10, S@1/5, MRR@10, nDCG@10 (gain 2^grade − 1).
  * Bootstrap 95% CI of the full system's R@5; paired permutation tests; a per-category table.
  * Systems: `bm25`, `bm25+soundslike`, `dense`, `hybrid-rrf`, `hybrid-rrf+coverage`, `full`,
    `full-no-snap`, `full-no-nms`.
* `tests/test_quality.py` (30): floors just below what you measure (reference floors: R@1 ≥ 0.60,
  R@5 ≥ 0.87, R@10 ≥ 0.92, MRR ≥ 0.86). Add the API contract test here in B7.

**Checkpoint:** `audiosearch eval --out reports/evaluation.md`. The whole table takes about 1 min.

### B7 · API + UI (2:55–3:20, ~140 lines)

* `api.py` (54) serves:
  * `GET /api/search?q&k&role&mode`: hits with `file_id, title, timestamp, start, end, match_time,
    speaker, role, name, text, highlights, channels, audio_url`
  * `GET /api/files/{id}/transcript`
  * `GET /media/{id}`: `FileResponse` honours HTTP Range, so the player can seek; reject paths
    that leave the audio directory
  * `GET /healthz` and `GET /` (the UI)
* `static/index.html` (85): search box, role dropdown, and result cards (timestamp · file ·
  speaker (role) · channels) with `<mark>` highlights.
  * Clicking a card sets `audio.src`, calls `audio.load()`, awaits `loadedmetadata`, then sets
    `currentTime = start − 0.3` and plays.
  * Escape every string before using `innerHTML`.
* Extend `test_quality.py` with the API contract test: fields present, `Range: bytes=0-99` → 206,
  traversal → 404.

**Checkpoint:** `audiosearch serve`, open http://localhost:8000, search, click, and hear the moment.
Take 2 screenshots for the README and submission.

### B8 · Packaging (3:20–3:30)

* `Dockerfile` (12): python:3.11-slim, CPU torch, `pip install ".[api]"`, pre-download bge-base,
  `CMD audiosearch init-db && audiosearch index && audiosearch serve`.
* Add the `api` service to compose. Validate with `docker compose config -q`. Start `docker compose
  up --build` in the background while you write docs. The build downloads CPU torch and bge-base,
  so allow several minutes.
* README: a one-paragraph pitch, the results table, quickstart (Docker and local), CLI examples, a screenshot.
* CI (`.github/workflows/ci.yml`, 24 lines) only if you are on time: pgvector service, install,
  ruff, `init-db`, `index`, pytest.

### B9 · Submission (3:30–3:50)

`SUBMISSION.md` covers everything the brief asks for:

1. **Results at a glance.** Paste the `full` / `bm25` / `dense` / `hybrid-rrf` rows plus latency.
2. **Design.** The pipeline diagram:
   `audio → faster-whisper (word timestamps) → ECAPA + spectral k=2 + boundary Viterbi → utterances → roles → 50-word chunks → bge-base + BM25 postings`;
   and at query time `query → BM25 (+ sounds-like) ‖ pgvector HNSW → IDF-coverage RRF → moment snapping → NMS`.
3. **Answers to the brief:** hybrid strategy; diarization; database; embeddings; indexing and how
   it scales (incremental BM25 stats, HNSW, partitioning by file); production metrics (latency
   p50/p95, zero-result rate, click-through on the top 3, Recall@K on a growing labelled set).
4. **Rationale:** why moments rather than passages, why coverage-weighted fusion, why local models.
5. **Success criteria and achievement.** A table with a ✅/⚠️ per criterion.
6. **Limitations.** Pre-fill these:
   * The labels were written by the builders (frozen before evaluation, with auditable quotes).
   * 56 minutes of audio, so the full-vs-dense difference is not significant (p ≈ 0.38).
   * Exactly two speakers are assumed, and overlapping speech isn't modelled.
   * English only.
   * The index is rebuilt in full.
   * On pgvector < 0.8, HNSW with a filter can return fewer results.
   * No auth or rate limiting.
   * CPU-only latency figures.
7. **Novelty.** One paragraph covering N1–N6.
8. **Coding-agent disclosure.** The tools, the prompts (paste your log), and how you verified the
   output: tests, the eval gate, browser checks.

If Tier B allows, adapt this repo's `SUBMISSION.md`, `docs/PRD.md` and `docs/TDD.md`. They describe
the same design, so update the numbers and move the dropped features into limitations or future work.

---

## 5. Parameters (fixed; there is nothing to tune tomorrow)

| Area | Setting |
|---|---|
| ASR | faster-whisper `large-v3-turbo`, CPU int8 (fallback `base.en`); `language=en`, beam 5, word timestamps, VAD (min silence 500 ms), `condition_on_previous_text=False` |
| Speaker windows | 1.5 s, hop 0.75 s, inside ASR speech regions (gap ≤ 0.5 s; skip < 0.4 s); ECAPA `speechbrain/spkrec-ecapa-voxceleb` |
| Clustering | spectral k = 2, cosine affinity pruned to the top 30% per row, `cluster_qr`, `random_state=0`, 5 k-means refinement steps |
| Posteriors | softmax(10 × cosine to centroids), 0.1 s frames |
| Smoothing | Viterbi over words, switch cost 4.0, × 0.15 at sentence end / ASR segment end / pause ≥ 0.6 s |
| Utterances | split on speaker change, sentence end (abbreviation-aware), pause ≥ 1.5 s, > 50 words (last comma in the second half) |
| Roles | 3·question rate + 1.5·(0.5 − word share) + 0.5·[spoke first]; names if logistic(4·margin) ≥ 0.75 |
| Chunks | ~50 words, stride ~25, whole utterances, no context augmentation |
| Embeddings | `BAAI/bge-base-en-v1.5`, 768-d, normalised; the BGE query prefix goes on queries only |
| Vector index | pgvector HNSW `vector_cosine_ops` (default m 16, ef_construction 64), `hnsw.ef_search` 100 |
| Lexical | Postgres `english` tsvector; BM25 k1 1.2, b 0.75 over `chunk_terms` |
| Fusion | RRF k = 10, 50 candidates per channel, equal channel weights, lexical votes × IDF coverage |
| Moments | pool max(3k, k + 20); snap weights 0.8 / 0.5 / 0.3; NMS 10 s, 50% passage overlap |
| Evaluation | ±5 s tolerance; each labelled moment credited once; label mentions < 10 s apart merged |

## 6. Expected numbers (sanity checks)

| Stage | Reference / dress rehearsal |
|---|---|
| ASR | 6 files, 56.0 min of audio, 8,996 words. Turbo: 11.1 min at RTF 0.19–0.21 (this VM measured 0.26 later) |
| Diarization | ~70–75 s per file in the original run (7.3 min in total), up to ~2 min per file on a busier VM; roles 6/6; word attribution 99.9% (vs NASA transcripts) |
| Index | 242 chunks (38/43/51/37/40/33), 570 utterances, 5,690 postings, 1,323 vocabulary terms; about 1 min |
| Test split (52 queries, 96 moments) | full R@1 0.660 · R@5 **0.915** · R@10 0.966 · MRR **0.908** · nDCG@10 0.905 · S@1 0.865 · S@5 0.962 |
| Baselines (test) | bm25 R@5 0.721 · bm25+soundslike 0.788 · dense 0.887 · hybrid-rrf 0.855 · hybrid-rrf+coverage 0.896 · full-no-snap 0.804 · full-no-nms 0.874 |
| Dev split (29 queries) | full R@5 0.934, MRR 0.937 |
| Significance (test) | full R@5 95% CI [0.849, 0.971]; full vs bm25 p = 0.001; full vs hybrid-rrf p = 0.12; full vs dense p = 0.38 |
| Latency (4 vCPU) | full p50 ≈ 40–50 ms (query embedding ≈ 30 ms of that); bm25 ≈ 5 ms |

If full-system R@5 comes out below about 0.85, check these first (section 7):

* the BGE query prefix
* `normalize_embeddings`
* OR (not AND) lexical queries
* coverage applied to lexical votes only
* the snapping pool size
* NMS applied *after* snapping

## 7. Gotchas from the first build, and the cut ladder

**Gotchas.** Every one of these cost time in the original build:

1. Whisper splits tokens ("F" + "-15"): merge them using the segment text (B3).
2. `condition_on_previous_text=True` can make Whisper loop on long audio.
3. pyannote needs a gated Hugging Face token. ECAPA + spectral clustering needs no token.
4. SpeechBrain 1.x: import from `speechbrain.inference.speaker`. Its `torch.load` FutureWarnings are harmless.
5. `register_vector(conn)` fails until `CREATE EXTENSION vector` has run, so run `init-db` first.
6. `vector(768)` must match the model. Changing the model means dropping the tables.
7. In psycopg SQL with parameters, pg_trgm's `%` operator must be written `%%`.
8. `metaphone()` needs a length argument: `metaphone(term, 12)`.
9. Don't use `plainto_tsquery` or `websearch_to_tsquery` for ranking; they AND the terms. Analyze
   the query into lexemes and OR them. In tsquery text, quote the lexemes (`'clock' | 'tick'`) so
   Postgres doesn't stem them twice.
10. Plain RRF lost to dense-only on paraphrases because BM25 matched filler words. Coverage
    multipliers + k = 10 fixed it.
11. Unrestricted sounds-like expansion rewrote ordinary words ("inside" → "insights"). The lexicon policy fixed it.
12. BGE: the query prefix goes on queries only; normalise the vectors; use cosine ops.
13. `set_config('hnsw.ef_search', …, true)` is transaction-local. Run it in the same
    transaction as the query.
14. FastAPI: don't put `from __future__ import annotations` in the API module. With `Query(...)`
    defaults it caused a `PydanticUserError`.
15. Audio won't seek before its metadata loads. Use `preload="metadata"`, call `load()`, await
    `loadedmetadata`, then set `currentTime`. Serve audio with Range support.
16. `scrollIntoView` scrolls the whole page. Use `container.scrollTo` for a transcript panel.
17. YAML reads `2026-05-29` as a date, which `json.dumps` can't serialise. Quote dates or call `.isoformat()`.
18. Never `pkill -f <pattern>` from a shell whose own command line contains the pattern; it can kill
    your shell. Stop servers with `fuser -k 8000/tcp`.
19. The Snowball stemmer maps "guppie" and "guppy" to the same stem. Use truly unknown strings in
    misspelling tests.
20. Query-embedding caches flatter latency. Clear them before benchmarking.
21. ruff's SIM905 rewrites `"""…""".split()` word lists. Add `# noqa: SIM905`.
22. Time-interval labels need an identical audio timeline. Re-cut with the same offsets and ffmpeg
    command, never with different trimming.

**Cut ladder.** When you're behind, drop items in this order; each saves 5–15 min:

1. CI workflow (keep local `pytest`)
2. Docker API image (keep compose `db` + local run; note it in the limitations)
3. Transcript panel and other UI polish (keep search + play)
4. Ablation systems beyond `bm25`, `dense`, `hybrid-rrf`, `full`
5. Bootstrap CI and permutation tests
6. The Viterbi, replaced by `argmax` (−0.02 pts of word accuracy)
7. Sounds-like expansion (misspelled R@5 0.95 → 0.81)
8. IDF coverage; last resort (plain RRF measured R@5 0.855 vs 0.915)

**Never cut:**

* the dataset contract
* local transcripts with speakers
* BM25 + dense + RRF
* file/timestamp/speaker in results
* the Recall@K test
* `SUBMISSION.md` with limitations and agent disclosure
* the pushed repo

**Recovery rules:**

* ASR not started by 1:00? Switch to `base.en` (about 4 min for the corpus).
* Search not working at 2:30? Ship steps 1–5 of B5b and move on.
* Stop coding at 3:30, whatever state you're in.

## 8. If you use a coding agent (allowed by the brief; disclose it)

Type the algorithmic core yourself (B4, B5b); that's what you'll be asked about. Delegate
boilerplate: API, UI, Docker, CI, docs. Keep a running `docs/AGENT_LOG.md` with each prompt and a
one-line outcome; section 8 of the submission quotes it. Prompts that worked:

* **B1:** "Create a Python 3.11 package `audiosearch` (src layout, hatchling, typer CLI
  `audiosearch`). Add `schema.sql` for Postgres 16 + pgvector with these tables: …(paste B1)…,
  `db.py` with `init_db()`/`connect()` (register_vector), and a compose file with
  `pgvector/pgvector:pg16`."
* **B4:** "Implement `diarize.py` exactly as specified: …(paste B4 steps 1–6 and section 5
  rows)…. Add unit tests for the Viterbi on synthetic log-likelihoods."
* **B5b:** "Implement `search.py` in the order of the B5b table. After each step, run
  `audiosearch search` with the test query from that row and show me the output."
* **B7:** "FastAPI app with these endpoints: …(paste B7)…, plus a single-file HTML/JS UI with a
  search box, role dropdown and result cards; clicking a card seeks an `<audio>` element after
  `loadedmetadata`. Escape all strings. Add an HTTP contract test."
* **B9:** "Draft `SUBMISSION.md` with these 8 sections, using the numbers in
  `reports/evaluation.md` only. Don't invent numbers."

Always check agent output against section 6 and run the tests before committing.

## 9. Submission checklist (the problem statement, item by item)

- [ ] Golden dataset: 5–6 files, 8–10 min each, a unique pair of two speakers per file (`tests/test_dataset.py`)
- [ ] Hybrid keyword + semantic search (BM25 + pgvector + RRF), plus the sounds-like channel
- [ ] Transcripts generated from each file, locally (`data/transcripts/*.json`)
- [ ] Results show file, timestamp and speaker (CLI, API, UI)
- [ ] Embeddings and indexing are local; Python + an RDBMS with embeddings (Postgres + pgvector)
- [ ] Automated tests measuring Recall@K on a labelled query set (`tests/test_quality.py`, `reports/evaluation.md`)
- [ ] Submission document: design, rationale, success criteria, achievement, limitations (`SUBMISSION.md`)
- [ ] Coding-agent disclosure, including how the agent was directed (`SUBMISSION.md` section 8 + `docs/AGENT_LOG.md`)
- [ ] GitHub repo with code, golden dataset (manifest, audio, queries) and tests, pushed and tagged

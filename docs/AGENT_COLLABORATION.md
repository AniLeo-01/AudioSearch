# Coding-Agent Disclosure

The submission format requires us to disclose coding-agent use and explain how we directed the
agent. This document covers the tools we used, the exact direction given, how the work was checked,
and a summarised trace of the session.

## 1. Tools

| Tool | Used for |
|---|---|
| **ChatGPT** | First draft of the PRD (`Audio_Search_PRD.md`, v0.1): goals, user stories, BM25 + embeddings + RRF baseline, Recall@K evaluation outline. |
| **Claude Code** (Anthropic's coding agent, cloud session with a Linux VM, PostgreSQL 16 and 4 vCPUs) | Refining the PRD, writing the TDD, building the dataset, implementing and testing the system, running the evaluation, writing these documents. |

No other agents or code generators were used. All models the system *runs* (Whisper, ECAPA, BGE,
MiniLM cross-encoder) are open-weight models used locally. None of them was used to write or label
anything.

## 2. How the agent was directed

### 2.1 The brief

The agent received two attachments, the **problem statement** and the **ChatGPT PRD draft**, and
this instruction:

> "Given the problem statement, start refining the PRD (as provided by ChatGPT) and the TDD for the
> project. Also make sure that the novelty of the project retains! Build it production ready as well
> so that I win in the hackathon!"

### 2.2 Constraints the agent was held to

The agent was required to work to the problem statement and the refined PRD. The constraints below
turned those documents into checkable rules; each is enforced in the repository:

1. **Problem-statement requirements are the contract.** They became tests where possible: the
   dataset contract test (5–6 files, 8–10 min, unique speaker pairs), result fields (file,
   timestamp, speaker) in the HTTP contract tests, and Recall@K in the CI gate. The traceability
   matrix is in PRD §13.
2. **Novelty must be measurable.** Every differentiator in PRD §5 has a hypothesis and an acceptance
   metric, and is ablated on the same labels. Features that did not help are reported as negative
   results, not hidden.
3. **No tuning on the test set.** The golden queries were written and committed *before* any
   retrieval tuning (commit `04a8f75`). All parameter choices were made on the 29-query dev split
   (`scripts/tune_fusion.py`, `scripts/tune_index.py`). The 52 test queries were evaluated once at
   the end.
4. **Independent ground truth for upstream stages.** ASR and diarization are scored against NASA's
   human transcripts, which the retrieval system never reads.
5. **Production means operable.** One-command deploy, migrations with a model-mismatch guard,
   idempotent ingestion, observability, security headers, CI.

### 2.3 How the agent verified its own work

* **Tests first-class:** 70 unit, integration and HTTP-contract tests plus the retrieval-quality gate,
  run after each substantive change.
* **Real data, real services:** every claim in the docs comes from a command that ran in the session
  (ASR benchmarks, stage metrics, sweeps, final evaluation). Reports are regenerated with
  `make report`.
* **UI checked in a real browser:** headless Chromium screenshots (`docs/img/`). They caught two
  real bugs, both fixed: the audio never seeked because `preload="none"` suppressed metadata, and
  transcript auto-scroll moved the whole page.
* **Evidence-driven debugging:** e.g. plain RRF was *worse* than dense-only on dev paraphrases
  (0.20 vs 1.00 Recall@5). Inspecting the failures led to the IDF-coverage weighting, which was then
  kept because the sweep confirmed it (§4 below).

## 3. Summarised trace

| Step | What the agent did | Outcome / evidence |
|---|---|---|
| 1 | Read the brief and draft PRD; probed the environment (Postgres 16, pgvector available, 4 vCPU, no GPU, HF reachable) | plan with 8 tracked tasks |
| 2 | Surveyed candidate public-domain two-speaker sources; scanned 200 NASA HWHAP episodes and parsed their human transcripts; selected 6 episodes with 6 distinct hosts and guests and diverse topics | `data/manifest.yaml`, `scripts/build_dataset.py` |
| 3 | Found the conversation start after the intro music with a fast ASR pass; cut 9.3-min excerpts on sentence boundaries | `data/audio/*.mp3` |
| 4 | Benchmarked Whisper variants on the VM (RTF: base.en 0.09, small.en 0.20, medium.en 0.42, **turbo 0.24**) and picked large-v3-turbo | TDD §4.2 |
| 5 | Implemented the pipeline: ASR, diarization (ECAPA + spectral + boundary-aware Viterbi, chosen because pyannote needs a gated token), alignment, roles, chunking. Measured **WER 4.3 %** and **99.9 %** speaker attribution; found and fixed Whisper's split hyphenated tokens ("F -15") | `audiosearch eval stages` |
| 6 | Implemented the Postgres schema (BM25 postings, pgvector HNSW, spoken vocabulary), the indexer with incremental statistics, and the search engine (BM25, dense, RRF, sounds-like, moments, NMS) | `src/audiosearch/**` |
| 7 | Read all six transcripts and authored 81 labelled queries (7 categories, time-interval labels, dev/test split); validated categories by lexical overlap (paraphrase 0.07 vs keyword 0.94); **committed the frozen golden set before tuning** | `04a8f75` |
| 8 | Fixed a generic bug seen before tuning: unrestricted sounds-like expansion ("inside" → "insights"). The fix is a lexicon-aware policy | `search/phonetic.py` |
| 9 | Dev evaluation: plain hybrid RRF < dense-only. Diagnosed BM25 filler-word matches dominating fusion and designed **IDF-coverage-weighted RRF**. The sweep over k, depth, intent weights and coverage confirmed it (dev R@5 0.74 → 0.86 with the configuration at the time; 0.80 → 0.93 when re-run on the final configuration) | `scripts/tune_fusion.py` |
| 10 | Merged gold moments whose ±tolerance windows overlap (a uniform rule, applied before any test-split run) | `scripts/label_helper.py` |
| 11 | Wrote unit, integration and API tests (70 passing), the FastAPI service, the web UI, Docker, compose, CI | `tests/`, `Dockerfile`, `.github/workflows/ci.yml` |
| 12 | Index-time study on dev: embedding model × dialogue context × chunk size. Adopted bge-base, 50/25 windows, **no dialogue context** (a negative result for an idea the agent itself had proposed, reported in the PRD and TDD) | `scripts/tune_index.py` |
| 13 | Diarization smoothing ablation; final test-split evaluation (all systems + index variants + stages); set CI floors just below the measured values | `reports/`, `docs/EVALUATION.md` |
| 14 | Wrote the refined PRD, TDD, dataset card, evaluation report, this disclosure, the submission document and the README | `docs/`, `SUBMISSION.md` |

## 4. Where human judgement matters (and what to review)

* **Relevance labels.** The agent authored the golden set by reading the transcripts. Protocols
  limit the bias: labels were frozen before tuning, labelling was exhaustive, and time intervals come
  with verbatim quotes that make them auditable. The authors are still the same party as the system
  builders. Before relying on the numbers, a human should spot-check `data/eval/queries.yaml`
  (every label carries its quote) and, ideally, add queries written by someone who has not seen the
  system.
* **Dataset licensing.** NASA material is generally public domain in the US. Confirm this fits the
  hackathon's rules.
* **Design ownership.** The PRD/TDD record *why* each choice was made, with measurements, so the team
  can defend or change any decision.

*(Team: add your own review notes and any further prompts you gave the agent below.)*

### Team review log

| Date | Reviewer | What was reviewed / changed |
|---|---|---|
| | | |

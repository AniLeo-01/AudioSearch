# Golden Dataset Card — `nasa-hwhap-two-speaker-v1`

Six 9.2–9.4-minute excerpts from NASA's *Houston We Have a Podcast* (HWHAP). Each excerpt contains
exactly two speakers, a host and a guest, and **no person appears in more than one file** (6 distinct hosts, 6 distinct
guests). The dataset also ships NASA's human transcripts as an independent reference and a labelled
query set of 81 queries for retrieval evaluation.

| file_id | Episode (date) | Host | Guest | Topic | Words | Turns | Host ? rate | Inter-speaker cos* |
|---|---|---|---|---|---:|---:|---:|---:|
| `ai_at_nasa` | #424 (2026-05-29) | Nilufar Ramji | Kevin Murphy | AI, machine learning & data science at NASA | 1,493 | 11 | 0.45 | 0.18 |
| `stem_cells` | #432 (2026-08-07) | Gary Jordan | Abba Zubair | Stem cells in microgravity, transfusion medicine | 1,318 | 6 | 0.06 | 0.05 |
| `wayfinding` | #427 (2026-06-26) | Leah Cheshier | Giuseppe Iaria | Neuroscience of spatial orientation | 1,338 | 12 | 0.50 | 0.03 |
| `telling_time` | #419 (2026-04-24) | Dane Turner | Kevin Coggins | Timekeeping & navigation on the Moon and Mars | 1,481 | 28 | 0.32 | 0.35 |
| `runway` | #389 (2025-06-13) | Courtney Beasley | David Johnson | Ellington Field & NASA aircraft operations | 1,718 | 33 | 0.48 | 0.13 |
| `artemis_launch` | #401 (2025-09-12) | Joseph Zakrzewski | Charlie Blackwell-Thompson | Career of the Artemis launch director | 1,648 | 10 | 0.35 | 0.24 |

\* Cosine similarity between the two speakers' mean ECAPA voice embeddings. It is an objective
proxy for how hard diarization is (higher = the voices sound more alike). We report this instead of
inferring any demographic attributes of the speakers.

Total: **56.0 minutes**, 8,996 words, 570 utterances, 100 speaker turns.

## Why these recordings (what makes the golden set "effective")

1. **Two speakers, verified.** We scanned 200 HWHAP episodes (`scripts/`, see *Construction*) and
   parsed NASA's published transcripts. We kept only episodes whose transcript has exactly one host
   and one guest. The excerpt starts at the first conversational turn after the intro music, so no
   narrated intro or archive audio clips are included.
2. **Unique speaker pairs.** HWHAP rotates hosts, so we could pick six episodes with six different hosts
   *and* six different guests. That rules out any shortcut where one voice spans several files.
3. **Topical diversity, with deliberate confusers.** The six topics are distinct (AI, regenerative
   medicine, neuroscience, physics of time, aviation, launch operations), but they share vocabulary
   in ways that make retrieval hard:
   * *navigation*: brain wayfinding in `wayfinding` vs GPS/clocks in `telling_time`
   * *gravity*: stem-cell growth, the vestibular system, and clock rates
   * *space station*, *Artemis* and *Moon* come up in several files
   * *career-path* stories appear in every interview
4. **Realistic conversational audio.** These are studio and remote podcast recordings with
   back-channels ("Yeah.", "Wow."), false starts and run-on answers. Turn structure varies from
   6 turns (a monologue-heavy guest) to 33 turns (a rapid back-and-forth).
5. **Independent human reference.** NASA's clean-verbatim transcripts let us measure ASR word error
   rate and diarization accuracy (see `audiosearch eval stages`). The retrieval system never reads them.
6. **Redistributable.** NASA audio and transcripts are US Government works, generally not subject to
   copyright in the US, so the audio can be committed to the repository and anyone can reproduce the
   results.

## Construction (reproducible)

```bash
python scripts/build_dataset.py        # downloads episodes, cuts excerpts, scrapes transcripts
```

* Offsets in `data/manifest.yaml` record exactly where each excerpt sits in the source episode
  (`excerpt.start`/`end`, seconds). Cuts were placed on sentence boundaries found with a fast ASR
  pass: the start is the host's first conversational line, and the end is the sentence end with a pause
  closest to +9.3 minutes.
* Encoding: mono, 44.1 kHz, 80 kbps MP3 (~5.6 MB per file). Every browser can play it, and ffmpeg
  decodes it to 16 kHz for the models.
* Speaker names in the manifest come from NASA's transcripts. The system never identifies voices. It
  only maps the diarization clusters it infers as *host* and *guest* to these metadata names.

## Layout

```
data/
  manifest.yaml                    provenance, offsets, host/guest names, topics
  audio/<file_id>.mp3              the six excerpts
  reference/<file_id>.json         NASA human transcript (full episode, speaker turns)
  transcripts/large-v3-turbo/      pipeline output used for retrieval
      <file_id>.asr.json           raw ASR: segments + word timestamps + probabilities
      <file_id>.json               canonical transcript: words (+speaker, confidence), utterances, turns, roles
  transcripts/base.en/             same, from a smaller ASR model (for the ASR-quality ablation)
  eval/queries.src.yaml            human-authored labels (utterance ranges)
  eval/queries.yaml                frozen golden set (audio-time intervals + verbatim quotes)
```

## Golden query set

81 queries across 7 categories. The split is **29 dev** (used for all tuning) and **52 test** (held out
and reported once). There are 154 labelled relevant moments; 35 queries have several relevant
moments and 9 span several recordings.

| Category | What it probes | Example | n (dev/test) | Mean lexical overlap† |
|---|---|---|---:|---:|
| `keyword` | a rare term said verbatim | `cesium`, `WB-57`, `Nigeria` | 18 (6/12) | 0.94 |
| `phrase` | quoted exact phrase | `"firing room one"` | 10 (4/6) | 1.00 |
| `paraphrase` | same meaning, different words | *people who get lost even in their own neighborhood* | 14 (5/9) | **0.07** |
| `question` | natural-language question | *Why do clocks on Mars run faster?* | 15 (5/10) | 0.58 |
| `cross_file` | topic discussed in several files | *mentors who shaped their career* | 7 (3/4) | 0.50 |
| `misspelled` | typos and **real ASR errors** | `apheresis` (ASR wrote *aphoresis*, *ismoresis*), `Pizzamiglio` (*Pizzamilio*), `Zubaire` | 11 (4/7) | 0.41 |
| `speaker` | role-scoped search | `Guppy` + `role:host` | 6 (2/4) | 1.00 |

† Fraction of the query's stemmed content words that appear in its relevant moments. It confirms
that each category tests what it claims to test (paraphrases share almost no words with their
answers; keywords share almost all of them).

### Labelling protocol

1. The queries were written by reading the transcripts **before any retrieval tuning**. The golden set
   was committed (`04a8f75`) before the first evaluation run.
2. Labels are **exhaustive**: every moment in the corpus that answers the query is listed. Partial
   relevance is graded 1 (default 2).
3. Relevance is stored as **audio-time intervals**, not segment or chunk IDs. Annotators reference
   utterance ranges in `queries.src.yaml`, and `scripts/label_helper.py` resolves them to times plus
   verbatim quotes. The same labels can therefore score any ASR model, diarization or chunking. (The
   draft PRD's segment-ID labels would silently break as soon as chunking changed.)
4. Same-file moments whose ±5 s tolerance windows overlap are merged (a uniform rule). Two mentions
   a few seconds apart are one place in the audio for a listener.
5. Speaker-scoped queries are labelled with the **true** speaker from NASA's transcript, even where our
   diarization is wrong. Diarization errors therefore show up as retrieval misses; they are not hidden.

### Known quirks (kept on purpose — they are realistic)

* ASR errors on rare names and jargon: "Abizubair" (Abba Zubair), "Eardia" (Iaria),
  "Pizzamilio" (Pizzamiglio), "ismoresis"/"aphoresis" (apheresis), "cod blood" (cord blood),
  "LCORNS" (LCRNS).
* One diarization error that matters for a query. In `runway`, the guest's "Thanks for having me."
  (00:04.5) is attributed to the host, so query `s04` cannot reach 100%.
* NASA's reference is clean verbatim (fillers and false starts removed), so the reported WER
  *overstates* the true ASR error.

## License and ethics

The audio and transcripts come from NASA. They are US Government works, generally not subject to copyright in the US
([NASA media guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/)). Their use here
implies no NASA endorsement. The recordings are public, professionally produced interviews. We store
no additional personal data and infer no demographic attributes.

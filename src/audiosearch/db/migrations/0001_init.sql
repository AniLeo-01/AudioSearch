-- 0001: core schema for conversation-aware hybrid audio search.
-- __EMBEDDING_DIM__ is rendered from the configured embedding model at migration time; the model
-- name/dimension are recorded in index_meta and verified on every start-up (fail fast on mismatch).

CREATE TABLE IF NOT EXISTS index_meta (
    key   text PRIMARY KEY,
    value text NOT NULL
);

-- One row per ingested recording (provenance + pipeline parameters for reproducibility).
CREATE TABLE IF NOT EXISTS audio_files (
    file_id          text PRIMARY KEY,
    title            text NOT NULL,
    audio_path       text NOT NULL,
    sha256           text NOT NULL,
    duration_sec     double precision NOT NULL CHECK (duration_sec > 0),
    sample_rate      integer,
    channels         integer,
    metadata         jsonb NOT NULL DEFAULT '{}'::jsonb,
    transcript_meta  jsonb NOT NULL DEFAULT '{}'::jsonb,
    index_signature  text NOT NULL,          -- hash(transcript, chunking config, embedding model)
    indexed_at       timestamptz NOT NULL DEFAULT now()
);

-- Anonymous diarization clusters with inferred conversational roles.
CREATE TABLE IF NOT EXISTS speakers (
    file_id          text NOT NULL REFERENCES audio_files(file_id) ON DELETE CASCADE,
    label            text NOT NULL,                     -- SPEAKER_00, SPEAKER_01
    role             text NOT NULL DEFAULT 'unknown',   -- host | guest | unknown
    role_confidence  real NOT NULL DEFAULT 0,
    display_name     text,
    talk_time        real NOT NULL DEFAULT 0,
    n_words          integer NOT NULL DEFAULT 0,
    question_rate    real NOT NULL DEFAULT 0,
    PRIMARY KEY (file_id, label)
);

-- Sentence-level, single-speaker units: the granularity at which results are localised.
CREATE TABLE IF NOT EXISTS utterances (
    id          text PRIMARY KEY,
    file_id     text NOT NULL REFERENCES audio_files(file_id) ON DELETE CASCADE,
    idx         integer NOT NULL,
    speaker     text NOT NULL,
    start_sec   double precision NOT NULL,
    end_sec     double precision NOT NULL CHECK (end_sec >= start_sec),
    text        text NOT NULL,
    words       jsonb NOT NULL,                         -- [[token, start, end], ...]
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding   vector(__EMBEDDING_DIM__),
    UNIQUE (file_id, idx)
);

-- Retrieval passages: windows of consecutive utterances (may span both speakers).
CREATE TABLE IF NOT EXISTS chunks (
    id          text PRIMARY KEY,
    file_id     text NOT NULL REFERENCES audio_files(file_id) ON DELETE CASCADE,
    idx         integer NOT NULL,
    start_sec   double precision NOT NULL,
    end_sec     double precision NOT NULL CHECK (end_sec >= start_sec),
    utt_start   integer NOT NULL,                       -- inclusive utterance idx
    utt_end     integer NOT NULL,                       -- exclusive utterance idx
    speakers    text[] NOT NULL,
    text        text NOT NULL,                          -- verbatim transcript (lexical index)
    embed_text  text NOT NULL,                          -- dialogue-context-augmented text (dense index)
    n_words     integer NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    doc_len     integer NOT NULL DEFAULT 0,             -- BM25 document length (lexeme occurrences)
    embedding   vector(__EMBEDDING_DIM__),
    UNIQUE (file_id, idx)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS utterances_embedding_hnsw ON utterances
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS utterances_tsv_gin ON utterances USING gin (tsv);

-- BM25 inverted index maintained incrementally at ingest time (portable: no pg_search needed).
CREATE TABLE IF NOT EXISTS chunk_terms (
    lexeme    text NOT NULL,
    chunk_id  text NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    tf        integer NOT NULL CHECK (tf > 0),
    PRIMARY KEY (lexeme, chunk_id)
);
CREATE INDEX IF NOT EXISTS chunk_terms_chunk ON chunk_terms (chunk_id);

CREATE TABLE IF NOT EXISTS term_stats (
    lexeme  text PRIMARY KEY,
    df      integer NOT NULL CHECK (df >= 0)
);

CREATE TABLE IF NOT EXISTS corpus_stats (
    id          integer PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    n_docs      bigint NOT NULL DEFAULT 0,
    sum_doc_len bigint NOT NULL DEFAULT 0,
    updated_at  timestamptz NOT NULL DEFAULT now()
);
INSERT INTO corpus_stats (id) VALUES (1) ON CONFLICT (id) DO NOTHING;

-- Spoken vocabulary for "sounds-like" (phonetic + trigram) query expansion.
CREATE TABLE IF NOT EXISTS vocabulary (
    term            text PRIMARY KEY,                   -- lower-cased surface form as transcribed
    df              integer NOT NULL CHECK (df >= 0),   -- number of chunks containing the term
    metaphone       text NOT NULL,
    dmetaphone      text NOT NULL,
    dmetaphone_alt  text NOT NULL
);
CREATE INDEX IF NOT EXISTS vocabulary_trgm ON vocabulary USING gin (term gin_trgm_ops);
CREATE INDEX IF NOT EXISTS vocabulary_metaphone ON vocabulary (metaphone);
CREATE INDEX IF NOT EXISTS vocabulary_dmetaphone ON vocabulary (dmetaphone);

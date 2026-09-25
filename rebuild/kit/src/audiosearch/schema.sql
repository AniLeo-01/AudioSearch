CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;

CREATE TABLE IF NOT EXISTS audio_files (
    file_id      text PRIMARY KEY,
    title        text NOT NULL,
    duration_sec float8 NOT NULL
);

CREATE TABLE IF NOT EXISTS speakers (
    file_id text REFERENCES audio_files ON DELETE CASCADE,
    label   text,                           -- SPEAKER_00 / SPEAKER_01
    role    text NOT NULL,                  -- host | guest
    name    text,
    PRIMARY KEY (file_id, label)
);

CREATE TABLE IF NOT EXISTS utterances (       -- the unit a result snaps to: one speaker, one sentence
    file_id   text REFERENCES audio_files ON DELETE CASCADE,
    idx       int,
    speaker   text NOT NULL,
    start_sec float8 NOT NULL,
    end_sec   float8 NOT NULL,
    text      text NOT NULL,
    words     jsonb NOT NULL,                 -- [[token, start, end], ...]
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding vector(768),
    PRIMARY KEY (file_id, idx)
);

CREATE TABLE IF NOT EXISTS chunks (           -- the unit we retrieve: ~50-word windows of utterances
    id        text PRIMARY KEY,
    file_id   text REFERENCES audio_files ON DELETE CASCADE,
    utt_start int NOT NULL,                   -- inclusive
    utt_end   int NOT NULL,                   -- exclusive
    start_sec float8 NOT NULL,
    end_sec   float8 NOT NULL,
    speakers  text[] NOT NULL,
    text      text NOT NULL,
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    doc_len   int NOT NULL DEFAULT 0,         -- BM25 document length
    embedding vector(768)
);
CREATE INDEX IF NOT EXISTS chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING gin (tsv);

CREATE TABLE IF NOT EXISTS chunk_terms (      -- BM25 postings
    lexeme   text,
    chunk_id text REFERENCES chunks ON DELETE CASCADE,
    tf       int NOT NULL,
    PRIMARY KEY (lexeme, chunk_id)
);

CREATE TABLE IF NOT EXISTS vocabulary (       -- spoken words, for sounds-like expansion
    term           text PRIMARY KEY,
    df             int NOT NULL,
    metaphone      text NOT NULL,
    dmetaphone     text NOT NULL,
    dmetaphone_alt text NOT NULL
);
CREATE INDEX IF NOT EXISTS vocabulary_trgm ON vocabulary USING gin (term gin_trgm_ops);
CREATE INDEX IF NOT EXISTS vocabulary_metaphone ON vocabulary (metaphone);

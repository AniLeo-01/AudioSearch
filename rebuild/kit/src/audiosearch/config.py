"""Every knob in one place. Values are the ones tuned on the dev split in the reference build."""

import os
from pathlib import Path

DATA = Path(os.getenv("AUDIOSEARCH_DATA", "data"))
DATABASE_URL = os.getenv("AUDIOSEARCH_DATABASE_URL", "postgresql://audiosearch:audiosearch@localhost:5432/audiosearch")

ASR_MODEL = os.getenv("AUDIOSEARCH_ASR_MODEL", "large-v3-turbo")  # fallback: base.en
EMBED_MODEL = "BAAI/bge-base-en-v1.5"  # 768-d: must match vector(768) in schema.sql

DIAR_WINDOW, DIAR_HOP = 1.5, 0.75  # ECAPA windows (s)
SWITCH_PENALTY, BOUNDARY_DISCOUNT = 4.0, 0.15  # Viterbi speaker-change costs
CHUNK_WORDS, CHUNK_STRIDE = 50, 25  # passage windows over whole utterances
DEPTH = 50  # candidates per retrieval channel
RRF_K = 10
EF_SEARCH = 100
NMS_GAP = 10.0  # seconds
TOLERANCE = 5.0  # eval: a hit may be this far outside the labelled interval

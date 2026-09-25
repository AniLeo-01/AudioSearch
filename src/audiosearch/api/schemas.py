"""Public HTTP contract (versioned via the /api prefix; additive changes only)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Expansion(BaseModel):
    source: str = Field(description="query token as typed")
    term: str = Field(description="spoken term it was matched to")
    score: float
    kind: str = Field(description="phonetic | spelling")


class SearchHit(BaseModel):
    rank: int
    file_id: str
    file_title: str
    audio_url: str
    start: float = Field(description="moment start (s) - where playback should begin")
    end: float
    match_time: float = Field(description="time of the first matching word (s)")
    timestamp: str = Field(description="start formatted as MM:SS.s")
    speaker: str = Field(description="anonymous diarization label, e.g. SPEAKER_01")
    speaker_role: str = Field(description="host | guest | unknown (inferred)")
    speaker_name: str | None = Field(description="display name from dataset metadata, if known")
    text: str = Field(description="the utterance that best matches the query")
    highlights: list[tuple[int, int]] = Field(description="[start, end) character spans of matched terms")
    passage_text: str
    passage_start: float
    passage_end: float
    score: float
    channels: dict[str, int] = Field(description="1-based rank of this passage in each retrieval channel")


class SearchResponse(BaseModel):
    query: str
    mode: str
    intent: str
    weights: dict[str, float]
    expansions: list[Expansion]
    hits: list[SearchHit]
    timings_ms: dict[str, float]
    total_candidates: int


class Speaker(BaseModel):
    label: str
    role: str
    role_confidence: float
    display_name: str | None
    talk_time: float
    n_words: int


class AudioFile(BaseModel):
    file_id: str
    title: str
    duration_sec: float
    audio_url: str
    topic: str | None = None
    page_url: str | None = None
    speakers: list[Speaker]


class Utterance(BaseModel):
    id: str
    idx: int
    speaker: str
    start: float
    end: float
    text: str


class Health(BaseModel):
    status: str
    detail: dict[str, str] = {}

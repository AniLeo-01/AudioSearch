"""Core domain objects shared by the ingestion pipeline, the indexer and the search engine.

Time is always expressed in seconds relative to the start of the audio file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TRANSCRIPT_SCHEMA_VERSION = 1


@dataclass(slots=True)
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0
    speaker: str | None = None
    speaker_conf: float | None = None  # posterior of the assigned speaker (diarization confidence)

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"w": self.text, "s": round(self.start, 3), "e": round(self.end, 3)}
        d["p"] = round(self.prob, 3)
        if self.speaker is not None:
            d["spk"] = self.speaker
        if self.speaker_conf is not None:
            d["sc"] = round(self.speaker_conf, 3)
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Word:
        return cls(d["w"], d["s"], d["e"], d.get("p", 1.0), d.get("spk"), d.get("sc"))


@dataclass(slots=True)
class Utterance:
    """A sentence-like unit spoken by exactly one speaker (the unit of moment localisation)."""

    id: str
    idx: int
    speaker: str
    start: float
    end: float
    text: str
    word_start: int  # inclusive index into Transcript.words
    word_end: int  # exclusive

    @property
    def is_question(self) -> bool:
        return self.text.rstrip().endswith("?")


@dataclass(slots=True)
class Turn:
    """A maximal run of consecutive utterances by the same speaker."""

    idx: int
    speaker: str
    start: float
    end: float
    utterance_start: int  # inclusive
    utterance_end: int  # exclusive


@dataclass(slots=True)
class SpeakerProfile:
    label: str  # anonymous diarization label, e.g. SPEAKER_00
    role: str = "unknown"  # host | guest | unknown  (inferred, see pipeline/roles.py)
    role_confidence: float = 0.0
    display_name: str | None = None  # optional, from dataset metadata (never from voice identity)
    talk_time: float = 0.0
    n_words: int = 0
    n_utterances: int = 0
    question_rate: float = 0.0


@dataclass
class Transcript:
    """Canonical, speaker-attributed, word-timed transcript of one audio file."""

    file_id: str
    duration: float
    words: list[Word]
    utterances: list[Utterance]
    turns: list[Turn]
    speakers: list[SpeakerProfile]
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- helpers -----------------------------------------------------------------------------------
    def speaker(self, label: str) -> SpeakerProfile | None:
        return next((s for s in self.speakers if s.label == label), None)

    def text(self) -> str:
        return " ".join(u.text for u in self.utterances)

    # ---- (de)serialisation ------------------------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": TRANSCRIPT_SCHEMA_VERSION,
            "file_id": self.file_id,
            "duration": round(self.duration, 3),
            "meta": self.meta,
            "speakers": [asdict(s) for s in self.speakers],
            "utterances": [{**asdict(u), "start": round(u.start, 3), "end": round(u.end, 3)} for u in self.utterances],
            "turns": [{**asdict(t), "start": round(t.start, 3), "end": round(t.end, 3)} for t in self.turns],
            "words": [w.to_json() for w in self.words],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Transcript:
        if d.get("schema_version") != TRANSCRIPT_SCHEMA_VERSION:
            raise ValueError(f"unsupported transcript schema_version: {d.get('schema_version')}")
        return cls(
            file_id=d["file_id"],
            duration=d["duration"],
            words=[Word.from_json(w) for w in d["words"]],
            utterances=[Utterance(**u) for u in d["utterances"]],
            turns=[Turn(**t) for t in d["turns"]],
            speakers=[SpeakerProfile(**s) for s in d["speakers"]],
            meta=d.get("meta", {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Transcript:
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))


@dataclass(slots=True)
class Chunk:
    """A retrieval passage: a window of consecutive utterances (may span both speakers)."""

    id: str
    file_id: str
    idx: int
    start: float
    end: float
    utterance_start: int  # inclusive
    utterance_end: int  # exclusive
    text: str  # verbatim transcript text -> lexical index + display
    embed_text: str  # text that is embedded (may carry dialogue context, see chunking.py)
    speakers: list[str]
    n_words: int

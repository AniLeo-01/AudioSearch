"""Automatic speech recognition with word-level timestamps.

The default engine is faster-whisper (CTranslate2).  ``large-v3-turbo`` in int8 runs at ~0.24x real
time on a 4-core CPU, which makes local transcription practical without a GPU.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from audiosearch.domain import Word

log = logging.getLogger(__name__)


@dataclass
class AsrSegment:
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    word_start: int
    word_end: int


@dataclass
class AsrResult:
    words: list[Word]
    segments: list[AsrSegment]
    language: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "meta": self.meta,
            "segments": [s.__dict__ for s in self.segments],
            "words": [w.to_json() for w in self.words],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> AsrResult:
        return cls(
            words=[Word.from_json(w) for w in d["words"]],
            segments=[AsrSegment(**s) for s in d["segments"]],
            language=d["language"],
            meta=d.get("meta", {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> AsrResult:
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))


class AsrEngine(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray) -> AsrResult: ...


def clean_words(words: list[Word]) -> list[Word]:
    """Strip whitespace, drop empty tokens and enforce monotonic, non-degenerate timestamps."""
    out: list[Word] = []
    prev_end = 0.0
    for w in words:
        text = w.text.strip()
        if not text:
            continue
        start = max(w.start, prev_end)
        end = max(w.end, start + 0.02)
        out.append(Word(text, round(start, 3), round(end, 3), w.prob))
        prev_end = start  # allow slight overlap in ends, but never go back in time for starts
    return out


class FasterWhisperAsr:
    name = "faster-whisper"

    def __init__(
        self,
        model: str = "large-v3-turbo",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        cpu_threads: int = 0,
    ) -> None:
        from faster_whisper import WhisperModel  # heavy import, keep lazy

        self.model_name = model
        self.params: dict[str, Any] = {
            "language": "en",
            "beam_size": beam_size,
            "word_timestamps": True,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500},
            # Conditioning on previous text makes Whisper prone to repetition loops on long audio;
            # disabling it trades a little fluency for robustness.
            "condition_on_previous_text": False,
            "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        }
        self.device = device
        self.compute_type = compute_type
        t0 = time.perf_counter()
        self._model = WhisperModel(model, device=device, compute_type=compute_type, cpu_threads=cpu_threads)
        log.info("loaded ASR model %s (%s/%s) in %.1fs", model, device, compute_type, time.perf_counter() - t0)

    def transcribe(self, audio: np.ndarray) -> AsrResult:
        t0 = time.perf_counter()
        seg_iter, info = self._model.transcribe(audio, **self.params)
        words: list[Word] = []
        segments: list[AsrSegment] = []
        for seg in seg_iter:
            w0 = len(words)
            for w in seg.words or []:
                words.append(Word(w.word, float(w.start), float(w.end), float(w.probability)))
            segments.append(
                AsrSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text.strip(),
                    avg_logprob=float(seg.avg_logprob),
                    no_speech_prob=float(seg.no_speech_prob),
                    word_start=w0,
                    word_end=len(words),
                )
            )
        elapsed = time.perf_counter() - t0
        duration = len(audio) / 16_000
        cleaned = clean_words(words)
        if len(cleaned) != len(words):  # re-index segments only if tokens were dropped
            segments = _reindex_segments(segments, words)
        log.info(
            "ASR %.1fs audio in %.1fs (RTF %.2f), %d words",
            duration,
            elapsed,
            elapsed / duration,
            len(cleaned),
        )
        return AsrResult(
            words=cleaned,
            segments=segments,
            language=info.language,
            meta={
                "engine": self.name,
                "model": self.model_name,
                "device": self.device,
                "compute_type": self.compute_type,
                "params": {k: v for k, v in self.params.items() if k != "temperature"},
                "elapsed_sec": round(elapsed, 2),
                "rtf": round(elapsed / max(duration, 1e-6), 3),
            },
        )


def merge_continuations(words: list[Word], segments: list[AsrSegment]) -> tuple[list[Word], set[int]]:
    """Re-join sub-word tokens that Whisper emits as separate "words" ("F" + "-15" -> "F-15").

    Whisper marks word starts with a leading space; tokens without one continue the previous word.
    The segment text preserves that spacing, so we recover it by walking each segment's text.
    Returns the merged words and the indices of words that end an ASR segment.
    """
    merged: list[Word] = []
    segment_ends: set[int] = set()
    for seg in segments:
        text = seg.text
        cursor = 0
        for j in range(seg.word_start, seg.word_end):
            w = words[j]
            tok = w.text.strip()
            pos = text.find(tok, cursor) if tok else -1
            attached = (
                j > seg.word_start
                and pos > 0
                and pos == cursor  # token starts exactly where the previous one ended
                and not text[pos - 1].isspace()
                and merged
            )
            if pos >= 0:
                cursor = pos + len(tok)
            if attached:
                prev = merged[-1]
                merged[-1] = Word(prev.text + tok, prev.start, max(prev.end, w.end), min(prev.prob, w.prob))
            elif tok:
                merged.append(Word(tok, w.start, w.end, w.prob))
        if merged:
            segment_ends.add(len(merged) - 1)
    return merged, segment_ends


def _reindex_segments(segments: list[AsrSegment], raw_words: list[Word]) -> list[AsrSegment]:
    keep = [bool(w.text.strip()) for w in raw_words]
    new_index = np.cumsum([0, *keep])
    for s in segments:
        s.word_start, s.word_end = int(new_index[s.word_start]), int(new_index[s.word_end])
    return segments

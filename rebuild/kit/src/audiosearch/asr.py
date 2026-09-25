"""Local ASR: faster-whisper with word timestamps, cached per file as JSON."""

from __future__ import annotations

import json
from pathlib import Path

from .models import Word

PARAMS = {
    "language": "en",
    "beam_size": 5,
    "word_timestamps": True,
    "vad_filter": True,
    "vad_parameters": {"min_silence_duration_ms": 500},
    "condition_on_previous_text": False,  # avoids repetition loops on long audio
}


def merge_continuations(seg_text: str, raw: list[tuple], min_start: float = 0.0) -> list[Word]:
    """Whisper emits sub-word pieces as separate words ("F", "-15"). A piece that starts exactly where
    the previous one ended in the segment text (no space before it) continues the previous word."""
    out: list[Word] = []
    cursor = 0
    for tok, start, end, prob in raw:
        tok = tok.strip()
        if not tok:
            continue
        pos = seg_text.find(tok, cursor)
        attached = bool(out) and pos > 0 and pos == cursor and not seg_text[pos - 1].isspace()
        if pos >= 0:
            cursor = pos + len(tok)
        if attached:
            p = out[-1]
            out[-1] = Word(p.text + tok, p.start, max(p.end, end), min(p.prob, prob))
        else:
            start = max(start, out[-1].start if out else min_start)  # keep starts monotonic
            out.append(Word(tok, start, max(end, start + 0.02), prob))
    return out


def transcribe(model, audio_path: Path, cache: Path) -> tuple[list[Word], set[int]]:
    """Words plus the indices of words that end an ASR segment (turn-boundary hints for diarization)."""
    if cache.exists():
        d = json.loads(cache.read_text())
        return [Word(**w) for w in d["words"]], set(d["segment_ends"])
    segments, _info = model.transcribe(str(audio_path), **PARAMS)
    words: list[Word] = []
    seg_ends: set[int] = set()
    for seg in segments:
        raw = [(w.word, w.start, w.end, w.probability) for w in seg.words or []]
        words += merge_continuations(seg.text, raw, words[-1].start if words else 0.0)
        if words:
            seg_ends.add(len(words) - 1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"words": [w.__dict__ for w in words], "segment_ends": sorted(seg_ends)}))
    return words, seg_ends

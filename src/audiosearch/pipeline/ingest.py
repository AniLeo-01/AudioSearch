"""Ingestion orchestrator: audio file -> canonical speaker-attributed transcript (cached on disk).

Every stage is content-addressed: the ASR cache is keyed by (audio sha256, ASR model) and the
transcript by (ASR output, diarization parameters).  Re-running ingestion is therefore idempotent and
cheap, and models are only loaded when a cache miss actually requires them.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audiosearch.audio import AudioInfo, decode, probe
from audiosearch.config import Settings
from audiosearch.dataset import ManifestEntry
from audiosearch.domain import Transcript, Word
from audiosearch.pipeline.alignment import build_turns, build_utterances
from audiosearch.pipeline.asr import AsrResult, FasterWhisperAsr, merge_continuations
from audiosearch.pipeline.diarization import EcapaDiarizer
from audiosearch.pipeline.roles import infer_roles, speaker_profiles

log = logging.getLogger(__name__)

WORD_NORMALIZATION_VERSION = 1  # bump when post-ASR word normalisation changes (invalidates transcripts)


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class IngestReport:
    file_id: str
    asr_cached: bool
    transcript_cached: bool
    n_words: int
    n_utterances: int
    speakers: dict[str, str]


class IngestPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._asr: FasterWhisperAsr | None = None
        self._diarizer: EcapaDiarizer | None = None

    # lazily constructed models ------------------------------------------------------------------
    @property
    def asr(self) -> FasterWhisperAsr:
        if self._asr is None:
            s = self.settings
            self._asr = FasterWhisperAsr(s.asr_model, s.asr_device, s.asr_compute_type, s.asr_beam_size, s.asr_cpu_threads)
        return self._asr

    @property
    def diarizer(self) -> EcapaDiarizer:
        if self._diarizer is None:
            s = self.settings
            self._diarizer = EcapaDiarizer(
                model=s.diarization_model,
                num_speakers=s.num_speakers,
                window_sec=s.diar_window_sec,
                hop_sec=s.diar_hop_sec,
                switch_penalty=s.diar_switch_penalty,
                boundary_discount=s.diar_boundary_discount,
            )
        return self._diarizer

    def diarization_params(self) -> dict[str, Any]:
        s = self.settings
        return {
            "model": s.diarization_model,
            "num_speakers": s.num_speakers,
            "window_sec": s.diar_window_sec,
            "hop_sec": s.diar_hop_sec,
            "switch_penalty": s.diar_switch_penalty,
            "boundary_discount": s.diar_boundary_discount,
        }

    # stages ---------------------------------------------------------------------------------------
    def run_asr(self, entry: ManifestEntry, info: AudioInfo, audio_loader: Callable[[], Any], force: bool) -> tuple[AsrResult, bool]:
        path = self.settings.transcripts_dir / f"{entry.file_id}.asr.json"
        if path.exists() and not force:
            cached = AsrResult.load(path)
            if cached.meta.get("audio_sha256") == info.sha256 and cached.meta.get("model") == self.settings.asr_model:
                return cached, True
            log.info("[%s] ASR cache stale (audio or model changed)", entry.file_id)
        result = self.asr.transcribe(audio_loader())
        result.meta.update(audio_sha256=info.sha256, audio_duration=round(info.duration, 3))
        result.save(path)
        return result, False

    def process(self, entry: ManifestEntry, force: bool = False) -> tuple[Transcript, IngestReport]:
        info = probe(entry.audio_path)
        audio_cache: dict[str, Any] = {}

        def load_audio() -> Any:
            if "a" not in audio_cache:
                audio_cache["a"] = decode(entry.audio_path)
            return audio_cache["a"]

        asr, asr_cached = self.run_asr(entry, info, load_audio, force)
        diar_params = self.diarization_params()
        signature = _digest(
            {"asr": asr.meta, "n_words": len(asr.words), "diar": diar_params, "norm": WORD_NORMALIZATION_VERSION}
        )
        out = self.settings.transcripts_dir / f"{entry.file_id}.json"
        if out.exists() and not force:
            cached_t = Transcript.load(out)
            if cached_t.meta.get("signature") == signature:
                return cached_t, self._report(cached_t, asr_cached, True)

        asr_words, segment_ends = merge_continuations(asr.words, asr.segments)
        diar = self.diarizer.diarize(load_audio(), asr_words, segment_ends)
        words = [
            Word(w.text, w.start, w.end, w.prob, spk, conf)
            for w, spk, conf in zip(asr_words, diar.labels, diar.confidences, strict=True)
        ]
        utterances = build_utterances(words, entry.file_id)
        turns = build_turns(utterances)
        speakers = infer_roles(speaker_profiles(utterances, turns), utterances, entry.host, entry.guest)
        transcript = Transcript(
            file_id=entry.file_id,
            duration=info.duration,
            words=words,
            utterances=utterances,
            turns=turns,
            speakers=speakers,
            meta={
                "signature": signature,
                "audio": {"sha256": info.sha256, "duration": round(info.duration, 3), "sample_rate": info.sample_rate,
                          "channels": info.channels, "codec": info.codec},
                "asr": asr.meta,
                "diarization": diar.meta,
            },
        )  # fmt: skip
        transcript.save(out)
        return transcript, self._report(transcript, asr_cached, False)

    @staticmethod
    def _report(t: Transcript, asr_cached: bool, transcript_cached: bool) -> IngestReport:
        return IngestReport(
            file_id=t.file_id,
            asr_cached=asr_cached,
            transcript_cached=transcript_cached,
            n_words=len(t.words),
            n_utterances=len(t.utterances),
            speakers={s.label: f"{s.role}:{s.display_name or '-'}" for s in t.speakers},
        )


def transcript_path(settings: Settings, file_id: str) -> Path:
    return settings.transcripts_dir / f"{file_id}.json"

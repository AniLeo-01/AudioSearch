"""Audio I/O via ffmpeg: validation, metadata probing, decoding to 16 kHz mono float32."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


class AudioError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AudioInfo:
    path: Path
    duration: float
    sample_rate: int
    channels: int
    codec: str
    sha256: str
    size_bytes: int


def _require(binary: str) -> str:
    exe = shutil.which(binary)
    if exe is None:
        raise AudioError(f"'{binary}' not found on PATH; install ffmpeg")
    return exe


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def probe(path: Path) -> AudioInfo:
    """Validate that ``path`` is decodable audio and return its metadata."""
    if not path.is_file():
        raise AudioError(f"audio file not found: {path}")
    cmd = [
        _require("ffprobe"),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,codec_name:format=duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise AudioError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    meta = json.loads(proc.stdout or "{}")
    streams = meta.get("streams") or []
    if not streams:
        raise AudioError(f"no audio stream in {path}")
    duration = float(meta.get("format", {}).get("duration", 0.0))
    if duration <= 0:
        raise AudioError(f"zero-length audio: {path}")
    return AudioInfo(
        path=path,
        duration=duration,
        sample_rate=int(streams[0].get("sample_rate", 0)),
        channels=int(streams[0].get("channels", 0)),
        codec=str(streams[0].get("codec_name", "unknown")),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def decode(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any ffmpeg-readable file to a mono float32 waveform in [-1, 1]."""
    cmd = [
        _require("ffmpeg"),
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg decode failed for {path}: {proc.stderr.decode(errors='ignore').strip()}")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0

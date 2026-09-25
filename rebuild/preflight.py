#!/usr/bin/env python
"""Night-before check for the 4-hour rebuild: tools, database, models and ASR speed on *your* machine.

    python rebuild/preflight.py [--db URL] [--audio FILE.mp3] [--skip-models] [--skip-bench]

1. Tools     python >= 3.11, ffmpeg, ffprobe, docker, git, uv
2. Database  connects, creates vector / pg_trgm / fuzzystrmatch, prints the pgvector version
3. Models    downloads and loads every model the build uses, so tomorrow runs offline (~2.3 GB)
4. ASR       transcribes a 60 s clip with large-v3-turbo and base.en and projects the time for the
             whole corpus (6 x 9.3 min), so you can pick the ASR model before the clock starts

Standalone on purpose: needs only the packages the build installs (faster-whisper, speechbrain,
sentence-transformers, psycopg). Each section degrades to a warning instead of stopping the run.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

CORPUS_MIN = 56.0  # 6 excerpts x ~9.3 min
DB_DEFAULT = "postgresql://audiosearch:audiosearch@localhost:5432/audiosearch"
ASR_PARAMS = {
    "language": "en",
    "beam_size": 5,
    "word_timestamps": True,
    "vad_filter": True,
    "vad_parameters": {"min_silence_duration_ms": 500},
    "condition_on_previous_text": False,
}
problems: list[str] = []


def report(ok: bool, what: str, detail: str = "") -> None:
    print(f"  [{'ok' if ok else '!!'}] {what}{'  ' + detail if detail else ''}")
    if not ok:
        problems.append(what)


def check_tools() -> None:
    print("1. Tools")
    report(sys.version_info >= (3, 11), "python >= 3.11", sys.version.split()[0])
    for tool in ("ffmpeg", "ffprobe", "docker", "git", "uv"):
        report(shutil.which(tool) is not None, tool, shutil.which(tool) or "not on PATH")


def check_db(url: str) -> None:
    print("2. Database")
    try:
        import psycopg

        with psycopg.connect(url, autocommit=True, connect_timeout=5) as conn:
            for ext in ("vector", "pg_trgm", "fuzzystrmatch"):
                conn.execute(f"CREATE EXTENSION IF NOT EXISTS {ext}")
            row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
            report(True, "postgres + extensions", f"pgvector {row[0] if row else '?'}")
    except Exception as exc:
        report(False, "postgres + extensions", f"{type(exc).__name__}: {exc}".strip()[:160])


def fetch_models() -> None:
    print("3. Models (first run downloads; later runs just load from the cache)")
    steps = [
        ("faster-whisper large-v3-turbo (1.6 GB)", lambda: _whisper("large-v3-turbo")),
        ("faster-whisper base.en fallback (0.14 GB)", lambda: _whisper("base.en")),
        ("ECAPA speaker encoder (85 MB)", _ecapa),
        ("BAAI/bge-base-en-v1.5 embeddings (0.42 GB)", _bge),
    ]
    for name, load in steps:
        t0 = time.perf_counter()
        try:
            load()
            report(True, name, f"{time.perf_counter() - t0:.0f}s")
        except Exception as exc:
            report(False, name, f"{type(exc).__name__}: {exc}"[:160])


def _whisper(name: str):
    from faster_whisper import WhisperModel

    return WhisperModel(name, device="cpu", compute_type="int8")


def _ecapa() -> None:
    from speechbrain.inference.speaker import EncoderClassifier

    savedir = Path.home() / ".cache" / "ecapa"  # the kit loads it from here
    EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(savedir))


def _bge() -> None:
    from sentence_transformers import SentenceTransformer

    SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu").encode(["warm-up"])


def bench_asr(audio: Path | None) -> None:
    print("4. ASR speed")
    if audio is None or not audio.exists():
        report(False, "benchmark clip", "pass --audio FILE (any speech recording longer than 3 minutes)")
        return
    try:
        from faster_whisper import decode_audio

        wav = decode_audio(str(audio), sampling_rate=16_000)
        clip = wav[120 * 16_000 : 180 * 16_000] if len(wav) >= 180 * 16_000 else wav[: 60 * 16_000]
        seconds = len(clip) / 16_000
        for name in ("large-v3-turbo", "base.en"):
            model = _whisper(name)
            t0 = time.perf_counter()
            segments, _ = model.transcribe(clip, **ASR_PARAMS)
            n_words = sum(len(s.words or []) for s in segments)  # the generator does the work
            rtf = (time.perf_counter() - t0) / seconds
            print(f"  {name:15s} RTF {rtf:.2f}  ({n_words} words)  ->  ~{rtf * CORPUS_MIN:.0f} min for the corpus")
            if name == "large-v3-turbo":
                pick = "large-v3-turbo" if rtf <= 0.35 else "base.en (costs ~3 pts Recall@5 in the reference)"
                print(f"  recommendation: {pick}")
    except Exception as exc:
        report(False, "ASR benchmark", f"{type(exc).__name__}: {exc}"[:160])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--audio", type=Path, default=next(iter(sorted(Path("data/audio").glob("*.mp3"))), None))
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()
    check_tools()
    check_db(args.db)
    if not args.skip_models:
        fetch_models()
    if not args.skip_bench:
        bench_asr(args.audio)
    print("\nREADY" if not problems else f"\nFIX BEFORE TOMORROW: {', '.join(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

"""CLI: init-db | ingest [--asr-only] | index | search | eval | serve"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from . import config

app = typer.Typer(add_completion=False, no_args_is_help=True)


def manifest() -> list[dict]:
    return yaml.safe_load((config.DATA / "manifest.yaml").read_text())["files"]


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


@app.command()
def init_db() -> None:
    from .db import init_db as _init

    _init()
    typer.echo("schema ready")


@app.command()
def ingest(only: list[str] = typer.Option(None), asr_only: bool = False) -> None:
    """audio -> words (ASR, cached) -> speakers -> utterances + roles -> data/transcripts/<id>.json"""
    from faster_whisper import WhisperModel, decode_audio

    from .asr import transcribe
    from .diarize import diarize
    from .models import Transcript
    from .transcript import build_utterances, infer_roles

    whisper = enc = None
    for e in manifest():
        fid = e["file_id"]
        out = config.DATA / "transcripts" / f"{fid}.json"
        if (only and fid not in only) or (out.exists() and not asr_only):
            continue
        audio_path = config.DATA / "audio" / f"{fid}.mp3"
        cache = out.with_suffix(".asr.json")
        if whisper is None and not cache.exists():
            whisper = WhisperModel(config.ASR_MODEL, device="cpu", compute_type="int8")
        words, seg_ends = transcribe(whisper, audio_path, cache)
        typer.echo(f"{fid}: {len(words)} words")
        if asr_only:
            continue
        if enc is None:
            from speechbrain.inference.speaker import EncoderClassifier

            enc = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(Path.home() / ".cache/ecapa")
            )
        audio = decode_audio(str(audio_path), sampling_rate=16_000)
        diarize(enc, audio, words, seg_ends)
        utts = build_utterances(words)
        speakers = infer_roles(utts, e["speakers"]["host"], e["speakers"]["guest"])
        Transcript(fid, len(audio) / 16_000, utts, speakers).save(out)
        typer.echo(f"{fid}: {len(utts)} utterances, {[(s.label, s.role, s.name) for s in speakers]}")


@app.command()
def index() -> None:
    from .db import connect
    from .embed import Embedder
    from .index import index_all
    from .models import Transcript

    titles = {e["file_id"]: e["title"] for e in manifest()}
    ts = [Transcript.load(config.DATA / "transcripts" / f"{fid}.json") for fid in titles]
    with connect() as conn:
        index_all(conn, ts, titles, Embedder())
        n = conn.execute(
            "SELECT (SELECT count(*) FROM chunks), (SELECT count(*) FROM utterances),"
            " (SELECT count(*) FROM chunk_terms), (SELECT count(*) FROM vocabulary)"
        ).fetchone()
    typer.echo(f"indexed {len(ts)} files: {n[0]} chunks, {n[1]} utterances, {n[2]} postings, {n[3]} vocabulary terms")


@app.command()
def show(file_id: str) -> None:
    """Print a transcript with utterance numbers, for writing golden labels as file:first-last."""
    from .models import Transcript

    t = Transcript.load(config.DATA / "transcripts" / f"{file_id}.json")
    roles = {s.label: s.role for s in t.speakers}
    for u in t.utterances:
        typer.echo(f"[{u.idx:3d}] {mmss(u.start)} {roles.get(u.speaker, u.speaker):5s} {u.text}")


@app.command()
def label(src: Path = typer.Option(None)) -> None:
    """queries.src.yaml (labels as utterance ranges "runway:70-72@1") -> queries.yaml (time intervals + quotes)."""
    import re

    from .models import Transcript

    src = src or config.DATA / "eval" / "queries.src.yaml"
    spec = yaml.safe_load(src.read_text())
    tol = float(spec.get("tolerance_sec", config.TOLERANCE))
    cache: dict[str, Transcript] = {}
    for q in spec["queries"]:
        rel: list[dict] = []
        for ref in q["relevant"]:
            m = re.fullmatch(r"(\w+):(\d+)(?:-(\d+))?(?:@([12]))?", ref)
            if not m:
                raise typer.BadParameter(f"{q['id']}: bad reference {ref!r}")
            fid, a, b = m[1], int(m[2]), int(m[3] or m[2])
            if fid not in cache:
                cache[fid] = Transcript.load(config.DATA / "transcripts" / f"{fid}.json")
            us = cache[fid].utterances[a : b + 1]
            quote = " ".join(u.text for u in us)[:200]
            rel.append({"file": fid, "start": us[0].start, "end": us[-1].end, "grade": int(m[4] or 2), "quote": quote})
        merged: list[dict] = []  # mentions closer than 2 x tolerance are one place in the audio
        for r in sorted(rel, key=lambda r: (r["file"], r["start"])):
            if merged and merged[-1]["file"] == r["file"] and r["start"] - merged[-1]["end"] < 2 * tol:
                merged[-1].update(end=max(merged[-1]["end"], r["end"]), grade=max(merged[-1]["grade"], r["grade"]))
            else:
                merged.append(r)
        q["relevant"] = merged
    out = src.with_name("queries.yaml")
    out.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
    typer.echo(f"wrote {out}: {len(spec['queries'])} queries")


@app.command()
def search(query: str, k: int = 5, role: str = typer.Option(None), mode: str = "hybrid") -> None:
    from .search import Searcher

    for h in Searcher.create().search(query, k=k, role=role, mode=mode):
        who = f"{h.role} {h.name or h.speaker}"
        typer.echo(f"{h.rank}. [{h.file_id} {mmss(h.start)}] {who}: {h.text}  {h.channels}")


@app.command("eval")
def eval_(split: str = "test", out: Path = Path("reports/evaluation.md")) -> None:
    from .evaluate import run
    from .search import Searcher

    report = run(Searcher.create(), config.DATA / "eval" / "queries.yaml", split)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n")
    typer.echo(report)


@app.command()
def serve(port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(f"{__package__}.api:app", host="0.0.0.0", port=port)

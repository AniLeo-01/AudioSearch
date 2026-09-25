"""Command-line interface: ``audiosearch --help``."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from audiosearch.config import Settings, get_settings
from audiosearch.dataset import ManifestEntry
from audiosearch.logging_setup import setup_logging

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Conversation-aware hybrid audio search.")
db_app = typer.Typer(no_args_is_help=True, help="Database schema management.")
eval_app = typer.Typer(no_args_is_help=True, help="Retrieval and pipeline evaluation.")
app.add_typer(db_app, name="db")
app.add_typer(eval_app, name="eval")
console = Console()
log = logging.getLogger("audiosearch.cli")

FileOpt = Annotated[list[str] | None, typer.Option("--file", "-f", help="Limit to these file_ids (repeatable).")]


def _settings() -> Settings:
    s = get_settings()
    setup_logging(s.log_level, s.log_format)
    return s


def _entries(s: Settings, only: list[str] | None) -> list[ManifestEntry]:
    from audiosearch.dataset import load_manifest

    entries = load_manifest(s.manifest_path, s.audio_dir)
    if only:
        unknown = set(only) - {e.file_id for e in entries}
        if unknown:
            raise typer.BadParameter(f"unknown file_id(s): {', '.join(sorted(unknown))}")
        entries = [e for e in entries if e.file_id in only]
    return entries


def fmt_ts(sec: float) -> str:
    m, s = divmod(max(sec, 0.0), 60)
    return f"{int(m):02d}:{s:04.1f}"


# ---------------------------------------------------------------------------------------------------------
@db_app.command("migrate")
def db_migrate() -> None:
    """Create extensions/schema and apply pending migrations."""
    from audiosearch.db import migrate
    from audiosearch.embeddings import Embedder

    s = _settings()
    dim = Embedder(s.embedding_model, s.embedding_device).dim
    applied = migrate(s, s.embedding_model, dim)
    console.print(f"[green]schema '{s.db_schema}' up to date[/] (applied: {applied or 'none'})")


@db_app.command("reset")
def db_reset(yes: Annotated[bool, typer.Option("--yes", help="Confirm destructive reset.")] = False) -> None:
    """Drop all AudioSearch tables in the configured schema."""
    from audiosearch.db import reset_schema

    s = _settings()
    if not yes:
        typer.confirm(f"Drop all AudioSearch data in schema '{s.db_schema}'?", abort=True)
    reset_schema(s)
    console.print("[yellow]schema dropped[/]")


# ---------------------------------------------------------------------------------------------------------
@app.command()
def ingest(files: FileOpt = None, force: bool = typer.Option(False, help="Ignore caches.")) -> None:
    """Audio -> speaker-attributed transcripts (ASR + diarization + roles), cached under data/transcripts."""
    from audiosearch.pipeline.ingest import IngestPipeline

    s = _settings()
    pipe = IngestPipeline(s)
    table = Table("file_id", "ASR", "transcript", "words", "utterances", "speakers")
    for e in _entries(s, files):
        _, rep = pipe.process(e, force=force)
        table.add_row(
            e.file_id,
            "cached" if rep.asr_cached else "new",
            "cached" if rep.transcript_cached else "new",
            str(rep.n_words),
            str(rep.n_utterances),
            ", ".join(f"{k}={v}" for k, v in rep.speakers.items()),
        )
    console.print(table)


@app.command()
def index(files: FileOpt = None, force: bool = typer.Option(False, help="Re-index even if unchanged.")) -> None:
    """Transcripts -> PostgreSQL (chunks, embeddings, BM25 postings, vocabulary)."""
    from audiosearch.db import connect, migrate
    from audiosearch.domain import Transcript
    from audiosearch.embeddings import Embedder
    from audiosearch.indexing import Indexer

    s = _settings()
    embedder = Embedder(s.embedding_model, s.embedding_device, s.embedding_batch_size)
    migrate(s, s.embedding_model, embedder.dim)
    indexer = Indexer(s, embedder)
    table = Table("file_id", "status", "utterances", "chunks", "seconds")
    with connect(s) as conn:
        for e in _entries(s, files):
            path = s.transcripts_dir / f"{e.file_id}.json"
            if not path.exists():
                console.print(f"[red]missing transcript {path}; run `audiosearch ingest` first[/]")
                raise typer.Exit(1)
            rep = indexer.index(conn, Transcript.load(path), e, force=force)
            table.add_row(
                e.file_id,
                "unchanged" if rep.skipped else "indexed",
                str(rep.n_utterances),
                str(rep.n_chunks),
                f"{rep.seconds:.1f}",
            )
    console.print(table)


@app.command()
def build(force: bool = typer.Option(False)) -> None:
    """ingest + index for every file in the manifest."""
    ingest(None, force)
    index(None, force)


# ---------------------------------------------------------------------------------------------------------
@app.command()
def search(
    query: Annotated[str, typer.Argument(help='Free text; supports "phrases", role:guest, file:<id>.')],
    k: Annotated[int, typer.Option("-k", help="Number of results.")] = 5,
    mode: Annotated[str, typer.Option(help="hybrid | lexical | semantic")] = "hybrid",
    role: Annotated[str | None, typer.Option(help="host | guest")] = None,
    rerank: Annotated[bool | None, typer.Option(help="Force cross-encoder reranking on/off.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    explain: Annotated[bool, typer.Option(help="Show channel ranks and timings.")] = False,
) -> None:
    """Search the indexed corpus."""
    from audiosearch.search.engine import SearchOptions
    from audiosearch.services import build_engine

    s = _settings()
    engine = build_engine(s, with_reranker=bool(rerank) or s.rerank)
    resp = engine.search(query, k, role=role, options=SearchOptions(mode=mode, rerank=rerank))  # type: ignore[arg-type]
    if as_json:
        json.dump(asdict(resp), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return
    console.print(
        f"[bold]{len(resp.hits)} results[/] for [cyan]{query!r}[/]  intent={resp.intent} mode={resp.mode} "
        f"({resp.timings_ms.get('total', 0):.0f} ms)"
    )
    if resp.expansions:
        console.print("  sounds-like: " + ", ".join(f"{e.source}→{e.term} ({e.score:.2f})" for e in resp.expansions))
    for h in resp.hits:
        who = h.speaker_name or h.speaker
        header = Text(f"{h.rank:>2}. {h.file_id}.mp3  {fmt_ts(h.start)}–{fmt_ts(h.end)}  ", style="bold")
        header.append(f"{who} ({h.speaker_role}, {h.speaker})", style="magenta")
        console.print(header)
        body = Text("    “")
        pos = 0
        for a, b in h.highlights:
            body.append(h.text[pos:a])
            body.append(h.text[a:b], style="bold yellow")
            pos = b
        body.append(h.text[pos:] + "”")
        console.print(body)
        if explain:
            console.print(
                f"    [dim]score={h.score:.4f} channels={h.channels} match@{fmt_ts(h.match_time)} "
                f"passage {fmt_ts(h.passage_start)}–{fmt_ts(h.passage_end)} chunk={h.chunk_id}[/]"
            )
    if explain:
        console.print(f"[dim]weights={resp.weights} timings_ms={resp.timings_ms}[/]")


@app.command()
def stats() -> None:
    """Index statistics."""
    from audiosearch.db import connect

    s = _settings()
    with connect(s) as conn:
        q = {
            "files": "SELECT count(*) FROM audio_files",
            "hours": "SELECT round((coalesce(sum(duration_sec),0)/3600)::numeric, 2) FROM audio_files",
            "utterances": "SELECT count(*) FROM utterances",
            "chunks": "SELECT count(*) FROM chunks",
            "postings": "SELECT count(*) FROM chunk_terms",
            "lexemes": "SELECT count(*) FROM term_stats",
            "vocabulary": "SELECT count(*) FROM vocabulary",
        }
        table = Table("metric", "value")
        for name, sql_text in q.items():
            table.add_row(name, str(conn.execute(sql_text).fetchone()[0]))  # type: ignore[index]
    console.print(table)


@eval_app.command("stages")
def eval_stages() -> None:
    """ASR word error rate and speaker-attribution accuracy against NASA's human transcripts."""
    from audiosearch.dataset import load_manifest
    from audiosearch.eval.report import stage_table
    from audiosearch.eval.stage_metrics import evaluate_stages

    s = _settings()
    hosts = {e.file_id: e.host for e in load_manifest(s.manifest_path, s.audio_dir)}
    console.print(stage_table(evaluate_stages(s.transcripts_dir, s.reference_dir, hosts)))


@eval_app.command("run")
def eval_run(
    split: Annotated[str, typer.Option(help="dev | test | all")] = "test",
    systems: Annotated[str, typer.Option(help="'all', 'none' or comma-separated system names")] = "all",
    variants: Annotated[str, typer.Option(help="'none', 'all' or comma-separated index variants")] = "none",
    out: Annotated[str, typer.Option(help="Output directory for the Markdown/JSON report")] = "reports",
    stages: Annotated[bool, typer.Option(help="Include ASR/diarization stage metrics")] = True,
) -> None:
    """Score systems on the golden query set; writes reports/evaluation_<split>.{md,json}."""
    from pathlib import Path

    from audiosearch.dataset import load_manifest
    from audiosearch.eval.golden import load_golden
    from audiosearch.eval.report import write_report
    from audiosearch.eval.runner import INDEX_VARIANTS, QUERY_SYSTEMS, run_evaluation
    from audiosearch.eval.stage_metrics import evaluate_stages

    s = _settings()
    golden = load_golden(s.golden_queries_path)

    def pick(spec: str, pool: tuple[Any, ...], label: str) -> tuple[Any, ...]:
        if spec == "all":
            return pool
        if spec == "none":
            return ()
        names = [x.strip() for x in spec.split(",") if x.strip()]
        known = {p.name: p for p in pool}
        missing = [n for n in names if n not in known]
        if missing:
            raise typer.BadParameter(f"unknown {label}: {missing}; choose from {sorted(known)}")
        return tuple(known[n] for n in names)

    sys_specs = pick(systems, QUERY_SYSTEMS, "systems")
    if not any(sp.name == "full" for sp in sys_specs):
        sys_specs = (*sys_specs, next(sp for sp in QUERY_SYSTEMS if sp.name == "full"))
    results = run_evaluation(
        s, golden, None if split == "all" else split, sys_specs, pick(variants, INDEX_VARIANTS, "variants")
    )
    stage_reports = None
    if stages:
        hosts = {e.file_id: e.host for e in load_manifest(s.manifest_path, s.audio_dir)}
        stage_reports = evaluate_stages(s.transcripts_dir, s.reference_dir, hosts)
    md, js = write_report(Path(out), results, split, golden.tolerance_sec, stage_reports)
    table = Table("system", "R@1", "R@5", "R@10", "MRR", "nDCG@10", "p50 ms")
    for name, res in results.items():
        sm = res.summary()
        table.add_row(
            name,
            f"{sm['recall@1']:.3f}",
            f"{sm['recall@5']:.3f}",
            f"{sm['recall@10']:.3f}",
            f"{sm['mrr']:.3f}",
            f"{sm['ndcg@10']:.3f}",
            f"{sm['latency_p50_ms']:.0f}",
        )
    console.print(table)
    console.print(f"wrote {md} and {js}")


@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = typer.Option(1, help="Uvicorn worker processes (each loads its own models)."),
) -> None:
    """Run the HTTP API + web UI."""
    import uvicorn

    s = _settings()
    uvicorn.run(
        "audiosearch.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        workers=workers,
        log_level=s.log_level.lower(),
    )


def main() -> None:  # pragma: no cover
    from audiosearch.db import SchemaMismatchError
    from audiosearch.search.query import QueryError

    try:
        app()
    except (SchemaMismatchError, QueryError) as e:
        console.print(f"[red]error:[/] {e}")
        raise SystemExit(2) from None


if __name__ == "__main__":  # pragma: no cover
    main()

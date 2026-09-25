"""Render evaluation results as Markdown (for humans / the submission) and JSON (for tooling)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from audiosearch.eval.golden import CATEGORIES
from audiosearch.eval.metrics import paired_permutation_pvalue
from audiosearch.eval.runner import SystemResult
from audiosearch.eval.stage_metrics import StageReport


def _f(x: Any, digits: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _ts(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:04.1f}"


def systems_table(results: dict[str, SystemResult]) -> str:
    rows = [
        "| System | R@1 | R@3 | R@5 | R@10 | MRR | nDCG@10 | P@5 | median offset (s) | p50 latency (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, res in results.items():
        s = res.summary()
        rows.append(
            f"| `{name}` | {_f(s['recall@1'])} | {_f(s['recall@3'])} | **{_f(s['recall@5'])}** | {_f(s['recall@10'])} "
            f"| {_f(s['mrr'])} | {_f(s['ndcg@10'])} | {_f(s['precision@5'])} | {_f(s['median_offset_sec'], 1)} "
            f"| {_f(s['latency_p50_ms'], 0)} |"
        )
    return "\n".join(rows)


def category_table(results: dict[str, SystemResult], systems: list[str], metric: str = "recall@5") -> str:
    present = [c for c in CATEGORIES if any(c in results[s].summary()["by_category"] for s in systems if s in results)]
    head = "| Category | n | " + " | ".join(f"`{s}`" for s in systems if s in results) + " |"
    sep = "|---|---:|" + "---:|" * len([s for s in systems if s in results])
    rows = [head, sep]
    for cat in present:
        cells, n = [], 0
        best = max(results[s].summary()["by_category"].get(cat, {}).get(metric, 0.0) for s in systems if s in results)
        for s in systems:
            if s not in results:
                continue
            c = results[s].summary()["by_category"].get(cat)
            n = c["n"] if c else n
            v = c[metric] if c else None
            cell = _f(v)
            cells.append(f"**{cell}**" if v is not None and abs(v - best) < 1e-9 else cell)
        rows.append(f"| {cat} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def significance_table(results: dict[str, SystemResult], ref: str, others: list[str]) -> str:
    rows = [
        f"| `{ref}` vs | ΔR@5 | p (R@5) | ΔMRR | p (MRR) |",
        "|---|---:|---:|---:|---:|",
    ]
    a5, am = results[ref].values("recall", 5), results[ref].values("mrr")
    for o in others:
        if o not in results or o == ref:
            continue
        b5, bm = results[o].values("recall", 5), results[o].values("mrr")
        d5 = sum(a5) / len(a5) - sum(b5) / len(b5)
        dm = sum(am) / len(am) - sum(bm) / len(bm)
        rows.append(
            f"| `{o}` | {d5:+.3f} | {_f(paired_permutation_pvalue(a5, b5), 4)} | {dm:+.3f} | "
            f"{_f(paired_permutation_pvalue(am, bm), 4)} |"
        )
    return "\n".join(rows)


def failures(res: SystemResult, k: int = 5) -> str:
    rows = ["| Query | Category | R@5 | first hit | top result |", "|---|---|---:|---:|---|"]
    for o in res.outcomes:
        if o.metrics.recall[k] >= 1.0:
            continue
        top = o.top[0] if o.top else None
        top_s = f"{top['file']} {_ts(top['start'])} “{top['text'][:70]}…”" if top else "–"
        rows.append(
            f"| {o.query.query} | {o.query.category} | {_f(o.metrics.recall[k], 2)} | "
            f"{o.metrics.first_hit_rank or '–'} | {top_s} |"
        )
    return "\n".join(rows) if len(rows) > 2 else "_None: every labelled moment was retrieved in the top 5._"


def stage_table(reports: list[StageReport]) -> str:
    rows = [
        "| File | ref words | WER | S / D / I | speaker attribution acc. | host role correct | inter-speaker cos |",
        "|---|---:|---:|---|---:|:---:|---:|",
    ]
    for r in reports:
        rows.append(
            f"| {r.file_id} | {r.ref_words} | {r.wer:.3f} | {r.substitutions} / {r.deletions} / {r.insertions} | "
            f"{r.speaker_accuracy:.3f} | {'✓' if r.host_role_correct else '✗'} | {_f(r.inter_speaker_cosine)} |"
        )
    if reports:
        tot = sum(r.ref_words for r in reports)
        wer = sum(r.wer * r.ref_words for r in reports) / tot
        acc = sum(r.speaker_accuracy * r.aligned_pairs for r in reports) / sum(r.aligned_pairs for r in reports)
        roles = sum(r.host_role_correct for r in reports)
        rows.append(f"| **all** | {tot} | **{wer:.3f}** | | **{acc:.3f}** | {roles}/{len(reports)} | |")
    return "\n".join(rows)


def write_report(
    out_dir: Path,
    results: dict[str, SystemResult],
    split: str,
    tolerance: float,
    stage_reports: list[StageReport] | None = None,
    reference_system: str = "full",
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = [
        f"# Retrieval evaluation — split `{split}`",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by `audiosearch eval run`. "
        f"A result counts as a hit when its moment overlaps a labelled interval (±{tolerance:g}s) in the same file; "
        "each labelled moment is credited once._",
        "",
        "## Systems",
        "",
        systems_table({k: v for k, v in results.items() if not k.startswith("variant:")}),
        "",
    ]
    variants = {k: v for k, v in results.items() if k.startswith("variant:")}
    if variants:
        md += [
            "## Index-time variants (each vs the default `full` system)",
            "",
            systems_table(
                {reference_system: results[reference_system], **variants} if reference_system in results else variants
            ),
            "",
        ]
    key: list[str] = [
        s for s in ("bm25", "bm25+soundslike", "dense", "hybrid-rrf", "full", "full+rerank") if s in results
    ]
    md += ["## Recall@5 by query category", "", category_table(results, key), ""]
    md += ["## MRR by query category", "", category_table(results, key, "mrr"), ""]
    if reference_system in results:
        s = results[reference_system].summary()
        lo, hi = s["recall@5_ci95"]
        mlo, mhi = s["mrr_ci95"]
        md += [
            f"## Uncertainty and significance (`{reference_system}`)",
            "",
            f"95% bootstrap CI over queries — Recall@5: [{lo:.3f}, {hi:.3f}], MRR: [{mlo:.3f}, {mhi:.3f}].",
            "",
            significance_table(results, reference_system, [k for k in results if k != reference_system]),
            "",
            "_p-values: two-sided paired permutation test over queries (5,000 permutations)._",
            "",
            f"## Misses of `{reference_system}` (labelled moments not all found in the top 5)",
            "",
            failures(results[reference_system]),
            "",
        ]
    if stage_reports:
        md += ["## Upstream stages vs NASA human transcripts", "", stage_table(stage_reports), ""]
    md_path = out_dir / f"evaluation_{split}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    payload = {
        "split": split,
        "tolerance_sec": tolerance,
        "systems": {k: v.summary() for k, v in results.items()},
        "per_query": {
            k: [
                {
                    "id": o.query.id,
                    "category": o.query.category,
                    "recall@5": o.metrics.recall[5],
                    "recall@10": o.metrics.recall[10],
                    "mrr": o.metrics.mrr,
                    "first_hit_rank": o.metrics.first_hit_rank,
                    "latency_ms": o.latency_ms,
                    "top": o.top,
                }
                for o in v.outcomes
            ]
            for k, v in results.items()
        },
        "stages": [asdict(r) for r in stage_reports or []],
    }
    json_path = out_dir / f"evaluation_{split}.json"
    json_path.write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")
    return md_path, json_path

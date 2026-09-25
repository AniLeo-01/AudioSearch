"""Golden query set: queries labelled with *time intervals* of relevant moments.

Labels are (file, start, end) intervals in audio time, not chunk or segment IDs.  That makes the
ground truth independent of every modelling choice - ASR model, diarization, chunk size - so the same
labels can score any configuration fairly (segment-ID labels silently break when chunking changes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

CATEGORIES = (
    "keyword",  # a rare term / named entity said verbatim
    "phrase",  # quoted multi-word phrase
    "paraphrase",  # same meaning, little or no lexical overlap with the transcript
    "question",  # natural-language question
    "cross_file",  # topic discussed in several recordings
    "misspelled",  # misspelled or ASR-mangled names/terms (sounds-like channel)
    "speaker",  # role-scoped query (role:host / role:guest)
)


@dataclass(frozen=True)
class RelevantMoment:
    file: str
    start: float
    end: float
    grade: int = 2  # 2 = directly relevant, 1 = partially relevant
    quote: str = ""


@dataclass
class GoldenQuery:
    id: str
    query: str
    category: str
    split: str
    relevant: list[RelevantMoment]
    role: str | None = None
    notes: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class GoldenSet:
    version: int
    tolerance_sec: float
    queries: list[GoldenQuery]

    def split(self, name: str | None) -> list[GoldenQuery]:
        if name in (None, "all"):
            return list(self.queries)
        return [q for q in self.queries if q.split == name]


def load_golden(path: Path) -> GoldenSet:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    queries: list[GoldenQuery] = []
    seen: set[str] = set()
    for item in raw["queries"]:
        qid = str(item["id"])
        if qid in seen:
            raise ValueError(f"duplicate query id {qid}")
        seen.add(qid)
        if item["category"] not in CATEGORIES:
            raise ValueError(f"{qid}: unknown category {item['category']}")
        if item.get("split") not in ("dev", "test"):
            raise ValueError(f"{qid}: split must be dev or test")
        rel = [
            RelevantMoment(
                file=str(r["file"]),
                start=float(r["start"]),
                end=float(r["end"]),
                grade=int(r.get("grade", 2)),
                quote=str(r.get("quote", "")),
            )
            for r in item["relevant"]
        ]
        if not rel:
            raise ValueError(f"{qid}: at least one relevant moment is required")
        for r in rel:
            if r.end < r.start:
                raise ValueError(f"{qid}: interval end before start")
        queries.append(
            GoldenQuery(
                id=qid,
                query=str(item["query"]),
                category=item["category"],
                split=item["split"],
                relevant=rel,
                role=item.get("role"),
                notes=str(item.get("notes", "")),
                tags=list(item.get("tags", [])),
            )
        )
    return GoldenSet(
        version=int(raw.get("version", 1)),
        tolerance_sec=float(raw.get("tolerance_sec", 5.0)),
        queries=queries,
    )

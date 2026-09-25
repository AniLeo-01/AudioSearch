"""Dataset manifest: which audio files exist, where they came from, and who is speaking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    file_id: str
    title: str
    audio_path: Path
    page_url: str | None = None
    source_audio_url: str | None = None
    excerpt_start: float | None = None
    excerpt_end: float | None = None
    host: str | None = None
    guest: str | None = None
    topic: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def load_manifest(manifest_path: Path, audio_dir: Path) -> list[ManifestEntry]:
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    entries: list[ManifestEntry] = []
    seen: set[str] = set()
    for item in raw.get("files", []):
        fid = str(item["file_id"])
        if fid in seen:
            raise ValueError(f"duplicate file_id in manifest: {fid}")
        seen.add(fid)
        excerpt = item.get("excerpt") or {}
        speakers = item.get("speakers") or {}
        entries.append(
            ManifestEntry(
                file_id=fid,
                title=str(item.get("title", fid)),
                audio_path=audio_dir / item.get("audio", f"{fid}.mp3"),
                page_url=item.get("page_url"),
                source_audio_url=item.get("source_audio_url"),
                excerpt_start=excerpt.get("start"),
                excerpt_end=excerpt.get("end"),
                host=speakers.get("host"),
                guest=speakers.get("guest"),
                topic=item.get("topic"),
                extra={
                    k: (v.isoformat() if hasattr(v, "isoformat") else v)  # YAML dates -> JSON-safe strings
                    for k, v in item.items()
                    if k in {"episode", "published"}
                },
            )
        )
    return entries

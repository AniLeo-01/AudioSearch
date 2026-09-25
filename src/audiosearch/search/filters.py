"""Filters shared by every retrieval channel, compiled to parameterised SQL (never string-formatted)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from psycopg import sql


@dataclass(frozen=True)
class SearchFilters:
    file_ids: tuple[str, ...] = ()
    role: str | None = None  # host | guest
    speaker: str | None = None  # SPEAKER_xx
    phrases: tuple[str, ...] = field(default_factory=tuple)  # must match in the chunk (stemmed phrase)

    def chunk_clause(self, alias: str = "c") -> tuple[sql.Composable, dict[str, Any]]:
        """WHERE-clause fragment over the chunks table (always valid SQL, 'TRUE' when empty)."""
        a = sql.Identifier(alias)
        parts: list[sql.Composable] = [sql.SQL("TRUE")]
        params: dict[str, Any] = {}
        if self.file_ids:
            parts.append(sql.SQL("{}.file_id = ANY(%(f_file_ids)s)").format(a))
            params["f_file_ids"] = list(self.file_ids)
        if self.speaker:
            parts.append(sql.SQL("%(f_speaker)s = ANY({}.speakers)").format(a))
            params["f_speaker"] = self.speaker
        if self.role:
            parts.append(
                sql.SQL(
                    "EXISTS (SELECT 1 FROM speakers s WHERE s.file_id = {a}.file_id "
                    "AND s.role = %(f_role)s AND s.label = ANY({a}.speakers))"
                ).format(a=a)
            )
            params["f_role"] = self.role
        for i, phrase in enumerate(self.phrases):
            key = f"f_phrase_{i}"
            parts.append(sql.SQL("{}.tsv @@ phraseto_tsquery('english', %({})s)").format(a, sql.SQL(key)))
            params[key] = phrase
        return sql.SQL(" AND ").join(parts), params

    @property
    def restricts_speaker(self) -> bool:
        return bool(self.role or self.speaker)

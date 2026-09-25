"""Query parsing: quoted phrases, inline filters and query-intent classification.

Syntax (all optional, combinable with free text):
    "exact phrase"        phrase must occur verbatim (stemmed) in the passage
    role:host|guest       only moments spoken by the host / the guest
    speaker:SPEAKER_01    only moments spoken by that diarization label
    file:ai_at_nasa       restrict to one recording (repeatable)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from audiosearch.textutil import content_tokens, tokens

PHRASE_RE = re.compile(r'"([^"]{1,200})"')
FILTER_RE = re.compile(r"\b(role|speaker|file):([A-Za-z0-9_\-]+)", re.IGNORECASE)
QUESTION_STARTERS = frozenset(
    "what how why when where who which whose whom is are was were do does did can could should would will "
    "has have had tell explain describe".split()
)
ROLES = frozenset({"host", "guest"})

# Fusion weights per intent: exact-looking queries lean lexical, natural-language questions lean dense.
INTENT_WEIGHTS: dict[str, dict[str, float]] = {
    "phrase": {"lexical": 1.0, "dense": 0.3},
    "keyword": {"lexical": 1.0, "dense": 0.7},
    "question": {"lexical": 0.6, "dense": 1.0},
    "topic": {"lexical": 0.8, "dense": 1.0},
}


class QueryError(ValueError):
    pass


@dataclass
class ParsedQuery:
    raw: str
    text: str  # free text (quotes stripped, filters removed)
    phrases: list[str] = field(default_factory=list)
    role: str | None = None
    speaker: str | None = None
    file_ids: list[str] = field(default_factory=list)
    intent: str = "topic"
    content_tokens: list[str] = field(default_factory=list)


def classify_intent(text: str, phrases: list[str]) -> str:
    if phrases:
        return "phrase"
    toks = tokens(text)
    if not toks:
        return "keyword"
    if text.strip().endswith("?") or toks[0] in QUESTION_STARTERS:
        return "question"
    if len(content_tokens(text)) <= 2 and len(toks) <= 3:
        return "keyword"
    return "topic"


def parse_query(raw: str, max_chars: int = 512) -> ParsedQuery:
    if raw is None or not raw.strip():
        raise QueryError("query must not be empty")
    if len(raw) > max_chars:
        raise QueryError(f"query too long ({len(raw)} > {max_chars} characters)")
    role = speaker = None
    file_ids: list[str] = []
    for key, value in FILTER_RE.findall(raw):
        key = key.lower()
        if key == "role":
            if value.lower() not in ROLES:
                raise QueryError(f"unknown role '{value}' (expected host or guest)")
            role = value.lower()
        elif key == "speaker":
            speaker = value.upper()
        else:
            file_ids.append(value)
    body = FILTER_RE.sub(" ", raw)
    phrases = [p.strip() for p in PHRASE_RE.findall(body) if p.strip()]
    text = re.sub(r"\s+", " ", body.replace('"', " ")).strip()
    if not text:
        raise QueryError("query has filters but no search terms")
    return ParsedQuery(
        raw=raw,
        text=text,
        phrases=phrases,
        role=role,
        speaker=speaker,
        file_ids=file_ids,
        intent=classify_intent(text, phrases),
        content_tokens=content_tokens(text),
    )


def fusion_weights(intent: str, adaptive: bool) -> dict[str, float]:
    return dict(INTENT_WEIGHTS[intent]) if adaptive else {"lexical": 1.0, "dense": 1.0}

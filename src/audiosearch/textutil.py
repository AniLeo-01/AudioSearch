"""Small text utilities shared by indexing and query processing."""

from __future__ import annotations

import re
import unicodedata

# A compact English stopword list (Postgres' `english` config removes these from tsvectors too).
STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had
    has have having he her here hers herself him himself his how i if in into is it its itself just
    let me more most my myself no nor not now of off on once only or other our ours ourselves out
    over own same she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when where which
    while who whom why will with would you your yours yourself yourselves also yeah okay oh um uh
    like really got get gets going go know think thing things lot kind sort well right mean
    actually""".split()  # noqa: SIM905 - compact word list
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_POSSESSIVE_RE = re.compile(r"'s\b")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")
    return text.lower()


def tokens(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens (possessive 's dropped, other apostrophes removed)."""
    return _TOKEN_RE.findall(_POSSESSIVE_RE.sub("", normalize(text)).replace("'", ""))


def content_tokens(text: str, min_len: int = 3) -> list[str]:
    return [t for t in tokens(text) if len(t) >= min_len and t not in STOPWORDS and not t.isdigit()]


def vocabulary_terms(text: str) -> set[str]:
    """Distinct spoken terms eligible for sounds-like matching."""
    return {t for t in content_tokens(text, min_len=3) if any(c.isalpha() for c in t)}

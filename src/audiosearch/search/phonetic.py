"""Sounds-like query expansion: recover from ASR misrecognitions and user misspellings.

ASR systems mangle rare proper nouns ("Iaria" -> "Aria", "Coggins" -> "Coggin's"), and users misspell
them ("Zubaire", "Ellington Feild").  Exact lexical search then silently returns nothing useful, while
dense embeddings of rare names are unreliable.  For every query term that does **not** occur in the
spoken vocabulary we look for spoken terms that are orthographically (pg_trgm trigram similarity,
Levenshtein) or phonetically (Metaphone / Double Metaphone via fuzzystrmatch) close, and add them to
the BM25 query with a confidence-scaled weight.  In-vocabulary terms are never expanded, so precise
queries are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

MIN_TOKEN_LEN = 4
MAX_PER_TOKEN = 3
MIN_SCORE = 0.62
EXPANSION_WEIGHT = 0.9  # expansions never outweigh an exact hit


@dataclass(frozen=True)
class Expansion:
    source: str  # query token as typed
    term: str  # spoken term found in the corpus vocabulary
    score: float  # 0..1 confidence
    kind: str  # "phonetic" | "spelling"


def score_candidate(tok: str, term: str, trgm: float, lev: int, metaphone_eq: bool, dmeta_eq: bool) -> tuple[float, str]:
    """Combine orthographic and phonetic evidence into a single confidence score."""
    lev_sim = 1.0 - lev / max(len(tok), len(term), 1)
    ortho = max(trgm, lev_sim)
    if metaphone_eq:
        return min(1.0, max(ortho, 0.55) + 0.25), "phonetic"
    if dmeta_eq:
        return min(1.0, ortho + 0.12), "phonetic"
    return ortho, "spelling"


def expand_terms(conn: psycopg.Connection, query_tokens: list[str]) -> list[Expansion]:
    cands = sorted({t for t in query_tokens if len(t) >= MIN_TOKEN_LEN and t.isalpha()})
    if not cands:
        return []
    known = {
        r[0] for r in conn.execute("SELECT term FROM vocabulary WHERE term = ANY(%s)", (cands,)).fetchall()
    }
    oov = [t for t in cands if t not in known]
    if not oov:
        return []
    rows = conn.execute(
        """
        SELECT q.tok, v.term, v.df,
               similarity(v.term, q.tok) AS trgm,
               levenshtein(v.term, q.tok) AS lev,
               v.metaphone = metaphone(q.tok, 12) AS mph,
               (dmetaphone(q.tok) IN (v.dmetaphone, v.dmetaphone_alt)
                OR dmetaphone_alt(q.tok) IN (v.dmetaphone, v.dmetaphone_alt)) AS dm
        FROM unnest(%s::text[]) AS q(tok)
        JOIN vocabulary v
          ON v.term %% q.tok
          OR v.metaphone = metaphone(q.tok, 12)
          OR v.dmetaphone = dmetaphone(q.tok)
        """,
        (oov,),
    ).fetchall()
    best: dict[str, list[Expansion]] = {}
    for tok, term, _df, trgm, lev, mph, dm in rows:
        # Double Metaphone codes are only 4 characters long: require some orthographic support too.
        if dm and not mph and float(trgm) < 0.25:
            continue
        score, kind = score_candidate(tok, term, float(trgm), int(lev), bool(mph), bool(dm))
        if score >= MIN_SCORE:
            best.setdefault(tok, []).append(Expansion(tok, term, round(score, 3), kind))
    out: list[Expansion] = []
    for tok in oov:
        out.extend(sorted(best.get(tok, []), key=lambda e: (-e.score, e.term))[:MAX_PER_TOKEN])
    return out

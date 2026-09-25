"""Sounds-like query expansion: recover from ASR misrecognitions and user misspellings.

ASR systems mangle rare proper nouns and jargon ("apheresis" -> "aphoresis"/"ismoresis",
"Pizzamiglio" -> "Pizzamilio", "Abba Zubair" -> "Abizubair"), and users misspell them ("Zubaire",
"vestibuler").  Exact lexical search then silently misses, and dense embeddings of rare names are
unreliable.  We expand such query terms with *spoken* terms from the corpus vocabulary that are
orthographically (pg_trgm trigram similarity, Levenshtein, longest common substring) or phonetically
(Metaphone / Double Metaphone via fuzzystrmatch) close, weighted by confidence.

Expansion policy per query token (``lexicon`` = common English words, from the embedding model's
tokenizer vocabulary):

=====================  ==================  ================================================
token in corpus?       common English?     expansion
=====================  ==================  ================================================
no                     no (name / typo)    phonetic *or* spelling neighbours  (score >= 0.62)
no                     yes ("inside")      strong phonetic identity only      (score >= 0.80)
yes, rare (df <= 2)    no ("apheresis")    phonetic variants only             (score >= 0.80)
yes                    otherwise           none - precise queries stay precise
=====================  ==================  ================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

import psycopg

MIN_TOKEN_LEN = 4
MAX_PER_TOKEN = 3
MIN_SCORE_OPEN = 0.62
MIN_SCORE_STRICT = 0.80
RARE_DF = 2
EXPANSION_WEIGHT = 0.9  # expansions never outweigh an exact hit


@dataclass(frozen=True)
class Expansion:
    source: str  # query token as typed
    term: str  # spoken term found in the corpus vocabulary
    score: float  # 0..1 confidence
    kind: str  # "phonetic" | "spelling"


def substring_ratio(tok: str, term: str) -> float:
    """Longest common substring relative to the query token (captures ASR word merges)."""
    m = SequenceMatcher(None, tok, term, autojunk=False).find_longest_match(0, len(tok), 0, len(term))
    return m.size / max(len(tok), 1)


def score_candidate(
    tok: str, term: str, trgm: float, lev: int, metaphone_eq: bool, dmeta_eq: bool
) -> tuple[float, str]:
    """Combine orthographic and phonetic evidence into a single confidence score in [0, 1]."""
    lev_sim = 1.0 - lev / max(len(tok), len(term), 1)
    ortho = max(trgm, lev_sim)
    if len(tok) >= 5 and substring_ratio(tok, term) >= 0.85:
        ortho = max(ortho, 0.70)
    if metaphone_eq:
        return min(1.0, max(ortho, 0.55) + 0.25), "phonetic"
    if dmeta_eq:
        return min(1.0, ortho + 0.12), "phonetic"
    return ortho, "spelling"


def expand_terms(
    conn: psycopg.Connection, query_tokens: list[str], lexicon: frozenset[str] = frozenset()
) -> list[Expansion]:
    cands = sorted({t for t in query_tokens if len(t) >= MIN_TOKEN_LEN and t.isalpha()})
    if not cands:
        return []
    known: dict[str, int] = dict(
        conn.execute("SELECT term, df FROM vocabulary WHERE term = ANY(%s)", (cands,)).fetchall()
    )
    policy: dict[str, str] = {}
    for t in cands:
        common = t in lexicon
        if t in known:
            if known[t] <= RARE_DF and not common:
                policy[t] = "strict"  # look for other ASR spellings of the same rare word
        else:
            policy[t] = "strict" if common else "open"
    if not policy:
        return []
    rows = conn.execute(
        """
        SELECT q.tok, v.term,
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
          OR (length(q.tok) >= 5 AND strpos(v.term, left(q.tok, length(q.tok) - 1)) > 0)
        WHERE v.term <> q.tok
        """,
        (sorted(policy),),
    ).fetchall()
    best: dict[str, list[Expansion]] = {}
    for tok, term, trgm, lev, mph, dm in rows:
        # Double Metaphone codes are only 4 characters long: require some orthographic support too.
        if dm and not mph and float(trgm) < 0.25:
            continue
        score, kind = score_candidate(tok, term, float(trgm), int(lev), bool(mph), bool(dm))
        ok = score >= MIN_SCORE_OPEN if policy[tok] == "open" else bool(mph) and score >= MIN_SCORE_STRICT
        if ok:
            best.setdefault(tok, []).append(Expansion(tok, term, round(score, 3), kind))
    out: list[Expansion] = []
    for tok in sorted(policy):
        out.extend(sorted(best.get(tok, []), key=lambda e: (-e.score, e.term))[:MAX_PER_TOKEN])
    return out

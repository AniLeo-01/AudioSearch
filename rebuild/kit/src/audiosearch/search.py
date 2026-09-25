"""Hybrid search: BM25 (+ sounds-like) || pgvector -> IDF-coverage-weighted RRF -> moment snapping -> NMS."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np
from psycopg_pool import ConnectionPool

from . import config
from .embed import Embedder
from .models import Chunk
from .text import content_tokens, tokens

QUESTION_WORDS = frozenset(
    """what how why when where who which whose whom is are was were do does did can could should would
    will has have had tell explain describe""".split()  # noqa: SIM905 - compact word list
)
PHRASE_RE = re.compile(r'"([^"]{1,200})"')
ROLE_RE = re.compile(r"\brole:(host|guest)\b", re.I)


# ---------------------------------------------------------------------------------------------- query
@dataclass
class Query:
    text: str
    phrases: list[str]
    role: str | None
    intent: str  # phrase | question | keyword | topic
    content: list[str]


def parse(raw: str, role: str | None = None) -> Query:
    m = ROLE_RE.search(raw)
    role = role or (m.group(1).lower() if m else None)
    body = ROLE_RE.sub(" ", raw)
    phrases = [p.strip() for p in PHRASE_RE.findall(body) if p.strip()]
    text = " ".join(body.replace('"', " ").split())
    toks, content = tokens(text), content_tokens(text)
    if phrases:
        intent = "phrase"
    elif text.endswith("?") or (toks and toks[0] in QUESTION_WORDS):
        intent = "question"
    elif len(content) <= 2 and len(toks) <= 3:
        intent = "keyword"
    else:
        intent = "topic"
    return Query(text, phrases, role, intent, content)


def where_clause(q: Query) -> tuple[str, dict]:
    """Filters shared by both channels; user input only ever travels as bound parameters."""
    parts, params = ["TRUE"], {}
    for i, p in enumerate(q.phrases):
        parts.append(f"c.tsv @@ phraseto_tsquery('english', %(ph{i})s)")
        params[f"ph{i}"] = p
    if q.role:
        parts.append(
            "EXISTS (SELECT 1 FROM speakers s WHERE s.file_id = c.file_id"
            " AND s.role = %(role)s AND s.label = ANY(c.speakers))"
        )
        params["role"] = q.role
    return " AND ".join(parts), params


# -------------------------------------------------------------------------------------------- lexical
def analyze(conn, text: str) -> list[str]:
    """Stemmed, stopword-free lexemes exactly as Postgres indexes them (one per occurrence)."""
    rows = conn.execute(
        "SELECT lexeme, coalesce(array_length(positions, 1), 1) FROM unnest(to_tsvector('english', %s))", (text,)
    ).fetchall()
    return [lx for lx, n in rows for _ in range(n)]


BM25_SQL = """
WITH q(lexeme, w) AS (SELECT * FROM unnest(%(lex)s::text[], %(w)s::float8[])),
     cs AS (SELECT count(*)::float8 AS n, avg(doc_len)::float8 AS avgdl FROM chunks),
     df AS (SELECT lexeme, count(*)::float8 AS df FROM chunk_terms WHERE lexeme = ANY(%(lex)s) GROUP BY lexeme)
SELECT c.id,
       sum(q.w * ln(1 + (cs.n - df.df + 0.5) / (df.df + 0.5))
           * ct.tf * 2.2 / (ct.tf + 1.2 * (0.25 + 0.75 * c.doc_len / cs.avgdl))) AS score,
       array_agg(q.lexeme) AS matched
FROM q JOIN df USING (lexeme) JOIN chunk_terms ct USING (lexeme) JOIN chunks c ON c.id = ct.chunk_id CROSS JOIN cs
WHERE {where}
GROUP BY c.id ORDER BY score DESC, c.id LIMIT %(n)s
"""  # BM25 with k1 = 1.2, b = 0.75


def bm25(conn, lexemes: list[tuple[str, float, str]], where: str, params: dict, n: int) -> list[tuple]:
    w: dict[str, float] = {}
    for lx, wt, _unit in lexemes:  # repeated query terms add up
        w[lx] = w.get(lx, 0.0) + wt
    if not w:
        return []
    rows = conn.execute(BM25_SQL.format(where=where), {**params, "lex": list(w), "w": list(w.values()), "n": n})
    return [(cid, float(score), list(matched)) for cid, score, matched in rows.fetchall()]


def idf(conn, lexemes: list[str]) -> tuple[dict[str, float], int]:
    n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    rows = conn.execute(
        "SELECT lexeme, count(*) FROM chunk_terms WHERE lexeme = ANY(%s) GROUP BY lexeme", (lexemes,)
    ).fetchall()
    return {lx: math.log(1 + (n - df + 0.5) / (df + 0.5)) for lx, df in rows}, n


# ------------------------------------------------------------------------------------ sounds-like (N1)
PHONETIC_SQL = """
SELECT q.tok, v.term, similarity(v.term, q.tok), levenshtein(v.term, q.tok),
       v.metaphone = metaphone(q.tok, 12),
       dmetaphone(q.tok) IN (v.dmetaphone, v.dmetaphone_alt)
         OR dmetaphone_alt(q.tok) IN (v.dmetaphone, v.dmetaphone_alt)
FROM unnest(%s::text[]) AS q(tok)
JOIN vocabulary v ON v.term %% q.tok OR v.metaphone = metaphone(q.tok, 12) OR v.dmetaphone = dmetaphone(q.tok)
                  OR (length(q.tok) >= 5 AND strpos(v.term, left(q.tok, length(q.tok) - 1)) > 0)
WHERE v.term <> q.tok
"""


def substring_ratio(tok: str, term: str) -> float:
    m = SequenceMatcher(None, tok, term, autojunk=False).find_longest_match(0, len(tok), 0, len(term))
    return m.size / len(tok)


def expand(conn, toks: list[str], lexicon: frozenset[str]) -> list[tuple[str, str, float]]:
    """(query token, spoken term, score) for query words that are unknown to the corpus or rare in it.

    unknown + not common English (name/typo) -> spelling or phonetic neighbours, score >= 0.62
    unknown + common English ("inside")      -> only a Metaphone-identical term, score >= 0.80
    rare in corpus (df <= 2) + not common    -> other ASR spellings: Metaphone-identical, score >= 0.80
    """
    cands = sorted({t for t in toks if len(t) >= 4 and t.isalpha()})
    if not cands:
        return []
    df = dict(conn.execute("SELECT term, df FROM vocabulary WHERE term = ANY(%s)", (cands,)).fetchall())
    policy = {}
    for t in cands:
        if t not in df:
            policy[t] = "strict" if t in lexicon else "open"
        elif df[t] <= 2 and t not in lexicon:
            policy[t] = "strict"
    if not policy:
        return []
    best: dict[str, list[tuple[float, str]]] = {}
    for tok, term, trgm, lev, mph, dm in conn.execute(PHONETIC_SQL, (sorted(policy),)).fetchall():
        if dm and not mph and trgm < 0.25:  # 4-character Double Metaphone codes collide easily
            continue
        ortho = max(trgm, 1 - lev / max(len(tok), len(term)))
        if len(tok) >= 5 and substring_ratio(tok, term) >= 0.85:  # ASR word merges: "abizubair"
            ortho = max(ortho, 0.70)
        score = min(1.0, max(ortho, 0.55) + 0.25) if mph else min(1.0, ortho + 0.12) if dm else ortho
        if (score >= 0.62) if policy[tok] == "open" else (mph and score >= 0.80):
            best.setdefault(tok, []).append((round(score, 3), term))
    return [(t, term, s) for t in sorted(best) for s, term in sorted(best[t], key=lambda x: (-x[0], x[1]))[:3]]


def expansion_lexemes(conn, exps: list[tuple[str, str, float]], seen: set[str]) -> list[tuple[str, float, str]]:
    """(lexeme, weight, unit): an expansion stands in for its source token's lexeme, at weight 0.9 * score."""
    terms = sorted({e[0] for e in exps} | {e[1] for e in exps})
    rows = conn.execute(
        """SELECT t.term, array_remove(array_agg(x.lexeme), NULL) FROM unnest(%s::text[]) AS t(term)
           LEFT JOIN LATERAL unnest(to_tsvector('english', t.term)) AS x ON TRUE GROUP BY t.term""",
        (terms,),
    ).fetchall()
    lex = {term: lxs for term, lxs in rows}
    seen, out = set(seen), []
    for src, term, score in exps:
        unit = (lex.get(src) or [src])[0]
        for lx in lex.get(term, []):
            if lx not in seen:
                seen.add(lx)
                out.append((lx, 0.9 * score, unit))
    return out


# ------------------------------------------------------------------------------------- fusion (N2)
def idf_coverage(idf_map: dict[str, float], n: int, lexemes, hits) -> dict[str, float]:
    """Share of the query's IDF mass each lexical hit covers; sounds-like stand-ins earn partial credit."""
    units = {lx for lx, _w, unit in lexemes if lx == unit}
    if not units:
        return {}
    idf_max = math.log(1 + (n + 0.5) / 0.5)  # query words absent from the corpus count as maximally specific
    unit_w = {u: idf_map.get(u, idf_max) for u in units}
    total = sum(unit_w.values())
    stand_in = {lx: (unit, w) for lx, w, unit in lexemes if lx != unit}
    cov = {}
    for cid, _score, matched in hits:
        credit: dict[str, float] = {}
        for m in matched:
            if m in unit_w:
                credit[m] = 1.0
            elif m in stand_in and stand_in[m][0] in unit_w:
                u, w = stand_in[m]
                credit[u] = max(credit.get(u, 0.0), w)
        cov[cid] = sum(unit_w[u] * c for u, c in credit.items()) / total
    return cov


def rrf(channels: dict[str, list[str]], k: int, mult: dict[str, dict[str, float]]):
    score: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, ids in channels.items():
        for r, cid in enumerate(ids, 1):
            score[cid] = score.get(cid, 0.0) + mult.get(name, {}).get(cid, 1.0) / (k + r)
            ranks.setdefault(cid, {})[name] = r
    return sorted(score, key=lambda c: (-score[c], min(ranks[c].values()), c)), score, ranks


# ------------------------------------------------------------------------------------ moments (N3)
@dataclass
class Utt:
    file_id: str
    idx: int
    speaker: str
    start: float
    end: float
    text: str
    words: list
    dsim: float | None
    matched: list[str]
    headline: str | None


@dataclass
class Moment:
    chunk: Chunk
    utt: Utt
    match_time: float
    highlights: list[tuple[int, int]] = field(default_factory=list)


UTT_SQL = """
WITH r AS (SELECT * FROM unnest(%(f)s::text[], %(a)s::int[], %(b)s::int[]) AS r(fid, a, b)),
     u AS (SELECT DISTINCT ON (u.file_id, u.idx) u.* FROM r
           JOIN utterances u ON u.file_id = r.fid AND u.idx >= r.a AND u.idx < r.b)
SELECT u.file_id, u.idx, u.speaker, u.start_sec, u.end_sec, u.text, u.words,
       CASE WHEN %(has_q)s THEN 1 - (u.embedding <=> %(q)s::vector) END,
       ARRAY(SELECT unnest(tsvector_to_array(u.tsv)) INTERSECT SELECT unnest(%(lex)s::text[])),
       CASE WHEN %(tsq)s <> '' AND u.tsv @@ %(tsq)s::tsquery
            THEN ts_headline('english', u.text, %(tsq)s::tsquery,
                             'HighlightAll=true, StartSel=' || chr(2) || ', StopSel=' || chr(3)) END
FROM u
"""


def tsquery(lexemes: list[str]) -> str:
    """OR-query over already-stemmed lexemes (quoted, so Postgres does not stem them again)."""
    return " | ".join("'" + lx.replace("\\", "\\\\").replace("'", "''") + "'" for lx in dict.fromkeys(lexemes))


def parse_headline(h: str) -> list[tuple[int, int]]:
    spans, out, start = [], 0, None
    for ch in h:
        if ch == "\x02":
            start = out
        elif ch == "\x03":
            if start is not None and out > start:
                spans.append((start, out))
            start = None
        else:
            out += 1
    return spans


def first_match_time(u: Utt, spans: list[tuple[int, int]]) -> float:
    """Start of the first highlighted word (the utterance text is its words joined by spaces)."""
    if spans:
        offset = 0
        for tok, start, _end in u.words:
            if offset + len(tok) > spans[0][0]:
                return start
            offset += len(tok) + 1
    return u.start


def snap(c: Chunk, utts: dict, idf_w: dict[str, float], lex_w: float, allowed: set[str] | None) -> Moment | None:
    """Best utterance of a passage: IDF-weighted matched lexemes mixed with query-utterance cosine."""
    cands = [
        u
        for i in range(c.utt_start, c.utt_end)
        if (u := utts.get((c.file_id, i))) and (allowed is None or u.speaker in allowed)
    ]
    if not cands:
        return None
    lex = np.array([sum(idf_w.get(m, 0.0) for m in u.matched) for u in cands])
    sem = np.array([u.dsim or 0.0 for u in cands])
    lex_n = lex / lex.max() if lex.max() > 0 else lex
    sem_n = np.zeros_like(sem)
    if np.ptp(sem) > 0:
        sem_n = (sem - sem.min()) / np.ptp(sem)
        sem_n[np.array([len(u.text.split()) < 4 for u in cands])] *= 0.5  # "Yeah." resembles everything
    w = lex_w if lex.max() > 0 else 0.0
    best = cands[int(np.argmax(w * lex_n + (1 - w) * sem_n))]
    spans = parse_headline(best.headline) if best.headline else []
    return Moment(c, best, first_match_time(best, spans), spans)


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0)) / max(min(a1 - a0, b1 - b0), 1e-6)


def temporal_nms(moments: list[Moment], gap: float) -> list[Moment]:
    kept: list[Moment] = []
    for m in moments:
        if not any(
            k.chunk.file_id == m.chunk.file_id
            and (
                k.utt.idx == m.utt.idx
                or abs(k.utt.start - m.utt.start) < gap
                or overlap(k.chunk.start, k.chunk.end, m.chunk.start, m.chunk.end) > 0.5
            )
            for k in kept
        ):
            kept.append(m)
    return kept


# --------------------------------------------------------------------------------------------- engine
@dataclass
class Hit:
    rank: int
    file_id: str
    title: str
    start: float
    end: float
    match_time: float
    speaker: str
    role: str
    name: str | None
    text: str
    highlights: list[tuple[int, int]]
    score: float
    channels: dict[str, int]


class Searcher:
    def __init__(self, pool: ConnectionPool, emb: Embedder) -> None:
        self.pool, self.emb, self.lexicon = pool, emb, emb.lexicon()

    @classmethod
    def create(cls) -> Searcher:
        from pgvector.psycopg import register_vector

        pool = ConnectionPool(config.DATABASE_URL, min_size=1, max_size=4, configure=register_vector, open=True)
        return cls(pool, Embedder())

    def search(self, raw: str, k: int = 10, role: str | None = None, mode: str = "hybrid", coverage: bool = True,
               phonetic: bool = True, snap: bool = True, nms: bool = True) -> list[Hit]:  # fmt: skip
        q = parse(raw, role)
        where, params = where_clause(q)
        channels: dict[str, list[str]] = {}
        mult: dict[str, dict[str, float]] = {}
        lexemes: list[tuple[str, float, str]] = []
        qvec = None
        with self.pool.connection() as conn:
            if mode in ("hybrid", "lexical"):
                lexemes = [(lx, 1.0, lx) for lx in analyze(conn, q.text)]
                if phonetic:
                    exps = expand(conn, q.content, self.lexicon)
                    lexemes += expansion_lexemes(conn, exps, {lx for lx, _, _ in lexemes})
                hits = bm25(conn, lexemes, where, params, config.DEPTH)
                channels["lexical"] = [h[0] for h in hits]
                if coverage and mode == "hybrid" and hits:
                    idf_map, n = idf(conn, sorted({u for _, _, u in lexemes}))
                    mult["lexical"] = idf_coverage(idf_map, n, lexemes, hits)
            if mode in ("hybrid", "semantic"):
                qvec = self.emb.query(q.text)
                conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(config.EF_SEARCH),))
                rows = conn.execute(
                    f"SELECT c.id FROM chunks c WHERE {where} ORDER BY c.embedding <=> %(q)s LIMIT %(n)s",
                    {**params, "q": qvec, "n": config.DEPTH},
                ).fetchall()
                channels["dense"] = [r[0] for r in rows]
            order, score, ranks = rrf(channels, config.RRF_K, mult)
            pool_ids = order[: max(3 * k, k + 20)]
            lex_w = 0.3 if mode == "semantic" else 0.8 if q.intent in ("keyword", "phrase") else 0.5
            moments = self._moments(conn, pool_ids, qvec, lexemes, lex_w, q.role, snap)
            if nms:
                moments = temporal_nms(moments, config.NMS_GAP)
            return self._hits(conn, moments[:k], score, ranks)

    def _moments(self, conn, ids, qvec, lexemes, lex_w, role, do_snap) -> list[Moment]:
        rows = conn.execute(
            "SELECT id, file_id, utt_start, utt_end, start_sec, end_sec, speakers, text FROM chunks WHERE id = ANY(%s)",
            (ids,),
        ).fetchall()
        by_id = {r[0]: Chunk(*r) for r in rows}
        chunks = [by_id[i] for i in ids if i in by_id]
        lex_list = sorted({lx for lx, _, _ in lexemes})
        utt_rows = conn.execute(
            UTT_SQL,
            {
                "f": [c.file_id for c in chunks],
                "a": [c.utt_start for c in chunks],
                "b": [c.utt_end for c in chunks],
                "has_q": qvec is not None,
                "q": qvec if qvec is not None else np.zeros(1, np.float32),
                "lex": lex_list,
                "tsq": tsquery(lex_list),
            },
        ).fetchall()
        utts = {(r[0], r[1]): Utt(*r) for r in utt_rows}
        weight: dict[str, float] = {}
        for lx, w, _ in lexemes:  # snapping IDF, scaled by the (expansion) weight of each lexeme
            weight[lx] = max(weight.get(lx, 0.0), w)
        idf_map, _ = idf(conn, list(weight))
        idf_w = {lx: v * weight[lx] for lx, v in idf_map.items()}
        allowed: dict[str, set[str]] | None = None
        if role:
            allowed = {}
            for f, label in conn.execute("SELECT file_id, label FROM speakers WHERE role = %s", (role,)).fetchall():
                allowed.setdefault(f, set()).add(label)
        out = []
        for c in chunks:
            if not do_snap:  # ablation: report the passage's first utterance
                if (u := utts.get((c.file_id, c.utt_start))) is not None:
                    out.append(Moment(c, u, c.start))
                continue
            m = snap(c, utts, idf_w, lex_w, None if allowed is None else allowed.get(c.file_id, set()))
            if m is not None:
                out.append(m)
        return out

    @staticmethod
    def _hits(conn, moments: list[Moment], score, ranks) -> list[Hit]:
        fids = sorted({m.chunk.file_id for m in moments})
        titles = dict(
            conn.execute("SELECT file_id, title FROM audio_files WHERE file_id = ANY(%s)", (fids,)).fetchall()
        )
        spk = {
            (f, lab): (role, name)
            for f, lab, role, name in conn.execute(
                "SELECT file_id, label, role, name FROM speakers WHERE file_id = ANY(%s)", (fids,)
            ).fetchall()
        }
        hits = []
        for rank, m in enumerate(moments, 1):
            u = m.utt
            role, name = spk.get((u.file_id, u.speaker), ("unknown", None))
            hits.append(
                Hit(rank, u.file_id, titles.get(u.file_id, u.file_id), round(u.start, 2), round(u.end, 2),
                    round(m.match_time, 2), u.speaker, role, name, u.text, m.highlights,
                    round(score[m.chunk.id], 5), ranks[m.chunk.id])
            )  # fmt: skip
        return hits

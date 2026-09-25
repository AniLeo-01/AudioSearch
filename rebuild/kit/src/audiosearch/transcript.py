"""Speaker-labelled words -> sentence-like utterances -> host/guest roles."""

from __future__ import annotations

import math
import re

from .models import Speaker, Utterance, Word

ABBREV = {"dr.", "mr.", "mrs.", "ms.", "st.", "jr.", "sr.", "prof.", "vs.", "etc.", "e.g.", "i.e.", "u.s.",
          "u.k.", "a.m.", "p.m.", "no.", "approx.", "inc.", "co.", "lt.", "col.", "gen.", "sgt."}  # fmt: skip
END_RE = re.compile(r"[.?!][\"')\]]*$")


def ends_sentence(tok: str) -> bool:
    t = tok.lower()
    return t not in ABBREV and not re.fullmatch(r"(?:[a-z]\.){2,}", t) and bool(END_RE.search(tok))


def build_utterances(words: list[Word], max_words: int = 50, max_gap: float = 1.5) -> list[Utterance]:
    """New utterance on speaker change, sentence end, pause >= max_gap, or > max_words (split at a comma)."""
    utts: list[Utterance] = []
    cur: list[Word] = []

    def flush() -> None:
        if cur:
            text = " ".join(w.text for w in cur)
            words_json = [[w.text, w.start, w.end] for w in cur]
            utts.append(Utterance(len(utts), cur[0].speaker, cur[0].start, cur[-1].end, text, words_json))
            cur.clear()

    for w in words:
        if cur and (w.speaker != cur[-1].speaker or w.start - cur[-1].end >= max_gap):
            flush()
        cur.append(w)
        if ends_sentence(w.text):
            flush()
        elif len(cur) >= max_words:  # split at the last comma in the second half, else hard split
            k = next((k for k in range(len(cur) - 1, len(cur) // 2, -1) if cur[k].text.endswith(",")), None)
            if k is None:
                flush()
            else:
                tail = cur[k + 1 :]
                del cur[k + 1 :]
                flush()
                cur.extend(tail)
    flush()
    return utts


def infer_roles(utts: list[Utterance], host: str | None = None, guest: str | None = None) -> list[Speaker]:
    """Host = asks more questions, talks less, usually opens. Names attached only when confident."""
    total = sum(len(u.text.split()) for u in utts) or 1
    scored = []
    for label in dict.fromkeys(u.speaker for u in utts):
        us = [u for u in utts if u.speaker == label]
        n_words = sum(len(u.text.split()) for u in us)
        q_rate = sum(u.is_question for u in us) / len(us)
        score = 3.0 * q_rate + 1.5 * (0.5 - n_words / total) + 0.5 * (label == utts[0].speaker)
        scored.append((score, Speaker(label, n_words=n_words, question_rate=round(q_rate, 4))))
    scored.sort(key=lambda t: -t[0])
    margin = scored[0][0] - scored[1][0] if len(scored) > 1 else 1.0
    conf = 1 / (1 + math.exp(-4 * margin))
    for i, (_, s) in enumerate(scored):
        s.role, s.confidence = ("host" if i == 0 else "guest"), round(conf, 4)
        if conf >= 0.75:
            s.name = host if i == 0 else guest
    return [s for _, s in scored]

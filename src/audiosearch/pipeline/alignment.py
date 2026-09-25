"""Merge word-level ASR output and speaker labels into utterances and turns."""

from __future__ import annotations

import re

from audiosearch.domain import Turn, Utterance, Word

# Tokens ending in "." that do not end a sentence.
ABBREVIATIONS = {
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "st.",
    "jr.",
    "sr.",
    "prof.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "u.s.",
    "u.k.",
    "a.m.",
    "p.m.",
    "no.",
    "approx.",
    "inc.",
    "co.",
    "lt.",
    "col.",
    "gen.",
    "sgt.",
}
SENTENCE_END_RE = re.compile(r"[.?!][\"')\]]*$")


def ends_sentence(token: str) -> bool:
    t = token.lower()
    if t in ABBREVIATIONS:
        return False
    if re.fullmatch(r"(?:[a-z]\.){2,}", t):  # initialisms such as "u.s."
        return False
    return bool(SENTENCE_END_RE.search(token))


def join_words(words: list[Word]) -> str:
    return " ".join(w.text for w in words).strip()


def build_utterances(
    words: list[Word],
    file_id: str,
    max_words: int = 50,
    max_gap: float = 1.5,
) -> list[Utterance]:
    """Split the word stream into single-speaker, sentence-like utterances.

    A new utterance starts when the speaker changes, after sentence-final punctuation, after a long
    pause, or when an utterance exceeds ``max_words`` (split at the last comma if there is one).
    """
    utts: list[Utterance] = []
    cur: list[int] = []

    def flush() -> None:
        if not cur:
            return
        ws = [words[i] for i in cur]
        utts.append(
            Utterance(
                id=f"{file_id}_u{len(utts):04d}",
                idx=len(utts),
                speaker=ws[0].speaker or "SPEAKER_00",
                start=ws[0].start,
                end=ws[-1].end,
                text=join_words(ws),
                word_start=cur[0],
                word_end=cur[-1] + 1,
            )
        )
        cur.clear()

    for i, w in enumerate(words):
        if cur:
            prev = words[cur[-1]]
            if w.speaker != prev.speaker or (w.start - prev.end) >= max_gap:
                flush()
        cur.append(i)
        if ends_sentence(w.text):
            flush()
        elif len(cur) >= max_words:
            # split at the last comma in the second half of the utterance, else hard split
            comma = next((k for k in range(len(cur) - 1, len(cur) // 2, -1) if words[cur[k]].text.endswith(",")), None)
            if comma is None:
                flush()
            else:
                tail = cur[comma + 1 :]
                del cur[comma + 1 :]
                flush()
                cur.extend(tail)
    flush()
    return utts


def build_turns(utterances: list[Utterance]) -> list[Turn]:
    turns: list[Turn] = []
    for u in utterances:
        if turns and turns[-1].speaker == u.speaker:
            t = turns[-1]
            t.end = u.end
            t.utterance_end = u.idx + 1
        else:
            turns.append(Turn(len(turns), u.speaker, u.start, u.end, u.idx, u.idx + 1))
    return turns

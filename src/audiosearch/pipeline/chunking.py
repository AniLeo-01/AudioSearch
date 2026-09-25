"""Conversation-aware retrieval chunks.

Chunks are sliding windows of whole utterances (never cutting a sentence) that may span both
speakers, so a question and the start of its answer stay together.  Speaker attribution is not lost:
every search hit is later *snapped* to a single utterance, which has exactly one speaker.

**Dialogue-context augmentation.**  In interviews, answers routinely omit the topic words that were
in the question ("It was incredible - we saw the cells grow three times faster").  When a window
starts inside an answer, the interviewer's most recent question is prepended to the text that gets
*embedded* (never to the lexically indexed text, so keyword hits always point at words actually
spoken in the window).  The effect is measured in the ablation study (docs/EVALUATION.md).
"""

from __future__ import annotations

from audiosearch.domain import Chunk, Transcript, Utterance

MAX_CONTEXT_LOOKBACK_SEC = 240.0


def _wc(u: Utterance) -> int:
    return len(u.text.split())


def find_context_question(utts: list[Utterance], start_idx: int) -> Utterance | None:
    """The other speaker's most recent question before ``start_idx`` (skipping the current turn).

    Returns None when the window already begins with that speaker, when no question exists in the
    preceding turn, or when it is too far in the past to plausibly frame the window.
    """
    if start_idx <= 0:
        return None
    speaker = utts[start_idx].speaker
    k = start_idx - 1
    while k >= 0 and utts[k].speaker == speaker:  # walk back to the start of the current turn
        k -= 1
    if k < 0:
        return None
    other = utts[k].speaker
    while k >= 0 and utts[k].speaker == other:  # scan the other speaker's preceding turn
        if utts[k].is_question:
            if utts[start_idx].start - utts[k].end > MAX_CONTEXT_LOOKBACK_SEC:
                return None
            return utts[k]
        k -= 1
    return None


def build_chunks(
    transcript: Transcript,
    target_words: int = 70,
    stride_words: int = 35,
    context: str = "question",
    context_max_words: int = 40,
    title: str | None = None,
) -> list[Chunk]:
    if target_words <= 0 or stride_words <= 0:
        raise ValueError("target_words and stride_words must be positive")
    if context not in {"none", "question", "question+title"}:
        raise ValueError(f"unknown chunk context mode: {context}")
    utts = transcript.utterances
    counts = [_wc(u) for u in utts]
    chunks: list[Chunk] = []
    i, n = 0, len(utts)
    while i < n:
        j, words = i, 0
        while j < n and words < target_words:
            words += counts[j]
            j += 1
        window = utts[i:j]
        text = " ".join(u.text for u in window)
        embed_text = text
        if context != "none":
            q = find_context_question(utts, i)
            if q is not None:
                q_words = q.text.split()[-context_max_words:]
                embed_text = " ".join(q_words) + "\n" + text
            if context == "question+title" and title:
                embed_text = f"{title}.\n{embed_text}"
        speakers: list[str] = []
        for u in window:
            if u.speaker not in speakers:
                speakers.append(u.speaker)
        chunks.append(
            Chunk(
                id=f"{transcript.file_id}_c{len(chunks):04d}",
                file_id=transcript.file_id,
                idx=len(chunks),
                start=window[0].start,
                end=window[-1].end,
                utterance_start=i,
                utterance_end=j,
                text=text,
                embed_text=embed_text,
                speakers=speakers,
                n_words=words,
            )
        )
        if j >= n:
            break
        k, acc = i, 0
        while k < j and acc < stride_words:
            acc += counts[k]
            k += 1
        i = max(k, i + 1)
    return chunks

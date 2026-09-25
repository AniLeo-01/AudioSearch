"""Upstream-stage quality against NASA's human transcripts: ASR WER and speaker attribution.

The reference covers the full episode while the audio is an excerpt, so we first locate the excerpt in
the reference by fuzzy-matching the first/last words of the hypothesis, then align word by word.

* **WER** - (S + D + I) / N after light normalisation (case, punctuation, hyphens).  NASA transcripts
  are "clean verbatim" (false starts and fillers removed), so this *over*-states true ASR error.
* **Speaker attribution accuracy** - over aligned word pairs, the fraction whose diarization label maps
  to the reference speaker under the best label->name assignment (1 - word diarization error rate).
* **Role accuracy** - whether the label inferred as "host" is the reference host.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path

from audiosearch.domain import Transcript

_NUM_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_FILLERS = {"um", "uh", "hmm", "mm", "mhm", "ah", "er"}


def norm_tokens(text: str) -> list[str]:
    text = text.lower().replace("’", "'").replace("-", " ").replace("—", " ").replace("–", " ")
    toks = re.findall(r"[a-z0-9']+", text)
    out = []
    for t in toks:
        t = t.strip("'")
        if not t or t in _FILLERS:
            continue
        out.append(_NUM_WORDS.get(t, t))
    return out


@dataclass
class StageReport:
    file_id: str
    ref_words: int
    hyp_words: int
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    speaker_accuracy: float
    aligned_pairs: int
    mapping: dict[str, str]
    host_role_correct: bool
    inter_speaker_cosine: float | None


def _locate(ref: list[str], probe: list[str], from_end: bool = False) -> int:
    """Index in ``ref`` where ``probe`` (a short token list) best matches."""
    from rapidfuzz import fuzz

    m = len(probe)
    target = " ".join(probe)
    best, best_i = -1.0, 0
    for i in range(0, max(1, len(ref) - m + 1)):
        score = fuzz.ratio(target, " ".join(ref[i : i + m]))
        if score > best:
            best, best_i = score, i
    return best_i + m if from_end else best_i


def evaluate_file(transcript: Transcript, reference: dict, host_name: str | None) -> StageReport:
    import jiwer

    ref_tokens: list[str] = []
    ref_spk: list[str] = []
    for turn in reference["turns"]:
        toks = norm_tokens(turn["text"])
        ref_tokens += toks
        ref_spk += [turn["speaker"]] * len(toks)
    hyp_tokens: list[str] = []
    hyp_spk: list[str] = []
    for w in transcript.words:
        toks = norm_tokens(w.text)
        hyp_tokens += toks
        hyp_spk += [w.speaker or "?"] * len(toks)

    probe = 12
    start = _locate(ref_tokens, hyp_tokens[:probe])
    end = _locate(ref_tokens[start:], hyp_tokens[-probe:], from_end=True) + start
    ref_win, spk_win = ref_tokens[start:end], ref_spk[start:end]

    out = jiwer.process_words(" ".join(ref_win), " ".join(hyp_tokens))
    pairs: list[tuple[int, int]] = []
    for chunk in out.alignments[0]:
        if chunk.type in ("equal", "substitute"):
            pairs += list(
                zip(
                    range(chunk.ref_start_idx, chunk.ref_end_idx),
                    range(chunk.hyp_start_idx, chunk.hyp_end_idx),
                    strict=True,
                )
            )
    labels = sorted({hyp_spk[h] for _, h in pairs})
    names = sorted({spk_win[r] for r, _ in pairs})
    best_acc, best_map = 0.0, {}
    for perm in itertools.permutations(names, min(len(labels), len(names))):
        mapping = dict(zip(labels, perm, strict=False))
        acc = sum(mapping.get(hyp_spk[h]) == spk_win[r] for r, h in pairs) / max(len(pairs), 1)
        if acc > best_acc:
            best_acc, best_map = acc, mapping
    host_label = next((s.label for s in transcript.speakers if s.role == "host"), None)
    return StageReport(
        file_id=transcript.file_id,
        ref_words=len(ref_win),
        hyp_words=len(hyp_tokens),
        wer=round(out.wer, 4),
        substitutions=out.substitutions,
        deletions=out.deletions,
        insertions=out.insertions,
        speaker_accuracy=round(best_acc, 4),
        aligned_pairs=len(pairs),
        mapping=best_map,
        host_role_correct=bool(host_name and host_label and best_map.get(host_label) == host_name),
        inter_speaker_cosine=transcript.meta.get("diarization", {}).get("inter_speaker_cosine"),
    )


def evaluate_stages(transcripts_dir: Path, reference_dir: Path, hosts: dict[str, str | None]) -> list[StageReport]:
    reports = []
    for fid, host in hosts.items():
        tp, rp = transcripts_dir / f"{fid}.json", reference_dir / f"{fid}.json"
        if tp.exists() and rp.exists():
            reports.append(evaluate_file(Transcript.load(tp), json.loads(rp.read_text()), host))
    return reports

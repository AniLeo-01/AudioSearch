#!/usr/bin/env python
"""Diarization ablation: how much does linguistically-aware smoothing matter?

For each recording the ECAPA window embeddings and cluster posteriors are computed once; then several
ways of turning them into per-word speaker labels are scored against NASA's human transcripts
(word-level speaker attribution accuracy, see audiosearch.eval.stage_metrics):

  raw-window-vote     argmax of the word's frame posterior (no smoothing)
  segment-majority    majority vote inside each ASR segment (common WhisperX-style assignment)
  viterbi-uniform     Viterbi with the same switch penalty everywhere
  viterbi-boundary    Viterbi where switches are cheap at sentence ends / segment ends / pauses (ours)

Usage: python scripts/diarization_ablation.py [--out reports/diarization_ablation.md]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

from audiosearch.audio import decode
from audiosearch.config import get_settings
from audiosearch.dataset import load_manifest
from audiosearch.domain import Transcript, Word
from audiosearch.eval.stage_metrics import evaluate_file
from audiosearch.logging_setup import setup_logging
from audiosearch.pipeline.alignment import build_turns, build_utterances
from audiosearch.pipeline.asr import AsrResult, merge_continuations
from audiosearch.pipeline.diarization import (
    EcapaDiarizer,
    frame_posteriors,
    is_boundary,
    make_windows,
    relabel_by_first_appearance,
    spectral_two_way,
    speech_regions,
    viterbi_smooth,
    word_log_likelihoods,
)
from audiosearch.pipeline.roles import infer_roles, speaker_profiles


def labels_to_transcript(fid: str, duration: float, words: list[Word], path: np.ndarray, entry) -> Transcript:
    path, _ = relabel_by_first_appearance(path)
    ws = [dataclasses.replace(w, speaker=f"SPEAKER_{int(c):02d}") for w, c in zip(words, path, strict=True)]
    utts = build_utterances(ws, fid)
    turns = build_turns(utts)
    spk = infer_roles(speaker_profiles(utts, turns), utts, entry.host, entry.guest)
    return Transcript(fid, duration, ws, utts, turns, spk, meta={})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/diarization_ablation.md")
    args = ap.parse_args()
    setup_logging("WARNING")
    s = get_settings()
    diar = EcapaDiarizer(s.diarization_model, s.num_speakers, s.diar_window_sec, s.diar_hop_sec)
    variants = ["raw-window-vote", "segment-majority", "viterbi-uniform", "viterbi-boundary"]
    rows: dict[str, dict[str, float]] = {}
    totals = {v: [0.0, 0] for v in variants}
    for e in load_manifest(s.manifest_path, s.audio_dir):
        asr = AsrResult.load(s.transcripts_dir / f"{e.file_id}.asr.json")
        words, seg_ends = merge_continuations(asr.words, asr.segments)
        ref = json.loads((s.reference_dir / f"{e.file_id}.json").read_text())
        audio = decode(e.audio_path)
        duration = len(audio) / 16_000
        windows = make_windows(speech_regions(words), s.diar_window_sec, s.diar_hop_sec)
        emb = diar.embed_windows(audio, windows)
        labels = spectral_two_way(emb, s.num_speakers)
        x = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        cents = np.stack([x[labels == k].mean(0) for k in range(s.num_speakers)])
        cents /= np.linalg.norm(cents, axis=1, keepdims=True)
        logits = diar.temperature * (x @ cents.T)
        post_w = np.exp(logits - logits.max(1, keepdims=True))
        post_w /= post_w.sum(1, keepdims=True)
        post, covered = frame_posteriors(windows, post_w, duration)
        ll = word_log_likelihoods(words, post, covered)
        bounds = np.array([is_boundary(words, i, seg_ends) for i in range(len(words))])

        seg_major = np.argmax(ll, axis=1).copy()
        start = 0
        for end in sorted(seg_ends):
            seg = slice(start, end + 1)
            seg_major[seg] = int(np.argmax(ll[seg].sum(0)))
            start = end + 1
        paths = {
            "raw-window-vote": np.argmax(ll, axis=1),
            "segment-majority": seg_major,
            "viterbi-uniform": viterbi_smooth(ll, bounds, s.diar_switch_penalty, 1.0),
            "viterbi-boundary": viterbi_smooth(ll, bounds, s.diar_switch_penalty, s.diar_boundary_discount),
        }
        rows[e.file_id] = {}
        for name, path in paths.items():
            rep = evaluate_file(labels_to_transcript(e.file_id, duration, words, path, e), ref, e.host)
            rows[e.file_id][name] = rep.speaker_accuracy
            totals[name][0] += rep.speaker_accuracy * rep.aligned_pairs
            totals[name][1] += rep.aligned_pairs
        print(e.file_id, {k: round(v, 4) for k, v in rows[e.file_id].items()}, flush=True)

    lines = [
        "# Diarization ablation — word-level speaker attribution accuracy vs NASA transcripts",
        "",
        "| File | " + " | ".join(variants) + " |",
        "|---|" + "---:|" * len(variants),
    ]
    for fid, r in rows.items():
        lines.append(f"| {fid} | " + " | ".join(f"{r[v]:.4f}" for v in variants) + " |")
    overall = {v: totals[v][0] / max(totals[v][1], 1) for v in variants}
    lines.append("| **all (word-weighted)** | " + " | ".join(f"**{overall[v]:.4f}**" for v in variants) + " |")
    errors = {v: (1 - overall[v]) for v in variants}
    lines += [
        "",
        "Word diarization error rate (1 - accuracy): "
        + ", ".join(f"{v} {100 * errors[v]:.2f}%" for v in variants)
        + ".",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

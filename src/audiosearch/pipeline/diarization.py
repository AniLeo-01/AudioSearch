"""ASR-guided speaker diarization for two-party conversations.

Pipeline (all local, no gated models):

1. **Speech windows** - sliding 1.5 s windows (0.75 s hop) over regions where the ASR found words.
2. **Speaker embeddings** - ECAPA-TDNN (SpeechBrain, VoxCeleb) embedding per window.
3. **Clustering** - spectral clustering with affinity pruning into ``num_speakers`` clusters (the
   corpus contract says every conversation has two speakers, so we do not estimate the count), then
   centroid refinement.
4. **Frame posteriors** - soft speaker posteriors on a 0.1 s grid, averaged over covering windows.
5. **Linguistically-aware Viterbi smoothing over words** - a speaker change costs
   ``switch_penalty`` inside a sentence but only ``switch_penalty * boundary_discount`` after
   sentence-final punctuation, an ASR segment end, or a long pause.  Turn-taking in conversation
   happens at sentence/pause boundaries, so this removes most single-word "speaker flicker" errors
   that plain window voting produces.

The output is a speaker label and a confidence for every ASR word.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from audiosearch.domain import Word

log = logging.getLogger(__name__)

SR = 16_000
FRAME = 0.1  # seconds per posterior frame
SENTENCE_END = (".", "?", "!")


@dataclass
class DiarizationResult:
    labels: list[str]  # one label per word, e.g. "SPEAKER_00"
    confidences: list[float]
    meta: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------------
# Pure functions (unit-tested without any model)
# ----------------------------------------------------------------------------------------------------
def speech_regions(words: list[Word], max_gap: float = 0.5) -> list[tuple[float, float]]:
    """Merge word spans into contiguous speech regions."""
    regions: list[tuple[float, float]] = []
    for w in words:
        if regions and w.start - regions[-1][1] <= max_gap:
            regions[-1] = (regions[-1][0], max(regions[-1][1], w.end))
        else:
            regions.append((w.start, w.end))
    return regions


def make_windows(
    regions: list[tuple[float, float]], win: float, hop: float, min_len: float = 0.4
) -> list[tuple[float, float]]:
    """Sliding windows inside speech regions; short regions become a single (shorter) window."""
    out: list[tuple[float, float]] = []
    for s, e in regions:
        if e - s < min_len:
            continue
        if e - s <= win:
            out.append((s, e))
            continue
        t = s
        while t + win < e:
            out.append((t, t + win))
            t += hop
        out.append((max(s, e - win), e))  # flush the tail so the region end is covered
    return out


def spectral_two_way(emb: np.ndarray, n_clusters: int = 2, prune: float = 0.3, seed: int = 0) -> np.ndarray:
    """Spectral clustering on a pruned cosine-affinity graph, followed by centroid refinement."""
    from sklearn.cluster import SpectralClustering

    x = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)
    sim = x @ x.T
    n = sim.shape[0]
    if n <= n_clusters:
        return np.arange(n) % n_clusters
    # Row-wise pruning: keep the strongest `prune` fraction of affinities per row (Park et al. 2019).
    keep = max(2, int(np.ceil(prune * n)))
    thresh = np.partition(sim, n - keep, axis=1)[:, n - keep][:, None]
    aff = np.where(sim >= thresh, np.clip(sim, 0.0, None), 0.0)
    aff = (aff + aff.T) / 2.0
    np.fill_diagonal(aff, 1.0)
    labels: np.ndarray = SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed", assign_labels="cluster_qr", random_state=seed
    ).fit_predict(aff)
    # Refine: re-assign every window to the nearest centroid (a few spherical k-means steps).
    for _ in range(5):
        cents = np.stack(
            [x[labels == k].mean(0) if np.any(labels == k) else x[np.random.default_rng(seed).integers(n)]
             for k in range(n_clusters)]
        )  # fmt: skip
        cents /= np.linalg.norm(cents, axis=1, keepdims=True).clip(min=1e-8)
        new = np.argmax(x @ cents.T, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
    return labels


def frame_posteriors(
    windows: list[tuple[float, float]], win_post: np.ndarray, duration: float
) -> tuple[np.ndarray, np.ndarray]:
    """Average window posteriors onto a regular frame grid. Returns (posteriors[T,K], covered[T])."""
    n_frames = int(np.ceil(duration / FRAME)) + 1
    k = win_post.shape[1]
    acc = np.zeros((n_frames, k))
    cnt = np.zeros(n_frames)
    for (s, e), p in zip(windows, win_post, strict=True):
        a, b = int(s / FRAME), max(int(s / FRAME) + 1, int(np.ceil(e / FRAME)))
        acc[a:b] += p
        cnt[a:b] += 1
    covered = cnt > 0
    post = np.full((n_frames, k), 1.0 / k)
    post[covered] = acc[covered] / cnt[covered, None]
    return post, covered


def word_log_likelihoods(words: list[Word], post: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Mean frame posterior over each word's span (nearest covered frame if the span is uncovered)."""
    k = post.shape[1]
    out = np.zeros((len(words), k))
    covered_idx = np.flatnonzero(covered)
    for i, w in enumerate(words):
        a = int(w.start / FRAME)
        b = max(a + 1, int(np.ceil(w.end / FRAME)))
        seg = post[a:b][covered[a:b]]
        if len(seg) == 0 and len(covered_idx):
            j = covered_idx[np.argmin(np.abs(covered_idx - (a + b) // 2))]
            seg = post[j : j + 1]
        p = seg.mean(0) if len(seg) else np.full(k, 1.0 / k)
        out[i] = np.log(np.clip(p, 1e-4, 1.0))
    return out


def is_boundary(words: list[Word], i: int, segment_ends: set[int], pause: float = 0.6) -> bool:
    """True if a speaker change *before* word ``i`` would fall on a natural turn boundary."""
    if i == 0:
        return True
    prev = words[i - 1]
    return (
        prev.text.endswith(SENTENCE_END)
        or (i - 1) in segment_ends
        or (words[i].start - prev.end) >= pause
    )


def viterbi_smooth(
    loglik: np.ndarray, boundaries: np.ndarray, switch_penalty: float, boundary_discount: float
) -> np.ndarray:
    """Most likely speaker sequence given per-word log-likelihoods and boundary-aware switch costs."""
    n, k = loglik.shape
    if n == 0:
        return np.zeros(0, dtype=int)
    score = loglik[0].copy()
    back = np.zeros((n, k), dtype=int)
    for i in range(1, n):
        cost = switch_penalty * (boundary_discount if boundaries[i] else 1.0)
        # trans[prev, cur] = 0 if same speaker else -cost
        trans = -cost * (1 - np.eye(k))
        cand = score[:, None] + trans
        back[i] = np.argmax(cand, axis=0)
        score = cand[back[i], np.arange(k)] + loglik[i]
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def relabel_by_first_appearance(path: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    mapping: dict[int, int] = {}
    for c in path:
        if int(c) not in mapping:
            mapping[int(c)] = len(mapping)
    return np.array([mapping[int(c)] for c in path], dtype=int), mapping


# ----------------------------------------------------------------------------------------------------
# Model-backed diarizer
# ----------------------------------------------------------------------------------------------------
class EcapaDiarizer:
    name = "ecapa-spectral-viterbi"

    def __init__(
        self,
        model: str = "speechbrain/spkrec-ecapa-voxceleb",
        num_speakers: int = 2,
        window_sec: float = 1.5,
        hop_sec: float = 0.75,
        switch_penalty: float = 4.0,
        boundary_discount: float = 0.15,
        temperature: float = 10.0,
        device: str = "cpu",
        cache_dir: str | None = None,
    ) -> None:
        import os
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # torch.load(weights_only=...) notices
            from speechbrain.inference.speaker import EncoderClassifier
        for name in list(logging.root.manager.loggerDict):
            if name.startswith("speechbrain"):
                logging.getLogger(name).setLevel(logging.WARNING)

        self.model_name = model
        self.num_speakers = num_speakers
        self.window_sec, self.hop_sec = window_sec, hop_sec
        self.switch_penalty, self.boundary_discount = switch_penalty, boundary_discount
        self.temperature = temperature
        savedir = cache_dir or os.path.join(
            os.path.expanduser("~/.cache/audiosearch/speechbrain"), model.replace("/", "--")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._enc = EncoderClassifier.from_hparams(source=model, savedir=savedir, run_opts={"device": device})

    def embed_windows(self, audio: np.ndarray, windows: list[tuple[float, float]], batch: int = 64) -> np.ndarray:
        import torch

        max_len = int(round(self.window_sec * SR))
        embs = []
        for b in range(0, len(windows), batch):
            chunk = windows[b : b + batch]
            wavs = np.zeros((len(chunk), max_len), dtype=np.float32)
            lens = np.zeros(len(chunk), dtype=np.float32)
            for j, (s, e) in enumerate(chunk):
                seg = audio[int(s * SR) : int(e * SR)][:max_len]
                wavs[j, : len(seg)] = seg
                lens[j] = max(len(seg), 1) / max_len
            with torch.inference_mode():
                out = self._enc.encode_batch(torch.from_numpy(wavs), torch.from_numpy(lens))
            embs.append(out.squeeze(1).cpu().numpy())
        return np.concatenate(embs) if embs else np.zeros((0, 192), dtype=np.float32)

    def diarize(self, audio: np.ndarray, words: list[Word], segment_ends: set[int]) -> DiarizationResult:
        t0 = time.perf_counter()
        duration = len(audio) / SR
        windows = make_windows(speech_regions(words), self.window_sec, self.hop_sec)
        emb = self.embed_windows(audio, windows)
        labels = spectral_two_way(emb, self.num_speakers)
        x = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)
        cents = np.stack([x[labels == k].mean(0) for k in range(self.num_speakers)])
        cents /= np.linalg.norm(cents, axis=1, keepdims=True).clip(min=1e-8)
        logits = self.temperature * (x @ cents.T)
        win_post = np.exp(logits - logits.max(1, keepdims=True))
        win_post /= win_post.sum(1, keepdims=True)

        post, covered = frame_posteriors(windows, win_post, duration)
        loglik = word_log_likelihoods(words, post, covered)
        bounds = np.array([is_boundary(words, i, segment_ends) for i in range(len(words))])
        cluster_path = viterbi_smooth(loglik, bounds, self.switch_penalty, self.boundary_discount)
        conf = [float(np.exp(loglik[i, c])) for i, c in enumerate(cluster_path)]
        flicker_fixed = int(np.sum(np.argmax(loglik, axis=1) != cluster_path))
        path, _ = relabel_by_first_appearance(cluster_path)
        meta = {
            "engine": self.name,
            "model": self.model_name,
            "num_speakers": self.num_speakers,
            "window_sec": self.window_sec,
            "hop_sec": self.hop_sec,
            "switch_penalty": self.switch_penalty,
            "boundary_discount": self.boundary_discount,
            "n_windows": len(windows),
            "inter_speaker_cosine": round(float(cents[0] @ cents[1]), 4) if self.num_speakers == 2 else None,
            "speaker_changes": int(np.sum(path[1:] != path[:-1])),
            "words_changed_by_smoothing": flicker_fixed,
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        }
        log.info("diarized %d words / %d windows in %.1fs", len(words), len(windows), meta["elapsed_sec"])
        return DiarizationResult([f"SPEAKER_{int(c):02d}" for c in path], conf, meta)

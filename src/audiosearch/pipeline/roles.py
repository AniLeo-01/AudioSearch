"""Infer conversational roles (host / guest) for anonymous diarization clusters.

Interviews have a strong behavioural signature: the host asks most of the questions, talks less, and
usually opens the conversation.  Roles make results readable ("Guest, 04:12") and power role filters
(``role:guest``) without any voice-identity recognition.  Display names are attached only when the
dataset metadata provides them.
"""

from __future__ import annotations

import math

from audiosearch.domain import SpeakerProfile, Turn, Utterance

QUESTION_WEIGHT = 3.0
SHARE_WEIGHT = 1.5
OPENER_WEIGHT = 0.5


def speaker_profiles(utterances: list[Utterance], turns: list[Turn]) -> list[SpeakerProfile]:
    labels: list[str] = []
    for u in utterances:
        if u.speaker not in labels:
            labels.append(u.speaker)
    profiles = []
    for lab in labels:
        us = [u for u in utterances if u.speaker == lab]
        n_words = sum(len(u.text.split()) for u in us)
        profiles.append(
            SpeakerProfile(
                label=lab,
                talk_time=round(sum(u.end - u.start for u in us), 2),
                n_words=n_words,
                n_utterances=len(us),
                question_rate=round(sum(u.is_question for u in us) / max(len(us), 1), 4),
            )
        )
    return profiles


def infer_roles(
    profiles: list[SpeakerProfile],
    utterances: list[Utterance],
    host_name: str | None = None,
    guest_name: str | None = None,
) -> list[SpeakerProfile]:
    """Assign host/guest roles in place (and return the list)."""
    if not profiles:
        return profiles
    total_words = sum(p.n_words for p in profiles) or 1
    opener = utterances[0].speaker if utterances else None
    scores = {
        p.label: QUESTION_WEIGHT * p.question_rate
        + SHARE_WEIGHT * (0.5 - p.n_words / total_words)
        + OPENER_WEIGHT * (p.label == opener)
        for p in profiles
    }
    host = max(scores, key=lambda k: scores[k])
    ranked = sorted(scores.values(), reverse=True)
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else 1.0
    confidence = 1.0 / (1.0 + math.exp(-4.0 * margin))  # logistic on the score margin
    for p in profiles:
        p.role = "host" if p.label == host else "guest"
        p.role_confidence = round(confidence, 4)
        if confidence >= 0.75:
            name = host_name if p.role == "host" else guest_name
            p.display_name = name if (p.role == "host" or len(profiles) == 2) else None
    return profiles

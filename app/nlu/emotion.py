"""Lightweight emotion/tone detection used to shape empathetic phrasing.

Rule-based by default (always available). If the LLM is enabled, the
orchestrator may ask it to double-check ambiguous cases, but the gate
logic never depends on the LLM's opinion here.
"""
from __future__ import annotations

import re

from app.nlu.extraction import FRUSTRATION_WORDS, ANXIETY_WORDS, CONFUSION_WORDS, ANGER_MARKERS_RE, REFUSAL_RE

EMOTIONS = ["frustrated", "anxious", "confused", "refusing", "angry", "neutral", "positive"]

POSITIVE_WORDS = ["thanks", "thank you", "great", "appreciate", "awesome", "perfect", "helpful"]


def classify_emotion(text: str) -> str:
    low = text.lower()
    if REFUSAL_RE.search(text):
        return "refusing"
    if ANGER_MARKERS_RE.search(text) or any(w in low for w in FRUSTRATION_WORDS):
        return "frustrated" if not ANGER_MARKERS_RE.search(text) else "angry"
    if any(w in low for w in ANXIETY_WORDS):
        return "anxious"
    if any(w in low for w in CONFUSION_WORDS):
        return "confused"
    if any(w in low for w in POSITIVE_WORDS):
        return "positive"
    return "neutral"


EMPATHY_PREFIXES = {
    "frustrated": [
        "I hear you, and I'm sorry this has been frustrating.",
        "I understand this is frustrating — thank you for staying with me.",
    ],
    "angry": [
        "I'm sorry — I can tell this is really upsetting, and that's completely fair.",
        "I understand you're upset, and I want to make this right as quickly as I can.",
    ],
    "anxious": [
        "I can hear this is stressful, and I want to help you get clarity as fast as possible.",
        "That sounds worrying — let's get this sorted out together.",
    ],
    "confused": [
        "No problem at all, let me make that clearer.",
        "That's a fair question — let me walk through it simply.",
    ],
    "refusing": [
        "I hear you, and I know repeating information is frustrating.",
    ],
    "angry_refusal_combo": [
        "I completely understand your frustration, and I'm sorry for the back-and-forth.",
    ],
}


def empathy_prefix(emotion: str, refusal: bool = False) -> str:
    if refusal and emotion in ("angry", "frustrated"):
        options = EMPATHY_PREFIXES["angry_refusal_combo"]
    else:
        options = EMPATHY_PREFIXES.get(emotion, [])
    if not options:
        return ""
    # deterministic choice (first) keeps behavior reproducible/testable
    return options[0]

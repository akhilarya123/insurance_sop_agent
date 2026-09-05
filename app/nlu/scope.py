"""
Keeps the agent talking only about the insurance customer-service domain.
Rule-based so it works without a model; the orchestrator may also pass
ambiguous cases through the LLM for a second opinion when available, but
never *relaxes* a rule-based "out of scope" verdict based on the LLM.
"""
from __future__ import annotations

import re

IN_SCOPE_KEYWORDS = [
    "claim", "policy", "coverage", "deductible", "premium", "denial", "denied",
    "appeal", "document", "documents", "submit", "eob", "copay", "co-pay",
    "insurance", "representative", "agent", "human", "verify", "identity",
    "refund", "reimburse", "reimbursement", "hospital", "doctor", "treatment",
    "status", "case", "email", "summary", "policyholder", "ssn", "dob",
    "date of birth", "phone", "social security", "authorize", "authorization",
    "consent", "beneficiary", "medical", "dental", "auto", "accident",
]

OUT_OF_SCOPE_TOPIC_MARKERS = [
    "reinforcement learning", "machine learning", "neural network", "programming",
    "python code", "javascript", "recipe", "cook", "weather forecast", "sports score",
    "football score", "basketball score", "stock price", "cryptocurrency", "bitcoin",
    "capital of", "president of", "who won", "movie recommendation", "tv show",
    "song lyrics", "write me a poem", "math homework", "solve for x", "translate this",
    "joke", "riddle", "what is rl", "what is ai", "define ", "who is the ceo",
    "history of", "geography", "define photosynthesis",
]

GENERIC_QUESTION_RE = re.compile(r"^\s*(what is|what'?s|who is|explain|define|tell me about|how does)\b", re.IGNORECASE)

SHORT_ACK_RE = re.compile(
    r"^\s*(ok(?:ay)?|sure|alright|got it|understood|hi|hello|hey|thanks|thank you|yes|no|yeah|yep|nope)\.?\s*$",
    re.IGNORECASE,
)


def is_in_scope(text: str) -> bool:
    low = text.lower().strip()
    if not low:
        return True
    if SHORT_ACK_RE.match(low):
        return True
    if any(kw in low for kw in IN_SCOPE_KEYWORDS):
        return True
    if any(marker in low for marker in OUT_OF_SCOPE_TOPIC_MARKERS):
        return False
    if GENERIC_QUESTION_RE.match(low):
        # A generic "what is / explain / who is ..." question with none of our
        # in-scope keywords is very likely off-topic trivia.
        return False
    return True


def redirect_message(strikes: int, offer_human: bool) -> str:
    base = ("I can only help with questions related to your insurance policy, "
            "claims, or this call — I'm not able to help with that.")
    if offer_human:
        return (base + " It sounds like you may need something outside what I can "
                "assist with here — would you like me to transfer you to a human "
                "representative instead?")
    return base + " Is there anything about your policy or claim I can help with?"

"""
Polish layer: takes an already-decided, already-fact-checked draft reply
(produced entirely by deterministic SOP code) and asks the local LLM to
phrase it more naturally. The LLM is never shown anything the SOP hasn't
already cleared, and its output is re-checked before use -- any sign that
it introduced new facts, numbers, or forbidden terms causes the draft to
be used verbatim instead.
"""
from __future__ import annotations

import json
import re

from app.llm import client

NUMBER_RE = re.compile(r"\d{2,}")
CASE_ID_RE = re.compile(r"\bCL-?\d{3,7}\b", re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are helping phrase a customer-service reply for an insurance claims call. "
    "You will be given a DRAFT reply that already contains every fact you're allowed to state. "
    "Rewrite the DRAFT in a warm, natural, conversational tone, as a live phone agent would speak. "
    "Rules you must follow exactly:\n"
    "1. Do not add any fact, number, date, name, or claim detail that is not already in the DRAFT.\n"
    "2. Do not remove any fact that is in the DRAFT.\n"
    "3. Do not mention that you are an AI, a draft, instructions, or this rewriting process.\n"
    "4. Keep it concise -- roughly the same length as the DRAFT.\n"
    "5. Reply with ONLY the rewritten text, nothing else."
)


def _extract_numbers(text: str) -> set:
    return set(NUMBER_RE.findall(text))


def _passes_guardrails(candidate: str, draft: str, facts: dict, forbidden_terms: list, resolved_case_id: str | None) -> bool:
    low = candidate.lower()

    for term in forbidden_terms:
        if term.lower() in low:
            return False

    # Case-id leakage: any mentioned case id must be the one already resolved
    # (or already present in the draft, e.g. when listing candidate claims).
    for m in CASE_ID_RE.findall(candidate):
        cid = m.upper()
        if cid not in draft.upper():
            return False

    # Numeric leakage: every number in the candidate should already appear
    # somewhere in the draft or the supporting facts.
    allowed_numbers = _extract_numbers(draft) | _extract_numbers(json.dumps(facts, default=str))
    for n in _extract_numbers(candidate):
        if n not in allowed_numbers:
            return False

    if not candidate.strip():
        return False
    return True


def polish(draft: str, facts: dict | None = None, forbidden_terms: list | None = None,
           resolved_case_id: str | None = None, allow: bool = True) -> tuple[str, bool]:
    """Returns (final_text, was_polished)."""
    facts = facts or {}
    forbidden_terms = forbidden_terms or []

    if not allow or not client.is_available():
        return draft, False

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"DRAFT:\n{draft}"},
    ]
    candidate = client.chat(messages)
    if candidate is None:
        return draft, False

    if _passes_guardrails(candidate, draft, facts, forbidden_terms, resolved_case_id):
        return candidate, True
    return draft, False

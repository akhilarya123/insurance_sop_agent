"""
RESOLVE_INTENT phase handler.

Freer than VERIFY_ID: the caller's language can be messy, and hints
gathered *before* verification (memory) should be used automatically so
the caller never has to repeat themselves. Still SOP-bounded, though:
the only allowed outputs are (a) a resolved case_id + intent, or (b) a
clarifying question — never claim content, which only PROCESS_CASE may
reveal.
"""
from __future__ import annotations

from app import data_loader as dl
from app.models import SessionState, TurnResult, Phase
from app.nlu.extraction import ExtractedSignals

INTENT_TYPES = [
    "denial_question", "status_inquiry", "document_submission",
    "next_steps", "general_claim_question",
]


def classify_intent(text: str, status_hint: str = None) -> str:
    low = text.lower()
    if any(k in low for k in ["why was", "why is", "why did", "reason for denial", "denied", "denial"]):
        return "denial_question"
    if any(k in low for k in ["status", "update", "what's happening", "where does it stand", "any progress"]):
        return "status_inquiry"
    if any(k in low for k in ["document", "upload", "submit", "attach", "send you", "portal", "paperwork"]):
        return "document_submission"
    if any(k in low for k in ["next step", "what do i do", "what should i do", "what now"]):
        return "next_steps"
    if status_hint == "denied":
        return "denial_question"
    return "general_claim_question"


def _month_of(date_str: str) -> str:
    try:
        return date_str.split("-")[1]
    except Exception:
        return ""


def filter_candidates(party_id: str, sig: ExtractedSignals, session: SessionState):
    memory = session.memory
    case_type_hint = sig.case_type_hint or memory.case_type_hint
    status_hint = sig.status_hint or memory.status_hint
    date_hint = sig.date_hint or memory.date_hint
    explicit_case_id = sig.explicit_case_id or memory.explicit_case_id_hint

    candidates = dl.find_claims_by_party(party_id)

    if explicit_case_id:
        exact = [c for c in candidates if c["case_id"].lower() == explicit_case_id.lower()]
        if exact:
            return exact, case_type_hint, status_hint, date_hint

    working = candidates
    for hint, keyfn in [
        (case_type_hint, lambda c: c["case_type"] == case_type_hint),
        (status_hint, lambda c: c["status"] == status_hint),
    ]:
        if hint:
            narrowed = [c for c in working if keyfn(c)]
            if narrowed:
                working = narrowed
            if len(working) == 1:
                return working, case_type_hint, status_hint, date_hint

    if date_hint:
        from app.nlu.extraction import MONTHS
        month_num = MONTHS.get(date_hint)
        if month_num:
            narrowed = [c for c in working if _month_of(c["created_at"]) == month_num]
            if narrowed:
                working = narrowed

    return working, case_type_hint, status_hint, date_hint


def try_auto_resolve(session: SessionState, sig: ExtractedSignals):
    """Attempt silent resolution purely from memory/extraction, no new
    clarifying question needed. Returns (case, intent) or (None, None)."""
    if not session.matched_party_id:
        return None, None
    candidates, case_type_hint, status_hint, date_hint = filter_candidates(session.matched_party_id, sig, session)
    if len(candidates) == 1:
        intent_text = sig.intent_hint_text or session.memory.intent_hint_text or ""
        intent = classify_intent(intent_text, status_hint)
        return candidates[0], intent
    return None, None


def _describe_claim_option(c: dict) -> str:
    return f"{c['case_id']} ({c['case_type']}, {c['status']}, opened {c['created_at']})"


def handle(session: SessionState, user_message: str, sig: ExtractedSignals) -> TurnResult:
    if not session.matched_party_id:
        session.log("resolve_intent called without a verified party -- returning to VERIFY_ID")
        session.phase = Phase.VERIFY_ID
        return TurnResult(reply_template="Let's first finish verifying your identity.")

    case, intent = try_auto_resolve(session, sig)
    if case:
        session.resolved_case_id = case["case_id"]
        session.resolved_intent = intent
        session.phase = Phase.PROCESS_CASE
        return TurnResult(
            reply_template=f"__AUTO_RESOLVED__:{case['case_id']}:{intent}",
            control_flags={"auto_resolved": True},
            phase_notes=f"Resolved case {case['case_id']} intent={intent} from hints/memory.",
        )

    candidates, case_type_hint, status_hint, date_hint = filter_candidates(session.matched_party_id, sig, session)
    session.candidate_case_ids = [c["case_id"] for c in candidates]

    if not candidates:
        all_claims = dl.find_claims_by_party(session.matched_party_id)
        if not all_claims:
            return TurnResult(
                reply_template=(
                    "I don't see any claims on file for your account yet. Is there something else I can help "
                    "you with, such as a policy or coverage question?"
                ),
                phase_notes="Verified party has zero claims on file.",
            )
        options = "; ".join(_describe_claim_option(c) for c in all_claims)
        return TurnResult(
            reply_template=(
                f"I couldn't match that to a specific claim. Here's what's on your account: {options}. "
                "Which one would you like to talk about?"
            ),
            phase_notes="No hint matched any claim; offering full list.",
        )

    if len(candidates) > 1:
        options = "; ".join(_describe_claim_option(c) for c in candidates)
        return TurnResult(
            reply_template=f"I see a couple of claims that could match: {options}. Which one did you mean?",
            phase_notes="Multiple candidates matched hints; asking to disambiguate.",
        )

    # exactly one candidate but classify_intent needs the live message
    case = candidates[0]
    intent = classify_intent(user_message, status_hint)
    session.resolved_case_id = case["case_id"]
    session.resolved_intent = intent
    session.phase = Phase.PROCESS_CASE
    return TurnResult(
        reply_template=f"__AUTO_RESOLVED__:{case['case_id']}:{intent}",
        control_flags={"auto_resolved": True},
        phase_notes=f"Resolved case {case['case_id']} intent={intent} from current turn.",
    )

"""
POST_PROCESS phase handler.

Offers an email summary of the call (what was discussed, claim status/
outcome, follow-up items) and lets the caller choose to send it or skip it.
If the caller instead asks another substantive claim question, we loop back
to PROCESS_CASE for that, then can re-offer the summary at the end again.
"""
from __future__ import annotations

from app import data_loader as dl
from app.models import SessionState, TurnResult, Phase
from app.nlu.extraction import ExtractedSignals
from app.nlu.scope import is_in_scope


def _followup_items(claim: dict) -> list[str]:
    items = []
    if claim.get("documents_needed"):
        items.append(f"Submit: {', '.join(claim['documents_needed'])}")
    if claim.get("appeal_deadline"):
        items.append(f"Appeal/submission deadline: {claim['appeal_deadline']}")
    if claim["status"] == "open":
        items.append("Continue monitoring claim status for updates.")
    if not items:
        items.append("No further action needed at this time.")
    return items


def compose_summary_text(session: SessionState) -> str:
    claim = dl.find_claim_by_id(session.resolved_case_id) if session.resolved_case_id else None
    party = dl.find_party(session.matched_party_id) if session.matched_party_id else None
    lines = [f"Call summary for {party['name'] if party else 'your account'}"]
    if claim:
        lines.append(f"- Claim discussed: {claim['case_id']} ({claim['case_type']})")
        lines.append(f"- Status/outcome: {claim['status']}")
        if claim.get("denial_reason"):
            lines.append(f"- Reason: {claim['denial_reason']}")
        lines.append("- Follow-up items:")
        for item in _followup_items(claim):
            lines.append(f"  - {item}")
    else:
        lines.append("- No specific claim was resolved during this call.")
    return "\n".join(lines)


def build_offer(session: SessionState) -> TurnResult:
    session.email_offered = True
    party = dl.find_party(session.matched_party_id) if session.matched_party_id else None
    email_hint = f" to {party['email']}" if party and party.get("email") else ""
    return TurnResult(
        reply_template=(
            f"would you like me to email you a summary of what we discussed today{email_hint}, "
            "including the claim status and next steps? Just say yes or no."
        ),
        phase_notes="Offering email summary; awaiting yes/no.",
    )


def handle(session: SessionState, user_message: str, sig: ExtractedSignals) -> TurnResult:
    if not session.email_offered:
        return build_offer(session)

    if session.email_decision is None:
        if sig.yes_no == "yes":
            summary = compose_summary_text(session)
            session.email_decision = "sent"
            session.email_summary_text = summary
            session.phase = Phase.ENDED
            party = dl.find_party(session.matched_party_id) if session.matched_party_id else None
            addr = party["email"] if party and party.get("email") else "the email on file"
            return TurnResult(
                reply_template=(
                    f"Done — I've sent the summary to {addr}. Is there anything else I can help you with today?"
                ),
                facts={"summary": summary, "sent_to": addr},
                phase_notes="Email summary sent (simulated).",
            )
        if sig.yes_no == "no":
            session.email_decision = "skipped"
            session.phase = Phase.ENDED
            return TurnResult(
                reply_template="No problem, I won't send anything. Is there anything else I can help you with today?",
                phase_notes="Email summary skipped by caller choice.",
            )
        # Ambiguous reply: if it looks like a substantive new question, handle it;
        # otherwise re-ask the yes/no clearly.
        if is_in_scope(user_message) and session.resolved_case_id:
            from app.sop import process_case
            result = process_case.handle(session, user_message, sig)
            session.phase = Phase.POST_PROCESS  # stay in post-process after answering
            result.phase_notes += " (answered during post-process before email decision)"
            return result
        return TurnResult(
            reply_template="Just to confirm — would you like me to email you that summary? Yes or no works fine.",
            phase_notes="Re-prompting for yes/no on email offer.",
        )

    # Email decision already made this call; caller is continuing the conversation.
    if sig.wants_to_end or not user_message.strip():
        session.phase = Phase.ENDED
        return TurnResult(reply_template="Thank you for calling — have a great day!", phase_notes="Call ended.")

    if session.resolved_case_id and is_in_scope(user_message):
        from app.sop import process_case
        result = process_case.handle(session, user_message, sig)
        session.phase = Phase.POST_PROCESS
        return result

    return TurnResult(
        reply_template="Is there anything else I can help you with today?",
        phase_notes="Post-decision small talk.",
    )

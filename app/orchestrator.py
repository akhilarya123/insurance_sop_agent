"""
Ties every module together into one per-turn handler. This is the only
place that decides *which* deterministic phase module runs, and it never
lets the LLM influence that decision -- the LLM is invoked once, at the
very end, purely to phrase the already-decided reply.
"""
from __future__ import annotations

from app.models import SessionState, TurnResult, Phase
from app.nlu.extraction import extract_all
from app.nlu.emotion import classify_emotion, empathy_prefix
from app.nlu.scope import is_in_scope, redirect_message
from app.sop import verify_id, resolve_intent, process_case, post_process
from app.llm import compose
from app.config import (
    MAX_OFF_TOPIC_STRIKES_BEFORE_ESCALATE,
    MAX_OFF_TOPIC_STRIKES_BEFORE_OFFER,
)

NEGATIVE_EMOTIONS = {"frustrated", "angry", "anxious", "confused", "refusing"}


def greeting_message() -> str:
    return (
        "Thanks for calling in! To protect your account, I'll first need to verify your identity. "
        "Could you share your full name along with at least two more of the following: date of birth, "
        "phone number, email, or the last 4 digits of your SSN? You're also welcome to tell me what "
        "you're calling about in the meantime."
    )


def _has_useful_signal(sig) -> bool:
    return any([
        sig.full_name, sig.dob, sig.phone, sig.email, sig.ssn_last4, sig.policy_number,
        sig.case_type_hint, sig.status_hint, sig.date_hint, sig.explicit_case_id,
        sig.yes_no, sig.wants_to_end, sig.is_representative_signal, sig.refusal_signal,
        sig.wants_human_transfer,
    ])


def _apply_memory(session: SessionState, sig):
    mem = session.memory
    if sig.case_type_hint:
        mem.case_type_hint = sig.case_type_hint
    if sig.status_hint:
        mem.status_hint = sig.status_hint
    if sig.date_hint:
        mem.date_hint = sig.date_hint
    if sig.explicit_case_id:
        mem.explicit_case_id_hint = sig.explicit_case_id
    if sig.intent_hint_text and not mem.intent_hint_text:
        mem.intent_hint_text = sig.intent_hint_text


def _resolve_marker(session: SessionState, result: TurnResult) -> TurnResult:
    """resolve_intent.handle()/try_auto_resolve() signal a resolved case via a
    magic-string marker so the orchestrator can immediately cascade into the
    PROCESS_CASE initial presentation within the same turn."""
    if not result.reply_template.startswith("__AUTO_RESOLVED__:"):
        return result
    presentation = process_case.build_initial_presentation(session)
    return TurnResult(
        reply_template=presentation.reply_template,
        facts=presentation.facts,
        forbidden_terms=[],
        phase_notes=result.phase_notes + " -> " + presentation.phase_notes,
        control_flags={**result.control_flags, **presentation.control_flags},
    )


def _dispatch(session: SessionState, user_message: str, sig) -> TurnResult:
    phase = session.phase

    if phase == Phase.VERIFY_ID:
        if not session.verified:
            verify_id.apply_identity_extraction(session, sig)
        result = verify_id.handle(session, user_message, sig)
        if result.control_flags.get("just_verified"):
            case, intent = resolve_intent.try_auto_resolve(session, sig)
            if case:
                session.resolved_case_id = case["case_id"]
                session.resolved_intent = intent
                session.phase = Phase.PROCESS_CASE
                presentation = process_case.build_initial_presentation(session)
                return TurnResult(
                    reply_template=(result.reply_template + " " + presentation.reply_template).strip(),
                    facts=presentation.facts,
                    forbidden_terms=[],
                    phase_notes=result.phase_notes + " -> " + presentation.phase_notes,
                    control_flags=result.control_flags,
                )
            # Not enough memory to auto-resolve -- ask what they need help with.
            next_result = resolve_intent.handle(session, user_message, sig)
            next_result = _resolve_marker(session, next_result)
            return TurnResult(
                reply_template=(result.reply_template + " " + next_result.reply_template).strip(),
                facts=next_result.facts,
                forbidden_terms=next_result.forbidden_terms,
                phase_notes=result.phase_notes + " -> " + next_result.phase_notes,
                control_flags={**result.control_flags, **next_result.control_flags},
            )
        return result

    if phase == Phase.RESOLVE_INTENT:
        result = resolve_intent.handle(session, user_message, sig)
        return _resolve_marker(session, result)

    if phase == Phase.PROCESS_CASE:
        return process_case.handle(session, user_message, sig)

    if phase == Phase.POST_PROCESS:
        return post_process.handle(session, user_message, sig)

    return TurnResult(
        reply_template="This call has concluded. Thank you for contacting us -- start a new session for another call.",
        phase_notes="Phase is ENDED; no further processing.",
    )


def handle_turn(session: SessionState, user_message: str, consent_scenario: str | None = None) -> dict:
    session.touch()
    if consent_scenario and not session.consent.active and not session.verified:
        session.consent.scenario = consent_scenario

    session.add_turn("user", user_message)
    sig = extract_all(user_message)
    _apply_memory(session, sig)

    # --- Highest-priority, phase-independent gates -----------------------------------
    if sig.wants_human_transfer and session.phase != Phase.ENDED:
        session.escalate_to_human = True
        reply = ("Of course -- I'll connect you with a human representative now. "
                 "Please hold for a moment.")
        session.add_turn("agent", reply, meta={"escalated": True})
        return _package(session, reply, polished=False, emotion="neutral")

    emotion = classify_emotion(user_message)

    if session.phase not in (Phase.ENDED,) and not _has_useful_signal(sig) and not is_in_scope(user_message):
        session.off_topic_strikes += 1
        offer_human = session.off_topic_strikes >= MAX_OFF_TOPIC_STRIKES_BEFORE_OFFER
        if session.off_topic_strikes >= MAX_OFF_TOPIC_STRIKES_BEFORE_ESCALATE:
            session.escalate_to_human = True
        draft = redirect_message(session.off_topic_strikes, offer_human)
        prefix = empathy_prefix(emotion, sig.refusal_signal)
        draft = (prefix + " " + draft).strip() if prefix else draft
        final_text, polished = compose.polish(draft, facts={}, forbidden_terms=[], allow=True)
        session.add_turn("agent", final_text, meta={"off_topic": True})
        return _package(session, final_text, polished, emotion)

    session.off_topic_strikes = 0

    result = _dispatch(session, user_message, sig)
    session.log(result.phase_notes)

    prefix = empathy_prefix(emotion, sig.refusal_signal) if emotion in NEGATIVE_EMOTIONS else ""
    draft = (prefix + " " + result.reply_template).strip() if prefix else result.reply_template

    final_text, polished = compose.polish(
        draft,
        facts=result.facts,
        forbidden_terms=result.forbidden_terms,
        resolved_case_id=session.resolved_case_id,
        allow=result.allow_llm_polish,
    )
    session.add_turn("agent", final_text, meta={"phase_notes": result.phase_notes})
    return _package(session, final_text, polished, emotion)


def _package(session: SessionState, reply: str, polished: bool, emotion: str) -> dict:
    return {
        "session_id": session.session_id,
        "reply": reply,
        "phase": session.phase.value,
        "verified": session.verified,
        "matched_party_id": session.matched_party_id,
        "representative_mode": session.representative_mode,
        "resolved_case_id": session.resolved_case_id,
        "resolved_intent": session.resolved_intent,
        "escalate_to_human": session.escalate_to_human,
        "email_decision": session.email_decision,
        "emotion": emotion,
        "polished_by_llm": polished,
        "debug_log": session.debug_log[-6:],
    }

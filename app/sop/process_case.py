"""
PROCESS_CASE phase handler.

Freer LLM reasoning is allowed here too (interpreting messy follow-up
questions), but every fact that can appear in a reply is assembled by this
deterministic module first and handed to the polish layer as `facts`. The
polish layer is instructed to phrase-only; it is not allowed to introduce
new figures, dates, or document names, and the orchestrator's guardrail
double-checks that.
"""
from __future__ import annotations

from app import data_loader as dl
from app.models import SessionState, TurnResult, Phase
from app.nlu.extraction import ExtractedSignals, detect_end_call


def _fuzzy_lookup(mapping: dict, name: str):
    name_low = name.lower().strip()
    for key, val in mapping.items():
        key_low = key.lower()
        if name_low == key_low or name_low in key_low or key_low in name_low:
            return val
    return None


def build_document_context(claim: dict) -> list[dict]:
    guideline = dl.required_document_guideline()
    guidance_map = guideline.get("document_guidance", {})
    alt_map = guideline.get("document_alternative_guidance", {})
    default_alt = alt_map.get("default", {}).get("en")

    out = []
    for doc_name in claim.get("documents_needed", []):
        g = _fuzzy_lookup(guidance_map, doc_name)
        a = _fuzzy_lookup(alt_map, doc_name)
        out.append({
            "name": doc_name,
            "guidance": g.get("en") if g else None,
            "alternative": (a.get("en") if a else None) or default_alt,
        })
    return out


ALTERNATIVE_SEEKING_PATTERNS = [
    "what if i", "can't get", "cannot get", "don't have", "do not have",
    "alternative", "substitute", "instead of", "another way", "not available",
]


def pick_followup_guidance(claim: dict, resolved_intent: str, user_message: str):
    guideline = dl.required_document_guideline()
    entries = guideline.get("claim_followup_guidance", [])
    has_docs = bool(claim.get("documents_needed"))
    low = user_message.lower()

    eligible = [
        e for e in entries
        if resolved_intent in e.get("intent_hints", [])
        and (not e.get("requires_documents") or has_docs)
    ]

    # Prefer a specific keyword-matched entry over a generic one.
    keyword_hits = [e for e in eligible if e.get("match_any") and any(p in low for p in e["match_any"])]
    if keyword_hits:
        return keyword_hits[0], guideline

    generic = [e for e in eligible if not e.get("match_any")]
    if generic and (not low.strip() or any(p in low for p in ALTERNATIVE_SEEKING_PATTERNS)):
        # Used for a silent/initial presentation, or when the caller is
        # explicitly asking about not having / substituting a document.
        return generic[0], guideline

    return None, guideline


def _format_template(template: str, claim: dict, guideline: dict) -> str:
    docs = claim.get("documents_needed", [])
    settings = guideline.get("claim_followup_settings", {})
    avg_time = settings.get("average_processing_time_after_submission", {}).get("en", "typically within a week")
    return template.format(
        case_id=claim["case_id"],
        documents=", ".join(docs) if docs else "the requested items",
        average_processing_time_after_submission=avg_time,
    )


def build_case_facts(session: SessionState, user_message: str, is_initial: bool) -> dict:
    claim = dl.find_claim_by_id(session.resolved_case_id)
    guideline = dl.required_document_guideline()
    facts = {
        "case_id": claim["case_id"],
        "case_type": claim["case_type"],
        "status": claim["status"],
        "summary": claim["summary"],
        "created_at": claim["created_at"],
        "intent": session.resolved_intent,
    }
    if claim.get("denial_reason"):
        facts["denial_reason"] = claim["denial_reason"]
    if claim.get("appeal_deadline"):
        facts["appeal_deadline"] = claim["appeal_deadline"]
    if claim.get("documents_needed"):
        facts["documents_needed"] = claim["documents_needed"]
        facts["document_context"] = build_document_context(claim)

    case_type_guidance = guideline.get("case_type_guidance", {}).get(claim["case_type"], {}).get("en")
    if case_type_guidance:
        facts["case_type_guidance"] = case_type_guidance
    facts["default_guidance"] = guideline.get("default_guidance", {}).get("en")

    entry, _ = pick_followup_guidance(claim, session.resolved_intent, "" if is_initial else user_message)
    if entry:
        facts["followup_topic"] = entry["topic"]
        facts["followup_answer"] = _format_template(entry["en"], claim, guideline)
    elif not is_initial:
        facts["followup_fallback"] = guideline.get("claim_followup_fallback", {}).get("en")

    return facts, claim


def _render_template_reply(facts: dict, is_initial: bool) -> str:
    parts = []
    intent = facts["intent"]

    if is_initial:
        if intent == "denial_question" and facts.get("denial_reason"):
            parts.append(
                f"I found it — claim {facts['case_id']} ({facts['case_type']}) was denied because "
                f"{facts['denial_reason']}."
            )
            if facts.get("appeal_deadline"):
                parts.append(f"You have until {facts['appeal_deadline']} to appeal or submit the missing items.")
        elif intent == "status_inquiry":
            parts.append(f"Claim {facts['case_id']} is currently **{facts['status']}**. {facts['summary']}.")
        else:
            parts.append(f"Here's claim {facts['case_id']}: {facts['summary']} (status: {facts['status']}).")

        if facts.get("documents_needed"):
            docs = ", ".join(facts["documents_needed"])
            parts.append(f"The outstanding items are: {docs}.")
        parts.append("What would you like to know — how to submit those, how long review takes, or anything else about this claim?")
    else:
        if facts.get("followup_answer"):
            parts.append(facts["followup_answer"])
        elif facts.get("followup_fallback"):
            parts.append(facts["followup_fallback"])
        else:
            parts.append(
                f"Regarding claim {facts['case_id']}: {facts['summary']} (status: {facts['status']})."
            )
        if facts.get("document_context") and any(d["alternative"] for d in facts["document_context"]) and \
           any(k in facts.get("followup_topic", "") for k in ("alternatives",)):
            for d in facts["document_context"]:
                if d.get("alternative"):
                    parts.append(f"For the {d['name']}: {d['alternative']}")

    return " ".join(parts)


def build_initial_presentation(session: SessionState) -> TurnResult:
    facts, claim = build_case_facts(session, "", is_initial=True)
    reply = _render_template_reply(facts, is_initial=True)
    return TurnResult(
        reply_template=reply,
        facts=facts,
        forbidden_terms=[],
        phase_notes=f"Initial grounded presentation of {claim['case_id']}.",
    )


def handle(session: SessionState, user_message: str, sig: ExtractedSignals) -> TurnResult:
    if detect_end_call(user_message) or sig.wants_to_end:
        from app.sop import post_process
        session.phase = Phase.POST_PROCESS
        offer = post_process.build_offer(session)
        return TurnResult(
            reply_template="Of course — before we wrap up, " + offer.reply_template,
            facts=offer.facts,
            control_flags={"moved_to_post_process": True},
            phase_notes="User signaled end of case discussion; moving to POST_PROCESS.",
        )

    # Caller may pivot to a different claim mid-conversation.
    if sig.explicit_case_id and sig.explicit_case_id != session.resolved_case_id:
        claim = dl.find_claim_by_id(sig.explicit_case_id)
        if claim and claim["party_id"] == session.matched_party_id:
            session.resolved_case_id = claim["case_id"]
            session.resolved_intent = None
            from app.sop.resolve_intent import classify_intent
            session.resolved_intent = classify_intent(user_message)
            return build_initial_presentation(session)

    # Follow-up questions can shift topic turn to turn (freer reasoning here
    # than in VERIFY_ID) -- reclassify intent from the live message so e.g.
    # "how do I submit that?" is treated as document_submission even though
    # the case was originally opened as a denial_question.
    from app.sop.resolve_intent import classify_intent
    session.resolved_intent = classify_intent(user_message, None) or session.resolved_intent

    facts, claim = build_case_facts(session, user_message, is_initial=False)
    reply = _render_template_reply(facts, is_initial=False)
    return TurnResult(
        reply_template=reply,
        facts=facts,
        forbidden_terms=[],
        phase_notes=f"Answered follow-up on {claim['case_id']} (topic={facts.get('followup_topic')}).",
    )

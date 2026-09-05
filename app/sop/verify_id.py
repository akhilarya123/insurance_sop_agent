"""
VERIFY_ID phase handler.

This is the strictest part of the SOP: the agent must not disclose any
claim details and must not advance to the next phase until at least
MIN_ID_MATCHES of the five whitelisted PII categories match a single
policyholder record (or, for a representative caller, until a simulated
consent check from the policyholder resolves to "approved").

Everything that decides *whether* verification succeeded is plain
deterministic Python. The LLM (used later, in the polish layer) is only
ever handed a `facts` dict that has already been filtered by this module,
so it is structurally impossible for a model hallucination to leak claim
data before verification.
"""
from __future__ import annotations

from typing import Optional

from app import data_loader as dl
from app.config import REQUIRED_ID_FIELDS, MIN_ID_MATCHES, MAX_CONSENT_POLLS_BEFORE_ESCALATE
from app.models import SessionState, TurnResult, Phase
from app.nlu.extraction import ExtractedSignals

FIELD_LABELS = {
    "full_name": "your full name",
    "dob": "your date of birth",
    "phone": "the phone number on file",
    "email": "the email on file",
    "ssn_last4": "the last 4 digits of your SSN (or national ID)",
}
FIELD_PRIORITY = ["full_name", "dob", "ssn_last4", "phone", "email"]


def apply_identity_extraction(session: SessionState, sig: ExtractedSignals):
    ident = session.identity
    if sig.full_name:
        ident.full_name = sig.full_name
    if sig.dob:
        ident.dob = sig.dob
    if sig.phone:
        ident.phone = sig.phone
    if sig.email:
        ident.email = sig.email
    if sig.ssn_last4:
        ident.ssn_last4 = sig.ssn_last4
    if sig.policy_number:
        ident.policy_number = sig.policy_number


def _norm_phone(p: Optional[str]) -> str:
    return "".join(ch for ch in (p or "") if ch.isdigit())[-10:]


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def score_against_party(identity, party: dict) -> list[str]:
    """Returns the list of the 5 whitelisted categories that match this party."""
    matched = []
    if identity.full_name:
        names = [party.get("name", "")] + party.get("name_aliases", [])
        if _norm(identity.full_name) in [_norm(n) for n in names]:
            matched.append("full_name")
    if identity.dob and identity.dob == party.get("dob"):
        matched.append("dob")
    if identity.phone:
        phones = [party.get("phone", "")] + party.get("phone_aliases", [])
        if _norm_phone(identity.phone) and _norm_phone(identity.phone) in [_norm_phone(p) for p in phones]:
            matched.append("phone")
    if identity.email:
        emails = [party.get("email", "")] + party.get("email_aliases", [])
        if _norm(identity.email) in [_norm(e) for e in emails]:
            matched.append("email")
    if identity.ssn_last4 and identity.ssn_last4 == party.get("id_last4"):
        matched.append("ssn_last4")
    return matched


def find_best_match(identity):
    candidates = dl.policyholders()
    # Narrow by policy number first if given -- it's an unambiguous key.
    if identity.policy_number:
        narrowed = [p for p in candidates if p.get("policy_number", "").upper() == identity.policy_number.upper()]
        if narrowed:
            candidates = narrowed
    best_party, best_matched = None, []
    for p in candidates:
        matched = score_against_party(identity, p)
        if len(matched) > len(best_matched):
            best_party, best_matched = p, matched
    return best_party, best_matched


def _missing_field_prompt(matched: list[str]) -> str:
    remaining = [f for f in FIELD_PRIORITY if f not in matched]
    # ask for two options so the caller has a choice of alternate fields
    options = remaining[:2] if remaining else FIELD_PRIORITY[:2]
    labels = [FIELD_LABELS[o] for o in options]
    if len(labels) == 1:
        return f"could you confirm {labels[0]}?"
    return f"could you confirm {labels[0]}, or alternatively {labels[1]}?"


def _start_consent_flow(session: SessionState, rep_entry: dict, rep_name: Optional[str]):
    session.representative_mode = True
    session.consent.active = True
    session.consent.status = "pending"
    session.consent.poll_index = 0
    session.consent.buyer_party_id = rep_entry["buyer_party_id"]
    session.consent.rep_name = rep_name or rep_entry["rep_name"]
    session.consent.relationship = rep_entry["relationship"]


def _poll_consent(session: SessionState) -> str:
    from app import data_loader as _dl
    scenarios = _dl.consent_scenarios()
    seq = scenarios.get(session.consent.scenario, scenarios["default"])["status_sequence"]
    idx = session.consent.poll_index
    if idx >= len(seq) or idx >= MAX_CONSENT_POLLS_BEFORE_ESCALATE:
        session.consent.status = "timed_out"
    else:
        session.consent.status = seq[idx]
    session.consent.poll_index += 1
    return session.consent.status


def handle(session: SessionState, user_message: str, sig: ExtractedSignals) -> TurnResult:
    session.verification_attempts += 1

    # ---- Representative / consent path -------------------------------------------------
    if session.consent.active and not session.verified:
        status = _poll_consent(session)
        if status == "approved":
            session.verified = True
            session.matched_party_id = session.consent.buyer_party_id
            session.match_breakdown = ["representative_consent"]
            session.phase = Phase.RESOLVE_INTENT
            buyer = dl.find_party(session.consent.buyer_party_id)
            buyer_name = buyer["name"] if buyer else "the policyholder"
            return TurnResult(
                reply_template=(
                    f"Thanks for your patience — {buyer_name} has approved you to discuss this account as "
                    f"their {session.consent.relationship}. You're verified. Let me pull up what you called about."
                ),
                control_flags={"just_verified": True, "representative": True},
                phase_notes="Representative consent approved; identity verified via consent, not PII match.",
            )
        if status == "timed_out":
            session.escalate_to_human = True
            return TurnResult(
                reply_template=(
                    "I'm sorry, but I wasn't able to get confirmation from the policyholder in time, so I "
                    "can't grant account access as a representative right now. I can transfer you to a human "
                    "representative who can arrange manual authorization, or the policyholder can call in "
                    "directly and add you as an authorized contact. Which would you prefer?"
                ),
                control_flags={"escalate_offered": True},
                phase_notes="Consent scenario timed out; offering human escalation without disclosing anything.",
            )
        # still pending
        return TurnResult(
            reply_template=(
                "I've sent a consent request to the policyholder and I'm still waiting to hear back — this "
                "usually only takes a moment. I'll check again as soon as you're ready, or I can transfer you "
                "to a human representative if you'd rather not wait."
            ),
            phase_notes=f"Consent still pending (poll {session.consent.poll_index}).",
        )

    if sig.is_representative_signal and not session.representative_mode and not session.verified:
        rep_entry = dl.find_representative(rep_name=sig.on_behalf_of_name, buyer_name=sig.on_behalf_of_name)
        # also try matching by the caller's own extracted name against rep_name
        if not rep_entry and sig.full_name:
            rep_entry = dl.find_representative(rep_name=sig.full_name)
        if rep_entry:
            _start_consent_flow(session, rep_entry, sig.full_name)
            return TurnResult(
                reply_template=(
                    f"Thanks for letting me know — since you're calling as {rep_entry['buyer_name']}'s "
                    f"{rep_entry['relationship']}, I do need the policyholder's consent before I can share any "
                    "account details with you. I'm sending a consent request now; this usually only takes a "
                    "moment, so go ahead and let me know when you're ready and I'll check the status."
                ),
                control_flags={"consent_started": True},
                phase_notes="Recognized registered representative; starting consent flow.",
            )
        else:
            return TurnResult(
                reply_template=(
                    "I understand you're calling on someone else's behalf. I don't have a record of you as an "
                    "authorized representative on this account yet, so I'm not able to share any details "
                    "without the policyholder's consent. The policyholder can call in directly to add you as "
                    "an authorized contact, or I can transfer you to a human representative to arrange manual "
                    "authorization. Which would you prefer?"
                ),
                control_flags={"escalate_offered": True},
                phase_notes="Unregistered representative attempt; declined without confirming/denying account existence.",
            )

    # ---- Standard PII verification path -------------------------------------------------
    if sig.refusal_signal:
        session.verification_refusal_strikes += 1

    party, matched = find_best_match(session.identity)
    session.match_breakdown = matched

    if party and len(matched) >= MIN_ID_MATCHES:
        session.verified = True
        session.matched_party_id = party["party_id"]
        session.phase = Phase.RESOLVE_INTENT
        return TurnResult(
            reply_template=(
                "Thank you, that's everything I need — you're verified. Let me pull up what you called about."
            ),
            control_flags={"just_verified": True},
            phase_notes=f"Verified via {matched} for party {party['party_id']}.",
        )

    # not yet verified
    still_need = _missing_field_prompt(matched)
    refusal_note = ""
    if sig.refusal_signal:
        refusal_note = (
            "I'm not able to share any claim details until I can confirm at least three pieces of identifying "
            "information — this protects your private claim data in case someone else has your name. "
        )
    offer_escalate = session.verification_refusal_strikes >= 2 or session.verification_attempts >= 5
    escalate_note = ""
    if offer_escalate:
        escalate_note = " If you'd rather not continue with this, I can transfer you to a human representative instead."

    if not any(session.identity.as_dict().values()):
        body = f"To protect your account, I first need to verify your identity — {still_need}"
    elif party is None:
        body = (
            f"I wasn't able to match those details together. Could you double-check them, or share a "
            f"different piece of ID — {still_need}"
        )
    else:
        got = ", ".join(FIELD_LABELS[m].replace("your ", "").replace("the ", "") for m in matched) or "none yet"
        body = f"I've confirmed {got} so far, but I need at least one more to verify you — {still_need}"

    return TurnResult(
        reply_template=(refusal_note + body + escalate_note),
        control_flags={"escalate_offered": offer_escalate},
        forbidden_terms=["CL-", "denied", "denial", "approved claim", "case_id"],
        phase_notes=f"Not yet verified. matched={matched} attempts={session.verification_attempts}.",
    )

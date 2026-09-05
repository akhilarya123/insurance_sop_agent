"""
Deterministic, regex/keyword based extraction of structured signals from
free-form caller text. This is the reliability backbone of the harness:
it works identically whether or not an LLM is available, and the LLM
(when used) is only ever asked to *supplement* or *phrase*, never to
replace, this layer for anything safety- or gate-relevant.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

MONTHS = {
    "january": "01", "jan": "01", "february": "02", "feb": "02", "march": "03",
    "mar": "03", "april": "04", "apr": "04", "may": "05", "june": "06", "jun": "06",
    "july": "07", "jul": "07", "august": "08", "aug": "08", "september": "09",
    "sep": "09", "sept": "09", "october": "10", "oct": "10", "november": "11",
    "nov": "11", "december": "12", "dec": "12",
}

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
POLICY_RE = re.compile(r"\bPOL-?\s?(\d{3,7})\b", re.IGNORECASE)
CASE_ID_RE = re.compile(r"\bCL-?\s?(\d{3,7})\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")
ISO_DATE_RE = re.compile(r"\b(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")
SLASH_DATE_RE = re.compile(r"\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-]((19|20)\d{2})\b")
WORDY_DATE_RE = re.compile(
    r"\b(" + "|".join(MONTHS.keys()) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+((19|20)\d{2})\b",
    re.IGNORECASE,
)
SSN_CONTEXT_RE = re.compile(
    r"(?:ssn|social security|social|security number|id number|national id)[^0-9]{0,25}(\d{4})\b",
    re.IGNORECASE,
)
LAST4_GENERIC_RE = re.compile(r"last\s*(?:four|4)[^0-9]{0,20}(\d{4})\b", re.IGNORECASE)

NAME_PATTERNS = [
    re.compile(r"\bmy name is\s+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,3})", re.IGNORECASE),
    re.compile(r"\bthis is\s+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,3})\s+(?:calling|speaking)?", re.IGNORECASE),
    re.compile(r"\bi'?m\s+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,3})\b(?!\s+(?:calling|the|a|an|so|just|not|going))", re.IGNORECASE),
    re.compile(r"\bname'?s\s+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,3})", re.IGNORECASE),
]

ON_BEHALF_RE = re.compile(r"(?:on behalf of|calling for)\s+(.*?)(?:[.!?]|$)", re.IGNORECASE)
REP_FALLBACK_PATTERNS = [
    re.compile(r"\bi'?m\s+(?:her|his|their)\s+(son|daughter|husband|wife|spouse|child|caregiver|representative|power of attorney)\b", re.IGNORECASE),
    re.compile(r"\bas\s+(?:her|his|their)\s+(son|daughter|husband|wife|spouse|child|caregiver|representative)\b", re.IGNORECASE),
    re.compile(r"\bauthorized representative\b", re.IGNORECASE),
    re.compile(r"\bpower of attorney\b", re.IGNORECASE),
]
RELATIONSHIP_WORDS = ["son", "daughter", "husband", "wife", "spouse", "child", "caregiver",
                      "representative", "power of attorney", "mother", "father", "mom", "dad"]
PROPER_NAME_RE = re.compile(r"([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,3})")

CASE_TYPE_KEYWORDS = {
    "healthcare": ["health", "medical", "healthcare", "doctor", "hospital", "surgery", "pathology"],
    "dental": ["dental", "dentist", "tooth", "teeth"],
    "auto": ["auto", "car", "vehicle", "accident", "collision", "fender"],
}
STATUS_KEYWORDS = {
    "denied": ["denied", "denial", "rejected", "refused", "turned down"],
    "closed": ["closed", "settled", "completed", "finished", "paid out"],
    "open": ["open", "in progress", "pending review", "still processing"],
}

HUMAN_TRANSFER_RE = re.compile(
    r"\b(talk to|speak (?:to|with)|transfer me to|connect me to|get me)\b.{0,15}\b(human|person|representative|agent|supervisor|someone real)\b",
    re.IGNORECASE,
)
END_CALL_RE = re.compile(
    r"\b(that'?s all|no more questions|nothing else|i'?m done|goodbye|good bye|bye now|that will be all|no thanks that'?s it)\b",
    re.IGNORECASE,
)
REFUSAL_RE = re.compile(
    r"\b(already told you|i said that already|why do you need|i'?m not (?:giving|telling)|this is ridiculous|i refuse|already gave you|already provided|i don'?t (?:want|feel comfortable) (?:to )?(?:giving|sharing))\b",
    re.IGNORECASE,
)
YES_RE = re.compile(r"^\s*(yes|yeah|yep|yup|sure|please do|go ahead|correct|that'?s right|affirmative|ok(?:ay)?)\b", re.IGNORECASE)
NO_RE = re.compile(r"^\s*(no|nope|nah|don'?t|do not|skip|not now|negative)\b", re.IGNORECASE)

FRUSTRATION_WORDS = ["ridiculous", "unacceptable", "angry", "furious", "annoyed", "annoying",
                      "frustrated", "frustrating", "fed up", "sick of", "unbelievable"]
ANXIETY_WORDS = ["worried", "scared", "anxious", "afraid", "nervous", "stressed", "desperate"]
CONFUSION_WORDS = ["confused", "don't understand", "not sure what", "i'm lost", "what does that mean"]
ANGER_MARKERS_RE = re.compile(r"[A-Z]{4,}|!{2,}")


@dataclass
class ExtractedSignals:
    full_name: Optional[str] = None
    dob: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    ssn_last4: Optional[str] = None
    policy_number: Optional[str] = None

    case_type_hint: Optional[str] = None
    status_hint: Optional[str] = None
    date_hint: Optional[str] = None
    explicit_case_id: Optional[str] = None
    intent_hint_text: Optional[str] = None

    is_representative_signal: bool = False
    on_behalf_of_name: Optional[str] = None
    relationship: Optional[str] = None

    wants_human_transfer: bool = False
    wants_to_end: bool = False
    refusal_signal: bool = False
    yes_no: Optional[str] = None


def _normalize_date(y: str, m: str, d: str) -> str:
    return f"{y}-{int(m):02d}-{int(d):02d}"


def extract_dob(text: str) -> Optional[str]:
    m = ISO_DATE_RE.search(text)
    if m:
        return m.group(0)
    m = SLASH_DATE_RE.search(text)
    if m:
        mm, dd, yyyy = m.group(1), m.group(2), m.group(3)
        return _normalize_date(yyyy, mm, dd)
    m = WORDY_DATE_RE.search(text)
    if m:
        month_word, day, year = m.group(1).lower(), m.group(2), m.group(3)
        return _normalize_date(year, MONTHS[month_word], day)
    return None


def extract_phone(text: str) -> Optional[str]:
    m = PHONE_RE.search(text)
    if not m:
        return None
    digits = re.sub(r"\D", "", m.group(0))
    return digits


def extract_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else None


def extract_policy_number(text: str) -> Optional[str]:
    m = POLICY_RE.search(text)
    return f"POL-{m.group(1)}" if m else None


def extract_case_id(text: str) -> Optional[str]:
    m = CASE_ID_RE.search(text)
    return f"CL-{m.group(1)}" if m else None


def extract_ssn_last4(text: str) -> Optional[str]:
    m = SSN_CONTEXT_RE.search(text)
    if m:
        return m.group(1)
    m = LAST4_GENERIC_RE.search(text)
    if m:
        return m.group(1)
    return None


def extract_full_name(text: str) -> Optional[str]:
    for pat in NAME_PATTERNS:
        m = pat.search(text)
        if m:
            candidate = m.group(1).strip()
            # avoid false positives like "I'm calling" leaking through
            words = candidate.split()
            if 1 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
                return candidate
    return None


def extract_representative_signal(text: str):
    m = ON_BEHALF_RE.search(text)
    if m:
        segment = m.group(1)
        seg_low = segment.lower()
        relationship = None
        for word in RELATIONSHIP_WORDS:
            if re.search(rf"\b{re.escape(word)}\b", seg_low):
                relationship = word
                break
        name_m = PROPER_NAME_RE.search(segment)
        on_behalf_of_name = name_m.group(1).strip() if name_m else None
        return True, on_behalf_of_name, relationship

    for pat in REP_FALLBACK_PATTERNS:
        mm = pat.search(text)
        if mm:
            groups = [g for g in mm.groups() if g]
            relationship = groups[0].lower() if groups else None
            return True, None, relationship

    return False, None, None


def extract_case_hints(text: str):
    low = text.lower()
    case_type_hint = None
    for ctype, kws in CASE_TYPE_KEYWORDS.items():
        if any(kw in low for kw in kws):
            case_type_hint = ctype
            break
    status_hint = None
    for status, kws in STATUS_KEYWORDS.items():
        if any(kw in low for kw in kws):
            status_hint = status
            break
    date_hint = None
    for month_word in MONTHS:
        if re.search(rf"\b{re.escape(month_word)}\b", low):
            date_hint = month_word
            break
    return case_type_hint, status_hint, date_hint


def detect_human_transfer(text: str) -> bool:
    return bool(HUMAN_TRANSFER_RE.search(text))


def detect_end_call(text: str) -> bool:
    return bool(END_CALL_RE.search(text))


def detect_refusal(text: str) -> bool:
    return bool(REFUSAL_RE.search(text))


def detect_yes_no(text: str) -> Optional[str]:
    if YES_RE.search(text.strip()):
        return "yes"
    if NO_RE.search(text.strip()):
        return "no"
    return None


def maybe_intent_hint_text(text: str, case_type_hint, status_hint, date_hint) -> Optional[str]:
    if case_type_hint or status_hint or date_hint:
        return text.strip()
    return None


def extract_all(text: str) -> ExtractedSignals:
    sig = ExtractedSignals()
    sig.full_name = extract_full_name(text)
    sig.dob = extract_dob(text)
    sig.phone = extract_phone(text)
    sig.email = extract_email(text)
    sig.ssn_last4 = extract_ssn_last4(text)
    sig.policy_number = extract_policy_number(text)

    sig.explicit_case_id = extract_case_id(text)
    sig.case_type_hint, sig.status_hint, sig.date_hint = extract_case_hints(text)
    sig.intent_hint_text = maybe_intent_hint_text(text, sig.case_type_hint, sig.status_hint, sig.date_hint)

    is_rep, on_behalf_of_name, relationship = extract_representative_signal(text)
    sig.is_representative_signal = is_rep
    sig.on_behalf_of_name = on_behalf_of_name
    sig.relationship = relationship

    sig.wants_human_transfer = detect_human_transfer(text)
    sig.wants_to_end = detect_end_call(text)
    sig.refusal_signal = detect_refusal(text)
    sig.yes_no = detect_yes_no(text)
    return sig

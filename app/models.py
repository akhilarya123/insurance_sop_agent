"""
Framework-agnostic data structures shared by the SOP engine. Deliberately
implemented with plain dataclasses (not pydantic) so the core logic can be
unit-tested without any web-framework dependency installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time
import uuid


class Phase(str, Enum):
    VERIFY_ID = "VERIFY_ID"
    RESOLVE_INTENT = "RESOLVE_INTENT"
    PROCESS_CASE = "PROCESS_CASE"
    POST_PROCESS = "POST_PROCESS"
    ENDED = "ENDED"


@dataclass
class TranscriptTurn:
    role: str            # "user" | "agent" | "system"
    text: str
    phase: str
    ts: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)


@dataclass
class ConsentState:
    active: bool = False
    scenario: str = "default"
    poll_index: int = -1
    status: str = "not_requested"     # not_requested | pending | approved | denied | timed_out
    buyer_party_id: Optional[str] = None
    rep_name: Optional[str] = None
    relationship: Optional[str] = None


@dataclass
class IdentitySlots:
    full_name: Optional[str] = None
    dob: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    ssn_last4: Optional[str] = None
    policy_number: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "full_name": self.full_name,
            "dob": self.dob,
            "phone": self.phone,
            "email": self.email,
            "ssn_last4": self.ssn_last4,
            "policy_number": self.policy_number,
        }

    def filled_required_count(self, required_fields) -> int:
        d = self.as_dict()
        return sum(1 for f in required_fields if d.get(f))


@dataclass
class Memory:
    """Cross-phase memory captured opportunistically, regardless of the
    current SOP phase. E.g. an intent hint volunteered during VERIFY_ID."""
    intent_hint_text: Optional[str] = None
    case_type_hint: Optional[str] = None
    status_hint: Optional[str] = None
    date_hint: Optional[str] = None
    explicit_case_id_hint: Optional[str] = None
    notes: list = field(default_factory=list)

    def has_case_hint(self) -> bool:
        return any([
            self.intent_hint_text, self.case_type_hint,
            self.status_hint, self.date_hint, self.explicit_case_id_hint,
        ])


@dataclass
class SessionState:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    phase: Phase = Phase.VERIFY_ID
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    identity: IdentitySlots = field(default_factory=IdentitySlots)
    verified: bool = False
    matched_party_id: Optional[str] = None
    match_breakdown: list = field(default_factory=list)   # which categories matched
    verification_attempts: int = 0
    verification_refusal_strikes: int = 0

    representative_mode: bool = False
    consent: ConsentState = field(default_factory=ConsentState)

    memory: Memory = field(default_factory=Memory)

    resolved_intent: Optional[str] = None
    resolved_case_id: Optional[str] = None
    candidate_case_ids: list = field(default_factory=list)

    off_topic_strikes: int = 0
    human_transfer_offered: bool = False
    escalate_to_human: bool = False

    email_offered: bool = False
    email_decision: Optional[str] = None   # "sent" | "skipped" | None
    email_summary_text: Optional[str] = None

    transcript: list = field(default_factory=list)  # list[TranscriptTurn]
    debug_log: list = field(default_factory=list)

    def touch(self):
        self.last_active = time.time()

    def add_turn(self, role: str, text: str, meta: Optional[dict] = None):
        self.transcript.append(TranscriptTurn(role=role, text=text, phase=self.phase.value, meta=meta or {}))

    def log(self, msg: str):
        self.debug_log.append(f"[{self.phase.value}] {msg}")


@dataclass
class TurnResult:
    """What a phase handler returns to the orchestrator."""
    reply_template: str                     # deterministic, always-safe text
    facts: dict = field(default_factory=dict)   # facts allowed for LLM polish
    forbidden_terms: list = field(default_factory=list)  # must never appear unless verified/resolved
    allow_llm_polish: bool = True
    phase_notes: str = ""                   # short note for the debug panel
    control_flags: dict = field(default_factory=dict)

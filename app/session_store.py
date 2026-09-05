"""Simple in-memory session store, keyed by session_id.

This is intentionally minimal for a demo harness. Swap for Redis/DB for
multi-process production deployments.
"""
from __future__ import annotations

import time
from threading import Lock
from typing import Optional

from app.config import SESSION_IDLE_TTL_SECONDS
from app.models import SessionState

_LOCK = Lock()
_SESSIONS: dict[str, SessionState] = {}


def get_or_create(session_id: Optional[str]) -> SessionState:
    with _LOCK:
        _evict_stale()
        if session_id and session_id in _SESSIONS:
            s = _SESSIONS[session_id]
            s.touch()
            return s
        s = SessionState()
        if session_id:
            s.session_id = session_id
        _SESSIONS[s.session_id] = s
        return s


def reset(session_id: str) -> SessionState:
    with _LOCK:
        s = SessionState(session_id=session_id)
        _SESSIONS[session_id] = s
        return s


def _evict_stale():
    now = time.time()
    stale = [sid for sid, s in _SESSIONS.items() if now - s.last_active > SESSION_IDLE_TTL_SECONDS]
    for sid in stale:
        _SESSIONS.pop(sid, None)

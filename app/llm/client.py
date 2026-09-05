"""
Minimal Ollama chat client. No paid API keys; talks to a locally running
`ollama serve` instance. Every call is wrapped so that any failure (model
not pulled, server not running, timeout, bad JSON) degrades to `None`
rather than raising -- the orchestrator always has a deterministic
template reply to fall back on.
"""
from __future__ import annotations

import time
import requests

from app import config

_AVAILABILITY_CACHE = {"ts": 0.0, "ok": False}
_AVAILABILITY_TTL = 15.0


def is_available() -> bool:
    now = time.time()
    if now - _AVAILABILITY_CACHE["ts"] < _AVAILABILITY_TTL:
        return _AVAILABILITY_CACHE["ok"]
    ok = False
    try:
        r = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
        ok = r.status_code == 200
    except Exception:
        ok = False
    _AVAILABILITY_CACHE["ts"] = now
    _AVAILABILITY_CACHE["ok"] = ok
    return ok


def chat(messages: list[dict], temperature: float = None, timeout: float = None) -> str | None:
    """messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
    Returns the assistant's text, or None if the call failed for any reason.
    """
    if not config.LLM_POLISH_ENABLED:
        return None
    try:
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json={
                "model": config.OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature if temperature is not None else config.LLM_TEMPERATURE},
            },
            timeout=timeout or config.OLLAMA_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        content = (data.get("message") or {}).get("content")
        if not content or not content.strip():
            return None
        return content.strip()
    except Exception:
        return None

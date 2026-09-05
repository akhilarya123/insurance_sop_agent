"""
FastAPI wrapper around the orchestrator. Kept intentionally thin: this file
owns HTTP concerns only (request/response schemas, routing, static files).
All SOP logic lives in app/orchestrator.py and app/sop/*.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config, session_store, orchestrator
from app.llm import client as ollama_client

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Insurance Claims SOP Agent", version="1.0.0")


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    consent_scenario: Optional[str] = None  # "default" | "timeout" (demo/testing hook)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    phase: str
    verified: bool
    matched_party_id: Optional[str] = None
    representative_mode: bool = False
    resolved_case_id: Optional[str] = None
    resolved_intent: Optional[str] = None
    escalate_to_human: bool = False
    email_decision: Optional[str] = None
    emotion: str
    polished_by_llm: bool
    debug_log: list[str] = []


class ResetRequest(BaseModel):
    session_id: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ollama_reachable": ollama_client.is_available(),
        "ollama_model": config.OLLAMA_MODEL,
        "llm_polish_enabled": config.LLM_POLISH_ENABLED,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session = session_store.get_or_create(req.session_id)
    if not session.transcript:
        # first turn of a brand new session: greet before processing, unless
        # the caller's very first message already carries real content.
        pass
    result = orchestrator.handle_turn(session, req.message, consent_scenario=req.consent_scenario)
    return result


@app.post("/api/reset")
def reset(req: ResetRequest):
    session = session_store.reset(req.session_id)
    return {"session_id": session.session_id, "reply": orchestrator.greeting_message(), "phase": session.phase.value}


@app.get("/api/greeting")
def greeting():
    return {"reply": orchestrator.greeting_message()}


# --- Static test UI ---------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Static UI not found; see /health and POST /api/chat."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT, reload=True)

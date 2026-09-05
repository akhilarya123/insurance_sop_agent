"""
Central configuration, all overridable via environment variables / .env file.

No paid API keys are required. The agent talks to a locally running Ollama
server. If Ollama is unreachable or misbehaves, the harness automatically
falls back to a deterministic, template-based response generator so the SOP
demo always works end-to-end even without a model running.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("SOP_DATA_DIR", BASE_DIR / "data"))

# --- LLM (Ollama) settings -------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "20"))
# If false, the agent never calls the LLM and runs on the deterministic
# template engine only (useful for CI / offline testing).
LLM_POLISH_ENABLED = os.getenv("LLM_POLISH_ENABLED", "true").lower() != "false"
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))

# --- SOP thresholds ---------------------------------------------------------
REQUIRED_ID_FIELDS = ["full_name", "dob", "phone", "email", "ssn_last4"]
MIN_ID_MATCHES = 3

MAX_OFF_TOPIC_STRIKES_BEFORE_OFFER = 2
MAX_OFF_TOPIC_STRIKES_BEFORE_ESCALATE = 4

MAX_VERIFICATION_REFUSAL_STRIKES_BEFORE_ESCALATE_OFFER = 2
MAX_CONSENT_POLLS_BEFORE_ESCALATE = 3

SESSION_IDLE_TTL_SECONDS = int(os.getenv("SESSION_IDLE_TTL_SECONDS", str(60 * 60 * 4)))

HOST = os.getenv("SOP_HOST", "0.0.0.0")
PORT = int(os.getenv("SOP_PORT", "8000"))

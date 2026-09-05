#!/usr/bin/env bash
# Convenience script: create/activate a venv, install deps, run the API.
set -e
cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  python3.11 -m venv .venv
fi
source .venv/bin/activate
pip3.11 install -q -r requirements.txt

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "Created .env from .env.example -- edit OLLAMA_MODEL if needed."
fi

echo "Starting server on http://localhost:8000 ..."
python3.11 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

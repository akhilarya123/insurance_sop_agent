"""Loads the read-only demo fixtures and exposes small lookup helpers."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Optional

from app.config import DATA_DIR


def _load(name: str):
    with open(DATA_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def policyholders() -> list:
    return _load("policyholders.json")


@lru_cache(maxsize=1)
def claims() -> list:
    return _load("claims.json")


@lru_cache(maxsize=1)
def representatives() -> list:
    return _load("representatives.json")


@lru_cache(maxsize=1)
def consent_scenarios() -> dict:
    return _load("consent_scenarios.json")


@lru_cache(maxsize=1)
def required_document_guideline() -> dict:
    return _load("required_document_guideline.json")


@lru_cache(maxsize=1)
def claim_schema() -> dict:
    return _load("claim_schema.json")


def find_party(party_id: str) -> Optional[dict]:
    for p in policyholders():
        if p["party_id"] == party_id:
            return p
    return None


def find_claims_by_party(party_id: str) -> list:
    return [c for c in claims() if c["party_id"] == party_id]


def find_claim_by_id(case_id: str) -> Optional[dict]:
    for c in claims():
        if c["case_id"].lower() == case_id.lower():
            return c
    return None


def find_representative(rep_name: Optional[str] = None, buyer_name: Optional[str] = None) -> Optional[dict]:
    for r in representatives():
        if rep_name and r["rep_name"].lower() == rep_name.lower():
            return r
        if buyer_name and r["buyer_name"].lower() == buyer_name.lower():
            return r
    return None

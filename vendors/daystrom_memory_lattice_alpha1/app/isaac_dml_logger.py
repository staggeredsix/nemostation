"""Helper wrappers for logging/querying robotics traces in DML."""
from __future__ import annotations

import os
from typing import Any, Dict

import requests

SERVICE_URL = os.getenv("DML_SERVICE_URL", "http://localhost:8000")


def log_event(tenant_id: str, robot_id: str, mission_id: str, text: str, meta: Dict[str, Any] | None = None) -> None:
    payload = {
        "tenant_id": tenant_id,
        "client_id": robot_id,
        "session_id": mission_id,
        "kind": "memory",
        "text": text,
        "meta": meta or {},
    }
    requests.post(f"{SERVICE_URL}/v1/memory/ingest", json=payload, timeout=15).raise_for_status()


def query_robot_history(tenant_id: str, robot_id: str, query: str, top_k: int = 5) -> Dict[str, Any]:
    payload = {
        "tenant_id": tenant_id,
        "client_id": robot_id,
        "query": query,
        "top_k": top_k,
        "scope": "personal",
        "include_workflows": False,
    }
    resp = requests.post(f"{SERVICE_URL}/v1/memory/retrieve", json=payload, timeout=20)
    resp.raise_for_status()
    return resp.json()

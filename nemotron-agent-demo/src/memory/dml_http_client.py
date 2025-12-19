from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

DML_BASE_URL = os.getenv("DML_BASE_URL", "http://dml-service:9001").rstrip("/")
DEFAULT_TIMEOUT_SEC = float(os.getenv("DML_TIMEOUT_SEC", "6"))


class DMLServiceError(RuntimeError):
    pass


@dataclass
class DMLReport:
    entries: list[Dict[str, Any]]
    preamble: str
    tokens: int
    latency_ms: int
    error: Optional[str] = None


def _request(method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{DML_BASE_URL}{path}"
    try:
        resp = requests.request(method, url, json=payload, timeout=DEFAULT_TIMEOUT_SEC)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise DMLServiceError(f"DML service request failed: {exc}") from exc
    if not resp.content:
        return {}
    try:
        return resp.json()
    except ValueError as exc:
        raise DMLServiceError("DML service returned invalid JSON") from exc


def check_health() -> Tuple[bool, Optional[str]]:
    try:
        _request("GET", "/health")
    except DMLServiceError as exc:
        return False, str(exc)
    return True, None


def ingest(text: str, meta: Optional[Dict[str, Any]] = None) -> None:
    payload = {"text": text, "meta": meta}
    _request("POST", "/ingest", payload)


def retrieval_report(prompt: str, top_k: int = 6) -> DMLReport:
    payload = {"prompt": prompt, "top_k": top_k}
    data = _request("POST", "/retrieval_report", payload)
    return DMLReport(
        entries=data.get("entries", []),
        preamble=data.get("preamble", ""),
        tokens=int(data.get("tokens", 0)),
        latency_ms=int(data.get("latency_ms", 0)),
        error=data.get("error"),
    )


def build_preamble(prompt: str, top_k: int = 6) -> str:
    payload = {"prompt": prompt, "top_k": top_k}
    data = _request("POST", "/build_preamble", payload)
    return str(data.get("preamble", ""))

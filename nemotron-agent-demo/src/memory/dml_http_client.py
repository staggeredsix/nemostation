from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

DML_BASE_URL = os.getenv("DML_BASE_URL", "http://dml-service:9001").rstrip("/")

DEFAULT_CONNECT_TIMEOUT_SEC = 3.0


def _read_timeout() -> float:
    value = os.getenv("DML_HTTP_TIMEOUT_READ", "45")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 45.0


DEFAULT_READ_TIMEOUT_SEC = _read_timeout()

_retry_config = Retry(
    total=3,
    connect=3,
    read=3,
    status=3,
    backoff_factor=0.5,
    status_forcelist=(502, 503, 504),
    allowed_methods=frozenset(["GET", "POST"]),
    raise_on_status=False,
)
_session = requests.Session()
_adapter = HTTPAdapter(max_retries=_retry_config)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)


class DMLServiceError(RuntimeError):
    pass


@dataclass
class DMLReport:
    entries: list[Dict[str, Any]]
    preamble: str
    tokens: int
    latency_ms: int
    error: Optional[str] = None


@dataclass
class DMLCookbook:
    found: bool
    cookbook_text: str
    sources: list[Any]
    latency_ms: int


@dataclass
class DMLRunReportResult:
    ok: bool
    ingested_id: str
    summary_id: str
    summary_latency_ms: int


def _request(
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    url = f"{DML_BASE_URL}{path}"
    try:
        resp = _session.request(
            method,
            url,
            json=payload,
            timeout=timeout or (DEFAULT_CONNECT_TIMEOUT_SEC, DEFAULT_READ_TIMEOUT_SEC),
        )
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


def get_cookbook(scenario_key: str, goal: str, top_k: int = 6) -> DMLCookbook:
    payload = {"scenario_key": scenario_key, "goal": goal, "top_k": top_k}
    data = _request("POST", "/cookbook/get", payload, timeout=(3.0, 20.0))
    return DMLCookbook(
        found=bool(data.get("found", False)),
        cookbook_text=str(data.get("cookbook_text", "")),
        sources=list(data.get("sources", []) or []),
        latency_ms=int(data.get("latency_ms", 0)),
    )


def ingest_run_report(payload: Dict[str, Any]) -> DMLRunReportResult:
    data = _request("POST", "/run_report/ingest", payload, timeout=(3.0, 120.0))
    return DMLRunReportResult(
        ok=bool(data.get("ok", False)),
        ingested_id=str(data.get("ingested_id", "")),
        summary_id=str(data.get("summary_id", "")),
        summary_latency_ms=int(data.get("summary_latency_ms", 0)),
    )

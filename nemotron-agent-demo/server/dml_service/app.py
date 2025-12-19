from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger("dml-service")
logging.basicConfig(level=logging.INFO)

try:
    import faiss  # noqa: F401

    logger.info("FAISS import succeeded")
except Exception as exc:  # noqa: BLE001
    logger.exception("FAISS import failed: %s", exc)
    raise

from daystrom_dml.dml_adapter import DMLAdapter


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    meta: Optional[Dict[str, Any]] = None


class RetrievalRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    top_k: int = Field(6, ge=1, le=10)


class CookbookGetRequest(BaseModel):
    scenario_key: str = Field(..., min_length=1)
    goal: str = Field(..., min_length=1)
    top_k: int = Field(6, ge=1, le=10)


class CookbookGetResponse(BaseModel):
    found: bool
    cookbook_text: str
    sources: list[Any]
    latency_ms: int


class RunReportRequest(BaseModel):
    scenario_key: str = Field(..., min_length=1)
    goal: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    trace: Dict[str, Any]
    final: str
    success: bool
    meta: Optional[Dict[str, Any]] = None


class RunReportResponse(BaseModel):
    ok: bool
    ingested_id: str
    summary_id: str
    summary_latency_ms: int


class RetrievalResponse(BaseModel):
    entries: list[Dict[str, Any]]
    preamble: str
    tokens: int
    latency_ms: int
    error: Optional[str] = None


class PreambleResponse(BaseModel):
    preamble: str


def _build_adapter() -> DMLAdapter:
    storage_root = Path(os.getenv("DML_STORAGE_DIR", "/data/dml")).expanduser()
    storage_root.mkdir(parents=True, exist_ok=True)
    overrides = {
        "storage_dir": str(storage_root),
        "persistence": {"enable": True, "path": "dml_state.jsonl", "interval_sec": 300},
        "rag_store": {
            "enable": True,
            "path": "rag_index.faiss",
            "meta_path": "rag_meta.json",
            "backend": "faiss",
            "dim": 384,
        },
    }
    adapter = DMLAdapter(config_overrides=overrides)
    logger.info("DMLAdapter initialized")
    if adapter.persistent_rag_store is not None:
        logger.info("PersistentRAGStore: enabled")
    else:
        logger.warning("PersistentRAGStore: disabled")
    return adapter


def _openai_client() -> OpenAI:
    base_url = os.getenv("DML_OPENAI_BASE_URL", "").strip()
    if not base_url:
        raise RuntimeError("DML_OPENAI_BASE_URL is not set")
    api_key = os.getenv("DML_OPENAI_API_KEY", "EMPTY")
    return OpenAI(base_url=base_url, api_key=api_key)


def _summary_settings() -> Dict[str, Any]:
    return {
        "max_tokens": 512,
        "temperature": 0,
        "enable_thinking": False,
    }


def _build_cookbook_prompt(payload: RunReportRequest) -> str:
    trace_json = json.dumps(payload.trace, ensure_ascii=False, sort_keys=True)
    meta_json = json.dumps(payload.meta or {}, ensure_ascii=False, sort_keys=True)
    return "\n".join(
        [
            "Summarize the following agent run into the cookbook format.",
            "Keep it short (<= 2k chars) and high-signal.",
            "Cookbook summary format (must be consistent):",
            "TITLE: <scenario_key>",
            "WHAT WORKED:",
            "- ...",
            "WHAT FAILED:",
            "- ...",
            "DECISION HEURISTICS:",
            "- ...",
            "PITFALLS:",
            "- ...",
            "NEXT TIME TRY:",
            "- ...",
            "KEY ARTIFACTS / LINKS (if present):",
            "- ...",
            "",
            f"SCENARIO_KEY: {payload.scenario_key}",
            f"GOAL: {payload.goal}",
            f"RUN_ID: {payload.run_id}",
            f"SUCCESS: {payload.success}",
            f"FINAL: {payload.final}",
            f"META: {meta_json}",
            f"TRACE: {trace_json}",
        ]
    )


_EMPTY_SUMMARY_LOGGED = False


def _summarize_cookbook(payload: RunReportRequest) -> tuple[str, int]:
    start = time.perf_counter()
    settings = _summary_settings()
    model = os.getenv("DML_OPENAI_MODEL", "").strip()
    if not model:
        raise RuntimeError("DML_OPENAI_MODEL is not set")
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    client = _openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a precise summariser. Output only the cookbook summary.",
            },
            {"role": "user", "content": _build_cookbook_prompt(payload)},
        ],
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
        extra_body=extra_body,
    )
    message = response.choices[0].message
    summary = (message.content or "").strip()
    if not summary:
        summary = (getattr(message, "reasoning_content", None) or "").strip()
    if not summary:
        global _EMPTY_SUMMARY_LOGGED
        if not _EMPTY_SUMMARY_LOGGED:
            _EMPTY_SUMMARY_LOGGED = True
            try:
                logger.warning("Empty cookbook summary response: %s", response.model_dump_json())
            except Exception:  # noqa: BLE001
                logger.warning("Empty cookbook summary response: %s", response)
        summary = _fallback_cookbook(payload).strip()
    latency_ms = int((time.perf_counter() - start) * 1000)
    return summary.strip(), latency_ms


def _fallback_cookbook(payload: RunReportRequest) -> str:
    status = "success" if payload.success else "failure"
    meta_json = json.dumps(payload.meta or {}, ensure_ascii=False, sort_keys=True)
    return "\n".join(
        [
            f"TITLE: {payload.scenario_key}",
            "WHAT WORKED:",
            f"- Run status: {status}",
            "WHAT FAILED:",
            "- No model summary available; using basic run metadata.",
            "DECISION HEURISTICS:",
            f"- Goal: {payload.goal}",
            "PITFALLS:",
            "- Review run trace for missing steps or tool errors.",
            "NEXT TIME TRY:",
            "- Re-run summarisation or provide more detailed run metadata.",
            "KEY ARTIFACTS / LINKS (if present):",
            f"- RUN_ID: {payload.run_id}",
            f"- META: {meta_json}",
        ]
    )


app = FastAPI(title="DML Service", version="1.0")

adapter = _build_adapter()


@app.on_event("startup")
def warmup_embedder() -> None:
    start = time.perf_counter()
    try:
        adapter.embedder.embed("warmup")
    except Exception:  # noqa: BLE001
        logger.exception("DML warmup embed failed")
        return
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("DML warmup embed completed in %.2f ms", duration_ms)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "persistent_rag_store": adapter.persistent_rag_store is not None,
        "storage_dir": str(adapter.storage_dir),
    }


@app.post("/ingest")
def ingest(request: IngestRequest) -> Dict[str, str]:
    try:
        adapter.ingest(request.text, meta=request.meta)
    except Exception as exc:  # noqa: BLE001
        logger.exception("DML ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "ok"}


@app.post("/retrieval_report", response_model=RetrievalResponse)
def retrieval_report(request: RetrievalRequest) -> RetrievalResponse:
    try:
        report = adapter.retrieval_report(request.prompt, top_k=request.top_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("DML retrieval_report failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RetrievalResponse(
        entries=report.get("entries", []),
        preamble=report.get("preamble", ""),
        tokens=int(report.get("tokens", 0)),
        latency_ms=int(report.get("latency_ms", 0)),
        error=report.get("error"),
    )


@app.post("/build_preamble", response_model=PreambleResponse)
def build_preamble(request: RetrievalRequest) -> PreambleResponse:
    try:
        preamble = adapter.build_preamble(request.prompt, top_k=request.top_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("DML build_preamble failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PreambleResponse(preamble=preamble)


@app.post("/cookbook/get", response_model=CookbookGetResponse)
def get_cookbook(request: CookbookGetRequest) -> CookbookGetResponse:
    start = time.perf_counter()
    items = adapter.store.items()
    candidates = [
        item
        for item in items
        if (item.meta or {}).get("kind") == "cookbook"
        and (item.meta or {}).get("scenario_key") == request.scenario_key
        and not (item.meta or {}).get("superseded", False)
    ]
    if not candidates:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return CookbookGetResponse(found=False, cookbook_text="", sources=[], latency_ms=latency_ms)
    latest = max(candidates, key=lambda item: item.timestamp)
    meta = latest.meta or {}
    sources = meta.get("cookbook_sources") or meta.get("sources") or [str(latest.id)]
    latency_ms = int((time.perf_counter() - start) * 1000)
    return CookbookGetResponse(
        found=True,
        cookbook_text=str(latest.text or ""),
        sources=list(sources),
        latency_ms=latency_ms,
    )


@app.post("/run_report/ingest", response_model=RunReportResponse)
def ingest_run_report(request: RunReportRequest) -> RunReportResponse:
    try:
        report_payload = {
            "scenario_key": request.scenario_key,
            "goal": request.goal,
            "run_id": request.run_id,
            "trace": request.trace,
            "final": request.final,
            "success": request.success,
            "meta": request.meta or {},
        }
        report_text = json.dumps(report_payload, ensure_ascii=False, sort_keys=True)
        report_meta = {
            "scenario_key": request.scenario_key,
            "run_id": request.run_id,
            "goal": request.goal,
            "success": request.success,
            "kind": "run_report",
        }
        if request.meta:
            report_meta.update(request.meta)
        report_item = adapter.ingest_memory(
            report_text,
            tenant_id="nemotron",
            client_id="demo",
            kind="run_report",
            meta=report_meta,
        )
        for item in adapter.store.items():
            if (item.meta or {}).get("kind") != "cookbook":
                continue
            if (item.meta or {}).get("scenario_key") != request.scenario_key:
                continue
            item.meta = dict(item.meta or {})
            item.meta["superseded"] = True
            item.meta["superseded_at"] = time.time()
        cookbook_text, summary_latency_ms = _summarize_cookbook(request)
        cookbook_meta = {
            "scenario_key": request.scenario_key,
            "run_id": request.run_id,
            "sources": [str(report_item.id)],
            "cookbook_sources": [str(report_item.id)],
            "summary_latency_ms": summary_latency_ms,
            "kind": "cookbook",
        }
        cookbook_item = adapter.ingest_memory(
            cookbook_text or "",
            tenant_id="nemotron",
            client_id="demo",
            kind="cookbook",
            meta=cookbook_meta,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("DML run report ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RunReportResponse(
        ok=True,
        ingested_id=str(report_item.id),
        summary_id=str(cookbook_item.id),
        summary_latency_ms=summary_latency_ms,
    )


@app.get("/debug/scenario/{scenario_key}")
def debug_scenario(scenario_key: str) -> Dict[str, Any]:
    items = adapter.store.items()
    run_reports = [
        item
        for item in items
        if (item.meta or {}).get("kind") == "run_report"
        and (item.meta or {}).get("scenario_key") == scenario_key
    ]
    run_reports_sorted = sorted(run_reports, key=lambda item: item.timestamp, reverse=True)
    cookbooks = [
        item
        for item in items
        if (item.meta or {}).get("kind") == "cookbook"
        and (item.meta or {}).get("scenario_key") == scenario_key
        and not (item.meta or {}).get("superseded", False)
    ]
    cookbook_latest = max(cookbooks, key=lambda item: item.timestamp) if cookbooks else None
    cookbook_meta = cookbook_latest.meta or {} if cookbook_latest else {}
    return {
        "run_report_count": len(run_reports),
        "latest_run_report_ids": [str(item.id) for item in run_reports_sorted[:5]],
        "cookbook_exists": {
            "id": str(cookbook_latest.id) if cookbook_latest else None,
            "timestamp": cookbook_latest.timestamp if cookbook_latest else None,
            "chars": len(cookbook_latest.text or "") if cookbook_latest else 0,
        },
        "last_summary_latency_ms": cookbook_meta.get("summary_latency_ms"),
    }

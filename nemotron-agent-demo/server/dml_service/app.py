from __future__ import annotations

import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from llm_client import create_chat_completion, get_vllm_model_id
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


def _summary_settings() -> Dict[str, Any]:
    return {
        "input_budget_tokens": int(os.getenv("DML_SUMMARY_INPUT_BUDGET_TOKENS", "12000")),
        "max_tokens": int(os.getenv("DML_SUMMARY_MAX_TOKENS", "512")),
        "chunk_max_tokens": int(os.getenv("DML_SUMMARY_CHUNK_MAX_TOKENS", "256")),
        "temperature": 0,
        "enable_thinking": False,
    }


def _build_cookbook_prompt(payload: RunReportRequest, trace_text: Optional[str] = None) -> str:
    trace_json = trace_text if trace_text is not None else json.dumps(payload.trace, ensure_ascii=False, sort_keys=True)
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


@lru_cache(maxsize=1)
def _get_tokenizer(model_name: str) -> Optional[Any]:
    if not model_name:
        return None
    try:
        local_only = os.getenv("DML_TOKENIZER_LOCAL_ONLY", "1") == "1"
        return AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tokenizer load failed for %s: %s", model_name, exc)
        return None


def _estimate_tokens(text: str, tokenizer: Optional[Any]) -> int:
    if not text:
        return 0
    if tokenizer is None:
        return max(1, len(text) // 4)
    return len(tokenizer.encode(text, add_special_tokens=False))


def _estimate_message_tokens(messages: Iterable[Dict[str, str]], tokenizer: Optional[Any]) -> int:
    return sum(_estimate_tokens(message.get("content", ""), tokenizer) for message in messages)


def _truncate_text_to_budget(text: str, tokenizer: Optional[Any], budget_tokens: int) -> str:
    if budget_tokens <= 0 or not text:
        return ""
    if _estimate_tokens(text, tokenizer) <= budget_tokens:
        return text
    if tokenizer is None:
        return text[: budget_tokens * 4]
    encoded = tokenizer.encode(text, add_special_tokens=False)
    truncated = encoded[:budget_tokens]
    return tokenizer.decode(truncated)


def _split_text_by_tokens(text: str, tokenizer: Optional[Any], budget_tokens: int) -> List[str]:
    if not text:
        return [""]
    if budget_tokens <= 0:
        return [text]
    if _estimate_tokens(text, tokenizer) <= budget_tokens:
        return [text]
    lines = text.splitlines()
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for line in lines:
        line_tokens = _estimate_tokens(f"{line}\n", tokenizer)
        if current and current_tokens + line_tokens > budget_tokens:
            chunks.append("\n".join(current))
            current = []
            current_tokens = 0
        if line_tokens > budget_tokens:
            chunks.append(_truncate_text_to_budget(line, tokenizer, budget_tokens))
        else:
            current.append(line)
            current_tokens += line_tokens
    if current:
        chunks.append("\n".join(current))
    return chunks


def _truncate_head_tail(text: str, tokenizer: Optional[Any], head_tokens: int, tail_tokens: int) -> str:
    if not text or (head_tokens <= 0 and tail_tokens <= 0):
        return ""
    if tokenizer is None:
        head_chars = max(head_tokens, 0) * 4
        tail_chars = max(tail_tokens, 0) * 4
        if head_chars + tail_chars >= len(text):
            return text
        if tail_chars <= 0:
            return text[:head_chars]
        return text[:head_chars] + text[-tail_chars:]
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if head_tokens + tail_tokens >= len(encoded):
        return text
    head_tokens = max(head_tokens, 0)
    tail_tokens = max(tail_tokens, 0)
    head = encoded[:head_tokens] if head_tokens > 0 else []
    tail = encoded[-tail_tokens:] if tail_tokens > 0 else []
    return tokenizer.decode(head + tail)


def _format_stage_chunk(stage_name: str, stage: Dict[str, Any]) -> str:
    parts = [f"STAGE: {stage_name.upper()}"]
    output = stage.get("output")
    if output:
        parts.append("OUTPUT:")
        parts.append(str(output))
    error = stage.get("error")
    if error:
        parts.append("ERROR:")
        parts.append(str(error))
    tokens = stage.get("tokens")
    if tokens is not None:
        parts.append(f"TOKENS: {tokens}")
    return "\n".join(parts)


def _build_stage_chunks(payload: RunReportRequest) -> List[str]:
    trace = payload.trace or {}
    stages = trace.get("stages", {}) if isinstance(trace, dict) else {}
    chunks: List[str] = []
    ordered = ["planner", "coder", "reviewer", "ops", "aggregator"]
    seen = set()
    for name in ordered:
        stage = stages.get(name)
        if stage:
            chunks.append(_format_stage_chunk(name, stage))
            seen.add(name)
    for name, stage in stages.items():
        if name in seen:
            continue
        chunks.append(_format_stage_chunk(name, stage))
    errors = trace.get("errors")
    if errors:
        chunks.append(f"TRACE ERRORS:\n{json.dumps(errors, ensure_ascii=False, sort_keys=True)}")
    return chunks


def _build_chunk_prompt(chunk_text: str) -> str:
    return "\n".join(
        [
            "Summarize this agent run chunk.",
            "Return bullet points of what worked and failed.",
            "Keep it brief and high-signal.",
            "CHUNK:",
            chunk_text,
        ]
    )


def _build_cookbook_prompt_from_chunks(payload: RunReportRequest, chunk_summaries: List[str]) -> str:
    meta_json = json.dumps(payload.meta or {}, ensure_ascii=False, sort_keys=True)
    summaries = "\n\n".join(chunk_summaries)
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
            "TRACE SUMMARIES:",
            summaries,
        ]
    )


def _call_summarizer(
    messages: List[Dict[str, str]],
    max_tokens: int,
    tokenizer: Optional[Any],
    summary_meta: Dict[str, Any],
    label: str,
) -> str:
    start = time.perf_counter()
    summary_meta.setdefault("per_call_input_tokens_est", []).append(_estimate_message_tokens(messages, tokenizer))
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    response = create_chat_completion(
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    latency_ms = int((time.perf_counter() - start) * 1000)
    summary_meta.setdefault("per_call_latency_ms", []).append({label: latency_ms})
    message = response.get("choices", [{}])[0].get("message", {})
    summary = (message.get("content") or "").strip()
    if not summary:
        summary = (message.get("reasoning_content") or "").strip()
    return summary


def _summarize_cookbook(payload: RunReportRequest) -> tuple[str, int, Dict[str, Any]]:
    start = time.perf_counter()
    settings = _summary_settings()
    summary_meta: Dict[str, Any] = {
        "input_budget_tokens": settings["input_budget_tokens"],
        "output_tokens": settings["max_tokens"],
        "chunk_count": 0,
    }
    try:
        model = get_vllm_model_id()
        tokenizer = _get_tokenizer(model)
        trace_json = json.dumps(payload.trace, ensure_ascii=False, sort_keys=True)
        raw_tokens_est = _estimate_tokens(trace_json, tokenizer)
        summary_meta["raw_trace_tokens_est"] = raw_tokens_est
        system_message = {"role": "system", "content": "You are a precise summariser. Output only the cookbook summary."}
        truncation_applied = False
        truncated_trace = trace_json
        head_tokens = 1500
        overhead_margin = 500
        tail_tokens = max(settings["input_budget_tokens"] - head_tokens - overhead_margin, 0)
        if raw_tokens_est > settings["input_budget_tokens"]:
            truncation_applied = True
            truncated_trace = _truncate_head_tail(trace_json, tokenizer, head_tokens, tail_tokens)
        truncated_tokens_est = _estimate_tokens(truncated_trace, tokenizer)
        summary_meta["truncated_trace_tokens_est"] = truncated_tokens_est
        summary_meta["trace_truncated"] = truncation_applied
        prompt = _build_cookbook_prompt(payload, trace_text=truncated_trace)
        messages = [system_message, {"role": "user", "content": prompt}]
        prompt_tokens_est = _estimate_message_tokens(messages, tokenizer)
        if prompt_tokens_est > settings["input_budget_tokens"]:
            messages[1]["content"] = _truncate_text_to_budget(
                messages[1]["content"],
                tokenizer,
                settings["input_budget_tokens"] - _estimate_tokens(system_message["content"], tokenizer),
            )
            prompt_tokens_est = _estimate_message_tokens(messages, tokenizer)
            summary_meta["hard_truncated"] = True
        summary_meta["prompt_tokens_est"] = prompt_tokens_est
        summary = _call_summarizer(
            messages,
            settings["max_tokens"],
            tokenizer,
            summary_meta,
            "cookbook_final",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Cookbook summarization failed, falling back: %s", exc)
        summary_meta["summary_source"] = "fallback"
        summary = _fallback_cookbook(payload).strip()
    if not summary:
        global _EMPTY_SUMMARY_LOGGED
        if not _EMPTY_SUMMARY_LOGGED:
            _EMPTY_SUMMARY_LOGGED = True
            logger.warning("Empty cookbook summary response; using fallback")
        summary_meta["summary_source"] = "fallback"
        summary = _fallback_cookbook(payload).strip()
    summary_meta.setdefault("summary_source", "llm")
    latency_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "DML cookbook summary meta: source=%s raw_trace_tokens_est=%s truncated_trace_tokens_est=%s "
        "trace_truncated=%s chunk_count=%s per_call_input_tokens_est=%s prompt_tokens_est=%s",
        summary_meta.get("summary_source"),
        summary_meta.get("raw_trace_tokens_est"),
        summary_meta.get("truncated_trace_tokens_est"),
        summary_meta.get("trace_truncated"),
        summary_meta.get("chunk_count"),
        summary_meta.get("per_call_input_tokens_est"),
        summary_meta.get("prompt_tokens_est"),
    )
    return summary.strip(), latency_ms, summary_meta


def _fallback_cookbook(payload: RunReportRequest) -> str:
    status = "success" if payload.success else "failure"
    meta_json = json.dumps(payload.meta or {}, ensure_ascii=False, sort_keys=True)
    trace = payload.trace if isinstance(payload.trace, dict) else {}
    errors = trace.get("errors") or []
    error_lines = []
    for error in errors:
        if isinstance(error, dict):
            stage = error.get("stage")
            message = error.get("error") or error.get("message") or ""
            if stage or message:
                error_lines.append(f"- {stage or 'stage'}: {message}".strip())
        else:
            error_lines.append(f"- {error}")
    if not error_lines:
        error_lines = ["- No error excerpts available."]
    artifacts = payload.meta or {}
    artifact_lines = []
    for key in ("artifacts", "links", "urls", "files"):
        value = artifacts.get(key) if isinstance(artifacts, dict) else None
        if value:
            artifact_lines.append(f"- {key}: {value}")
    if not artifact_lines:
        artifact_lines = [f"- RUN_ID: {payload.run_id}"]
    return "\n".join(
        [
            f"TITLE: {payload.scenario_key}",
            "WHAT WORKED:",
            f"- Run status: {status}",
            "WHAT FAILED:",
            *error_lines,
            "DECISION HEURISTICS:",
            f"- Goal: {payload.goal}",
            "PITFALLS:",
            "- Review run trace for missing steps or tool errors.",
            "NEXT TIME TRY:",
            "- Re-run summarisation or provide more detailed run metadata.",
            "KEY ARTIFACTS / LINKS (if present):",
            *artifact_lines,
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
        cookbook_text, summary_latency_ms, summary_meta = _summarize_cookbook(request)
        cookbook_meta = {
            "scenario_key": request.scenario_key,
            "run_id": request.run_id,
            "sources": [str(report_item.id)],
            "cookbook_sources": [str(report_item.id)],
            "summary_latency_ms": summary_latency_ms,
            "summary_meta": summary_meta,
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

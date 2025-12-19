from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
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

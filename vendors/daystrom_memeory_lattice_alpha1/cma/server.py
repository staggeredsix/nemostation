"""FastAPI server exposing CMA endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency
    from fastapi import FastAPI
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore

from .adapter import CMAAdapter


if FastAPI is not None:  # pragma: no cover - exercised in integration
    app = FastAPI()

    class IngestRequest(BaseModel):
        text: str
        meta: Optional[dict] = None

    class QueryRequest(BaseModel):
        prompt: str
        top_k: Optional[int] = None
        budget: Optional[int] = None

    class ReinforceRequest(BaseModel):
        text: str

    adapter = CMAAdapter(storage_path=Path("cma_store.json"))

    @app.post("/ingest")
    def ingest(payload: IngestRequest) -> dict:
        adapter.ingest(payload.text, payload.meta)
        return {"status": "ok"}

    @app.post("/augment")
    def augment(payload: QueryRequest) -> dict:
        preamble = adapter.augment_prompt(payload.prompt, payload.top_k, payload.budget)
        return {"preamble": preamble}

    @app.post("/reinforce")
    def reinforce(payload: ReinforceRequest) -> dict:
        adapter.reinforce(payload.text)
        return {"status": "ok"}

    @app.get("/stats")
    def stats() -> dict:
        return adapter.stats()

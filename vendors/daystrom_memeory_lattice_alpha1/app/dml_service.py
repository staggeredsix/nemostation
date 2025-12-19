"""FastAPI service exposing the Daystrom Memory Lattice as a multi-tenant API."""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from daystrom_dml.dml_adapter import DMLAdapter

LOGGER = logging.getLogger(__name__)

app = FastAPI(title="DML Memory Service", version="0.1")
adapter = DMLAdapter(start_aging_loop=False)


class Metadata(BaseModel):
    tenant_id: str
    client_id: Optional[str] = None
    session_id: Optional[str] = None
    instance_id: Optional[str] = None


class IngestRequest(Metadata):
    kind: str = Field("memory", description="memory|scratch|workflow|summary")
    text: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class RetrieveRequest(Metadata):
    query: str
    top_k: int = 6
    scope: str = Field("personal", description="personal|session|instance|tenant|global")
    include_workflows: bool = False


class InstanceCreateRequest(Metadata):
    task_description: Optional[str] = None


class InstanceIngestRequest(Metadata):
    text: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class InstanceContextRequest(Metadata):
    query: str
    top_k: int = 5


class InstancePromoteRequest(Metadata):
    promote_as: str = Field("workflow", description="workflow|summary")
    task_description: str
    outcome: str


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/memory/ingest")
def ingest_memory(payload: IngestRequest) -> Dict[str, Any]:
    item = adapter.ingest_memory(
        payload.text,
        tenant_id=payload.tenant_id,
        client_id=payload.client_id,
        session_id=payload.session_id,
        instance_id=payload.instance_id,
        kind=payload.kind,
        meta=payload.meta,
    )
    return {"id": str(item.id), "summary": item.cached_summary()}


@app.post("/v1/memory/retrieve")
def retrieve_memory(payload: RetrieveRequest) -> Dict[str, Any]:
    client_id = payload.client_id
    session_id = payload.session_id
    instance_id = payload.instance_id
    kinds: Optional[List[str]] = None
    if payload.include_workflows:
        kinds = ["workflow", "summary", "memory"]
    scope = payload.scope.lower()
    if scope == "personal":
        pass
    elif scope == "session":
        if session_id is None:
            raise HTTPException(status_code=400, detail="session_id required for session scope")
    elif scope == "instance":
        if instance_id is None:
            raise HTTPException(status_code=400, detail="instance_id required for instance scope")
    elif scope in {"tenant", "global"}:
        client_id = None
        session_id = None
        instance_id = None
    else:
        raise HTTPException(status_code=400, detail="invalid scope")
    return adapter.retrieve_context(
        payload.query,
        tenant_id=payload.tenant_id,
        client_id=client_id,
        session_id=session_id,
        instance_id=instance_id,
        kinds=kinds,
        top_k=payload.top_k,
    )


@app.post("/v1/instance/create")
def create_instance(payload: InstanceCreateRequest) -> Dict[str, str]:
    instance_id = payload.instance_id or str(uuid.uuid4())
    if payload.task_description:
        adapter.ingest_memory(
            payload.task_description,
            tenant_id=payload.tenant_id,
            client_id=payload.client_id,
            session_id=payload.session_id,
            instance_id=instance_id,
            kind="scratch",
            meta={"role": "task_description"},
        )
    return {"instance_id": instance_id}


@app.post("/v1/instance/ingest")
def ingest_instance(payload: InstanceIngestRequest) -> Dict[str, Any]:
    if not payload.instance_id:
        raise HTTPException(status_code=400, detail="instance_id is required")
    item = adapter.ingest_memory(
        payload.text,
        tenant_id=payload.tenant_id,
        client_id=payload.client_id,
        session_id=payload.session_id,
        instance_id=payload.instance_id,
        kind="scratch",
        meta=payload.meta,
    )
    return {"id": str(item.id), "summary": item.cached_summary()}


@app.post("/v1/instance/context")
def instance_context(payload: InstanceContextRequest) -> Dict[str, Any]:
    if not payload.instance_id:
        raise HTTPException(status_code=400, detail="instance_id is required")
    return adapter.retrieve_context(
        payload.query,
        tenant_id=payload.tenant_id,
        client_id=payload.client_id,
        instance_id=payload.instance_id,
        kinds=["scratch"],
        top_k=payload.top_k,
    )


@app.post("/v1/instance/promote")
def promote_instance(payload: InstancePromoteRequest) -> Dict[str, Any]:
    if not payload.instance_id:
        raise HTTPException(status_code=400, detail="instance_id is required")
    scratch = adapter.collect_instance_scratch(
        tenant_id=payload.tenant_id,
        client_id=payload.client_id,
        session_id=payload.session_id,
        instance_id=payload.instance_id,
    )
    if not scratch:
        raise HTTPException(status_code=404, detail="no scratch memories found")
    combined = "\n".join(item.text for item in scratch if item.text)
    summary_text = adapter.summarizer.summarize(combined, max_len=512) or combined[:512]
    created_ids: List[str] = []
    summary_item = adapter.ingest_memory(
        summary_text,
        tenant_id=payload.tenant_id,
        client_id=payload.client_id,
        session_id=payload.session_id,
        instance_id=payload.instance_id,
        kind="summary",
        meta={
            "source": "promotion",
            "task_description": payload.task_description,
            "outcome": payload.outcome,
        },
    )
    created_ids.append(str(summary_item.id))
    if payload.promote_as == "workflow":
        workflow_text = "Task: " + payload.task_description + "\nOutcome: " + payload.outcome + "\n" + combined
        workflow_item = adapter.ingest_memory(
            workflow_text,
            tenant_id=payload.tenant_id,
            client_id=payload.client_id,
            session_id=payload.session_id,
            instance_id=payload.instance_id,
            kind="workflow",
            meta={"visibility": "tenant", "task_description": payload.task_description},
        )
        created_ids.append(str(workflow_item.id))
    return {"ids": created_ids, "summary": summary_item.cached_summary()}


def main() -> None:
    uvicorn.run("app.dml_service:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()

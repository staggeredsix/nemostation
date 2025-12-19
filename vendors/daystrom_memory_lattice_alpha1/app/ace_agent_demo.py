"""Prototype ACE-like agent showing instance memory + workflow reuse."""
from __future__ import annotations

import os
import random
import sys
from typing import Any, Dict, List

import requests

SERVICE_URL = os.getenv("DML_SERVICE_URL", "http://localhost:8000")
LLM_BASE = os.getenv("OPENAI_API_BASE")
LLM_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")


def call_llm(system_prompt: str, user_prompt: str) -> str:
    if not LLM_BASE or not LLM_KEY:
        return f"[stubbed agent response]\n{user_prompt[:160]}"
    headers = {"Authorization": f"Bearer {LLM_KEY}"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    resp = requests.post(f"{LLM_BASE}/v1/chat/completions", json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def create_instance(tenant: str, client: str, task: str) -> str:
    resp = requests.post(
        f"{SERVICE_URL}/v1/instance/create",
        json={"tenant_id": tenant, "client_id": client, "task_description": task},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["instance_id"]


def ingest_step(tenant: str, client: str, instance: str, text: str, role: str) -> None:
    requests.post(
        f"{SERVICE_URL}/v1/instance/ingest",
        json={
            "tenant_id": tenant,
            "client_id": client,
            "instance_id": instance,
            "text": text,
            "meta": {"role": role},
        },
        timeout=15,
    ).raise_for_status()


def instance_context(tenant: str, client: str, instance: str, query: str) -> str:
    resp = requests.post(
        f"{SERVICE_URL}/v1/instance/context",
        json={
            "tenant_id": tenant,
            "client_id": client,
            "instance_id": instance,
            "query": query,
            "top_k": 5,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("raw_context", "")


def promote_instance(tenant: str, client: str, instance: str, task: str, outcome: str) -> None:
    requests.post(
        f"{SERVICE_URL}/v1/instance/promote",
        json={
            "tenant_id": tenant,
            "client_id": client,
            "instance_id": instance,
            "promote_as": "workflow",
            "task_description": task,
            "outcome": outcome,
        },
        timeout=30,
    ).raise_for_status()


def retrieve_workflows(tenant: str, query: str) -> str:
    resp = requests.post(
        f"{SERVICE_URL}/v1/memory/retrieve",
        json={
            "tenant_id": tenant,
            "query": query,
            "scope": "tenant",
            "include_workflows": True,
            "top_k": 3,
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("raw_context", "")


def run_task(task: str, *, tenant: str, client: str) -> None:
    instance_id = create_instance(tenant, client, task)
    steps = [
        "Collect flight options from API",
        "Filter by price delta and airline preferences",
        "Monitor price changes for 24 hours",
        "Alert user when optimal price found",
    ]
    for idx, step in enumerate(steps, start=1):
        context = instance_context(tenant, client, instance_id, "Summarize progress so far")
        system_prompt = "You are ACE-like agent reasoning through steps."
        user_prompt = f"Task: {task}\nContext:\n{context}\nNext step {idx}: {step}"
        thought = call_llm(system_prompt, user_prompt)
        ingest_step(tenant, client, instance_id, f"Step {idx}: {thought}", role="thought")
    promote_instance(tenant, client, instance_id, task, "Completed monitoring run")


def main() -> None:
    tenant = "ace_demo"
    client = os.getenv("ACE_USER_ID", f"user_{random.randint(100,999)}")
    first_task = "track flight prices and alert when optimal to buy"
    run_task(first_task, tenant=tenant, client=client)
    workflow_hint = retrieve_workflows(tenant, first_task)
    instance_id = create_instance(tenant, client, first_task)
    ingest_step(
        tenant,
        client,
        instance_id,
        f"Loaded workflow hint from prior runs:\n{workflow_hint}",
        role="hint",
    )
    context = instance_context(tenant, client, instance_id, "What should we do next?")
    user_prompt = f"Task: {first_task}\nWorkflow hints:\n{workflow_hint}\nContext:\n{context}\nPlan the first actionable step."
    response = call_llm("Reuse workflows when possible.", user_prompt)
    ingest_step(tenant, client, instance_id, response, role="plan")
    promote_instance(tenant, client, instance_id, first_task, "Second run seeded with workflow hints")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - demo utility
        sys.stderr.write(f"ace_agent_demo failed: {exc}\n")
        sys.exit(1)

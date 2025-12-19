"""Prototype NIM/OpenAI-style client that uses DML as its memory layer."""
from __future__ import annotations

import os
import random
import sys
from typing import Any, Dict

import requests

SERVICE_URL = os.getenv("DML_SERVICE_URL", "http://localhost:8000")
LLM_BASE = os.getenv("OPENAI_API_BASE")
LLM_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")


def call_llm(prompt: str) -> str:
    if not LLM_BASE or not LLM_KEY:
        return f"[stubbed LLM response]\n{prompt[:200]}"
    headers = {"Authorization": f"Bearer {LLM_KEY}"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    }
    resp = requests.post(f"{LLM_BASE}/v1/chat/completions", json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def retrieve_context(user_id: str, query: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{SERVICE_URL}/v1/memory/retrieve",
        json={
            "tenant_id": "nim_demo",
            "client_id": user_id,
            "query": query,
            "top_k": 6,
            "scope": "personal",
            "include_workflows": True,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def log_memory(user_id: str, text: str, role: str) -> None:
    requests.post(
        f"{SERVICE_URL}/v1/memory/ingest",
        json={
            "tenant_id": "nim_demo",
            "client_id": user_id,
            "kind": "memory",
            "text": text,
            "meta": {"role": role},
        },
        timeout=30,
    ).raise_for_status()


def main() -> None:
    user_id = os.getenv("NIM_USER_ID", f"user_{random.randint(100,999)}")
    questions = [
        "What is the status of our launch readiness?",
        "Why did we adjust the risk threshold yesterday?",
    ]
    for question in questions:
        context = retrieve_context(user_id, question)
        prompt_parts = [
            "You are answering on behalf of the NIM demo user.",
            "Relevant DML context:",
            context.get("raw_context", "(no context yet)") or "(no context yet)",
            "User question:",
            question,
        ]
        prompt = "\n\n".join(prompt_parts)
        answer = call_llm(prompt)
        print("---")
        print("Q:", question)
        print("Context:\n", context.get("raw_context"))
        print("A:", answer)
        log_memory(user_id, f"Q: {question}\nA: {answer}", role="interaction")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - demo utility
        sys.stderr.write(f"nim_demo failed: {exc}\n")
        sys.exit(1)

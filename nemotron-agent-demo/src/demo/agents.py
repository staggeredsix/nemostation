from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from openai import OpenAI

from .prompts import get_active_prompt, get_context_payload

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
API_KEY = "none"
MODEL_ID = os.getenv("MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")


@dataclass
class AgentResult:
    name: str
    output: str


def build_messages(role_prompt: str, goal: str, scenario: str | None = None, extra_context: str = "") -> List[Dict[str, str]]:
    user_content = f"Goal: {goal}\nScenario: {scenario or 'general'}"
    if extra_context:
        user_content += f"\nContext:\n{extra_context}"
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": get_active_prompt("system")},
        {
            "role": "system",
            "content": f"GB300 context:\n{get_context_payload()}",
        },
        {"role": "system", "content": role_prompt},
        {"role": "user", "content": user_content},
    ]
    return messages


def call_agent(
    role: str,
    goal: str,
    scenario: str | None = None,
    max_tokens: int = 512,
    extra_context: str = "",
) -> AgentResult:
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=API_KEY)
    prompt = get_active_prompt(role)
    messages = build_messages(prompt, goal, scenario, extra_context)
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens,
        stream=False,
    )
    content = response.choices[0].message.content or ""
    return AgentResult(name=role, output=content.strip())

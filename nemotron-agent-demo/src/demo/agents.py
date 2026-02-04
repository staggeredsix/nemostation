from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from .prompts import get_active_prompt, get_context_payload
from llm_client import create_chat_completion


@dataclass
class AgentResult:
    name: str
    output: str
    tokens: int = 0


def _extract_completion_tokens(response: dict) -> int:
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    if not isinstance(usage, dict):
        return 0
    for key in ("completion_tokens", "output_tokens", "generated_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return 0


def build_messages(
    role_prompt: str,
    goal: str,
    scenario: str | None = None,
    extra_context: str = "",
    system_messages: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    user_content = f"Goal: {goal}\nScenario: {scenario or 'general'}"
    if extra_context:
        user_content += f"\nContext:\n{extra_context}"
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": get_active_prompt("system")},
        {
            "role": "system",
            "content": f"GB300 context:\n{get_context_payload()}",
        },
    ]
    if system_messages:
        for message in system_messages:
            if message:
                messages.append({"role": "system", "content": message})
    messages.append({"role": "system", "content": role_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def call_agent(
    role: str,
    goal: str,
    scenario: str | None = None,
    max_tokens: int = 512,
    extra_context: str = "",
    system_messages: Optional[List[str]] = None,
) -> AgentResult:
    prompt = get_active_prompt(role)
    messages = build_messages(prompt, goal, scenario, extra_context, system_messages=system_messages)
    response = create_chat_completion(
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens,
        role=role,
    )
    message = response.get("choices", [{}])[0].get("message", {}) or {}
    content = message.get("content", "") or ""
    if not content:
        tool_calls = message.get("tool_calls")
        if tool_calls:
            content = json.dumps({"tool_calls": tool_calls}, indent=2)
        elif message.get("function_call"):
            content = json.dumps({"function_call": message.get("function_call")}, indent=2)
        else:
            finish_reason = response.get("choices", [{}])[0].get("finish_reason")
            if finish_reason:
                content = f"No content returned (finish_reason={finish_reason})."
    tokens = _extract_completion_tokens(response)
    return AgentResult(name=role, output=content.strip(), tokens=tokens)

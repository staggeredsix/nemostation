from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Iterable, Optional

import requests

logger = logging.getLogger("vllm-client")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://host.docker.internal:8000/v1").rstrip("/")
VLLM_MODEL_ID = os.getenv("VLLM_MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4")
VLLM_TIMEOUT_S = int(os.getenv("VLLM_TIMEOUT_S", "120"))

_startup_check_started = False
_resolved_model_id: Optional[str] = None
_model_id_lock = threading.Lock()


def get_vllm_base_url() -> str:
    return VLLM_BASE_URL


def get_vllm_model_id() -> str:
    return _resolve_vllm_model_id()


def get_vllm_timeout_s() -> int:
    return VLLM_TIMEOUT_S


def _extract_model_ids(models_payload: dict[str, Any]) -> list[str]:
    data = models_payload.get("data", [])
    return [item.get("id", "") for item in data if isinstance(item, dict) and item.get("id")]


def _resolve_vllm_model_id(timeout_s: int = 5, *, force_refresh: bool = False) -> str:
    global _resolved_model_id
    if _resolved_model_id is not None and not force_refresh:
        return _resolved_model_id
    with _model_id_lock:
        if _resolved_model_id is not None and not force_refresh:
            return _resolved_model_id
        payload = fetch_models(timeout_s=timeout_s)
        model_ids = _extract_model_ids(payload) if payload else []
        selected_model = VLLM_MODEL_ID
        if model_ids:
            if selected_model in model_ids:
                _resolved_model_id = selected_model
            else:
                _resolved_model_id = model_ids[0]
                logger.warning(
                    "vLLM model selected: %s (not found in /models; using=%s; available=%s)",
                    selected_model,
                    _resolved_model_id,
                    ", ".join(model_ids),
                )
        else:
            _resolved_model_id = selected_model
        return _resolved_model_id


def _display_name_for_model_id(model_id: str) -> str:
    if "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4" in model_id:
        return "Nemotron 3 Nano 30B NVFP4"
    if "Mistral-Large-3-675B-Instruct-2512-NVFP4" in model_id:
        return "Mistral 3 Large NVFP4"
    if model_id == "kimi-k2-nvfp4":
        return "Kimi K2 NVFP4"
    return model_id


def fetch_models(timeout_s: int = 5) -> Optional[dict[str, Any]]:
    url = f"{VLLM_BASE_URL}/models"
    try:
        response = requests.get(url, timeout=timeout_s)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("vLLM models check failed for %s: %s", url, exc)
        return None
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        logger.error("vLLM models response was not JSON: %s", exc)
        return None


def log_model_availability() -> None:
    payload = fetch_models()
    if payload is None:
        logger.error("vLLM unavailable at %s", VLLM_BASE_URL)
        return
    model_ids = _extract_model_ids(payload)
    resolved_model = _resolve_vllm_model_id(force_refresh=True)
    if resolved_model in model_ids:
        logger.info("vLLM model selected: %s (found in /models)", resolved_model)
    else:
        logger.warning(
            "vLLM model selected: %s (not found in /models; available=%s)",
            resolved_model,
            ", ".join(model_ids) if model_ids else "none",
        )


def start_vllm_startup_check() -> None:
    global _startup_check_started
    if _startup_check_started:
        return
    _startup_check_started = True
    thread = threading.Thread(target=log_model_availability, name="vllm-startup-check", daemon=True)
    thread.start()


def ping_vllm(timeout_s: int = 3) -> bool:
    url = f"{VLLM_BASE_URL}/models"
    try:
        response = requests.get(url, timeout=timeout_s)
    except requests.RequestException:
        return False
    return response.status_code == 200


def fetch_available_model_names(timeout_s: int = 5) -> list[str]:
    payload = fetch_models(timeout_s=timeout_s)
    if payload is None:
        return []
    model_ids = _extract_model_ids(payload)
    return [_display_name_for_model_id(model_id) for model_id in model_ids]


def build_chat_payload(
    messages: Iterable[dict[str, Any]],
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    model_id: Optional[str] = None,
    chat_template_kwargs: Optional[dict[str, Any]] = None,
    extra_body: Optional[dict[str, Any]] = None,
    extra_payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_id or get_vllm_model_id(),
        "messages": list(messages),
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    if extra_body:
        payload["extra_body"] = extra_body
    if extra_payload:
        payload.update(extra_payload)
    return payload


def post_chat_completion(payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{VLLM_BASE_URL}/chat/completions"
    response = requests.post(url, json=payload, timeout=VLLM_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def create_chat_completion(
    messages: Iterable[dict[str, Any]],
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    chat_template_kwargs: Optional[dict[str, Any]] = None,
    extra_body: Optional[dict[str, Any]] = None,
    extra_payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload = build_chat_payload(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        chat_template_kwargs=chat_template_kwargs,
        extra_body=extra_body,
        extra_payload=extra_payload,
    )
    return post_chat_completion(payload)


start_vllm_startup_check()

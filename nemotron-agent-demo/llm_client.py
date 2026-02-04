from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Iterable, Optional, Tuple

import requests

logger = logging.getLogger("vllm-client")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://host.docker.internal:8000/v1").rstrip("/")
VLLM_MODEL_ID = os.getenv("VLLM_MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4")
VLLM_TIMEOUT_S = int(os.getenv("VLLM_TIMEOUT_S", "120"))

_startup_check_started = False
_resolved_model_id: Optional[str] = None
_resolved_model_ids: dict[Tuple[str, str], str] = {}
_model_id_lock = threading.Lock()


def _normalize_role(role: Optional[str]) -> Optional[str]:
    if not role:
        return None
    role_clean = str(role).strip().upper()
    return role_clean if role_clean else None


def _get_role_env(prefix: str, role: Optional[str]) -> Optional[str]:
    role_key = _normalize_role(role)
    if not role_key:
        return None
    return os.getenv(f"ROLE_{prefix}_{role_key}")


def get_vllm_base_url(role: Optional[str] = None) -> str:
    role_url = _get_role_env("BASE_URL", role)
    if role_url:
        return role_url.rstrip("/")
    return VLLM_BASE_URL


def get_vllm_model_id(role: Optional[str] = None) -> str:
    return _resolve_vllm_model_id(role=role)


def get_vllm_timeout_s() -> int:
    return VLLM_TIMEOUT_S


def _extract_model_ids(models_payload: dict[str, Any]) -> list[str]:
    data = models_payload.get("data", [])
    return [item.get("id", "") for item in data if isinstance(item, dict) and item.get("id")]


def _resolve_vllm_model_id(
    role: Optional[str] = None, timeout_s: int = 5, *, force_refresh: bool = False
) -> str:
    global _resolved_model_id
    requested_model = _get_role_env("MODEL_ID", role) or VLLM_MODEL_ID
    base_url = get_vllm_base_url(role=role)
    cache_key = (base_url, requested_model)
    if not force_refresh:
        cached = _resolved_model_ids.get(cache_key)
        if cached:
            return cached
        if role is None and _resolved_model_id is not None:
            return _resolved_model_id
    with _model_id_lock:
        if not force_refresh:
            cached = _resolved_model_ids.get(cache_key)
            if cached:
                return cached
            if role is None and _resolved_model_id is not None:
                return _resolved_model_id
        payload = fetch_models(timeout_s=timeout_s, base_url=base_url)
        model_ids = _extract_model_ids(payload) if payload else []
        selected_model = requested_model
        if model_ids:
            if selected_model in model_ids:
                _resolved_model_ids[cache_key] = selected_model
                if role is None:
                    _resolved_model_id = selected_model
            else:
                _resolved_model_ids[cache_key] = model_ids[0]
                if role is None:
                    _resolved_model_id = model_ids[0]
                logger.warning(
                    "vLLM model selected: %s (not found in /models; using=%s; available=%s)",
                    selected_model,
                    _resolved_model_ids[cache_key],
                    ", ".join(model_ids),
                )
        else:
            _resolved_model_ids[cache_key] = selected_model
            if role is None:
                _resolved_model_id = selected_model
        return _resolved_model_ids[cache_key]


def _display_name_for_model_id(model_id: str) -> str:
    if "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4" in model_id:
        return "Nemotron 3 Nano 30B NVFP4"
    if "Mistral-Large-3-675B-Instruct-2512-NVFP4" in model_id:
        return "Mistral 3 Large NVFP4"
    if model_id == "kimi-k2-nvfp4":
        return "Kimi K2 NVFP4"
    return model_id


def fetch_models(timeout_s: int = 5, base_url: Optional[str] = None) -> Optional[dict[str, Any]]:
    url = f"{(base_url or VLLM_BASE_URL).rstrip('/')}/models"
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


def ping_vllm(timeout_s: int = 3, base_url: Optional[str] = None) -> bool:
    url = f"{(base_url or VLLM_BASE_URL).rstrip('/')}/models"
    try:
        response = requests.get(url, timeout=timeout_s)
    except requests.RequestException:
        return False
    return response.status_code == 200


def fetch_available_model_names(timeout_s: int = 5, base_url: Optional[str] = None) -> list[str]:
    payload = fetch_models(timeout_s=timeout_s, base_url=base_url)
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


def post_chat_completion(payload: dict[str, Any], *, base_url: Optional[str] = None) -> dict[str, Any]:
    url = f"{(base_url or VLLM_BASE_URL).rstrip('/')}/chat/completions"
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
    role: Optional[str] = None,
) -> dict[str, Any]:
    base_url = get_vllm_base_url(role=role)
    model_id = get_vllm_model_id(role=role)
    payload = build_chat_payload(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        chat_template_kwargs=chat_template_kwargs,
        extra_body=extra_body,
        extra_payload=extra_payload,
        model_id=model_id,
    )
    return post_chat_completion(payload, base_url=base_url)


start_vllm_startup_check()

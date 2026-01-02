from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import requests
from openai import OpenAI

DEFAULT_VLLM_BASE_URL = "http://host.docker.internal:8000/v1"
DEFAULT_MODEL_ID = "deepseek/DeepSeek-V3.1-NVFP4/"
DEFAULT_TIMEOUT_S = 120

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    model_id: str
    available_models: List[str]
    timeout_s: float


def get_vllm_base_url() -> str:
    base_url = os.getenv("VLLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or DEFAULT_VLLM_BASE_URL
    return base_url.rstrip("/")


def get_request_timeout_s() -> float:
    raw_value = os.getenv("VLLM_REQUEST_TIMEOUT_S")
    if raw_value is None or raw_value == "":
        return float(DEFAULT_TIMEOUT_S)
    try:
        return float(raw_value)
    except ValueError:
        raise ValueError(f"Invalid VLLM_REQUEST_TIMEOUT_S value: {raw_value}") from None


def create_openai_client(base_url: Optional[str] = None, timeout_s: Optional[float] = None) -> OpenAI:
    resolved_base_url = (base_url or get_vllm_base_url()).rstrip("/")
    resolved_timeout = timeout_s if timeout_s is not None else get_request_timeout_s()
    api_key = os.getenv("OPENAI_API_KEY", "none")
    return OpenAI(base_url=resolved_base_url, api_key=api_key, timeout=resolved_timeout)


def fetch_model_ids(base_url: Optional[str] = None, timeout_s: Optional[float] = None) -> List[str]:
    resolved_base_url = (base_url or get_vllm_base_url()).rstrip("/")
    resolved_timeout = timeout_s if timeout_s is not None else get_request_timeout_s()
    url = f"{resolved_base_url}/models"
    response = requests.get(url, timeout=resolved_timeout)
    if response.status_code != 200:
        raise RuntimeError(f"vLLM /models returned HTTP {response.status_code} from {url}")
    payload = response.json()
    data = payload.get("data", []) if isinstance(payload, dict) else []
    models = [item.get("id") for item in data if isinstance(item, dict) and item.get("id")]
    return [model for model in models if model]


def resolve_model_id(
    requested_model_id: Optional[str] = None,
    available_models: Optional[List[str]] = None,
) -> str:
    env_model_id = os.getenv("VLLM_MODEL_ID")
    fallback = env_model_id or DEFAULT_MODEL_ID
    candidate = requested_model_id or fallback
    if not available_models:
        return candidate
    if requested_model_id:
        if requested_model_id not in available_models:
            raise RuntimeError(
                "Requested model ID not found in /models response: "
                f"{requested_model_id}. Available: {', '.join(available_models)}"
            )
        return requested_model_id
    if env_model_id and env_model_id in available_models:
        return env_model_id
    if env_model_id and env_model_id not in available_models:
        logger.warning(
            "VLLM_MODEL_ID '%s' not found in /models response. Falling back to %s.",
            env_model_id,
            available_models[0],
        )
    return available_models[0]


def ensure_vllm_ready(requested_model_id: Optional[str] = None) -> ModelConfig:
    base_url = get_vllm_base_url()
    timeout_s = get_request_timeout_s()
    try:
        available_models = fetch_model_ids(base_url=base_url, timeout_s=timeout_s)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Unable to reach vLLM at {base_url}/models. "
            "Check VLLM_BASE_URL and ensure the server is running."
        ) from exc
    model_id = resolve_model_id(requested_model_id, available_models)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logger.info("Using vLLM model '%s' at %s", model_id, base_url)
    return ModelConfig(
        base_url=base_url,
        model_id=model_id,
        available_models=available_models,
        timeout_s=timeout_s,
    )

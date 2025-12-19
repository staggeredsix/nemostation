import importlib
import importlib.util
import inspect
import logging
import os
import sys
import threading
import time
import uuid
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_ID = os.getenv("MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME") or MODEL_ID
HF_TOKEN = os.getenv("HF_TOKEN") or None

_tokenizer = None
_model = None
_model_lock = threading.Lock()
_cache_warmed = False
_nemotron_cache_class = None


def _check_mamba_ssm() -> None:
    try:
        importlib.import_module("mamba_ssm")
        logging.info("mamba_ssm import check: OK")
    except Exception as exc:
        logging.error(
            "mamba_ssm import check failed. Install mamba-ssm and causal-conv1d. Error: %s",
            exc,
        )
        raise


def _warm_cache() -> None:
    global _cache_warmed
    if _cache_warmed:
        return
    logging.info("Warming HF cache…")
    snapshot_download(
        MODEL_ID,
        token=HF_TOKEN,
        allow_patterns=[
            "*.safetensors",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "*.model",
            "*.txt",
            "*.json",
        ],
    )
    _cache_warmed = True


def _load_model() -> None:
    global _tokenizer, _model, _nemotron_cache_class
    if _model is not None and _tokenizer is not None:
        return
    with _model_lock:
        if _model is not None and _tokenizer is not None:
            return
        _warm_cache()
        logging.info("Loading model to GPU…")
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            trust_remote_code=True,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            token=HF_TOKEN,
        )
        _model.eval()
        _nemotron_cache_class = _resolve_nemotron_cache_class(_model)
        if _nemotron_cache_class is None:
            logging.info("Nemotron cache enabled: False (class not found)")
        else:
            logging.info("Nemotron cache enabled: True")


def _resolve_nemotron_cache_class(model: Any) -> type | None:
    module_name = getattr(model.__class__, "__module__", "")
    model_module = _import_module_safely(module_name)
    cache_class = _resolve_cache_from_module_object(model_module)
    if cache_class is not None:
        return cache_class

    for loaded_name, loaded_module in list(sys.modules.items()):
        if not loaded_name or not loaded_name.endswith("modeling_nemotron_h"):
            continue
        cache_class = _resolve_cache_from_module_object(loaded_module)
        if cache_class is not None:
            return cache_class
    return None


def _resolve_cache_from_module(module_name: str) -> type | None:
    if not module_name:
        return None
    module = _import_module_safely(module_name)
    return _resolve_cache_from_module_object(module)


def _import_module_safely(module_name: str) -> Any | None:
    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        return None
    except (ImportError, Exception):
        return None
    if spec is None:
        return None
    try:
        return importlib.import_module(module_name)
    except (ModuleNotFoundError, ImportError, Exception):
        return None


def _resolve_cache_from_module_object(module: Any) -> type | None:
    if module is None:
        return None
    try:
        return getattr(module, "NemotronHHybridDynamicCache", None)
    except Exception:
        return None


def _generate_with_cache(
    input_ids: torch.Tensor, generate_kwargs: dict[str, Any]
) -> torch.Tensor:
    if _nemotron_cache_class is None:
        return _model.generate(input_ids, **generate_kwargs)

    try:
        cache = _nemotron_cache_class()
    except Exception as exc:
        logging.warning("Cache init failed, running without cache: %s", exc)
        return _model.generate(input_ids, **generate_kwargs)

    try:
        return _model.generate(input_ids, past_key_values=cache, **generate_kwargs)
    except TypeError as exc:
        last_exc = exc

    try:
        return _model.generate(input_ids, cache=cache, **generate_kwargs)
    except TypeError as exc:
        last_exc = exc

    logging.warning("Cache init failed, running without cache: %s", last_exc)
    return _greedy_decode_with_cache(input_ids, cache, generate_kwargs)


def _greedy_decode_with_cache(
    input_ids: torch.Tensor, cache: Any, generate_kwargs: dict[str, Any]
) -> torch.Tensor:
    signature = inspect.signature(_model.forward)
    cache_arg = None
    if "past_key_values" in signature.parameters:
        cache_arg = "past_key_values"
    elif "cache" in signature.parameters:
        cache_arg = "cache"

    use_cache_supported = "use_cache" in signature.parameters
    eos_token_id = generate_kwargs.get("eos_token_id")
    max_new_tokens = int(generate_kwargs.get("max_new_tokens", 0))
    generated = []
    next_input_ids = input_ids
    for _ in range(max_new_tokens):
        forward_kwargs: dict[str, Any] = {}
        if cache_arg is not None:
            forward_kwargs[cache_arg] = cache
        if use_cache_supported:
            forward_kwargs["use_cache"] = True
        outputs = _model.forward(next_input_ids, **forward_kwargs)
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated.append(next_token)
        next_input_ids = next_token
        updated_cache = getattr(outputs, "past_key_values", None) or getattr(outputs, "cache", None)
        if updated_cache is not None:
            cache = updated_cache
        if eos_token_id is not None and torch.eq(next_token, eos_token_id).all():
            break
    if generated:
        return torch.cat([input_ids, torch.cat(generated, dim=-1)], dim=-1)
    return input_ids


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    return ""


app = FastAPI()


@app.on_event("startup")
def _on_startup() -> None:
    _check_mamba_ssm()
    _warm_cache()
    logging.info("Server ready")


@app.get("/v1/models")
def list_models() -> dict:
    return {"object": "list", "data": [{"id": OPENAI_MODEL_NAME, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict:
    payload = await request.json()
    messages = payload.get("messages")
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    max_tokens = payload.get("max_tokens") or 256
    temperature = payload.get("temperature", 0.7)
    top_p = payload.get("top_p", 1.0)

    chat_template_kwargs = payload.get("chat_template_kwargs") or {}
    extra_body = payload.get("extra_body") or {}
    extra_chat_kwargs = extra_body.get("chat_template_kwargs") or {}
    merged_chat_kwargs = {**chat_template_kwargs, **extra_chat_kwargs}
    enable_thinking = bool(merged_chat_kwargs.get("enable_thinking", False))

    _load_model()

    input_ids = _tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_tensors="pt",
    )
    input_ids = input_ids.to(_model.device)

    do_sample = temperature is not None and temperature > 0
    generate_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "eos_token_id": _tokenizer.eos_token_id,
    }
    if temperature is not None:
        generate_kwargs["temperature"] = temperature
    if top_p is not None:
        generate_kwargs["top_p"] = top_p

    with torch.inference_mode():
        output_ids = _generate_with_cache(input_ids, generate_kwargs)

    generated_tokens = output_ids[0][input_ids.shape[-1] :]
    text = _tokenizer.decode(generated_tokens, skip_special_tokens=True)

    prompt_text = "\n".join(_extract_text(msg.get("content")) for msg in messages)
    prompt_tokens = max(1, len(prompt_text) // 4) if prompt_text else 0
    completion_tokens = max(1, len(text) // 4) if text else 0
    total_tokens = prompt_tokens + completion_tokens

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": OPENAI_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host=host, port=port)

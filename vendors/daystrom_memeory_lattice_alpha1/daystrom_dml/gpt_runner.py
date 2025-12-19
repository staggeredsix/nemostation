"""Wrapper around HuggingFace models with graceful degradation."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import requests

try:  # pragma: no cover - torch might be unavailable in minimal environments
    import torch
except ModuleNotFoundError:  # pragma: no cover - handled during tests
    torch = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


@dataclass
class GPTRunner:
    """Minimal wrapper providing ``generate`` and ``summarize``.

    The class attempts to instantiate a HuggingFace pipeline.  When the required
    dependencies or model weights are not available (as is often the case in
    offline tests) it falls back to a deterministic dummy backend.
    """

    model_name: str
    task: str = "text-generation"
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self._backend = None
        self._last_usage: Optional[dict] = None
        remote_base = os.getenv("DML_API_BASE") or os.getenv("OPENAI_API_BASE")
        remote_base = remote_base or os.getenv("NIM_API_BASE")
        remote_key = os.getenv("DML_API_KEY") or os.getenv("OPENAI_API_KEY")
        remote_key = remote_key or os.getenv("NIM_API_KEY")
        if remote_base:
            self._backend = _OpenAICompatibleBackend(
                base_url=remote_base,
                api_key=remote_key,
                model_name=self.model_name,
            )
            LOGGER.info("Configured remote backend at %s", remote_base)
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model_kwargs = {}
            if self._should_use_half_precision():
                model_kwargs.update(
                    {
                        "torch_dtype": torch.float16,
                        "low_cpu_mem_usage": True,
                    }
                )
                LOGGER.info("Loading model %s in float16 precision", self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
            self._backend = pipeline(
                self.task,
                model=model,
                tokenizer=tokenizer,
                device=self.device,
            )
            LOGGER.info("Loaded HF model %s", self.model_name)
        except Exception as exc:  # pragma: no cover - executed in offline tests
            LOGGER.warning("Using DummyGPT backend: %s", exc)
            self._backend = _DummyBackend()

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        self._last_usage = None
        approx_tokens = len(prompt.split())
        LOGGER.info(
            "Sending prompt to language model (model=%s, approx_tokens=%d)",
            self.model_name,
            approx_tokens,
        )
        LOGGER.debug("Prompt excerpt: %s", prompt[:400])
        if isinstance(self._backend, _DummyBackend):
            return self._backend.generate(prompt, max_new_tokens=max_new_tokens)
        if isinstance(self._backend, _OpenAICompatibleBackend):
            text, usage = self._backend.generate(prompt, max_new_tokens=max_new_tokens)
            self._last_usage = usage
            return text
        outputs = self._backend(prompt, max_new_tokens=max_new_tokens)
        if isinstance(outputs, list):
            return outputs[0]["generated_text"]
        return str(outputs)

    def summarize(self, text: str, max_len: int = 128) -> str:
        if isinstance(self._backend, _DummyBackend):
            return self._backend.summarize(text, max_len=max_len)
        if isinstance(self._backend, _OpenAICompatibleBackend):
            text, usage = self._backend.generate(
                (
                    "Summarise the following content in at most "
                    f"{max_len} characters.\n{text}"
                ),
                max_new_tokens=max_len,
                system_prompt="You are a precise summariser that responds with plain text.",
            )
            self._last_usage = usage
            return text.strip()
        prompt = (
            "Summarise the following content in at most"
            f" {max_len} characters:\n{text}\nSummary:"
        )
        output = self.generate(prompt, max_new_tokens=max_len)
        return output.split("Summary:")[-1].strip()

    @property
    def is_dummy(self) -> bool:
        """Expose whether the runner is using the dummy backend."""

        return isinstance(self._backend, _DummyBackend)

    @property
    def last_usage(self) -> Optional[dict]:
        """Return the token usage payload from the most recent call."""

        return self._last_usage

    def _should_use_half_precision(self) -> bool:
        """Return ``True`` when the model should load using float16 weights."""

        if torch is None or not torch.cuda.is_available():
            return False
        if self.device is None:
            return True
        if isinstance(self.device, str):
            return self.device.startswith("cuda")
        if isinstance(self.device, int):
            return self.device >= 0
        return False


class _DummyBackend:
    """Fallback backend used during tests."""

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        text = prompt.strip()
        suffix = "\n[Dummy completion truncated]"
        if len(text) > max_new_tokens:
            text = text[: max_new_tokens]
        return text + suffix

    def summarize(self, text: str, max_len: int = 128) -> str:
        text = text.strip().replace("\n", " ")
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."


class _OpenAICompatibleBackend:
    """Thin wrapper around OpenAI-compatible REST endpoints (incl. NVIDIA NIM)."""

    def __init__(self, *, base_url: str, api_key: Optional[str], model_name: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        system_prompt: Optional[str] = None,
    ) -> tuple[str, Optional[dict]]:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": 0.2,
        }
        LOGGER.info(
            "Dispatching completion request to NIM endpoint %s using model %s",
            url,
            self.model_name,
        )
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return "", data.get("usage")
        choice = choices[0] or {}
        message = choice.get("message") or {}
        content = message.get("content")
        if content is None:
            # Some OpenAI-compatible servers return ``null`` for empty content or use
            # the older ``text`` field.  Normalise both cases to an empty string so we
            # always return a ``str``.
            content = choice.get("text") or ""
        if not isinstance(content, str):
            content = str(content)
        LOGGER.info("Received response from NIM endpoint %s", url)
        return content.strip(), data.get("usage")

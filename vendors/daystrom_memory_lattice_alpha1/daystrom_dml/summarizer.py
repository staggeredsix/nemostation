"""Summarisation helpers for the Daystrom Memory Lattice."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .gpt_runner import GPTRunner

LOGGER = logging.getLogger(__name__)


class Summarizer:
    """Simple interface used by the memory store."""

    def summarize(self, text: str, max_len: int = 128) -> str:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class LLMSummarizer(Summarizer):
    """Summarise text using the configured :class:`GPTRunner`."""

    runner: GPTRunner

    def summarize(self, text: str, max_len: int = 128) -> str:
        if not text:
            return ""
        return self.runner.summarize(text, max_len=max_len)


class DummySummarizer(Summarizer):
    """Fallback summariser that performs a light-weight truncation."""

    def summarize(self, text: str, max_len: int = 128) -> str:
        if not text:
            return ""
        if len(text) <= max_len:
            return text
        LOGGER.debug("DummySummarizer truncating text to %d chars", max_len)
        return text[: max_len - 3] + "..."

"""Embedding backends."""
from __future__ import annotations

import random
from typing import Iterable, List, Protocol


class Embedder(Protocol):
    """Protocol for embedding text."""

    def encode(self, text: str | Iterable[str]) -> List[List[float]]:
        """Encode text(s) into vector representations."""
        ...


class SentenceTransformerEmbedder:
    """SentenceTransformer backed embedder with graceful fallback."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", seed: int | None = None):
        self.model_name = model_name
        self._model = None
        self._rng = random.Random(seed)

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
        except Exception:  # pragma: no cover - no dependency in tests
            self._model = None

    def encode(self, text: str | Iterable[str]) -> List[List[float]]:
        items = [text] if isinstance(text, str) else list(text)
        if not items:
            return []
        if self._model is not None:
            vectors = self._model.encode(items)
            return [list(map(float, vec)) for vec in vectors]
        # Deterministic random fallback
        vectors = [[self._rng.gauss(0.0, 1.0) for _ in range(384)] for _ in items]
        return vectors


class RandomEmbedder:
    """Simple deterministic random embedder."""

    def __init__(self, dim: int = 128, seed: int | None = None):
        self.dim = dim
        self._rng = random.Random(seed)

    def encode(self, text: str | Iterable[str]) -> List[List[float]]:
        items = [text] if isinstance(text, str) else list(text)
        vectors = [[self._rng.gauss(0.0, 1.0) for _ in range(self.dim)] for _ in items]
        return vectors

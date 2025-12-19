"""Embedding backends for the Daystrom Memory Lattice."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from . import utils

LOGGER = logging.getLogger(__name__)


class Embedder(Protocol):
    """Simple embedder protocol."""

    def embed(self, text: str) -> np.ndarray:
        ...


@dataclass
class SentenceTransformerEmbedder:
    """Embed text using a SentenceTransformer model.

    The class degrades gracefully when the ``sentence_transformers`` package is
    unavailable by falling back to :class:`RandomEmbedder`.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str | None = None

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            target_device = self._resolve_device()
            self._model = SentenceTransformer(self.model_name, device=target_device)
            self._dim = int(self._model.get_sentence_embedding_dimension())
            resolved_device = getattr(self._model, "_target_device", None)
            if resolved_device is not None:
                LOGGER.info(
                    "Loaded SentenceTransformer model %s on device %s",
                    self.model_name,
                    resolved_device,
                )
            else:
                LOGGER.info(
                    "Loaded SentenceTransformer model %s", self.model_name
                )
        except Exception as exc:  # pragma: no cover - exercised when dependency missing
            LOGGER.warning("Falling back to RandomEmbedder: %s", exc)
            self._model = None
            self._dim = 384

    def embed(self, text: str) -> np.ndarray:
        if self._model is None:
            return RandomEmbedder(self._dim).embed(text)
        if not text:
            return np.zeros(self._dim, dtype=np.float32)
        vector = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype=np.float32)

    @staticmethod
    def _autodetect_device() -> str | None:
        """Return the best available accelerator for SentenceTransformer."""

        try:
            import torch
        except Exception:  # pragma: no cover - torch is an optional dependency
            LOGGER.debug("Torch not available; defaulting embedder to CPU")
            return None

        if torch.cuda.is_available():
            # Respect CUDA_VISIBLE_DEVICES; SentenceTransformer handles device indices
            try:
                torch.cuda.current_device()
            except Exception:  # pragma: no cover - guard against lazy init errors
                pass
            return "cuda"

        # Fall back to Apple Metal if available (macOS environments)
        try:
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend and mps_backend.is_available():
                return "mps"
        except Exception:  # pragma: no cover - backend probing failures
            LOGGER.debug("Torch MPS probe failed; continuing with CPU", exc_info=True)

        return None

    def _resolve_device(self) -> str | None:
        """Resolve the execution device honouring explicit overrides."""

        requested = self.device
        if requested is None:
            return self._autodetect_device()
        cleaned = str(requested).strip()
        if not cleaned:
            return self._autodetect_device()
        lowered = cleaned.lower()
        if lowered in {"auto", "autodetect", "detect"}:
            return self._autodetect_device()
        if lowered in {"none", "cpu"}:
            return None if lowered == "none" else "cpu"
        if lowered.startswith("cuda") or lowered.startswith("mps") or lowered.startswith("xpu"):
            return cleaned
        LOGGER.warning(
            "Unknown embedding device '%s'; falling back to auto-detection.",
            requested,
        )
        return self._autodetect_device()


@dataclass
class RandomEmbedder:
    """Deterministic pseudo-random embeddings used in tests."""

    dim: int = 384

    def embed(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.dim, dtype=np.float32)
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        return utils.seeded_random_vector(self.dim, seed)


def create_embedder(model_name: str | None, *, device: str | None = None) -> Embedder:
    """Factory returning the best available embedder."""

    if model_name:
        normalised_device = (device or "").strip() or None
        return SentenceTransformerEmbedder(model_name, device=normalised_device)
    return RandomEmbedder()


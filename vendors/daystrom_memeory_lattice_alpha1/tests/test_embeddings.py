from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from daystrom_dml.embeddings import SentenceTransformerEmbedder, create_embedder


class _DummySentenceTransformer:
    """Lightweight stub that mimics the SentenceTransformer API."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.requested_device = device
        self._target_device = device or "cpu"

    def get_sentence_embedding_dimension(self) -> int:
        return 6

    def encode(self, text: str, normalize_embeddings: bool, show_progress_bar: bool) -> np.ndarray:
        del text, normalize_embeddings, show_progress_bar
        return np.arange(6, dtype=np.float32)


@pytest.fixture(autouse=True)
def _patch_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.SimpleNamespace(SentenceTransformer=_DummySentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_embedder_respects_explicit_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        SentenceTransformerEmbedder,
        "_autodetect_device",
        staticmethod(lambda: "cuda"),
        raising=False,
    )
    embedder = SentenceTransformerEmbedder("stub-model", device="cuda:1")
    assert embedder._model.requested_device == "cuda:1"  # type: ignore[attr-defined]
    vector = embedder.embed("hello world")
    assert np.allclose(vector, np.arange(6, dtype=np.float32))


def test_embedder_autodetects_for_auto_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        SentenceTransformerEmbedder,
        "_autodetect_device",
        staticmethod(lambda: "cuda"),
        raising=False,
    )
    embedder = SentenceTransformerEmbedder("stub-model", device="auto")
    assert embedder._model.requested_device == "cuda"  # type: ignore[attr-defined]


def test_embedder_warns_on_unknown_device(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _fake_autodetect() -> str:
        calls.append("auto")
        return "cuda"

    monkeypatch.setattr(
        SentenceTransformerEmbedder,
        "_autodetect_device",
        staticmethod(_fake_autodetect),
        raising=False,
    )
    embedder = SentenceTransformerEmbedder("stub-model", device="mystery-device")
    assert embedder._model.requested_device == "cuda"  # type: ignore[attr-defined]
    assert calls == ["auto"]


def test_create_embedder_passes_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        SentenceTransformerEmbedder,
        "_autodetect_device",
        staticmethod(lambda: "cpu"),
        raising=False,
    )
    embedder = create_embedder("stub-model", device="cuda:0")
    assert isinstance(embedder, SentenceTransformerEmbedder)
    assert embedder.device == "cuda:0"
    assert embedder._model.requested_device == "cuda:0"  # type: ignore[attr-defined]

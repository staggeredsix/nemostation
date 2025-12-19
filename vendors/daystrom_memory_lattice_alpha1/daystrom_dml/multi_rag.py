from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from . import utils

try:  # pragma: no cover - optional dependency for real vector databases
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except Exception:  # pragma: no cover - handled gracefully when unavailable
    chromadb = None
    ChromaSettings = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency for FAISS integration
    import faiss  # type: ignore[import]
except Exception:  # pragma: no cover - handled gracefully when unavailable
    faiss = None  # type: ignore[assignment]


class RAGBackendProtocol:
    """Protocol for concrete RAG vector database integrations."""

    identifier: str
    label: str
    description: str

    def add_document(
        self,
        text: str,
        embedding: np.ndarray,
        tokens: int,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:  # pragma: no cover - interface method
        raise NotImplementedError

    def clear(self) -> None:  # pragma: no cover - interface method
        raise NotImplementedError

    def retrieve(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int,
    ) -> List[Dict[str, Any]]:  # pragma: no cover - interface method
        raise NotImplementedError


class _UnavailableBackend(RAGBackendProtocol):
    """Placeholder used when the real backend cannot be initialised."""

    def __init__(self, identifier: str, label: str, description: str, error: str) -> None:
        self.identifier = identifier
        self.label = label
        self.description = description
        self.error = error

    def add_document(self, text: str, embedding: np.ndarray, tokens: int, meta: Optional[Dict[str, Any]] = None) -> None:
        raise RuntimeError(self.error)

    def clear(self) -> None:
        return

    def retrieve(self, query_embedding: np.ndarray, *, top_k: int) -> List[Dict[str, Any]]:
        raise RuntimeError(self.error)


class ChromaRAGBackend(RAGBackendProtocol):
    """Persistence-backed integration with Chroma DB."""

    def __init__(self, persist_dir: str = "./.chroma") -> None:
        if chromadb is None:
            raise RuntimeError("chromadb is not installed")
        self.identifier = "chroma"
        self.label = "Chroma"
        self.description = "Local Chroma collection backed by DuckDB."
        self._client = self._create_client(persist_dir)
        self._collection = self._client.get_or_create_collection(
            "daystrom-playground",
            metadata={"hnsw:space": "cosine"},
        )

    def _create_client(self, persist_dir: str) -> Any:
        """Initialise a Chroma client compatible with legacy and modern APIs."""

        errors: list[str] = []

        # Preferred path for modern Chroma releases.
        try:
            return chromadb.PersistentClient(path=persist_dir)
        except TypeError as exc:  # pragma: no cover - compatibility with older clients
            errors.append(str(exc))
        except Exception as exc:  # pragma: no cover - unexpected client errors
            errors.append(str(exc))

        # Backwards-compatible fallback for older releases still expecting Settings.
        if ChromaSettings is not None:
            settings = ChromaSettings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_dir,
            )
            try:
                return chromadb.PersistentClient(path=persist_dir, settings=settings)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover - unexpected client errors
                errors.append(str(exc))
            try:
                return chromadb.Client(settings)  # type: ignore[call-arg]
            except Exception as exc:  # pragma: no cover - compatibility fallback
                errors.append(str(exc))

        raise RuntimeError(
            "Unable to initialise Chroma client with the available configuration; "
            + "; ".join(errors)
        )

    def add_document(self, text: str, embedding: np.ndarray, tokens: int, meta: Optional[Dict[str, Any]] = None) -> None:
        payload = dict(meta or {})
        payload.setdefault("tokens", tokens)
        doc_id = payload.get("doc_id") or f"doc-{uuid.uuid4()}"
        payload["doc_id"] = doc_id
        self._collection.add(  # type: ignore[attr-defined]
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding.astype(np.float32).tolist()],
            metadatas=[payload],
        )

    def clear(self) -> None:
        try:
            self._client.delete_collection("daystrom-playground")  # type: ignore[attr-defined]
        finally:
            self._collection = self._client.get_or_create_collection(
                "daystrom-playground",
                metadata={"hnsw:space": "cosine"},
            )

    def retrieve(self, query_embedding: np.ndarray, *, top_k: int) -> List[Dict[str, Any]]:
        if not hasattr(self, "_collection"):
            return []
        result = self._collection.query(  # type: ignore[attr-defined]
            query_embeddings=[query_embedding.astype(np.float32).tolist()],
            n_results=max(1, top_k),
            include=["documents", "embeddings", "metadatas", "distances"],
        )
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        matches: List[Dict[str, Any]] = []
        for idx, text in enumerate(documents[0]):
            meta = metadatas[0][idx] if idx < len(metadatas[0]) else {}
            distance = float(distances[0][idx]) if idx < len(distances[0]) else 0.0
            score = 1.0 - distance
            tokens = int(meta.get("tokens") or utils.estimate_tokens(text))
            matches.append(
                {
                    "text": text,
                    "meta": meta or {},
                    "tokens": tokens,
                    "score": score,
                }
            )
        return matches


class FaissRAGBackend(RAGBackendProtocol):
    """In-process FAISS index backed by cosine similarity."""

    def __init__(self) -> None:
        if faiss is None:
            raise RuntimeError("faiss is not installed")
        self.identifier = "faiss"
        self.label = "Faiss"
        self.description = "In-memory FAISS index using cosine similarity."
        self._index = None
        self._documents: List[str] = []
        self._metadata: List[Dict[str, Any]] = []

    def _ensure_index(self, dim: int) -> None:
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)

    def add_document(self, text: str, embedding: np.ndarray, tokens: int, meta: Optional[Dict[str, Any]] = None) -> None:
        vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        dim = vector.shape[1]
        self._ensure_index(dim)
        assert self._index is not None
        self._index.add(vector)
        payload = dict(meta or {})
        payload.setdefault("tokens", tokens)
        self._documents.append(text)
        self._metadata.append(payload)

    def clear(self) -> None:
        if self._index is not None:
            self._index.reset()
        self._documents.clear()
        self._metadata.clear()

    def retrieve(self, query_embedding: np.ndarray, *, top_k: int) -> List[Dict[str, Any]]:
        if self._index is None or not self._documents:
            return []
        vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        limit = min(max(1, top_k), len(self._documents))
        scores, indices = self._index.search(vector, limit)
        matches: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            text = self._documents[idx]
            meta = self._metadata[idx]
            tokens = int(meta.get("tokens") or utils.estimate_tokens(text))
            matches.append(
                {
                    "text": text,
                    "meta": meta,
                    "tokens": tokens,
                    "score": float(score),
                }
            )
        return matches


@dataclass(frozen=True)
class RAGBackendDescriptor:
    """Configuration record for available RAG deployments."""

    identifier: str
    label: str
    description: str
    factory: Any


DEFAULT_BACKENDS: List[RAGBackendDescriptor] = [
    RAGBackendDescriptor(
        identifier="chroma",
        label="Chroma",
        description="Local persistent Chroma collection",
        factory=lambda: ChromaRAGBackend(),
    ),
    RAGBackendDescriptor(
        identifier="faiss",
        label="Faiss",
        description="In-memory FAISS cosine index",
        factory=lambda: FaissRAGBackend(),
    ),
]


class MultiRAGStore:
    """Fan out incoming documents and retrievals to multiple backends."""

    def __init__(
        self,
        embedder,
        backends: Optional[Iterable[RAGBackendDescriptor]] = None,
    ) -> None:
        self.embedder = embedder
        self._descriptors: List[RAGBackendDescriptor] = list(backends or DEFAULT_BACKENDS)
        self._backends: List[Dict[str, Any]] = []
        for descriptor in self._descriptors:
            try:
                backend = descriptor.factory()
                available = True
                error = None
            except Exception as exc:
                backend = _UnavailableBackend(
                    descriptor.identifier,
                    descriptor.label,
                    descriptor.description,
                    str(exc),
                )
                available = False
                error = str(exc)
            self._backends.append(
                {
                    "id": descriptor.identifier,
                    "label": descriptor.label,
                    "backend": backend,
                    "description": descriptor.description,
                    "available": available,
                    "error": error,
                }
            )
        self._raw_documents: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------
    def add_document(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if not text:
            return
        payload = {"text": text, "meta": meta or {}}
        embedding = np.asarray(self.embedder.embed(text), dtype=np.float32)
        tokens = utils.estimate_tokens(text)
        with self._lock:
            self._raw_documents.append(payload)
            backends = list(self._backends)
        for backend in backends:
            instance: RAGBackendProtocol = backend["backend"]
            if not backend["available"]:
                continue
            try:
                instance.add_document(text, embedding, tokens, meta=meta)
            except Exception:
                with self._lock:
                    backend["available"] = False
                    backend["error"] = "Failed to ingest document"

    def clear(self) -> None:
        with self._lock:
            self._raw_documents.clear()
            backends = list(self._backends)
        for backend in backends:
            try:
                backend["backend"].clear()
            except Exception:
                with self._lock:
                    backend["available"] = False
                    backend["error"] = "Failed to reset backend"

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def report_all(self, prompt: str, *, top_k: int = 4) -> List[Dict[str, Any]]:
        reports: List[Dict[str, Any]] = []
        with self._lock:
            backends = list(self._backends)
        if not backends:
            return reports
        query_embedding = np.asarray(self.embedder.embed(prompt), dtype=np.float32)
        for backend in backends:
            instance: RAGBackendProtocol = backend["backend"]
            start = time.perf_counter()
            if not backend["available"]:
                reports.append(
                    {
                        "id": backend["id"],
                        "label": backend["label"],
                        "strategy": backend["description"],
                        "available": False,
                        "error": backend.get("error"),
                        "documents": [],
                        "context": "",
                        "tokens": 0,
                        "latency_ms": 0,
                    }
                )
                continue
            try:
                matches = instance.retrieve(query_embedding, top_k=top_k)
                latency_ms = int((time.perf_counter() - start) * 1000.0)
                context_lines: List[str] = []
                for idx, match in enumerate(matches, start=1):
                    meta = match.get("meta") or {}
                    source = meta.get("doc_path") or meta.get("source") or f"Document {idx}"
                    header = f"Document {idx} (score={match.get('score', 0.0):.3f})"
                    context_lines.extend([header, f"Source: {source}", match.get("text", "").strip(), ""])
                context = "\n".join(context_lines).strip()
                tokens = utils.estimate_tokens(context)
                reports.append(
                    {
                        "id": backend["id"],
                        "label": backend["label"],
                        "strategy": backend["description"],
                        "documents": matches,
                        "context": context,
                        "tokens": tokens,
                        "latency_ms": latency_ms,
                        "available": True,
                        "error": None,
                    }
                )
            except Exception as exc:
                with self._lock:
                    backend["available"] = False
                    backend["error"] = str(exc)
                reports.append(
                    {
                        "id": backend["id"],
                        "label": backend["label"],
                        "strategy": backend["description"],
                        "documents": [],
                        "context": "",
                        "tokens": 0,
                        "latency_ms": 0,
                        "available": False,
                        "error": str(exc),
                    }
                )
        return reports

    def primary_report(self, prompt: str, *, top_k: int = 4) -> Optional[Dict[str, Any]]:
        reports = self.report_all(prompt, top_k=top_k)
        return reports[0] if reports else None

    def catalog_summary(self) -> Dict[str, Any]:
        documents: List[Dict[str, Any]] = []
        total_tokens = 0
        with self._lock:
            documents_snapshot = list(self._raw_documents)
            backends_snapshot = [
                {
                    "id": backend["id"],
                    "label": backend["label"],
                    "strategy": backend["description"],
                    "available": backend["available"],
                    "error": backend.get("error"),
                }
                for backend in self._backends
            ]
        for idx, entry in enumerate(documents_snapshot, start=1):
            meta = entry.get("meta") or {}
            source = meta.get("doc_path") or meta.get("source") or f"Document {idx}"
            tokens = utils.estimate_tokens(entry["text"])
            total_tokens += tokens
            documents.append({"index": idx, "source": source, "tokens": tokens})
        return {
            "documents": documents,
            "total_tokens": total_tokens,
            "count": len(documents),
            "backends": backends_snapshot,
        }

    def export_state(self) -> Dict[str, Any]:
        with self._lock:
            return {"documents": list(self._raw_documents)}

    def import_state(self, payload: Optional[Dict[str, Any]]) -> None:
        if not payload:
            return
        documents = payload.get("documents") or []
        self.clear()
        for entry in documents:
            text = entry.get("text")
            if not text:
                continue
            meta = entry.get("meta") or {}
            self.add_document(str(text), meta=meta)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def descriptors(self) -> List[Dict[str, str]]:
        with self._lock:
            return [
                {
                    "id": backend["id"],
                    "label": backend["label"],
                    "strategy": backend["description"],
                    "available": backend["available"],
                    "error": backend.get("error"),
                }
                for backend in self._backends
            ]


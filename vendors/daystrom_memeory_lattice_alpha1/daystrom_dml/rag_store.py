"""Lightweight persistent FAISS-backed retrieval store."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore[import]
except Exception:  # pragma: no cover - handled gracefully when unavailable
    faiss = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


@dataclass
class RAGRecord:
    """Single document stored in the persistent RAG index."""

    id: int
    text: str
    meta: Dict[str, Any]


class PersistentRAGStore:
    """Minimal persistent vector store backed by FAISS."""

    def __init__(
        self,
        *,
        enable: bool,
        index_path: Path,
        meta_path: Path,
        dim: int,
        backend: str = "faiss",
    ) -> None:
        self.enable = enable
        self.backend = backend
        self.index_path = index_path
        self.meta_path = meta_path
        self._records: List[RAGRecord] = []
        self._id_lookup: Dict[int, int] = {}
        self._index: Any = None
        self._next_id = 0
        self._dim = int(dim)
        if self.backend != "faiss":
            raise ValueError(f"Unsupported backend: {backend}")
        if self.enable and faiss is None:
            raise RuntimeError("faiss is required for the persistent RAG store")

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load an existing index from disk if present."""

        if not self.enable:
            return
        if not self.index_path.exists() or not self.meta_path.exists():
            return
        try:
            self._index = faiss.read_index(str(self.index_path))
            data = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.exception("Failed to load persistent RAG index from disk")
            return
        records = data.get("records", [])
        next_id = int(data.get("next_id", len(records)))
        dim = int(data.get("dim", getattr(self._index, "d", self._dim)))
        self._records = [
            RAGRecord(id=int(entry.get("id", idx)), text=entry.get("text", ""), meta=dict(entry.get("meta") or {}))
            for idx, entry in enumerate(records)
        ]
        self._id_lookup = {record.id: pos for pos, record in enumerate(self._records)}
        self._next_id = max(next_id, (max(self._id_lookup.keys()) + 1) if self._id_lookup else 0)
        self._dim = dim
        if self._index is not None and getattr(self._index, "d", dim) != dim:
            LOGGER.warning(
                "FAISS index dimension (%s) mismatches metadata (%s); using index dimension.",
                getattr(self._index, "d", "unknown"),
                dim,
            )
            self._dim = int(getattr(self._index, "d", dim))
        if self._index is not None and self._index.ntotal != len(self._records):
            LOGGER.warning(
                "FAISS index document count (%s) mismatches metadata (%s).",
                self._index.ntotal,
                len(self._records),
            )

    def persist(self) -> None:
        """Persist the FAISS index and metadata to disk."""

        if not self.enable or self._index is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": self._dim,
            "next_id": self._next_id,
            "records": [record.__dict__ for record in self._records],
        }
        faiss.write_index(self._index, str(self.index_path))
        self.meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # core operations
    # ------------------------------------------------------------------
    def add(self, text: str, embedding: Iterable[float], meta: Optional[Dict[str, Any]] = None) -> int:
        """Add a new document to the persistent index."""

        if not text or not self.enable:
            return -1
        vector = self._prepare_vector(embedding)
        if vector is None:
            return -1
        if self._index is None:
            self._index = faiss.IndexFlatIP(vector.shape[1])
            self._dim = vector.shape[1]
        if vector.shape[1] != self._dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dim}, received {vector.shape[1]}"
            )
        faiss.normalize_L2(vector)
        self._index.add(vector)
        record_id = self._next_id
        self._next_id += 1
        record = RAGRecord(id=record_id, text=text, meta=dict(meta or {}))
        self._records.append(record)
        self._id_lookup[record_id] = len(self._records) - 1
        return record_id

    def search(self, embedding: Iterable[float], top_k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve the closest matches for the supplied embedding."""

        if not self.enable or self._index is None or not self._records:
            return []
        vector = self._prepare_vector(embedding)
        if vector is None or vector.shape[1] != self._dim:
            return []
        faiss.normalize_L2(vector)
        top_k = max(1, min(int(top_k), len(self._records)))
        scores, indices = self._index.search(vector, top_k)
        results: List[Dict[str, Any]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self._records):
                continue
            record = self._records[idx]
            results.append(
                {
                    "id": record.id,
                    "text": record.text,
                    "meta": dict(record.meta),
                    "score": float(score),
                }
            )
        return results

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _prepare_vector(self, embedding: Iterable[float]) -> Optional[np.ndarray]:
        if embedding is None:
            return None
        vector = np.asarray(list(embedding), dtype=np.float32)
        if vector.size == 0:
            return None
        return vector.reshape(1, -1)

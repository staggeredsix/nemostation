"""Core Daystrom Memory Lattice implementation."""
from __future__ import annotations

import contextlib
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import utils
from .vector_backend import get_vector_backend
from .summarizer import Summarizer


@dataclass
class MemoryItem:
    """Container for a single memory element."""

    id: int
    text: str
    embedding: np.ndarray
    timestamp: float
    salience: float
    fidelity: float
    level: int
    meta: Optional[Dict] = field(default_factory=dict)
    summary_of: List[int] = field(default_factory=list)

    @property
    def children(self) -> List[int]:
        if self.summary_of:
            return list(self.summary_of)
        return [self.id]

    def cached_summary(self, max_len: int = 256) -> str:
        summary = ""
        if self.meta is not None:
            summary = str(self.meta.get("summary") or "").strip()
        if summary:
            if len(summary) > max_len:
                return summary[: max_len - 3] + "..."
            return summary
        text = (self.text or "").strip()
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "timestamp": self.timestamp,
            "salience": self.salience,
            "fidelity": self.fidelity,
            "level": self.level,
            "meta": self.meta,
            "summary_of": self.summary_of,
            "embedding": utils.ensure_serializable(self.embedding),
            "children": self.children,
        }


class MemoryStore:
    """Implements storage, retrieval and ageing for the DML."""

    def __init__(
        self,
        summarizer: Summarizer,
        *,
        beta_a: float,
        beta_r: float,
        eta: float,
        gamma: float,
        kappa: float,
        tau_s: float,
        theta_merge: float,
        K: int,
        capacity: int,
        start_aging_loop: bool = True,
        aging_interval_seconds: float = 5.0,
        enable_quality_on_retrieval: bool = False,
        similarity_threshold: float = 0.0,
    ) -> None:
        self.summarizer = summarizer
        self.beta_a = beta_a
        self.beta_r = beta_r
        self.eta = eta
        self.gamma = gamma
        self.kappa = kappa
        self.tau_s = tau_s
        self.theta_merge = theta_merge
        self.K = max(1, K)
        self.capacity = max(1, capacity)
        self._items: List[MemoryItem] = []
        self._id = 0
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._lineage: Dict[int, MemoryItem] = {}
        self._repair_queue: List[int] = []
        self.quality_threshold = -0.1
        self._aging_interval_seconds = max(1.0, float(aging_interval_seconds))
        self.similarity_threshold = float(max(-1.0, min(1.0, similarity_threshold)))
        # Expensive quality/repair checks can be deferred to a maintenance pass.
        self.enable_quality_on_retrieval = bool(enable_quality_on_retrieval)
        self._vector_backend = get_vector_backend()
        self._aging_thread: Optional[threading.Thread] = None
        if start_aging_loop:
            self._aging_thread = threading.Thread(
                target=self._aging_loop, name="dml-aging", daemon=True
            )
            self._aging_thread.start()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._stop_event.set()
        if self._aging_thread and self._aging_thread.is_alive():
            self._aging_thread.join(timeout=1.0)

    def ingest(
        self,
        text: str,
        embedding: np.ndarray,
        *,
        salience: float = 1.0,
        fidelity: float = 1.0,
        level: int = 0,
        meta: Optional[Dict] = None,
    ) -> Tuple[MemoryItem, bool]:
        now = time.time()
        enriched_meta = dict(meta or {})

        with self._lock:
            best_match, best_sim = self._best_match(embedding)

        with self._lock:
            merged = self._try_merge(
                text,
                embedding,
                salience,
                meta=meta,
                precomputed_match=(best_match, best_sim),
            )
            if merged:
                return merged, True
            item = MemoryItem(
                id=self._next_id(),
                text=text,
                embedding=np.asarray(embedding, dtype=np.float32),
                timestamp=now,
                salience=float(salience),
                fidelity=float(max(0.0, min(1.0, fidelity))),
                level=int(level),
                meta=enriched_meta,
            )
            self._cache_summary(item, text)
            item.summary_of = [item.id]
            self._register_lineage(item)
            self._items.append(item)
            self._enforce_capacity()
            return item, False

    def retrieve(
        self, query_embedding: np.ndarray, top_k: Optional[int] = 6
    ) -> List[MemoryItem]:
        with self._lock:
            now = time.time()
            if not self._items:
                return []
            query_vec = np.asarray(query_embedding, dtype=np.float32)
            scores, similarities = self._score_candidates(self._items, query_vec, now)
            return self._select_top_items(self._items, query_vec, top_k, now, scores, similarities)

    def retrieve_filtered(
        self,
        query_embedding: np.ndarray,
        *,
        tenant_id: str,
        client_id: Optional[str] = None,
        session_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        kinds: Optional[Iterable[str]] = None,
        top_k: Optional[int] = 6,
    ) -> List[MemoryItem]:
        """Retrieve memories scoped by tenant/client/session/instance/kind."""

        with self._lock:
            candidates = [
                item
                for item in self._items
                if self._matches_filters(
                    item,
                    tenant_id=tenant_id,
                    client_id=client_id,
                    session_id=session_id,
                    instance_id=instance_id,
                    kinds=kinds,
                )
            ]
            if not candidates:
                return []
            now = time.time()
            query_vec = np.asarray(query_embedding, dtype=np.float32)
            scores, similarities = self._score_candidates(candidates, query_vec, now)
            return self._select_top_items(
                candidates, query_vec, top_k, now, scores, similarities
            )

    def items(self) -> Sequence[MemoryItem]:
        with self._lock:
            return list(self._items)

    def add(
        self,
        text: str,
        embedding: np.ndarray,
        *,
        salience: float = 1.0,
        fidelity: float = 1.0,
        level: int = 0,
        meta: Optional[Dict] = None,
    ) -> MemoryItem:
        """Convenience wrapper around :meth:`ingest` that ignores merge output."""

        item, _ = self.ingest(
            text,
            embedding,
            salience=salience,
            fidelity=fidelity,
            level=level,
            meta=meta,
        )
        return item

    def retrieve_by_kind(
        self, query_embedding: np.ndarray, kind: str, top_k: Optional[int] = None
    ) -> List[MemoryItem]:
        """Retrieve nodes matching a specific ``meta['kind']`` value."""

        with self._lock:
            candidates = [
                item
                for item in self._items
                if (item.meta or {}).get("kind") == kind
            ]
            if not candidates:
                return []
            now = time.time()
            query_vec = np.asarray(query_embedding, dtype=np.float32)
            scores, similarities = self._score_candidates(candidates, query_vec, now)
            return self._select_top_items(
                candidates, query_vec, top_k, now, scores, similarities
            )

    def export_state(self) -> Dict[str, Any]:
        """Return a JSON serialisable snapshot of the memory lattice."""

        with self._lock:
            return {
                "items": [item.to_dict() for item in self._items],
                "lineage": [item.to_dict() for item in self._lineage.values()],
                "repair_queue": list(self._repair_queue),
                "next_id": self._id,
            }

    def import_state(self, payload: Optional[Dict[str, Any]]) -> None:
        """Restore the lattice from ``payload`` if provided."""

        if not payload:
            return

        items_data = payload.get("items") or []
        reconstructed: List[MemoryItem] = []
        lineage_entries = payload.get("lineage") or []
        lineage_map: Dict[int, MemoryItem] = {}
        for entry in lineage_entries:
            item = self._reconstruct_item(entry)
            if item is not None:
                lineage_map[item.id] = item
        for entry in items_data:
            item = self._reconstruct_item(entry)
            if item is None:
                continue
            reconstructed.append(item)
            lineage_map[item.id] = item

        with self._lock:
            self._items = reconstructed
            self._lineage = lineage_map
            if self._items:
                self._id = max(item.id for item in self._items) + 1
            else:
                self._id = int(payload.get("next_id") or 0)
            existing_queue = payload.get("repair_queue") or []
            self._repair_queue = [
                int(val) for val in existing_queue if int(val) in self._lineage
            ]

    def list_scratch(
        self,
        tenant_id: str,
        client_id: Optional[str],
        session_id: Optional[str],
        instance_id: Optional[str],
    ) -> List[MemoryItem]:
        """Return scratch memories for an instance within a tenant."""

        with self._lock:
            return [
                item
                for item in self._items
                if self._matches_filters(
                    item,
                    tenant_id=tenant_id,
                    client_id=client_id,
                    session_id=session_id,
                    instance_id=instance_id,
                    kinds={"scratch"},
                )
            ]

    def decay_step(self, now: Optional[float] = None) -> None:
        """Public hook used in tests to simulate ageing."""

        with self._lock:
            self._apply_decay(now=now)
            self._abstract_low_fidelity()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _next_id(self) -> int:
        value = self._id
        self._id += 1
        return value

    def _best_match(self, embedding: np.ndarray) -> Tuple[Optional[MemoryItem], float]:
        """Return the most similar existing item and its similarity."""

        if not self._items:
            return None, 0.0
        backend = self._vector_backend
        candidate = np.asarray(embedding, dtype=np.float32)
        key_matrix = np.stack([item.embedding for item in self._items]).astype(
            np.float32
        )
        similarities = backend.cosine_sim_matrix(candidate, key_matrix)[0]
        best_idx = int(np.argmax(similarities)) if similarities.size else -1
        if best_idx < 0:
            return None, 0.0
        return self._items[best_idx], float(similarities[best_idx])

    def _try_merge(
        self,
        text: str,
        embedding: np.ndarray,
        salience: float,
        meta: Optional[Dict],
        *,
        precomputed_match: Tuple[Optional[MemoryItem], float] | None = None,
    ) -> Optional[MemoryItem]:
        if not self._items:
            return None
        backend = self._vector_backend
        if precomputed_match is not None:
            best, best_sim = precomputed_match
        else:
            best_sim = 0.0
            best = None
            candidate = np.asarray(embedding, dtype=np.float32)
            key_matrix = np.stack([item.embedding for item in self._items]).astype(
                np.float32
            )
            similarities = backend.cosine_sim_matrix(candidate, key_matrix)[0]
            best_idx = int(np.argmax(similarities)) if similarities.size else -1
            if best_idx >= 0:
                best = self._items[best_idx]
                best_sim = float(similarities[best_idx])
        if best and best_sim >= self.theta_merge:
            combined_text = f"{best.text}\n{text}".strip()
            summary = self.summarizer.summarize(combined_text, max_len=256)
            summary = summary or combined_text[:253] + "..."
            best.meta["summary"] = summary
            best.text = combined_text
            best.embedding = (best.embedding + embedding) / 2.0
            best.timestamp = time.time()
            best.salience = max(best.salience, salience)
            best.meta.setdefault("merges", 0)
            best.meta["merges"] += 1
            if not best.summary_of:
                best.summary_of = [best.id]
            child = MemoryItem(
                id=self._next_id(),
                text=text,
                embedding=np.asarray(embedding, dtype=np.float32),
                timestamp=time.time(),
                salience=float(salience),
                fidelity=1.0,
                level=0,
                meta=meta or {},
            )
            self._cache_summary(child, text)
            child.summary_of = [child.id]
            self._register_lineage(child)
            for child_id in child.summary_of:
                if child_id not in best.summary_of:
                    best.summary_of.append(child_id)
            self._register_lineage(best)
            return best
        return None

    def _score_item(
        self,
        item: MemoryItem,
        query_embedding: np.ndarray,
        now: float,
        *,
        similarity: Optional[float] = None,
    ) -> float:
        if similarity is None:
            similarity = utils.cosine_similarity(item.embedding, query_embedding)
        age = utils.age_in_hours(item.timestamp, now)
        recency = 1.0 / (1.0 + age)
        return (
            similarity
            + self.eta * recency
            + self.gamma * item.salience
            + self.kappa * item.fidelity
        )

    def _enforce_capacity(self) -> None:
        if len(self._items) <= self.capacity:
            return
        now = time.time()
        self._items.sort(
            key=lambda item: (
                item.fidelity
                + 0.1 * item.salience
                - 0.01 * utils.age_in_hours(item.timestamp, now)
            )
        )
        while len(self._items) > self.capacity:
            self._items.pop(0)

    def _aging_loop(self) -> None:  # pragma: no cover - background thread
        while not self._stop_event.is_set():
            with self._lock:
                self._apply_decay()
                self._abstract_low_fidelity()
            self._stop_event.wait(self._aging_interval_seconds)

    def _apply_decay(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        for item in self._items:
            age = utils.age_in_hours(item.timestamp, now)
            recency = 1.0 / (1.0 + age)
            lambda_star = utils.sigmoid(self.beta_r * recency - self.beta_a * age)
            item.fidelity = float(max(0.0, min(1.0, lambda_star)))
            item.level = min(self.K, max(0, int((1.0 - item.fidelity) * self.K)))

    def _abstract_low_fidelity(self) -> None:
        new_items: List[MemoryItem] = []
        for item in list(self._items):
            if item.fidelity < self.tau_s and item.level < self.K:
                summary_text = self._generate_summary(item.text, max_len=256)
                if not summary_text:
                    summary_text = item.text[:253] + "..."
                summary_embedding = item.embedding.copy()
                new_item = MemoryItem(
                    id=self._next_id(),
                    text=summary_text,
                    embedding=summary_embedding,
                    timestamp=time.time(),
                    salience=item.salience * 0.9,
                    fidelity=min(1.0, item.fidelity + 0.5),
                    level=item.level + 1,
                    meta={"abstracted_from": item.id, "summary": summary_text},
                    summary_of=list({item.id, *item.summary_of}),
                )
                self._register_lineage(new_item)
                item.meta.setdefault("abstracted", True)
                item.meta.setdefault("summary", self._generate_summary(item.text, max_len=256))
                item.fidelity *= 0.5
                new_items.append(new_item)
        self._items.extend(new_items)
        self._enforce_capacity()

    # ------------------------------------------------------------------
    # lineage and quality helpers
    # ------------------------------------------------------------------
    def _register_lineage(self, item: MemoryItem) -> None:
        self._lineage[item.id] = item

    def _reconstruct_item(self, entry: Dict[str, Any]) -> Optional[MemoryItem]:
        try:
            embedding = np.asarray(entry.get("embedding") or [], dtype=np.float32)
            item = MemoryItem(
                id=int(entry.get("id", 0)),
                text=str(entry.get("text") or ""),
                embedding=embedding,
                timestamp=float(entry.get("timestamp") or 0.0),
                salience=float(entry.get("salience") or 0.0),
                fidelity=float(entry.get("fidelity") or 0.0),
                level=int(entry.get("level") or 0),
                meta=entry.get("meta") or {},
                summary_of=list(entry.get("summary_of") or []),
            )
            return item
        except Exception:
            return None

    def _cache_summary(self, item: MemoryItem, text: str) -> None:
        summary = self.summarizer.summarize(text, max_len=256)
        if not summary:
            summary = text[:253] + "..."
        item.meta["summary"] = summary

    def _generate_summary(self, text: str, *, max_len: int) -> str:
        summary = self.summarizer.summarize(text, max_len=max_len)
        if summary:
            return summary
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _score_candidates(
        self, items: Sequence[MemoryItem], query_vec: np.ndarray, now: float
    ) -> tuple[np.ndarray, np.ndarray]:
        key_matrix = np.stack([item.embedding for item in items]).astype(np.float32)
        backend = self._vector_backend
        similarities = backend.cosine_sim_matrix(query_vec, key_matrix)[0]
        ages = np.array(
            [utils.age_in_hours(item.timestamp, now) for item in items], dtype=np.float32
        )
        recency = 1.0 / (1.0 + ages)
        salience = np.array([item.salience for item in items], dtype=np.float32)
        fidelity = np.array([item.fidelity for item in items], dtype=np.float32)
        scores = (
            similarities
            + self.eta * recency
            + self.gamma * salience
            + self.kappa * fidelity
        )
        if self.enable_quality_on_retrieval:
            for item, similarity in zip(items, similarities):
                self._assess_quality(item, query_vec, now, similarity=float(similarity))
        return scores, similarities

    def _select_top_items(
        self,
        candidates: Sequence[MemoryItem],
        query_vec: np.ndarray,
        top_k: Optional[int],
        now: float,
        scores: Optional[np.ndarray] = None,
        similarities: Optional[np.ndarray] = None,
    ) -> List[MemoryItem]:
        """Rank candidates and return the highest scoring items above the similarity floor."""

        if not candidates:
            return []

        if scores is None or similarities is None:
            scores, similarities = self._score_candidates(candidates, query_vec, now)

        mask = similarities >= self.similarity_threshold
        if not bool(np.any(mask)):
            return []

        filtered_items = [item for item, keep in zip(candidates, mask) if keep]
        filtered_scores = scores[mask]
        limit = self._resolve_limit(top_k, len(filtered_scores))
        if limit <= 0:
            return []

        backend = self._vector_backend
        top_indices, _ = backend.top_k(filtered_scores, limit)
        return [filtered_items[idx] for idx in top_indices[0]]

    def _resolve_limit(self, top_k: Optional[int], available: int) -> int:
        if available <= 0:
            return 0
        if top_k is None:
            return available
        try:
            limit = int(top_k)
        except (TypeError, ValueError):
            return available
        if limit <= 0:
            return available
        return min(limit, available)

    def _matches_filters(
        self,
        item: MemoryItem,
        *,
        tenant_id: str,
        client_id: Optional[str],
        session_id: Optional[str],
        instance_id: Optional[str],
        kinds: Optional[Iterable[str]],
    ) -> bool:
        meta = item.meta or {}
        if meta.get("tenant_id") != tenant_id:
            return False
        if kinds is not None:
            allowed = set(kinds)
            if meta.get("kind") not in allowed:
                return False
        if client_id is not None and meta.get("client_id") != client_id:
            return False
        if session_id is not None and meta.get("session_id") != session_id:
            return False
        if instance_id is not None and meta.get("instance_id") != instance_id:
            return False
        return True

    def _assess_quality(
        self,
        item: MemoryItem,
        query_embedding: np.ndarray,
        now: float,
        *,
        similarity: Optional[float] = None,
    ) -> None:
        if similarity is None:
            similarity = utils.cosine_similarity(item.embedding, query_embedding)
        child_embeddings = self._child_embeddings(item.children)
        variance = float(np.var(child_embeddings)) if child_embeddings else 0.0
        age_hours = utils.age_in_hours(item.timestamp, now)
        aging_penalty = age_hours * 0.01
        quality = similarity - variance - aging_penalty
        if quality < self.quality_threshold:
            self._enqueue_repair(item.id)

    def _child_embeddings(self, child_ids: Iterable[int]) -> List[np.ndarray]:
        vectors: List[np.ndarray] = []
        for child_id in child_ids:
            child = self._lineage.get(child_id)
            if child is None:
                continue
            vectors.append(child.embedding)
        return vectors

    def _enqueue_repair(self, item_id: int) -> None:
        if item_id in self._repair_queue:
            return
        self._repair_queue.append(item_id)

    # ------------------------------------------------------------------
    # public maintenance hooks
    # ------------------------------------------------------------------
    def dequeue_repair_batch(self, limit: int = 5) -> List[MemoryItem]:
        with self._lock:
            batch_ids = self._repair_queue[:limit]
            self._repair_queue = self._repair_queue[limit:]
            return [self._lineage.get(idx) for idx in batch_ids if self._lineage.get(idx)]

    def lineage_items(self, ids: Iterable[int]) -> List[MemoryItem]:
        with self._lock:
            return [self._lineage[it] for it in ids if it in self._lineage]

    def update_node(self, item_id: int, *, summary: str, embedding: np.ndarray) -> None:
        with self._lock:
            target = next((it for it in self._items if it.id == item_id), None)
            if target is None:
                return
            target.meta["summary"] = summary
            target.embedding = np.asarray(embedding, dtype=np.float32)
            target.timestamp = time.time()
            self._register_lineage(target)

    def repair_queue(self) -> List[int]:
        with self._lock:
            return list(self._repair_queue)

    def maintenance_pass(self, sample_ratio: float = 0.1) -> None:
        """Run quality assessment on a sampled subset of items.

        This shifts expensive quality/repair checks out of the retrieval hot path.
        """

        ratio = max(0.0, min(1.0, float(sample_ratio)))
        with self._lock:
            if not self._items or ratio <= 0.0:
                return
            now = time.time()
            for item in self._items:
                if random.random() > ratio:
                    continue
                self._assess_quality(item, item.embedding, now)

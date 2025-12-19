"""Concept memory store."""
from __future__ import annotations

import heapq
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .compressors import Keywordizer, Summarizer, VectorQuantizer
from .config import CMAConfig
from .embeddings import Embedder
from .schemas import MemoryItem
from .utils import (
    add_vectors,
    cosine_similarity,
    estimate_tokens,
    normalize,
    rolling_mean,
    sigmoid,
    softmax,
    truncate_tokens,
)


class ConceptMemory:
    """In-memory store implementing the CMA policies."""

    def __init__(
        self,
        embedder: Embedder,
        summarizer: Summarizer,
        keywordizer: Keywordizer,
        quantizer: VectorQuantizer,
        config: CMAConfig,
    ) -> None:
        self.embedder = embedder
        self.summarizer = summarizer
        self.keywordizer = keywordizer
        self.quantizer = quantizer
        self.config = config
        self._items: Dict[str, MemoryItem] = {}
        self._alignment_history: List[float] = []

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add(self, text: str, meta: Optional[Dict[str, object]] = None) -> str:
        now = self.config.now()
        vector = self.embedder.encode(text)[0]
        item = MemoryItem(
            e=vector,
            s=text,
            tau=now,
            r=1.0,
            lam=1.0,
            k=0,
            meta=meta or {},
        )
        self._items[item.id] = item
        return item.id

    def get(self, item_id: str) -> Optional[MemoryItem]:
        return self._items.get(item_id)

    # ------------------------------------------------------------------
    # Aging & compression
    # ------------------------------------------------------------------
    def age_tick(self, now: Optional[float] = None) -> None:
        now = self.config.now() if now is None else now
        vectors = [item.e for item in self._items.values()]
        if vectors and len(vectors) >= self.config.min_vq_items:
            self.quantizer.fit(vectors)
        for item in self._items.values():
            age = now - item.tau
            target = sigmoid(self.config.beta_r * item.r - self.config.beta_a * age)
            item.lam = (1 - self.config.eta) * item.lam + self.config.eta * target
            new_level = self._level_from_lambda(item.lam)
            if new_level > item.k:
                self._compress_item(item, new_level)

    def _level_from_lambda(self, lam: float) -> int:
        level = int(math.floor((1 - lam) * self.config.K))
        return int(max(0, min(self.config.K, level)))

    def _compress_item(self, item: MemoryItem, new_level: int) -> None:
        item.k = new_level
        if new_level == 0:
            item.s = truncate_tokens(item.s, self.config.snippet_token_budget)
            item.code = None
            item.centroid = None
        elif new_level == 1:
            item.s = self.summarizer.summarize(item.s, mode="summary")
            item.code = None
            item.centroid = None
        elif new_level == 2:
            keywords = self.keywordizer.keywords([item.s])
            bullets = [f"- {kw}" for kw in keywords[: self.config.bullet_count]]
            item.s = "\n".join(bullets) or item.s
            if self.quantizer.is_fitted():
                item.code, item.centroid = self.quantizer.encode(item.e)
        else:
            label = "concept"
            if self.quantizer.is_fitted():
                item.code, item.centroid = self.quantizer.encode(item.e)
                label = self.quantizer.label(item.code)
            item.s = f"Concept: {label}\n- abstract summary"

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        now: Optional[float] = None,
        sample: bool = False,
    ) -> List[MemoryItem]:
        if not self._items:
            return []
        now = self.config.now() if now is None else now
        q_vec = self.embedder.encode(query)[0]
        items = list(self._items.values())
        sims = []
        for item in items:
            cos_raw = cosine_similarity(q_vec, item.e)
            if item.centroid is not None:
                cos_comp = cosine_similarity(q_vec, item.centroid)
            else:
                cos_comp = cos_raw
            alpha = self.config.alpha_raw if item.k <= 1 else self.config.alpha_compressed
            blended = alpha * cos_raw + (1 - alpha) * cos_comp
            sims.append(blended)
        prob = softmax(sims, self.config.tau_s)
        scores = []
        for item, sim_prob in zip(items, prob):
            age = now - item.tau
            recency = math.exp(-self.config.gamma * age)
            salience = 1 + self.config.kappa * item.r
            score = sim_prob * recency * salience
            scores.append(score)
        indices = list(range(len(items)))
        if sample:
            weights = list(scores)
            top_k = top_k or self.config.top_k
            rng = random.Random(self.config.random_seed)
            available = list(zip(indices, weights))
            chosen = []
            for _ in range(min(top_k, len(available))):
                total = sum(max(w, 0.0) for _, w in available)
                if total == 0:
                    idx = rng.randrange(len(available))
                else:
                    threshold = rng.random() * total
                    cumulative = 0.0
                    idx = 0
                    for j, (candidate, weight) in enumerate(available):
                        cumulative += max(weight, 0.0)
                        if cumulative >= threshold:
                            idx = j
                            break
                candidate, _ = available.pop(idx)
                chosen.append(candidate)
        else:
            chosen = heapq.nlargest(top_k or self.config.top_k, indices, key=lambda i: scores[i])
        return [items[i] for i in chosen]

    # ------------------------------------------------------------------
    # Reinforcement
    # ------------------------------------------------------------------
    def post_gen_update(self, generated: str, items: Sequence[MemoryItem]) -> None:
        if not items:
            return
        g_vec = self.embedder.encode(generated)[0]
        alignments = []
        for item in items:
            alignment = cosine_similarity(g_vec, item.e)
            alignments.append(alignment)
            delta = alignment - self._alignment_mean()
            item.r += self.config.eta_salience * delta
            item.lam += self.config.eta_fidelity * delta
            item.lam = float(min(max(item.lam, 0.0), 1.0))
        self._alignment_history.extend(alignments)

    def _alignment_mean(self) -> float:
        return rolling_mean(self._alignment_history, self.config.alignment_window)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def merge_similar(self) -> int:
        if len(self._items) < 2:
            return 0
        merged = 0
        ids = list(self._items.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = self._items.get(ids[i])
                b = self._items.get(ids[j])
                if a is None or b is None:
                    continue
                if cosine_similarity(a.e, b.e) > self.config.theta_merge:
                    vector = normalize(add_vectors(a.e, b.e))
                    a.e = vector
                    a.s = f"Merged: {a.s}\n{b.s}"
                    a.r = max(a.r, b.r)
                    a.tau = max(a.tau, b.tau)
                    a.lam = max(a.lam, b.lam)
                    if b.code is not None:
                        a.code = b.code
                        a.centroid = b.centroid
                    del self._items[b.id]
                    merged += 1
                    break
        return merged

    def evict_to_capacity(self, max_items: Optional[int] = None) -> int:
        max_items = max_items or self.config.capacity
        if len(self._items) <= max_items:
            return 0
        scored: List[Tuple[float, str]] = []
        for item in self._items.values():
            score = (
                self.config.gamma * (self.config.now() - item.tau)
                - self.config.kappa * item.r
                + item.k
            )
            scored.append((score, item.id))
        scored.sort(reverse=True)
        removed = 0
        to_remove = scored[: max(0, len(scored) - max_items)]
        for _, item_id in to_remove:
            del self._items[item_id]
            removed += 1
        return removed

    # ------------------------------------------------------------------
    # Presentation
    # ------------------------------------------------------------------
    def build_preamble(
        self,
        items: Sequence[MemoryItem],
        token_budget: Optional[int] = None,
        user_prompt: Optional[str] = None,
    ) -> str:
        token_budget = token_budget or self.config.token_budget
        lines = ["=== Concept Memory ==="]
        used = 0
        for item in items:
            block = self._render_item(item)
            block_tokens = estimate_tokens(block)
            if used + block_tokens > token_budget:
                break
            lines.append(block)
            used += block_tokens
        lines.append("=== User Prompt ===")
        if user_prompt is not None:
            lines.append(user_prompt)
        return "\n".join(lines)

    def _render_item(self, item: MemoryItem) -> str:
        if item.k == 0:
            return item.s
        if item.k == 1:
            return f"Summary: {item.s}"
        if item.k == 2:
            return f"Concept bullets:\n{item.s}"
        label = self.quantizer.label(item.code) if item.code is not None else "concept"
        return f"Concept {label}:\n{item.s}"

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def to_list(self) -> List[Dict[str, object]]:
        return [item.to_dict() for item in self._items.values()]

    def load(self, records: Iterable[Dict[str, object]]) -> None:
        self._items = {record["id"]: MemoryItem.from_dict(record) for record in records}

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._items)

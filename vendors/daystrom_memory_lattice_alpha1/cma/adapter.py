"""High level adapter tying the components together."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import json

from .compressors import DummySummarizer, Keywordizer, VectorQuantizer
from .config import CMAConfig
from .embeddings import RandomEmbedder, SentenceTransformerEmbedder
from .schemas import MemoryItem
from .store import ConceptMemory


class CMAAdapter:
    """Convenience facade for the Concept Memory system."""

    def __init__(
        self,
        config: Optional[CMAConfig] = None,
        storage_path: Optional[Path | str] = None,
    ) -> None:
        self.config = config or CMAConfig()
        embedder = SentenceTransformerEmbedder(seed=self.config.random_seed)
        if getattr(embedder, "_model", None) is None:
            embedder = RandomEmbedder(seed=self.config.random_seed)
        summarizer = DummySummarizer(sentence_count=self.config.summary_sentence_count)
        keywordizer = Keywordizer(max_keywords=self.config.bullet_count)
        quantizer = VectorQuantizer(
            n_codes=self.config.codebook_size,
            random_state=self.config.random_seed,
        )
        self.memory = ConceptMemory(embedder, summarizer, keywordizer, quantizer, self.config)
        self.storage_path = Path(storage_path) if storage_path else None
        if self.storage_path and self.storage_path.exists():
            self._load()

    # ------------------------------------------------------------------
    def ingest(self, text: str, meta: Optional[dict] = None) -> str:
        item_id = self.memory.add(text, meta)
        self._persist()
        return item_id

    def augment_prompt(self, user_prompt: str, top_k: Optional[int] = None, token_budget: Optional[int] = None) -> str:
        self.memory.age_tick()
        items = self.memory.retrieve(user_prompt, top_k=top_k, sample=False)
        preamble = self.memory.build_preamble(items, token_budget=token_budget, user_prompt=user_prompt)
        return preamble

    def reinforce(self, generated: str) -> None:
        items = list(self.memory._items.values())
        self.memory.post_gen_update(generated, items)
        self._persist()

    def stats(self) -> dict:
        return {
            "items": len(self.memory),
            "avg_lambda": float(sum(item.lam for item in self.memory._items.values()) / max(1, len(self.memory))),
        }

    # ------------------------------------------------------------------
    def _persist(self) -> None:
        if not self.storage_path:
            return
        records = self.memory.to_list()
        self.storage_path.write_text(json.dumps(records, indent=2))

    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        records = json.loads(self.storage_path.read_text())
        self.memory.load(records)

    # ------------------------------------------------------------------
    def items(self) -> Iterable[MemoryItem]:
        return self.memory._items.values()

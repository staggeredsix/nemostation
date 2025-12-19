"""Retrieval helpers for the Daystrom Memory Lattice."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from . import utils
from .embeddings import Embedder
from .memory_store import MemoryItem
from .summarizer import Summarizer


@dataclass
class LiteralResult:
    """Container for literal retrieval results."""

    item: MemoryItem
    snippet: str
    context: List[str]
    literal_score: float
    semantic_score: float
    source: str | None

    @property
    def combined_context(self) -> str:
        return "\n".join(self.context)


class LiteralRetriever:
    """Retrieve literal fragments from level-0 documents.

    The retriever prioritises exact phrase matches using regular expressions and
    fallbacks to cosine similarity for fuzzier alignment.  Only small fragments
    are returned to minimise token consumption while still carrying 1-2
    neighbouring fragments for relational context.
    """

    def __init__(
        self,
        embedder: Embedder,
        summarizer: Summarizer,
        *,
        context_window: int = 1,
        max_snippet_chars: int = 320,
    ) -> None:
        self.embedder = embedder
        self.summarizer = summarizer
        self.context_window = max(0, context_window)
        self.max_snippet_chars = max(64, max_snippet_chars)

    def retrieve(
        self,
        query: str,
        items: Sequence[MemoryItem],
        query_embedding: np.ndarray,
        *,
        top_k: int = 4,
    ) -> List[LiteralResult]:
        if not query or not items:
            return []
        phrase_pattern = self._build_pattern(query)
        literal_candidates: List[LiteralResult] = []
        level_zero_items = [item for item in items if item.level == 0]
        search_space = level_zero_items if level_zero_items else list(items)
        doc_index = self._build_doc_index(search_space)
        for item in search_space:
            snippet, regex_boost = self._extract_snippet(item.text, phrase_pattern)
            similarity = utils.cosine_similarity(item.embedding, query_embedding)
            literal_score = self._score_literal(regex_boost, similarity)
            if literal_score == 0.0:
                continue
            context_segments = [snippet] + self._neighbour_context(item, doc_index)
            source = self._resolve_source(item.meta)
            literal_candidates.append(
                LiteralResult(
                    item=item,
                    snippet=snippet,
                    context=context_segments,
                    literal_score=literal_score,
                    semantic_score=similarity,
                    source=source,
                )
            )
        literal_candidates.sort(key=lambda res: res.literal_score, reverse=True)
        return literal_candidates[: max(1, top_k)]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_pattern(self, query: str) -> re.Pattern | None:
        tokens = [tok for tok in re.findall(r"[A-Za-z0-9_]+", query) if len(tok) >= 3]
        if not tokens:
            return None
        pattern = "|".join(re.escape(tok) for tok in tokens)
        return re.compile(pattern, re.IGNORECASE)

    def _extract_snippet(
        self, text: str, pattern: re.Pattern | None
    ) -> tuple[str, float]:
        if not text:
            return "", 0.0
        snippet = text.strip()
        boost = 0.0
        if pattern:
            matches = list(pattern.finditer(text))
            if matches:
                # Take the smallest window around the first match.
                first = matches[0]
                start = max(0, first.start() - 80)
                end = min(len(text), first.end() + 80)
                snippet = text[start:end].strip()
                boost = len(matches)
        if len(snippet) > self.max_snippet_chars:
            snippet = snippet[: self.max_snippet_chars - 3] + "..."
        return snippet, float(boost)

    def _score_literal(self, regex_hits: float, similarity: float) -> float:
        if regex_hits == 0.0 and similarity == 0.0:
            return 0.0
        # Prioritise regex hits but keep cosine similarity as a backstop.
        scaled_regex = min(1.0, regex_hits / 3.0)
        return 0.7 * scaled_regex + 0.3 * max(0.0, similarity)

    def _build_doc_index(
        self, items: Sequence[MemoryItem]
    ) -> Dict[str, List[MemoryItem]]:
        grouped: Dict[str, List[MemoryItem]] = {}
        for item in items:
            source = self._resolve_source(item.meta)
            if source is None:
                continue
            grouped.setdefault(source, []).append(item)
        for collection in grouped.values():
            collection.sort(key=lambda it: it.meta.get("chunk_index", it.id))
        return grouped

    def _neighbour_context(
        self, item: MemoryItem, index: Dict[str, List[MemoryItem]]
    ) -> List[str]:
        source = self._resolve_source(item.meta)
        if source is None or source not in index or self.context_window == 0:
            return []
        neighbours = index[source]
        try:
            pos = neighbours.index(item)
        except ValueError:
            return []
        context_segments: List[str] = []
        for offset in range(1, self.context_window + 1):
            for direction in (-1, 1):
                idx = pos + (offset * direction)
                if 0 <= idx < len(neighbours):
                    text = neighbours[idx].text.strip()
                    if not text:
                        continue
                    if len(text) > self.max_snippet_chars:
                        text = text[: self.max_snippet_chars - 3] + "..."
                    context_segments.append(text)
                    if len(context_segments) >= 2:
                        return context_segments
        return context_segments

    def _resolve_source(self, meta: Dict | None) -> str | None:
        if not meta:
            return None
        for key in ("doc_path", "source", "file", "path"):
            if key in meta and meta[key]:
                return str(meta[key])
        return None

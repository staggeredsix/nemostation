"""Maintenance utilities for repairing low-quality nodes."""
from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np

from . import utils
from .embeddings import Embedder
from .memory_store import MemoryItem, MemoryStore
from .summarizer import Summarizer

LOGGER = logging.getLogger(__name__)


def _rank_lineage(node: MemoryItem, children: Sequence[MemoryItem]) -> List[MemoryItem]:
    ranked = sorted(
        children,
        key=lambda child: utils.cosine_similarity(node.embedding, child.embedding),
        reverse=True,
    )
    return ranked


def run_repair_cycle(
    store: MemoryStore,
    embedder: Embedder,
    summarizer: Summarizer,
    *,
    batch_size: int = 5,
    max_context_chars: int = 800,
) -> int:
    """Process queued nodes and refresh their summaries.

    The repair cycle pulls nodes from the queue, performs a local RAG-style
    fanout over their lineage, and regenerates cached summaries. No lattice
    structure changes occur here, and no LLM calls are made during retrieval.
    """

    repaired = 0
    candidates = store.dequeue_repair_batch(limit=batch_size)
    for node in candidates:
        if node is None:
            continue
        children = store.lineage_items(node.children)
        if not children:
            continue
        ranked_children = _rank_lineage(node, children)
        merged_segments: List[str] = []
        consumed = 0
        for child in ranked_children:
            text = (child.text or "").strip()
            if not text:
                continue
            remaining = max_context_chars - consumed
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = text[: max(3, remaining) - 3] + "..."
            merged_segments.append(text)
            consumed += len(text)
            if consumed >= max_context_chars:
                break
        if not merged_segments:
            continue
        merged_context = "\n".join(merged_segments)
        new_summary = summarizer.summarize(merged_context, max_len=256)
        if not new_summary:
            new_summary = merged_context[:253] + "..."
        new_embedding = embedder.embed(new_summary)
        store.update_node(node.id, summary=new_summary, embedding=np.asarray(new_embedding))
        repaired += 1
    return repaired

from __future__ import annotations

import tempfile
from pathlib import Path

from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.embeddings import RandomEmbedder
from daystrom_dml.summarizer import Summarizer


class RecordingSummarizer(Summarizer):
    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def summarize(self, text: str, max_len: int = 128) -> str:
        self.calls.append((text, max_len))
        return f"summary::{text[:max_len]}"


def make_adapter():
    storage_dir = Path(tempfile.mkdtemp(prefix="dml-summary-cache-"))
    summarizer = RecordingSummarizer()
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "capacity": 20,
            "token_budget": 120,
            "storage_dir": str(storage_dir),
            "persistence": {"enable": False},
            "similarity_threshold": 0.0,
        },
        embedder=RandomEmbedder(dim=32),
        summarizer=summarizer,
        start_aging_loop=False,
    )
    return adapter, summarizer


def test_retrieval_uses_cached_summaries():
    adapter, summarizer = make_adapter()
    adapter.ingest("First memory text with enough content to summarize.")
    adapter.ingest("Second memory text to check caching behavior.")

    # Summaries should be generated during ingest and cached on the memory items.
    items = adapter.store.items()
    assert all(item.meta.get("summary") for item in items)
    initial_calls = len(summarizer.calls)
    assert initial_calls >= 2

    report = adapter.retrieval_report("What memories are stored?")
    assert len(summarizer.calls) == initial_calls

    summary_lookup = {item.id: item.meta.get("summary") for item in items}
    assert all(
        entry["summary"] == summary_lookup.get(entry["id"])
        for entry in report["entries"]
    )

    semantic = adapter.query_database("Summaries please", mode="semantic")
    assert len(summarizer.calls) == initial_calls
    assert semantic["context"]

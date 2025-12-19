from __future__ import annotations

import tempfile
from pathlib import Path

from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.embeddings import RandomEmbedder
from daystrom_dml.summarizer import DummySummarizer


def make_adapter():
    storage_dir = Path(tempfile.mkdtemp(prefix="dml-test-storage-"))
    return DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "capacity": 100,
            "top_k": 4,
            "literal_context": 1,
            "storage_dir": str(storage_dir),
            "persistence": {"enable": False},
            "similarity_threshold": 0.0,
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )


def test_query_database_literal_mode_auto():
    adapter = make_adapter()
    adapter.ingest(
        "# User service documentation",
        meta={"doc_path": "docs/user_api.md", "chunk_index": 0},
    )
    adapter.ingest(
        (
            "def fetchUserProfile(user_id: str) -> dict:\n"
            '    """Fetches a single user profile."""\n'
            '    return client.get(f"/users/{user_id}")'
        ),
        meta={"doc_path": "docs/user_api.md", "chunk_index": 1},
    )
    adapter.ingest(
        "# Related helper",
        meta={"doc_path": "docs/user_api.md", "chunk_index": 2},
    )

    result = adapter.query_database("Show API call to fetchUserProfile")
    assert result["mode"] == "literal"
    assert "fetchUserProfile" in result["context"]
    assert "docs/user_api.md" in result["source_docs"]
    assert result["tokens"] > 0
    assert result["latency_ms"] >= 0


def test_query_database_semantic_summary():
    adapter = make_adapter()
    adapter.ingest(
        "January average temperature was 5C while February averaged 6C.",
        meta={"doc_path": "reports/weather_2023.txt", "chunk_index": 0},
    )
    adapter.ingest(
        "Summer months peaked at 30C on average, cooling to 10C in autumn.",
        meta={"doc_path": "reports/weather_2023.txt", "chunk_index": 1},
    )

    result = adapter.query_database("Summarize average temperatures from reports last year")
    assert result["mode"] == "semantic"
    assert "temperature" in result["context"].lower()
    assert "reports/weather_2023.txt" in result["source_docs"]
    assert result["tokens"] > 0

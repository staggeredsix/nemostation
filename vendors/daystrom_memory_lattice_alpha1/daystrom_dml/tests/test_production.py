import json
from pathlib import Path

import numpy as np
import pytest

from daystrom_dml import server
from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.persistent_index import PersistentVectorIndex
from daystrom_dml.config import load_config
from scripts.benchmark import run_benchmark

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


def build_adapter(tmp_path: Path) -> DMLAdapter:
    storage = tmp_path / "storage"
    return DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(storage),
            "checkpoint_interval_seconds": 0,
        },
        start_aging_loop=False,
    )


def test_checkpoint_creation(tmp_path):
    adapter = build_adapter(tmp_path)
    adapter.ingest("Quantum data bus alignment logs")
    checkpoint_path = adapter.create_checkpoint()
    adapter.close()
    assert checkpoint_path.exists()
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert "dml" in payload and "rag" in payload
    assert payload["stats"]["count"] >= 1


def test_persistent_vector_index_roundtrip(tmp_path):
    index_path = tmp_path / "index.json"
    index = PersistentVectorIndex(index_path)
    vec = np.ones(8, dtype=np.float32)
    index.add(vec, {"text": "alpha", "tokens": 4, "meta": {"source": "doc"}})
    results = index.search(vec, top_k=1)
    assert results and results[0]["meta"]["source"] == "doc"
    reloaded = PersistentVectorIndex(index_path)
    results_reload = reloaded.search(vec, top_k=1)
    assert results_reload and results_reload[0]["text"] == "alpha"


def test_router_auto_mode_prefers_literal(tmp_path):
    adapter = build_adapter(tmp_path)
    adapter.ingest(
        "function fetchUserProfile() calls the /api/users endpoint for account lookup.",
        meta={"doc_path": "app/api.py"},
    )
    report = adapter.query_database("show fetchUserProfile implementation", mode="auto")
    adapter.close()
    assert report["mode"] in {"literal", "hybrid"}
    assert "Source: app/api.py" in report["context"]


def test_metrics_endpoint_returns_payload(monkeypatch):
    monkeypatch.setattr(server, "VISUALIZER_URL", "http://dummy.local")
    with TestClient(server.app) as client:
        response = client.get("/metrics")
    assert response.status_code == 200


def test_visualizer_state_endpoint(monkeypatch):
    monkeypatch.setattr(server, "VISUALIZER_URL", "http://dummy.local")
    monkeypatch.setattr(
        server.visualizer_bridge,
        "latest_prompt",
        lambda: {"prompt": "hello", "timestamp": 1.0},
    )
    with TestClient(server.app) as client:
        response = client.get("/visualizer/state")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["payload"]["prompt"] == "hello"


def test_settings_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("DML_CHECKPOINT_INTERVAL_SECONDS", "15")
    settings = load_config(overrides={"storage_dir": str(tmp_path)})
    assert settings.checkpoint_interval_seconds == 15


def test_benchmark_runner(tmp_path):
    adapter = build_adapter(tmp_path)
    metrics = run_benchmark(adapter, iterations=3)
    adapter.close()
    assert metrics["iterations"] == 3.0
    assert metrics["avg_latency_ms"] >= 0

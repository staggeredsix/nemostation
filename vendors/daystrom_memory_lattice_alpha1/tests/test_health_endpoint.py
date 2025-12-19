from __future__ import annotations

from fastapi.testclient import TestClient

from daystrom_dml.server import app


def test_health_endpoint_reports_core_components() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] in {"ok", "degraded"}
    assert isinstance(payload["timestamp"], str)
    assert isinstance(payload["uptime_seconds"], (int, float))

    components = payload.get("components")
    assert isinstance(components, dict)

    adapter = components.get("adapter")
    assert isinstance(adapter, dict)
    assert adapter.get("status") in {"ok", "error"}
    if adapter.get("status") == "ok":
        assert isinstance(adapter.get("memories"), int)
        assert adapter.get("memories", -1) >= 0
        assert isinstance(adapter.get("storage_dir"), str)
        assert isinstance(adapter.get("rag_backends"), list)
        persistence = adapter.get("persistence")
        assert isinstance(persistence, dict)
        assert "enabled" in persistence
        checkpoints = adapter.get("checkpoints")
        assert isinstance(checkpoints, dict)
        assert "enabled" in checkpoints

    visualizer = components.get("visualizer")
    assert isinstance(visualizer, dict)
    assert visualizer.get("status") in {"external", "running", "idle", "error"}

    nim = components.get("nim")
    assert isinstance(nim, dict)
    assert nim.get("status") in {"unconfigured", "stopped", "starting", "running", "error"}

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from daystrom_dml import server  # noqa: E402


def test_compute_visualizer_extras_includes_faiss_for_supported_versions():
    extras = server._compute_visualizer_extras((3, 11, 5))
    assert "faiss" in extras


def test_compute_visualizer_extras_skips_faiss_for_python312(monkeypatch):
    extras = server._compute_visualizer_extras((3, 12, 0))
    assert "faiss" not in extras


def test_visualizer_redirect_gracefully_handles_launch_failure(monkeypatch):
    def fail_launch() -> None:
        raise HTTPException(status_code=500, detail="pip exploded")

    monkeypatch.setattr(server, "_launch_visualizer_server", fail_launch)

    with TestClient(server.app) as client:
        response = client.get("/visualizer/redirect")

    assert response.status_code == 503
    payload = response.json()
    assert "Visualizer unavailable" in payload["detail"]


def test_visualizer_launch_gracefully_handles_launch_failure(monkeypatch):
    def fail_launch() -> None:
        raise HTTPException(status_code=500, detail="pip exploded")

    monkeypatch.setattr(server, "_launch_visualizer_server", fail_launch)

    with TestClient(server.app) as client:
        response = client.post("/visualizer/launch")

    assert response.status_code == 503
    payload = response.json()
    assert "Visualizer unavailable" in payload["detail"]

from __future__ import annotations

from fastapi.testclient import TestClient

from daystrom_dml.server import app, adapter


def _extract_metric_value(metrics_text: str, prefix: str) -> float:
    for line in metrics_text.splitlines():
        if line.startswith(prefix):
            try:
                return float(line.split()[-1])
            except (ValueError, IndexError):
                continue
    raise AssertionError(f"Metric {prefix!r} not found in payload")


def test_query_updates_prometheus_metrics() -> None:
    client = TestClient(app)

    assert adapter.metrics_enabled, "Metrics must be enabled for the test"

    response = client.post(
        "/ingest",
        json={"text": "The Enterprise explores new worlds.", "meta": {"source": "memory"}},
    )
    assert response.status_code == 200

    query_response = client.post("/query", json={"prompt": "Who explores new worlds?"})
    assert query_response.status_code == 200
    assert query_response.json()["mode"] in {"semantic", "literal", "hybrid", "auto"}

    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    payload = metrics_response.text

    mode_count = _extract_metric_value("\n".join(sorted(payload.splitlines())), "dml_mode_count_total")
    assert mode_count > 0

    tokens_consumed = _extract_metric_value(payload, "dml_tokens_consumed_total")
    assert tokens_consumed > 0

    latency_count = _extract_metric_value(payload, "dml_retrieval_latency_ms_count")
    assert latency_count > 0

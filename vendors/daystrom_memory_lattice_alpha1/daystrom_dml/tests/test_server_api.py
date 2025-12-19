from __future__ import annotations

import pytest

requests = pytest.importorskip("requests")
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from daystrom_dml import server  # noqa: E402


def _client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create a TestClient with visualizer auto-launch disabled."""

    monkeypatch.setattr(server, "VISUALIZER_URL", "http://example.com")
    return TestClient(server.app)


def test_ingest_endpoint_invokes_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubAdapter:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict | None]] = []

        def ingest(self, text: str, meta: dict | None = None) -> None:
            self.calls.append((text, meta))

    stub = StubAdapter()
    monkeypatch.setattr(server, "adapter", stub)

    with _client(monkeypatch) as client:
        response = client.post(
            "/ingest",
            json={"text": "capture this", "meta": {"source": "unit-test"}},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert stub.calls == [("capture this", {"source": "unit-test"})]


def test_reinforce_endpoint_invokes_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubAdapter:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict | None]] = []

        def reinforce(
            self, prompt: str, response: str, meta: dict | None = None
        ) -> None:
            self.calls.append((prompt, response, meta))

    stub = StubAdapter()
    monkeypatch.setattr(server, "adapter", stub)

    with _client(monkeypatch) as client:
        response = client.post(
            "/reinforce",
            json={"text": "keep this", "meta": {"tags": ["test"]}},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert stub.calls == [("", "keep this", {"tags": ["test"]})]


def test_query_endpoint_uses_context_and_records_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubRunner:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return "final answer"

    class StubAdapter:
        def __init__(self) -> None:
            self.runner = StubRunner()
            self.metrics_enabled = True
            self.retrieval_prompts: list[str] = []
            self.reinforcements: list[tuple[str, str]] = []

        def query_database(self, prompt: str) -> dict:
            self.retrieval_prompts.append(prompt)
            return {
                "mode": "literal",
                "context": "System context",
                "tokens": 3,
                "latency_ms": 17,
            }

        def reinforce(self, prompt: str, response: str) -> None:
            self.reinforcements.append((prompt, response))

        def stats(self) -> dict:
            return {"memories": 5}

    stub = StubAdapter()
    monkeypatch.setattr(server, "adapter", stub)

    token_inputs: list[str] = []
    monkeypatch.setattr(
        server.utils,
        "estimate_tokens",
        lambda text: token_inputs.append(text) or 10,
    )

    recorded: list[tuple[int, int]] = []

    def record_tokens(consumed: int, saved: int) -> None:
        recorded.append((consumed, saved))

    monkeypatch.setattr(server, "record_tokens", record_tokens)

    with _client(monkeypatch) as client:
        response = client.post("/query", json={"prompt": "Explain warp drive"})

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "mode": "literal",
        "context": "System context",
        "response": "final answer",
        "stats": {"memories": 5},
    }
    assert stub.retrieval_prompts == ["Explain warp drive"]
    assert stub.runner.prompts == ["System context\n\nExplain warp drive"]
    assert stub.reinforcements == [("Explain warp drive", "final answer")]
    assert token_inputs == ["Explain warp drive"]
    assert recorded == [(13, 7)]


def test_rag_compare_translates_request_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubAdapter:
        def compare_responses(
            self, prompt: str, *, top_k: int | None = None, max_new_tokens: int
        ) -> dict:
            raise requests.RequestException("network down")

    stub = StubAdapter()
    monkeypatch.setattr(server, "adapter", stub)

    queue_calls: list[tuple] = []
    monkeypatch.setattr(
        server.visualizer_bridge,
        "queue_prompt",
        lambda *args, **kwargs: queue_calls.append((args, kwargs)),
    )

    with _client(monkeypatch) as client:
        response = client.post("/rag/compare", json={"prompt": "Check"})

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "NIM backend is unreachable" in detail
    assert queue_calls == []


def test_rag_compare_success_queues_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubAdapter:
        def __init__(self) -> None:
            self.config = {"top_k": 4}
            self.calls: list[tuple[str, int | None, int]] = []

        def compare_responses(
            self, prompt: str, *, top_k: int | None = None, max_new_tokens: int
        ) -> dict:
            self.calls.append((prompt, top_k, max_new_tokens))
            return {"candidates": ["ok"]}

    stub = StubAdapter()
    monkeypatch.setattr(server, "adapter", stub)

    token_inputs: list[str] = []
    monkeypatch.setattr(
        server.utils,
        "estimate_tokens",
        lambda text: token_inputs.append(text) or 12,
    )

    queue_calls: list[tuple] = []
    monkeypatch.setattr(
        server.visualizer_bridge,
        "queue_prompt",
        lambda *args, **kwargs: queue_calls.append((args, kwargs)),
    )

    with _client(monkeypatch) as client:
        response = client.post(
            "/rag/compare",
            json={"prompt": "Assemble", "top_k": 3, "max_new_tokens": 1024},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "candidates": ["ok"],
        "prompt_tokens_est": 12,
    }
    assert stub.calls == [("Assemble", 3, 1024)]
    assert token_inputs == ["Assemble"]
    assert queue_calls == [
        (
            ("Assemble",),
            {
                "top_k": 3,
                "mode": "auto",
                "metadata": {"source": "rag_compare"},
            },
        )
    ]


def test_rag_retrieve_collects_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubRagStore:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def report_all(self, prompt: str, top_k: int) -> dict:
            self.calls.append((prompt, top_k))
            return {"faiss": ["result"]}

    class StubAdapter:
        def __init__(self) -> None:
            self.config = {"top_k": 7}
            self.rag_store = StubRagStore()
            self.reports: list[str] = []

        def retrieval_report(self, prompt: str) -> dict:
            self.reports.append(prompt)
            return {"mode": "literal"}

    stub = StubAdapter()
    monkeypatch.setattr(server, "adapter", stub)

    queue_calls: list[tuple] = []
    monkeypatch.setattr(
        server.visualizer_bridge,
        "queue_prompt",
        lambda *args, **kwargs: queue_calls.append((args, kwargs)),
    )

    with _client(monkeypatch) as client:
        response = client.post("/rag/retrieve", json={"prompt": "Gather data"})

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "prompt": "Gather data",
        "rag_backends": {"faiss": ["result"]},
        "dml": {"mode": "literal"},
    }
    assert stub.rag_store.calls == [("Gather data", 7)]
    assert stub.reports == ["Gather data"]
    assert queue_calls == [
        (
            ("Gather data",),
            {
                "top_k": 7,
                "mode": "auto",
                "metadata": {"source": "rag_retrieve"},
            },
        )
    ]

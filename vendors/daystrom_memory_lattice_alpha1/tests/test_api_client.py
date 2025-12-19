"""Tests for the convenience HTTP client."""
from __future__ import annotations

import pytest

from daystrom_dml.api_client import DMLClient


class DummyResponse:
    def __init__(self, payload, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("HTTP error")

    def json(self):
        return self.payload


class DummySession:
    def __init__(self):
        self.calls: list[tuple[str, str, dict | None, float | None]] = []
        self.closed = False

    def post(self, url: str, *, json=None, timeout=None):
        self.calls.append(("post", url, json, timeout))
        return DummyResponse({"status": "ok", "url": url})

    def get(self, url: str, *, timeout=None):
        self.calls.append(("get", url, None, timeout))
        return DummyResponse({"status": "ok", "url": url})

    def close(self) -> None:
        self.closed = True


def test_client_strips_trailing_slash_and_posts_payload():
    session = DummySession()
    client = DMLClient("http://localhost:1234/", session=session, timeout=5.0)

    response = client.ingest("hello", meta={"source": "unit-test"})

    assert response["status"] == "ok"
    method, url, payload, timeout = session.calls[0]
    assert method == "post"
    assert url == "http://localhost:1234/ingest"
    assert payload == {"text": "hello", "meta": {"source": "unit-test"}}
    assert timeout == 5.0


def test_client_supports_queries_and_gets():
    session = DummySession()
    client = DMLClient("http://localhost:1234", session=session, timeout=None)

    result = client.query("summarise the latest report")
    stats = client.stats()
    knowledge = client.knowledge()

    assert result["status"] == "ok"
    assert stats["status"] == "ok"
    assert knowledge["status"] == "ok"
    assert session.calls[0][1] == "http://localhost:1234/query"
    assert session.calls[1][1] == "http://localhost:1234/stats"
    assert session.calls[2][1] == "http://localhost:1234/knowledge"


def test_client_rejects_non_dict_payloads():
    session = DummySession()
    client = DMLClient("http://localhost:1234", session=session)

    bad_response = DummyResponse(["unexpected"])
    with pytest.raises(ValueError):
        client._json(bad_response)


def test_client_closes_session():
    session = DummySession()
    client = DMLClient("http://localhost:1234", session=session)

    client.close()

    assert session.closed is True

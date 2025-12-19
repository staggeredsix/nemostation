"""Prometheus metrics instrumentation for the Daystrom Memory Lattice."""
from __future__ import annotations

from typing import Iterable, Optional

try:  # pragma: no cover - optional dependency for lean environments
    from prometheus_client import (  # type: ignore
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        CONTENT_TYPE_LATEST,
        generate_latest,
    )
except Exception:  # pragma: no cover - graceful degradation when dependency absent
    CollectorRegistry = None  # type: ignore[assignment]
    Counter = Gauge = Histogram = None  # type: ignore[assignment]
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"  # type: ignore[assignment]

    def generate_latest(_: Optional[CollectorRegistry] = None) -> bytes:  # type: ignore[misc]
        return b""


class _NoOpMetric:
    """Fallback metric implementation used when prometheus_client is unavailable."""

    def __init__(self, *_, **__):
        pass

    def inc(self, *_: float, **__: float) -> None:
        return

    def observe(self, *_: float, **__: float) -> None:
        return

    def set(self, *_: float, **__: float) -> None:
        return

    def labels(self, **__: str) -> "_NoOpMetric":
        return self


if CollectorRegistry is not None:  # pragma: no cover - executed when dependency present
    REGISTRY = CollectorRegistry()
else:  # pragma: no cover - fallback when dependency absent
    REGISTRY = None  # type: ignore[assignment]


def _build_counter(
    name: str, documentation: str, labels: Optional[Iterable[str]] = None
):
    if CollectorRegistry is None:
        return _NoOpMetric()
    return Counter(name, documentation, labelnames=tuple(labels or ()), registry=REGISTRY)


def _build_histogram(
    name: str,
    documentation: str,
    buckets: Optional[Iterable[float]] = None,
):
    if CollectorRegistry is None:
        return _NoOpMetric()
    return Histogram(
        name,
        documentation,
        buckets=tuple(buckets or ()),
        registry=REGISTRY,
    )


def _build_gauge(
    name: str,
    documentation: str,
    labels: Optional[Iterable[str]] = None,
):
    if CollectorRegistry is None:
        return _NoOpMetric()
    return Gauge(name, documentation, labelnames=tuple(labels or ()), registry=REGISTRY)


RETRIEVAL_LATENCY = _build_histogram(
    "dml_retrieval_latency_ms",
    "Latency of retrieval operations in milliseconds.",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
)
MODE_COUNTER = _build_counter(
    "dml_mode_count",
    "Number of retrieval queries handled per mode.",
    labels=["mode"],
)
TOKENS_CONSUMED = _build_counter(
    "dml_tokens_consumed",
    "Total tokens consumed when serving queries.",
)
TOKENS_SAVED = _build_counter(
    "dml_tokens_saved",
    "Estimated tokens saved due to Daystrom retrieval.",
)
DML_ITEMS = _build_gauge(
    "dml_items",
    "Number of items currently stored in the lattice.",
)


def record_retrieval(mode: str, latency_ms: float) -> None:
    """Record latency and mode information for a retrieval."""

    MODE_COUNTER.labels(mode=mode).inc()
    RETRIEVAL_LATENCY.observe(max(float(latency_ms), 0.0))


def record_tokens(consumed: int, saved: int) -> None:
    """Increment token consumption and savings counters."""

    if consumed > 0:
        TOKENS_CONSUMED.inc(consumed)
    if saved > 0:
        TOKENS_SAVED.inc(saved)


def update_memory_gauge(count: int) -> None:
    """Update the gauge tracking the number of stored items."""

    DML_ITEMS.set(max(0, int(count)))


def latest_metrics() -> tuple[bytes, str]:
    """Return the latest metrics payload and content type."""

    if CollectorRegistry is None:
        return b"", CONTENT_TYPE_LATEST
    payload = generate_latest(REGISTRY)
    return payload, CONTENT_TYPE_LATEST


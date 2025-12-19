"""Utility helpers for CMA."""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return float(1 / (1 + z))
    z = math.exp(x)
    return float(z / (1 + z))


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors."""
    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0
    return dot(a, b) / denom


def softmax(values: Sequence[float], temperature: float) -> List[float]:
    """Softmax with temperature."""
    arr = list(float(v) for v in values)
    if not arr:
        return []
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [v / temperature for v in arr]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    total = sum(exps)
    if total == 0:
        return [0.0 for _ in exps]
    return [v / total for v in exps]


def add_vectors(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def scale_vector(v: Sequence[float], scalar: float) -> List[float]:
    return [x * scalar for x in v]


def normalize(v: Sequence[float]) -> List[float]:
    n = norm(v)
    if n == 0:
        return list(v)
    return [x / n for x in v]


def estimate_tokens(text: str) -> int:
    """Estimate token usage (fallback heuristic)."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:  # pragma: no cover - fallback path
        return max(1, len(text) // 4)


def truncate_tokens(text: str, limit: int) -> str:
    """Truncate text to an approximate token limit."""
    if limit <= 0:
        return ""
    tokens = text.split()
    if len(tokens) <= limit:
        return text
    return " ".join(tokens[:limit])


def rolling_mean(values: Iterable[float], window: int) -> float:
    """Compute a simple rolling mean over the most recent values."""
    vals = list(values)
    if not vals:
        return 0.0
    if window <= 0 or len(vals) <= window:
        return float(sum(vals) / len(vals))
    return float(sum(vals[-window:]) / window)

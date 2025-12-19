"""Utility helpers for the Daystrom Memory Lattice."""
from __future__ import annotations

import math
import time
from typing import Iterable, List, Sequence

import numpy as np


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    The implementation is robust to zero vectors and returns ``0`` when either
    vector does not contain any magnitude.  The function accepts any
    ``numpy.ndarray``-like inputs and converts them into ``float32`` arrays for
    deterministic behaviour.
    """

    if vec_a is None or vec_b is None:
        return 0.0
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softmax(values: Iterable[float]) -> List[float]:
    """Return softmax of the input iterable."""

    vals = np.asarray(list(values), dtype=np.float64)
    if vals.size == 0:
        return []
    vals = vals - np.max(vals)
    exps = np.exp(vals)
    denom = np.sum(exps)
    if denom == 0:
        return [0.0 for _ in exps]
    return list(exps / denom)


def estimate_tokens(text: str) -> int:
    """Rough token estimation using a GPT-2 style heuristic."""

    if not text:
        return 0
    # Empirical: 1 token ~ 4 characters for english-like text.
    return max(1, int(len(text) / 4))


def age_in_hours(timestamp: float, now: float | None = None) -> float:
    """Return age in hours from ``timestamp`` to ``now``."""

    now = time.time() if now is None else now
    return max(0.0, (now - timestamp) / 3600.0)


def ensure_serializable(array: np.ndarray) -> List[float]:
    """Convert numpy array to plain python list for JSON serialization."""

    return np.asarray(array, dtype=np.float32).tolist()


def seeded_random_vector(dim: int, seed: int) -> np.ndarray:
    """Return a deterministic pseudo-random vector for offline embeddings."""

    rng = np.random.default_rng(seed)
    return rng.normal(size=dim).astype(np.float32)


def chunk_text(
    text: str,
    *,
    max_tokens: int = 480,
    overlap: int = 40,
) -> List[str]:
    """Split ``text`` into token-aware chunks.

    The helper performs a greedy segmentation using the coarse
    :func:`estimate_tokens` heuristic.  Each chunk overlaps with the previous
    one by ``overlap`` tokens to preserve context continuity.
    """

    if not text:
        return []
    max_tokens = max(1, int(max_tokens))
    overlap = max(0, min(int(overlap), max_tokens - 1))
    paragraphs = [
        para.strip()
        for para in text.replace("\r", "").split("\n\n")
        if para.strip()
    ]
    chunks: List[str] = []
    buffer: List[str] = []
    buffer_tokens = 0
    for para in paragraphs:
        segments = _split_paragraph(para, max_tokens)
        for segment in segments:
            segment_tokens = estimate_tokens(segment)
            if buffer and buffer_tokens + segment_tokens > max_tokens:
                chunks.append("\n\n".join(buffer).strip())
                if overlap:
                    buffer = _truncate_to_overlap(buffer, overlap)
                    buffer_tokens = sum(estimate_tokens(part) for part in buffer)
                else:
                    buffer = []
                    buffer_tokens = 0
            buffer.append(segment)
            buffer_tokens += segment_tokens
    if buffer:
        chunks.append("\n\n".join(buffer).strip())
    return chunks


def _truncate_to_overlap(parts: Sequence[str], overlap_tokens: int) -> List[str]:
    """Return tail of ``parts`` whose tokens sum to ``overlap_tokens``."""

    tail: List[str] = []
    total = 0
    for piece in reversed(parts):
        tokens = estimate_tokens(piece)
        if tokens > overlap_tokens and not tail:
            truncated = _split_paragraph(piece, overlap_tokens)
            if truncated:
                last = truncated[-1]
                tail.append(last)
                total += estimate_tokens(last)
            break
        if total + tokens > overlap_tokens and tail:
            break
        tail.append(piece)
        total += tokens
        if total >= overlap_tokens:
            break
    return list(reversed(tail))


def _split_paragraph(paragraph: str, max_tokens: int) -> List[str]:
    """Split a paragraph into smaller segments if it exceeds ``max_tokens``."""

    tokens = estimate_tokens(paragraph)
    if tokens <= max_tokens:
        return [paragraph]
    words = paragraph.split()
    if not words:
        return [paragraph]
    segments: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for word in words:
        word_tokens = estimate_tokens(word + " ")
        if current and current_tokens + word_tokens > max_tokens:
            segments.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(word)
        current_tokens += word_tokens
    if current:
        segments.append(" ".join(current))
    return segments

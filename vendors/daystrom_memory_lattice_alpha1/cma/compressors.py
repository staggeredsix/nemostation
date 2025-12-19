"""Compression utilities."""
from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Protocol, Sequence


class Summarizer(Protocol):
    """Protocol for text summarisation."""

    def summarize(self, text: str, mode: str = "summary") -> str:
        ...


@dataclass
class DummySummarizer:
    """Trivial summariser based on simple heuristics."""

    sentence_count: int = 2

    def summarize(self, text: str, mode: str = "summary") -> str:
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        if mode == "summary":
            return ". ".join(sentences[: self.sentence_count])[: len(text)]
        return text


class Keywordizer:
    """Extract keywords using a simple frequency heuristic."""

    def __init__(self, max_keywords: int = 5):
        self.max_keywords = max_keywords
        self._token_re = re.compile(r"[A-Za-z][A-Za-z0-9]+")

    def keywords(self, documents: Iterable[str]) -> List[str]:
        docs = list(documents)
        if not docs:
            return []
        tokens = [token.lower() for doc in docs for token in self._token_re.findall(doc)]
        counts = Counter(tokens)
        common = counts.most_common(self.max_keywords)
        return [word for word, _ in common]


class VectorQuantizer:
    """Lightweight KMeans-style vector quantizer."""

    def __init__(self, n_codes: int, dim: int | None = None, random_state: int | None = None, max_iter: int = 10):
        self.n_codes = n_codes
        self.dim = dim
        self.random_state = random_state
        self.max_iter = max_iter
        self._rng = random.Random(random_state)
        self._centroids: List[List[float]] = []

    def fit(self, vectors: Iterable[Sequence[float]]) -> None:
        vectors = [list(vec) for vec in vectors]
        if not vectors:
            return
        self.dim = len(vectors[0])
        k = min(self.n_codes, len(vectors))
        self._centroids = [list(vectors[i]) for i in range(k)]
        for _ in range(self.max_iter):
            clusters: List[List[List[float]]] = [[] for _ in range(k)]
            for vec in vectors:
                idx = self._nearest_index(vec)
                clusters[idx].append(vec)
            new_centroids: List[List[float]] = []
            for idx, cluster in enumerate(clusters):
                if cluster:
                    averaged = [sum(values) / len(cluster) for values in zip(*cluster)]
                    new_centroids.append(averaged)
                else:
                    new_centroids.append(list(self._rng.choice(vectors)))
            if all(self._distance(a, b) < 1e-6 for a, b in zip(self._centroids, new_centroids)):
                break
            self._centroids = new_centroids

    def _distance(self, a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _nearest_index(self, vec: Sequence[float]) -> int:
        if not self._centroids:
            return 0
        distances = [self._distance(vec, centroid) for centroid in self._centroids]
        return int(min(range(len(distances)), key=lambda i: distances[i]))

    def is_fitted(self) -> bool:
        return bool(self._centroids)

    def encode(self, vector: Sequence[float]) -> tuple[int, List[float]]:
        if not self._centroids:
            raise RuntimeError("VectorQuantizer not fitted")
        idx = self._nearest_index(vector)
        return idx, list(self._centroids[idx])

    def decode(self, code: int) -> List[float]:
        if not self._centroids:
            raise RuntimeError("VectorQuantizer not fitted")
        return list(self._centroids[code])

    def label(self, code: int) -> str:
        return f"code_{code}"

"""Backend abstractions for vector similarity operations.

The module exposes a small interface used by the DML retrieval components to
compute cosine similarity matrices and select top-k results.  Two
implementations are provided:

``NumpyVectorBackend``
    Always available and relies on NumPy for computation.  This is the default
    backend when CUDA acceleration is unavailable.

``CUDAVectorBackend``
    Thin wrapper around the optional ``daystrom_dml._cuda_backend`` extension
    compiled with CUDA.  If the extension cannot be imported, the system falls
    back to the NumPy implementation transparently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

import numpy as np


class VectorBackend(Protocol):
    """Protocol describing the minimal vector operations used by DML."""

    def cosine_sim_matrix(self, queries: np.ndarray, keys: np.ndarray) -> np.ndarray:
        """Return cosine similarity for each query/key pair.

        The inputs are expected to be two-dimensional ``float32`` arrays with
        shape ``(num_queries, dim)`` and ``(num_keys, dim)`` respectively.  Any
        zero vectors yield a similarity of ``0``.
        """

    def top_k(self, scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return indices and scores of the top ``k`` entries per query row."""


@dataclass(frozen=True)
class NumpyVectorBackend(VectorBackend):
    """NumPy implementation used as the default and fallback backend."""

    def cosine_sim_matrix(self, queries: np.ndarray, keys: np.ndarray) -> np.ndarray:
        q = _prepare_matrix(queries)
        k = _prepare_matrix(keys)
        if q.shape[1] != k.shape[1]:
            raise ValueError("Query and key dimensions must match")
        q_norms = np.linalg.norm(q, axis=1, keepdims=True)
        k_norms = np.linalg.norm(k, axis=1, keepdims=True)
        # Avoid division by zero; zero vectors receive a zero similarity score.
        q_safe = np.where(q_norms == 0, 0.0, q)
        k_safe = np.where(k_norms == 0, 0.0, k)
        denom = np.clip(q_norms, a_min=np.finfo(np.float32).eps, a_max=None) * np.clip(
            k_norms.T, a_min=np.finfo(np.float32).eps, a_max=None
        )
        sim = np.matmul(q_safe, k_safe.T) / denom
        # Zero out rows/columns where an input vector had zero magnitude.
        sim = np.where((q_norms == 0) | (k_norms.T == 0), 0.0, sim)
        return sim.astype(np.float32)

    def top_k(self, scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        matrix = _prepare_scores(scores)
        if k <= 0:
            empty_idx = np.zeros((matrix.shape[0], 0), dtype=np.int64)
            empty_scores = np.zeros((matrix.shape[0], 0), dtype=np.float32)
            return empty_idx, empty_scores
        k = min(int(k), matrix.shape[1])
        # Use argpartition for efficiency then fully sort the top-k slice.
        partition_indices = np.argpartition(matrix, -k, axis=1)[:, -k:]
        partition_scores = np.take_along_axis(matrix, partition_indices, axis=1)
        order = np.argsort(partition_scores, axis=1)[:, ::-1]
        top_indices = np.take_along_axis(partition_indices, order, axis=1)
        top_scores = np.take_along_axis(matrix, top_indices, axis=1)
        return top_indices.astype(np.int64), top_scores.astype(np.float32)


@dataclass(frozen=True)
class CUDAVectorBackend(VectorBackend):
    """Wrapper around the optional CUDA extension."""

    module: object

    def cosine_sim_matrix(self, queries: np.ndarray, keys: np.ndarray) -> np.ndarray:
        q = _prepare_matrix(queries)
        k = _prepare_matrix(keys)
        return self.module.cosine_sim_matrix(q, k)

    def top_k(self, scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        matrix = _prepare_scores(scores)
        return self.module.top_k(matrix, int(k))


_BACKEND: VectorBackend | None = None


def get_vector_backend() -> VectorBackend:
    """Return the best available vector backend.

    The function attempts to import the CUDA extension and will fall back to the
    NumPy implementation if the import fails for any reason.
    """

    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:  # pragma: no cover - exercised only when CUDA is available
        from . import _cuda_backend

        _BACKEND = CUDAVectorBackend(_cuda_backend)
    except Exception:  # pragma: no cover - intentional fallback path
        _BACKEND = NumpyVectorBackend()
    return _BACKEND


def _prepare_matrix(array: np.ndarray) -> np.ndarray:
    matrix = np.asarray(array, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D array")
    return np.ascontiguousarray(matrix)


def _prepare_scores(array: np.ndarray) -> np.ndarray:
    scores = np.asarray(array, dtype=np.float32)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    if scores.ndim != 2:
        raise ValueError("Scores must be a 2D array")
    return np.ascontiguousarray(scores)

import numpy as np
import pytest

from daystrom_dml.vector_backend import (
    CUDAVectorBackend,
    NumpyVectorBackend,
    get_vector_backend,
)


def test_numpy_backend_cosine_and_topk() -> None:
    backend = NumpyVectorBackend()
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    keys = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    sims = backend.cosine_sim_matrix(queries, keys)
    expected = np.array([[1.0, 0.70710677], [0.0, 0.70710677]], dtype=np.float32)
    assert np.allclose(sims, expected, atol=1e-5)

    indices, scores = backend.top_k(sims, k=1)
    assert indices.shape == (2, 1)
    assert scores.shape == (2, 1)
    assert np.all(indices == np.array([[0], [1]]))
    assert np.allclose(scores[:, 0], np.array([1.0, 0.70710677], dtype=np.float32))


def test_cuda_matches_numpy_when_available() -> None:
    try:
        import daystrom_dml._cuda_backend as cuda_module
    except Exception:
        pytest.skip("CUDA backend is not available in this environment")

    numpy_backend = NumpyVectorBackend()
    cuda_backend = CUDAVectorBackend(cuda_module)
    rng = np.random.default_rng(seed=42)
    queries = rng.standard_normal((3, 8), dtype=np.float32)
    keys = rng.standard_normal((5, 8), dtype=np.float32)

    numpy_sims = numpy_backend.cosine_sim_matrix(queries, keys)
    cuda_sims = cuda_backend.cosine_sim_matrix(queries, keys)
    assert np.allclose(numpy_sims, cuda_sims, atol=1e-4)

    numpy_indices, numpy_scores = numpy_backend.top_k(numpy_sims, k=2)
    cuda_indices, cuda_scores = cuda_backend.top_k(cuda_sims, k=2)
    assert np.all(numpy_indices == cuda_indices)
    assert np.allclose(numpy_scores, cuda_scores, atol=1e-4)


def test_get_vector_backend_returns_backend_instance() -> None:
    backend = get_vector_backend()
    assert hasattr(backend, "cosine_sim_matrix")
    assert hasattr(backend, "top_k")

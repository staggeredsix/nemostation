import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "daystrom_dml" / "rag_store.py"
SPEC = importlib.util.spec_from_file_location("daystrom_dml.rag_store", MODULE_PATH)
assert SPEC and SPEC.loader  # pragma: no cover - sanity check for test setup
rag_module = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("daystrom_dml.rag_store", rag_module)
SPEC.loader.exec_module(rag_module)
PersistentRAGStore = rag_module.PersistentRAGStore


@pytest.fixture(scope="module", autouse=True)
def _require_faiss():
    pytest.importorskip("faiss")


def _vec(values):
    return np.asarray(values, dtype=np.float32)


def test_roundtrip_add_and_search(tmp_path):
    index_path = tmp_path / "index.faiss"
    meta_path = tmp_path / "meta.json"
    store = PersistentRAGStore(enable=True, index_path=index_path, meta_path=meta_path, dim=3)
    alpha_id = store.add("def alpha():\n    return 1", _vec([1.0, 0.0, 0.0]), {"source": "alpha.py"})
    beta_id = store.add("def beta():\n    return 2", _vec([0.0, 1.0, 0.0]), {"source": "beta.py"})
    assert alpha_id != beta_id
    results = store.search(_vec([1.0, 0.0, 0.0]), top_k=1)
    assert results
    top = results[0]
    assert top["id"] == alpha_id
    assert top["meta"]["source"] == "alpha.py"
    assert "return 1" in top["text"]


def test_persist_and_reload(tmp_path):
    index_path = tmp_path / "store.faiss"
    meta_path = tmp_path / "store.json"
    store = PersistentRAGStore(enable=True, index_path=index_path, meta_path=meta_path, dim=3)
    store.add("class Gamma:\n    pass", _vec([0.0, 0.0, 1.0]), {"source": "gamma.py"})
    store.add("class Delta:\n    pass", _vec([0.7, 0.1, 0.2]), {"source": "delta.py"})
    store.persist()

    restored = PersistentRAGStore(enable=True, index_path=index_path, meta_path=meta_path, dim=3)
    restored.load()
    query = _vec([0.65, 0.15, 0.2])
    results = restored.search(query, top_k=1)
    assert results
    assert results[0]["meta"]["source"] == "delta.py"
    assert "Delta" in results[0]["text"]

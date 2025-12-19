import numpy as np

from daystrom_dml.maintenance import run_repair_cycle
from daystrom_dml.memory_store import MemoryStore
from daystrom_dml.summarizer import Summarizer


def make_store(summarizer: Summarizer) -> MemoryStore:
    return MemoryStore(
        summarizer=summarizer,
        beta_a=0.08,
        beta_r=0.2,
        eta=0.15,
        gamma=0.02,
        kappa=0.5,
        tau_s=0.3,
        theta_merge=0.92,
        K=4,
        capacity=20,
        start_aging_loop=False,
        enable_quality_on_retrieval=True,
        similarity_threshold=0.0,
    )


class CountingSummarizer(Summarizer):
    def __init__(self) -> None:
        self.calls = 0

    def summarize(self, text: str, max_len: int = 128) -> str:  # pragma: no cover - trivial
        self.calls += 1
        return f"summary:{text[:max_len//2]}"


class StaticEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str):  # pragma: no cover - trivial
        self.calls += 1
        return np.ones(8, dtype=np.float32)


def test_cached_summary_used_during_retrieval():
    summarizer = CountingSummarizer()
    store = make_store(summarizer)
    embedding = np.ones(8, dtype=np.float32)
    item, _ = store.ingest("Sample memory", embedding)
    assert summarizer.calls == 1
    retrieved = store.retrieve(embedding, top_k=1)
    assert retrieved[0].cached_summary() == item.meta["summary"]
    assert summarizer.calls == 1  # retrieval must not re-summarize


def test_repair_cycle_refreshes_summary():
    summarizer = CountingSummarizer()
    embedder = StaticEmbedder()
    store = make_store(summarizer)
    store.quality_threshold = 1.1
    emb_one = np.ones(8, dtype=np.float32)
    emb_two = np.ones(8, dtype=np.float32) * 0.5
    item, _ = store.ingest("First raw memory", emb_one)
    store.ingest("Second detail", emb_two)
    store.retrieve(emb_one, top_k=2)
    assert store.repair_queue()
    repaired = run_repair_cycle(store, embedder, summarizer, batch_size=2)
    assert repaired >= 1
    updated = next(it for it in store.items() if it.id == item.id)
    assert updated.meta.get("summary", "").startswith("summary:")
    assert embedder.calls >= 1


def test_repair_cycle_preserves_raw_text():
    summarizer = CountingSummarizer()
    embedder = StaticEmbedder()
    store = make_store(summarizer)
    store.quality_threshold = 1.1

    original_text = "Detailed original memory"
    emb_one = np.ones(8, dtype=np.float32)
    emb_two = np.array([1, 0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    item, _ = store.ingest(original_text, emb_one)
    store.ingest("Additional context", emb_two)

    store.retrieve(emb_one, top_k=2)
    repaired = run_repair_cycle(store, embedder, summarizer, batch_size=2)

    assert repaired >= 1
    updated = next(it for it in store.items() if it.id == item.id)
    assert updated.text == original_text

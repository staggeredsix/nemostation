from __future__ import annotations

import math
from typing import List

from cma.compressors import DummySummarizer, Keywordizer, VectorQuantizer
from cma.config import CMAConfig
from cma.embeddings import RandomEmbedder
from cma.store import ConceptMemory


def make_memory(seed: int = 0, **config_overrides) -> ConceptMemory:
    config = CMAConfig(**config_overrides)
    embedder = RandomEmbedder(dim=32, seed=seed)
    summarizer = DummySummarizer(sentence_count=1)
    keywordizer = Keywordizer(max_keywords=3)
    quantizer = VectorQuantizer(n_codes=4, random_state=seed)
    return ConceptMemory(embedder, summarizer, keywordizer, quantizer, config)


def test_embedding_shape() -> None:
    memory = make_memory()
    vector = memory.embedder.encode("hello world")
    assert len(vector) == 1
    assert len(vector[0]) == 32


def test_add_and_retrieve_roundtrip() -> None:
    memory = make_memory(beta_a=0.01, beta_r=0.2, tau_s=0.5)
    memory.add("We adopted a cat named Pixel.")
    memory.add("PCIe Gen5 lanes bottleneck observed.")
    results = memory.retrieve("cat behavior", top_k=1)
    assert len(results) == 1
    assert "cat" in results[0].s.lower()


def test_age_tick_advances_levels() -> None:
    # Provide stable time source
    times: List[float] = [0.0]

    def time_provider() -> float:
        return times[-1]

    memory = make_memory(beta_a=0.5, beta_r=0.1, eta=1.0, min_vq_items=100, time_provider=time_provider)
    memory.add("Test memory for compression.")
    item = next(iter(memory._items.values()))
    assert item.k == 0
    times.append(10.0)
    memory.age_tick()
    assert item.k >= 1


def test_build_preamble_budget() -> None:
    memory = make_memory()
    memory.add("This is a longish snippet about neural nets and transformers.")
    preamble = memory.build_preamble(memory.retrieve("transformers"), token_budget=30, user_prompt="Hi")
    assert preamble.endswith("Hi")


def test_merge_and_evict() -> None:
    memory = make_memory(theta_merge=-1.0, capacity=1)
    memory.add("Item one about the same topic.")
    memory.add("Item two about the same topic.")
    merged = memory.merge_similar()
    assert merged >= 1
    removed = memory.evict_to_capacity()
    assert removed >= 0
    assert len(memory) <= 1


def test_evict_to_capacity_prefers_stale_items() -> None:
    times = [0.0]

    def time_provider() -> float:
        return times[-1]

    memory = make_memory(capacity=2, gamma=1.0, kappa=0.0, time_provider=time_provider)
    stale_id = memory.add("Stale entry")
    times.append(5.0)
    fresh_id = memory.add("Fresher entry")
    times.append(6.0)
    newest_id = memory.add("Newest entry")
    times.append(50.0)
    removed = memory.evict_to_capacity()
    remaining = set(memory._items.keys())
    assert removed == 1
    assert stale_id not in remaining
    assert newest_id in remaining
    assert fresh_id in remaining


def test_reinforcement_updates_values() -> None:
    memory = make_memory()
    memory.add("Memory snippet for reinforcement.")
    items = list(memory._items.values())
    before_r = items[0].r
    memory.post_gen_update("reinforcement text", items)
    assert not math.isclose(items[0].r, before_r)

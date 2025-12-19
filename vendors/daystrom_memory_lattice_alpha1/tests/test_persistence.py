"""Tests for the durable persistence helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from daystrom_dml.memory_store import MemoryItem
from daystrom_dml.persistence import load_state, save_state


def _create_item(idx: int) -> MemoryItem:
    return MemoryItem(
        id=idx,
        text=f"memory-{idx}",
        embedding=np.asarray([idx * 0.1, idx * 0.2, idx * 0.3], dtype=np.float32),
        timestamp=1700000000.0 + idx,
        salience=0.5 + idx,
        fidelity=0.9 - (idx * 0.1),
        level=idx % 2,
        meta={"source": "test", "index": idx},
        summary_of=[idx - 1] if idx else [],
    )


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    items = [_create_item(0), _create_item(1)]
    target = tmp_path / "state.jsonl"

    saved_path = save_state(items, target)
    assert saved_path.exists()

    # Ensure header contains checksum metadata
    header = json.loads(saved_path.read_text(encoding="utf-8").splitlines()[0])
    assert header["version"] == 1
    assert header["count"] == len(items)

    loaded = load_state(saved_path)
    assert len(loaded) == len(items)

    original = items[1]
    restored = loaded[1]
    assert restored.id == original.id
    assert restored.text == original.text
    assert restored.level == original.level
    assert restored.summary_of == original.summary_of
    np.testing.assert_allclose(restored.embedding, original.embedding)


def test_load_state_missing_file(tmp_path: Path) -> None:
    assert load_state(tmp_path / "missing.jsonl") == []

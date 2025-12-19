"""Durable persistence helpers for the Daystrom Memory Lattice."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

import numpy as np

from . import utils
from .memory_store import MemoryItem

_PERSISTENCE_VERSION = 1
_PERSISTENCE_TYPE = "daystrom_dml.memory"


def _item_to_record(item: MemoryItem) -> dict:
    record = {
        "id": item.id,
        "text": item.text,
        "level": item.level,
        "fidelity": item.fidelity,
        "salience": item.salience,
        "timestamp": item.timestamp,
        "meta": item.meta or {},
        "summary_of": list(item.summary_of or []),
        "embedding": utils.ensure_serializable(item.embedding),
    }
    return record


def save_state(items: Sequence[MemoryItem], path: str | Path) -> Path:
    """Persist ``items`` to ``path`` as newline delimited JSON."""

    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    target.parent.mkdir(parents=True, exist_ok=True)
    records = [_item_to_record(item) for item in items]
    payload_lines = [json.dumps(record, separators=(",", ":"), sort_keys=True) for record in records]
    payload_bytes = "\n".join(payload_lines).encode("utf-8")
    checksum = hashlib.sha256(payload_bytes).hexdigest()
    header = {
        "type": _PERSISTENCE_TYPE,
        "version": _PERSISTENCE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "count": len(records),
        "checksum": checksum,
    }
    header_line = json.dumps(header, separators=(",", ":"), sort_keys=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp") if target.suffix else target.with_name(target.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(header_line)
        if payload_lines:
            handle.write("\n")
            handle.write("\n".join(payload_lines))
    tmp_path.replace(target)
    return target


def load_state(path: str | Path) -> List[MemoryItem]:
    """Load persisted memories from ``path``."""

    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]
    if not lines:
        return []
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:  # pragma: no cover - invalid persistence header
        raise ValueError("Invalid persistence header") from exc
    if header.get("type") != _PERSISTENCE_TYPE:
        raise ValueError(f"Unsupported persistence payload: {header.get('type')}")
    data_lines = lines[1:]
    expected_checksum = header.get("checksum")
    payload_bytes = "\n".join(data_lines).encode("utf-8")
    actual_checksum = hashlib.sha256(payload_bytes).hexdigest()
    if expected_checksum and expected_checksum != actual_checksum:
        raise ValueError("Persistence checksum mismatch")
    items: List[MemoryItem] = []
    for raw in data_lines:
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:  # pragma: no cover - invalid record skipped
            continue
        embedding = np.asarray(record.get("embedding") or [], dtype=np.float32)
        item = MemoryItem(
            id=int(record.get("id", 0)),
            text=str(record.get("text") or ""),
            level=int(record.get("level", 0)),
            fidelity=float(record.get("fidelity") or 0.0),
            salience=float(record.get("salience") or 0.0),
            timestamp=float(record.get("timestamp") or 0.0),
            meta=record.get("meta") or {},
            summary_of=list(record.get("summary_of") or []),
            embedding=embedding,
        )
        items.append(item)
    count = header.get("count")
    if isinstance(count, int) and count != len(items):  # pragma: no cover - diagnostic only
        raise ValueError("Persistence record count mismatch")
    return items

"""Shared bridge between the FastAPI service and Streamlit visualizer."""
from __future__ import annotations

import json
import time
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional

BRIDGE_PATH = Path(__file__).resolve().parent.parent / "data" / "visualizer_queue.json"
BRIDGE_LOCK = RLock()


def queue_prompt(
    prompt: str,
    *,
    top_k: Optional[int] = None,
    mode: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist the most recent prompt so the visualizer can react to it.

    The Streamlit app polls this file and automatically replays the retrieval
    animation whenever a new payload is detected.  Persisting to disk keeps the
    mechanism resilient across process boundaries without introducing more
    infrastructure.
    """

    if not prompt.strip():
        return

    payload = {
        "prompt": prompt,
        "top_k": int(top_k) if top_k is not None else None,
        "mode": mode,
        "metadata": metadata or {},
        "timestamp": time.time(),
    }

    BRIDGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BRIDGE_LOCK:
        tmp_path = BRIDGE_PATH.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(BRIDGE_PATH)


def latest_prompt() -> Optional[Dict[str, Any]]:
    """Load the most recently queued prompt payload if available."""

    if not BRIDGE_PATH.exists():
        return None
    with BRIDGE_LOCK:
        try:
            return json.loads(BRIDGE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None


"""Checkpointing utilities for the Daystrom Memory Lattice."""
from __future__ import annotations

import json
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Callable, Dict, Optional

StateProvider = Callable[[], Dict[str, object]]


class CheckpointManager:
    """Persist lattice state to disk on demand and at fixed intervals."""

    def __init__(
        self,
        directory: Path,
        provider: StateProvider,
        *,
        interval_seconds: int = 0,
        retention: int = 3,
        start: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.provider = provider
        self.interval_seconds = max(0, int(interval_seconds))
        self.retention = max(0, int(retention))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.directory.mkdir(parents=True, exist_ok=True)
        if start and self.interval_seconds > 0:
            self._thread = threading.Thread(
                target=self._loop,
                name="dml-checkpoint",
                daemon=True,
            )
            self._thread.start()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Stop the background loop if it is running."""

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    # operations
    # ------------------------------------------------------------------
    def checkpoint(self) -> Path:
        """Create a checkpoint immediately and return its path."""

        payload = self.provider()
        timestamp = int(time.time())
        path = self.directory / f"checkpoint-{timestamp}.json"
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(path)
        self._prune_history()
        return path

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _loop(self) -> None:  # pragma: no cover - exercised indirectly via tests
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self.checkpoint()
            except Exception:
                continue

    def _prune_history(self) -> None:
        if self.retention <= 0:
            return
        checkpoints = sorted(self.directory.glob("checkpoint-*.json"))
        if len(checkpoints) <= self.retention:
            return
        for stale in checkpoints[:-self.retention]:
            with suppress(FileNotFoundError):
                stale.unlink()

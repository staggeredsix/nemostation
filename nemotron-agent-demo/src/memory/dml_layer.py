from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

DML_AVAILABLE = False
DMLAdapter = None


def _add_vendor_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    vendor_path = repo_root / "vendors" / "daystrom_memeory_lattice_alpha1"
    if vendor_path.exists() and str(vendor_path) not in sys.path:
        sys.path.insert(0, str(vendor_path))


_add_vendor_path()

try:
    from daystrom_dml.dml_adapter import DMLAdapter

    DML_AVAILABLE = True
except Exception:
    DML_AVAILABLE = False


@dataclass
class DMLReport:
    entries: list[Dict[str, Any]]
    preamble: str
    tokens: int
    latency_ms: int
    error: Optional[str] = None


class DMLMemoryLayer:
    def __init__(self, enabled: bool = True, storage_dir: Optional[Path] = None) -> None:
        self.enabled = False
        self.error: Optional[str] = None
        self._adapter: Optional[DMLAdapter] = None
        if not enabled:
            return
        if DMLAdapter is None:
            self.error = "daystrom_dml is not available"
            return
        try:
            root = storage_dir or (Path.cwd() / "data" / "dml")
            root.mkdir(parents=True, exist_ok=True)
            overrides = {
                "storage_dir": str(root),
                "persistence": {"enable": True, "path": "dml_state.jsonl", "interval_sec": 300},
                "rag_store": {"enable": True, "path": "rag_index.faiss", "meta_path": "rag_meta.json", "backend": "faiss", "dim": 384},
            }
            self._adapter = DMLAdapter(config_overrides=overrides)
            self.enabled = True
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            self._adapter = None
            self.enabled = False

    def retrieval_report(self, prompt: str, top_k: int) -> DMLReport:
        if not self.enabled or self._adapter is None:
            return DMLReport(entries=[], preamble="", tokens=0, latency_ms=0, error=self.error)
        report = self._adapter.retrieval_report(prompt, top_k=top_k)
        return DMLReport(
            entries=report.get("entries", []),
            preamble=report.get("preamble", ""),
            tokens=int(report.get("tokens", 0)),
            latency_ms=int(report.get("latency_ms", 0)),
        )

    def build_preamble(self, prompt: str, top_k: int) -> str:
        if not self.enabled or self._adapter is None:
            return ""
        return self._adapter.build_preamble(prompt, top_k=top_k)

    def ingest(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or self._adapter is None:
            return
        self._adapter.ingest(text, meta=meta)

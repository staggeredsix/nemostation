"""Lightweight data structures for CMA."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryItem:
    """Structured memory representation."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    e: List[float] = field(default_factory=list)
    s: str = ""
    tau: float = 0.0
    r: float = 1.0
    lam: float = 1.0
    k: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)
    code: Optional[int] = None
    centroid: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "e": list(self.e),
            "s": self.s,
            "tau": self.tau,
            "r": self.r,
            "lam": self.lam,
            "k": self.k,
            "meta": dict(self.meta),
            "code": self.code,
            "centroid": list(self.centroid) if self.centroid is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        return cls(
            id=data.get("id", uuid.uuid4().hex),
            e=list(data.get("e", [])),
            s=str(data.get("s", "")),
            tau=float(data.get("tau", 0.0)),
            r=float(data.get("r", 1.0)),
            lam=float(data.get("lam", 1.0)),
            k=int(data.get("k", 0)),
            meta=dict(data.get("meta", {})),
            code=data.get("code"),
            centroid=list(data.get("centroid", [])) if data.get("centroid") is not None else None,
        )

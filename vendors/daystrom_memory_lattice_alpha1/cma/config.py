"""Configuration objects for the Concept Memory Adapter."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CMAConfig:
    """Runtime configuration parameters."""

    K: int = 4
    beta_a: float = 0.08
    beta_r: float = 0.2
    eta: float = 0.15
    gamma: float = 0.02
    kappa: float = 0.5
    tau_s: float = 0.1
    theta_merge: float = 0.92
    top_k: int = 6
    token_budget: int = 600
    capacity: int = 2000
    snippet_token_budget: int = 160
    summary_sentence_count: int = 2
    bullet_count: int = 4
    codebook_size: int = 16
    min_vq_items: int = 8
    random_seed: int = 7
    alpha_raw: float = 1.0
    alpha_compressed: float = 0.5
    eta_salience: float = 0.1
    eta_fidelity: float = 0.05
    alignment_window: int = 50
    time_provider: Optional[callable] = field(default=None, repr=False)

    def now(self) -> float:
        """Return the current timestamp."""
        if self.time_provider is not None:
            return float(self.time_provider())
        from time import time

        return float(time())

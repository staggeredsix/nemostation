from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class StageMetrics:
    ms: float
    ttft_ms: float
    tokens: int
    tok_s: float


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def compute_throughput(tokens: int, elapsed_ms: float) -> float:
    elapsed_s = max(1e-6, elapsed_ms / 1000)
    return tokens / elapsed_s

"""Synthetic benchmark runner for the Daystrom Memory Lattice."""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Dict

from daystrom_dml.dml_adapter import DMLAdapter


def run_benchmark(
    adapter: DMLAdapter,
    *,
    iterations: int = 10,
    prompt: str = "Explain the Daystrom Memory Lattice",
) -> Dict[str, float]:
    """Execute repeated retrievals and return latency statistics."""

    iterations = max(1, int(iterations))
    adapter.ingest("Daystrom Lattice overview: hierarchical compression of knowledge.")
    adapter.ingest("Literal record: function fetchUserProfile() hits /api/users/{id}.")
    durations = []
    for idx in range(iterations):
        start = time.perf_counter()
        adapter.query_database(f"{prompt} #{idx}")
        durations.append(time.perf_counter() - start)
    avg_ms = statistics.mean(durations) * 1000.0
    if len(durations) >= 20:
        p95_ms = statistics.quantiles(durations, n=20)[-1] * 1000.0
    else:
        p95_ms = max(durations) * 1000.0
    return {
        "iterations": float(iterations),
        "avg_latency_ms": avg_ms,
        "p95_latency_ms": p95_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DML microbenchmarks")
    parser.add_argument("--iterations", type=int, default=10, help="Number of retrieval iterations")
    parser.add_argument("--storage", type=Path, default=Path("./data"), help="Storage directory override")
    args = parser.parse_args()
    adapter = DMLAdapter(config_overrides={"storage_dir": str(args.storage)}, start_aging_loop=False)
    try:
        metrics = run_benchmark(adapter, iterations=args.iterations)
    finally:
        adapter.close()
    print(
        f"Iterations: {int(metrics['iterations'])}, avg latency: {metrics['avg_latency_ms']:.2f} ms, "
        f"p95 latency: {metrics['p95_latency_ms']:.2f} ms"
    )


if __name__ == "__main__":
    main()

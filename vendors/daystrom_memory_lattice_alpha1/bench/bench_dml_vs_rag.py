"""Synthetic performance benchmarks for DML retrieval modes."""
from __future__ import annotations

import argparse
import csv
import random
import statistics
import tempfile
import time
from pathlib import Path
from typing import Iterable, List

from daystrom_dml.dml_adapter import DMLAdapter

DEFAULT_MODES = ("semantic", "literal", "hybrid", "agent")
VOCAB = [
    "quantum", "warp", "lattice", "plasma", "neutrino", "protocol", "diagnostic",
    "hyperdrive", "tensor", "fusion", "relay", "synthesis", "analysis", "resonance",
]


def _synthetic_document(idx: int, rng: random.Random) -> str:
    words = [rng.choice(VOCAB) for _ in range(60)]
    words[idx % len(words)] = f"concept_{idx}"
    return " ".join(words)


def _synthetic_query(idx: int) -> str:
    return f"Explain concept_{idx} with historical context"


def _ingest_corpus(adapter: DMLAdapter, count: int, rng: random.Random) -> None:
    for idx in range(count):
        text = _synthetic_document(idx, rng)
        adapter.ingest(
            text,
            meta={
                "doc_path": f"synthetic/doc_{idx}.txt",
                "tenant_id": "bench",
                "client_id": "bench",
            },
        )


def _run_mode(
    adapter: "DMLAdapter", prompts: Iterable[str], mode: str
) -> dict[str, float | str | List[dict[str, str]]]:
    latencies: List[float] = []
    tokens: List[int] = []
    generation_latencies: List[float] = []
    outputs: List[dict[str, str]] = []
    sample_output: str | None = None
    for prompt in prompts:
        start = time.perf_counter()
        if mode == "agent":
            report = adapter.retrieve_context(
                prompt,
                tenant_id="bench",
                client_id="bench",
                session_id=None,
                instance_id=None,
                kinds=None,
                top_k=None,
            )
            duration_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(report.get("latency_ms", duration_ms))
            tokens.append(int(report.get("context_tokens", 0)))
            augmented = adapter._compose_prompt(prompt, report.get("raw_context", ""))
        else:
            report = adapter.query_database(prompt, mode=mode)
            duration_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(duration_ms if report.get("latency_ms") is None else report["latency_ms"])
            tokens.append(int(report.get("tokens", 0)))
            augmented = adapter._compose_prompt(prompt, report.get("context", ""))
        gen_start = time.perf_counter()
        response = adapter.runner.generate(augmented)
        generation_latencies.append((time.perf_counter() - gen_start) * 1000.0)
        outputs.append({"prompt": prompt, "response": response})
        if sample_output is None:
            sample_output = response
    avg_latency = statistics.mean(latencies)
    avg_tokens = statistics.mean(tokens) if tokens else 0.0
    avg_generation_latency = statistics.mean(generation_latencies) if generation_latencies else 0.0
    cost_estimate = (avg_tokens / 1000.0) * 0.002
    return {
        "mode": mode,
        "avg_latency_ms": round(avg_latency, 3),
        "avg_tokens": round(avg_tokens, 2),
        "avg_generation_latency_ms": round(avg_generation_latency, 3),
        "estimated_cost_usd": round(cost_estimate, 6),
        "sample_output": (sample_output or "").strip(),
        "outputs": outputs,
    }


def run_benchmark(
    *,
    corpus_size: int,
    query_count: int,
    seed: int,
    modes: Iterable[str] = DEFAULT_MODES,
) -> List[dict[str, float | str]]:
    rng = random.Random(seed)
    prompts = [_synthetic_query(idx) for idx in range(query_count)]
    with tempfile.TemporaryDirectory(prefix="dml-bench-") as tmpdir:
        adapter = DMLAdapter(
            config_overrides={"storage_dir": tmpdir, "model_name": "dummy", "embedding_model": None},
            start_aging_loop=False,
        )
        try:
            _ingest_corpus(adapter, corpus_size, rng)
            results: List[dict[str, float | str]] = []
            for mode in modes:
                results.append(_run_mode(adapter, prompts, mode))
            return results
        finally:
            adapter.close()


def write_csv(results: List[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "avg_latency_ms",
        "avg_generation_latency_ms",
        "avg_tokens",
        "estimated_cost_usd",
        "sample_output",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            row_for_csv = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(row_for_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DML retrieval performance")
    parser.add_argument("--corpus-size", type=int, default=100, help="Number of synthetic documents")
    parser.add_argument("--queries", type=int, default=10, help="Number of benchmark queries")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bench/results.csv"),
        help="Path to write CSV results",
    )
    args = parser.parse_args()

    results = run_benchmark(corpus_size=args.corpus_size, query_count=args.queries, seed=args.seed)
    write_csv(results, args.output)
    for row in results:
        print(
            f"{row['mode']}: avg latency={row['avg_latency_ms']:.2f} ms, "
            f"avg generation latency={row['avg_generation_latency_ms']:.2f} ms, "
            f"avg tokens={row['avg_tokens']:.1f}, est cost=${row['estimated_cost_usd']:.6f}"
        )
        if row.get("sample_output"):
            print("  Example LLM output:\n  " + str(row["sample_output"]).replace("\n", "\n  "))


if __name__ == "__main__":  # pragma: no cover
    main()

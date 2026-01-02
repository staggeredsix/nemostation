from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Tuple

from openai import OpenAI

from .openai_client import create_openai_client, ensure_vllm_ready


def _load_prompt(prompt: str | None, prompt_file: str | None) -> str:
    if prompt and prompt_file:
        raise ValueError("Provide either --prompt or --prompt-file, not both.")
    if prompt_file:
        return Path(prompt_file).read_text().strip()
    if prompt:
        return prompt.strip()
    return "Summarize the benefits of stress testing autonomous agents."


def _send_chat(client: OpenAI, model_id: str, prompt: str, max_tokens: int) -> None:
    client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=max_tokens,
        stream=False,
    )


def _send_completion(client: OpenAI, model_id: str, prompt: str, max_tokens: int) -> None:
    client.completions.create(
        model=model_id,
        prompt=prompt,
        temperature=0.2,
        max_tokens=max_tokens,
        stream=False,
    )


def _run_request(
    client: OpenAI,
    model_id: str,
    prompt: str,
    max_tokens: int,
    endpoint: str,
) -> Tuple[bool, float]:
    start = time.perf_counter()
    try:
        if endpoint == "completions":
            _send_completion(client, model_id, prompt, max_tokens)
        else:
            _send_chat(client, model_id, prompt, max_tokens)
        return True, time.perf_counter() - start
    except Exception:
        return False, time.perf_counter() - start


def _percentile(values: Iterable[float], pct: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    index = int(round((len(sorted_values) - 1) * pct))
    return sorted_values[index]


def run_stress_test(
    model_id: str | None,
    concurrency: int,
    num_requests: int,
    max_tokens: int,
    prompt: str,
    endpoint: str,
) -> None:
    model_config = ensure_vllm_ready(model_id)
    client = create_openai_client(base_url=model_config.base_url, timeout_s=model_config.timeout_s)
    resolved_model_id = model_config.model_id

    start_time = time.perf_counter()
    successes = 0
    failures = 0
    latencies: list[float] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(_run_request, client, resolved_model_id, prompt, max_tokens, endpoint)
            for _ in range(num_requests)
        ]
        for future in as_completed(futures):
            ok, latency = future.result()
            if ok:
                successes += 1
                latencies.append(latency)
            else:
                failures += 1

    elapsed = time.perf_counter() - start_time
    throughput = successes / elapsed if elapsed > 0 else 0.0
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    p95_latency = _percentile(latencies, 0.95)

    print("Autonomous Agents Stress Testing - Stress Test Results")
    print(f"Model: {resolved_model_id}")
    print(f"Endpoint: {endpoint}")
    print(f"Requests: {num_requests}")
    print(f"Concurrency: {concurrency}")
    print(f"Successes: {successes}")
    print(f"Failures: {failures}")
    print(f"Throughput: {throughput:.2f} req/s")
    print(f"Average latency: {avg_latency:.2f} s")
    print(f"P95 latency: {p95_latency:.2f} s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a stress test against a vLLM OpenAI-compatible endpoint.")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent requests.")
    parser.add_argument("--num-requests", type=int, default=20, help="Total number of requests to send.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens for each request.")
    parser.add_argument("--prompt", type=str, default=None, help="Inline prompt text.")
    parser.add_argument("--prompt-file", type=str, default=None, help="Path to a prompt file.")
    parser.add_argument("--model-id", type=str, default=None, help="Model ID to use (overrides VLLM_MODEL_ID).")
    parser.add_argument(
        "--endpoint",
        type=str,
        default="chat",
        choices=["chat", "completions"],
        help="OpenAI endpoint to call (default: chat).",
    )
    args = parser.parse_args(argv)

    try:
        prompt = _load_prompt(args.prompt, args.prompt_file)
        run_stress_test(
            model_id=args.model_id,
            concurrency=args.concurrency,
            num_requests=args.num_requests,
            max_tokens=args.max_tokens,
            prompt=prompt,
            endpoint=args.endpoint,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

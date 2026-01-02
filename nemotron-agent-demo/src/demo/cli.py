from __future__ import annotations

import argparse
import sys
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .openai_client import fetch_model_ids, get_vllm_base_url
from .orchestrator import run_demo_stream

console = Console()


def format_status(status: str) -> str:
    icons = {"queued": "…", "running": "⏳", "done": "✅", "failed": "❌"}
    return icons.get(status, status)


def run(goal: str, fast: bool = False, scenario: Optional[str] = None, model_id: Optional[str] = None) -> None:
    table = Table(title="Autonomous Agents Stress Testing", expand=True)
    table.add_column("Stage", style="cyan")
    table.add_column("Status")
    table.add_column("ms")
    table.add_column("Approx TTFT")
    table.add_column("Tok/s")
    table.add_column("Tokens")
    console.print(table)

    for state in run_demo_stream(goal, fast=fast, scenario=scenario, model_id=model_id):
        table.rows.clear()
        for stage in state["stages"]:
            status = format_status(stage["status"])
            table.add_row(
                stage["name"],
                status,
                f"{stage['ms']:.0f}",
                f"{stage['ttft_ms']:.0f}",
                f"{stage['tok_s']:.1f}",
                str(stage.get("tokens", 0)),
            )
        console.clear()
        console.print(table)
        metrics = state.get("metrics", {})
        console.print(
            Text(
                f"Total: {metrics.get('total_ms',0):.0f} ms | Approx TTFT: {metrics.get('approx_ttft_ms',0):.0f} ms | Approx tok/s: {metrics.get('approx_tok_s',0):.1f}",
                style="green",
            )
        )
        if state.get("final"):
            console.rule("Final")
            console.print(state["final"])


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Autonomous Agents Stress Testing demo via CLI")
    parser.add_argument("goal", nargs="?", default="Explain how to optimize a local LLM agent stack")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--model-id", default=None, help="Model ID to use (overrides VLLM_MODEL_ID)")
    parser.add_argument("--list-models", action="store_true", help="List available models from /models and exit")
    args = parser.parse_args(argv)

    try:
        if args.list_models:
            base_url = get_vllm_base_url()
            models = fetch_model_ids(base_url=base_url)
            console.print(f"Models at {base_url}/models:")
            for model in models:
                console.print(f"- {model}")
            return 0
        run(args.goal, fast=args.fast, scenario=args.scenario, model_id=args.model_id)
    except KeyboardInterrupt:
        console.print("Interrupted", style="red")
        return 1
    except Exception as exc:
        console.print(f"Error: {exc}", style="red")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

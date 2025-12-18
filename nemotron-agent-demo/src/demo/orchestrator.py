from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional

from .agents import AgentResult, call_agent
from .metrics import StageMetrics, compute_throughput, estimate_tokens


@dataclass
class StageState:
    name: str
    status: str = "queued"
    ms: float = 0.0
    ttft_ms: float = 0.0
    tok_s: float = 0.0
    tokens: int = 0
    output: str = ""
    error: Optional[str] = None


DEFAULT_STAGES = ["planner", "coder", "reviewer", "ops", "aggregator"]


def _initial_state(goal: str, scenario: Optional[str], fast: bool) -> Dict:
    stage_order = ["planner", "coder", "reviewer"]
    if not fast:
        stage_order.append("ops")
    stage_order.append("aggregator")
    stages = [StageState(name=s.title()) for s in stage_order]
    return {
        "goal": goal,
        "scenario": scenario,
        "stages": [stage.__dict__ for stage in stages],
        "metrics": {"total_ms": 0, "approx_tok_s": 0, "approx_ttft_ms": 0},
        "final": "",
    }


def _serialize(stages: List[StageState], goal: str, scenario: Optional[str], final: str, total_ms: float) -> Dict:
    completed = [s for s in stages if s.status == "done"]
    total_tokens = sum(s.tokens for s in completed)
    total_ttft = sum(s.ttft_ms for s in completed)
    approx_tok_s = compute_throughput(total_tokens, total_ms) if total_ms else 0.0
    approx_ttft = total_ttft / len(completed) if completed else 0.0
    return {
        "goal": goal,
        "scenario": scenario,
        "stages": [stage.__dict__ for stage in stages],
        "metrics": {
            "total_ms": total_ms,
            "approx_tok_s": approx_tok_s,
            "approx_ttft_ms": approx_ttft,
        },
        "final": final,
    }


def _update_stage(stages: List[StageState], name: str, **kwargs) -> None:
    for stage in stages:
        if stage.name.lower() == name.lower():
            for key, value in kwargs.items():
                setattr(stage, key, value)
            break


def run_demo_stream(goal: str, fast: bool = False, scenario: Optional[str] = None) -> Generator[Dict, None, None]:
    stages: List[StageState] = []
    stage_order = ["planner", "coder", "reviewer"]
    if not fast:
        stage_order.append("ops")
    stage_order.append("aggregator")
    stages = [StageState(name=s.title()) for s in stage_order]

    start_time = time.perf_counter()
    yield _serialize(stages, goal, scenario, final="", total_ms=0)

    outputs: Dict[str, AgentResult] = {}
    failed = False

    for stage_name in stage_order:
        _update_stage(stages, stage_name, status="running")
        yield _serialize(stages, goal, scenario, final="", total_ms=(time.perf_counter() - start_time) * 1000)

        stage_start = time.perf_counter()
        try:
            extra_context = ""
            if stage_name != "planner":
                context_parts = [f"{k.title()} Output:\n{v.output}" for k, v in outputs.items()]
                extra_context = "\n\n".join(context_parts)
            max_tokens = 384 if fast else 640
            if stage_name == "aggregator":
                max_tokens = 320 if fast else 512
            result = call_agent(stage_name, goal, scenario, max_tokens=max_tokens, extra_context=extra_context)
            elapsed_ms = (time.perf_counter() - stage_start) * 1000
            tokens = estimate_tokens(result.output)
            tok_s = compute_throughput(tokens, elapsed_ms)
            metrics = StageMetrics(ms=elapsed_ms, ttft_ms=elapsed_ms, tokens=tokens, tok_s=tok_s)
            outputs[stage_name] = result
            _update_stage(
                stages,
                stage_name,
                status="done",
                ms=metrics.ms,
                ttft_ms=metrics.ttft_ms,
                tok_s=metrics.tok_s,
                tokens=metrics.tokens,
                output=result.output,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - stage_start) * 1000
            _update_stage(
                stages,
                stage_name,
                status="failed",
                ms=elapsed_ms,
                ttft_ms=elapsed_ms,
                error=str(exc),
            )
            failed = True

        total_ms = (time.perf_counter() - start_time) * 1000
        final_text = outputs.get("aggregator", AgentResult(stage_name, "")).output if outputs else ""
        yield _serialize(stages, goal, scenario, final=final_text, total_ms=total_ms)

        if failed:
            if outputs:
                break
            return

    total_ms = (time.perf_counter() - start_time) * 1000
    final_text = outputs.get("aggregator", AgentResult("aggregator", "")).output
    yield _serialize(stages, goal, scenario, final=final_text, total_ms=total_ms)


def run_demo(goal: str, fast: bool = False, scenario: Optional[str] = None) -> Dict:
    last_state = {}
    for state in run_demo_stream(goal, fast=fast, scenario=scenario):
        last_state = deepcopy(state)
    return last_state

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional

from src.memory.dml_layer import DMLMemoryLayer

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
    dml: Dict = field(default_factory=dict)


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


def _serialize(
    stages: List[StageState],
    goal: str,
    scenario: Optional[str],
    final: str,
    total_ms: float,
    dml: Optional[Dict] = None,
) -> Dict:
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
        "dml": dml or {},
    }


def _update_stage(stages: List[StageState], name: str, **kwargs) -> None:
    for stage in stages:
        if stage.name.lower() == name.lower():
            for key, value in kwargs.items():
                setattr(stage, key, value)
            break


def _default_dml_payload(enabled: bool) -> Dict:
    return {
        "enabled": enabled,
        "retrieved_ids": [],
        "latency_ms": 0,
        "token_estimate": 0,
        "entries": [],
        "error": None,
    }


def _compact_text(text: str, limit: int = 420) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def run_demo_stream(
    goal: str,
    fast: bool = False,
    scenario: Optional[str] = None,
    use_dml: bool = False,
    dml_top_k: int = 6,
) -> Generator[Dict, None, None]:
    stages: List[StageState] = []
    stage_order = ["planner", "coder", "reviewer"]
    if not fast:
        stage_order.append("ops")
    stage_order.append("aggregator")
    stages = [StageState(name=s.title()) for s in stage_order]

    dml_layer = DMLMemoryLayer(enabled=use_dml)
    dml_info = {
        "requested": use_dml,
        "enabled": bool(use_dml and dml_layer.enabled),
        "top_k": dml_top_k,
        "error": dml_layer.error,
    }

    start_time = time.perf_counter()
    yield _serialize(stages, goal, scenario, final="", total_ms=0, dml=dml_info)

    outputs: Dict[str, AgentResult] = {}
    failed = False
    retrieved_ids_by_stage: Dict[str, List[str]] = {}

    for stage_name in stage_order:
        _update_stage(stages, stage_name, status="running")
        yield _serialize(stages, goal, scenario, final="", total_ms=(time.perf_counter() - start_time) * 1000, dml=dml_info)

        stage_start = time.perf_counter()
        try:
            extra_context = ""
            if stage_name != "planner":
                context_parts = [f"{k.title()} Output:\n{v.output}" for k, v in outputs.items()]
                extra_context = "\n\n".join(context_parts)
            max_tokens = 384 if fast else 640
            if stage_name == "aggregator":
                max_tokens = 320 if fast else 512
            dml_payload = _default_dml_payload(enabled=bool(use_dml))
            system_messages: List[str] = []
            if use_dml:
                if dml_layer.enabled:
                    try:
                        retrieval_prompt = f"Stage: {stage_name}\nGoal: {goal}\nScenario: {scenario or 'general'}"
                        if extra_context:
                            retrieval_prompt += f"\nContext:\n{extra_context}"
                        report = dml_layer.retrieval_report(retrieval_prompt, top_k=dml_top_k)
                        entries = report.entries
                        retrieved_ids = [
                            (entry.get("meta", {}).get("doc_path") or entry.get("id"))
                            for entry in entries
                            if entry.get("meta", {}).get("doc_path") or entry.get("id")
                        ]
                        dml_payload.update(
                            {
                                "retrieved_ids": retrieved_ids,
                                "latency_ms": report.latency_ms,
                                "token_estimate": report.tokens,
                                "entries": entries,
                                "error": report.error,
                            }
                        )
                        retrieved_ids_by_stage[stage_name] = retrieved_ids
                        if report.preamble:
                            system_messages.append(f"DML_MEMORY_CONTEXT:\n{report.preamble}")
                    except Exception as exc:  # noqa: BLE001
                        dml_payload["error"] = str(exc)
                else:
                    dml_payload["error"] = dml_layer.error or "DML unavailable"
            _update_stage(stages, stage_name, dml=dml_payload)
            result = call_agent(
                stage_name,
                goal,
                scenario,
                max_tokens=max_tokens,
                extra_context=extra_context,
                system_messages=system_messages or None,
            )
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
        yield _serialize(stages, goal, scenario, final=final_text, total_ms=total_ms, dml=dml_info)

        if failed:
            if outputs:
                break
            return

    total_ms = (time.perf_counter() - start_time) * 1000
    final_text = outputs.get("aggregator", AgentResult("aggregator", "")).output
    if use_dml and dml_layer.enabled:
        summary_lines = [
            f"Goal: {goal}",
            f"Final answer excerpt: {_compact_text(final_text, 360)}",
            "Agent outputs:",
        ]
        for stage, result in outputs.items():
            summary_lines.append(f"- {stage.title()}: {_compact_text(result.output, 240)}")
        summary_lines.append("Retrieved memory IDs:")
        for stage, ids in retrieved_ids_by_stage.items():
            summary_lines.append(f"- {stage.title()}: {', '.join(ids) if ids else 'None'}")
        meta = {
            "goal": goal,
            "final_excerpt": _compact_text(final_text, 360),
            "agent_outputs": {stage: _compact_text(result.output, 240) for stage, result in outputs.items()},
            "retrieved_ids_by_stage": retrieved_ids_by_stage,
        }
        dml_layer.ingest("\n".join(summary_lines), meta=meta)
    yield _serialize(stages, goal, scenario, final=final_text, total_ms=total_ms, dml=dml_info)


def run_demo(
    goal: str,
    fast: bool = False,
    scenario: Optional[str] = None,
    use_dml: bool = False,
    dml_top_k: int = 6,
) -> Dict:
    last_state = {}
    for state in run_demo_stream(goal, fast=fast, scenario=scenario, use_dml=use_dml, dml_top_k=dml_top_k):
        last_state = deepcopy(state)
    return last_state

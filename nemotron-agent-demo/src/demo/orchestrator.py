from __future__ import annotations

import logging
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

from src.memory import dml_http_client

from .agents import AgentResult, call_agent
from .metrics import StageMetrics, compute_throughput, estimate_tokens

logger = logging.getLogger(__name__)


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

    scenario_key = scenario or "general"
    dml_error: Optional[str] = None
    dml_enabled = False
    dml_get_calls = 0
    dml_ingest_calls = 0
    cookbook_info = {
        "found": False,
        "cookbook_text": "",
        "sources": [],
        "latency_ms": 0,
    }
    ingest_info = {
        "ok": False,
        "ingested_id": "",
        "summary_id": "",
        "summary_latency_ms": 0,
        "error": None,
    }
    if use_dml:
        try:
            dml_get_calls += 1
            if dml_get_calls > 1:
                logger.error("dml_get_calls_per_run exceeded: %d", dml_get_calls)
            cookbook = dml_http_client.get_cookbook(scenario_key, goal, dml_top_k)
            dml_enabled = True
            cookbook_info.update(
                {
                    "found": cookbook.found,
                    "cookbook_text": cookbook.cookbook_text,
                    "sources": cookbook.sources,
                    "latency_ms": cookbook.latency_ms,
                }
            )
        except dml_http_client.DMLServiceError as exc:
            dml_error = str(exc)
            dml_enabled = False
    dml_info = {
        "requested": use_dml,
        "enabled": bool(use_dml and dml_enabled),
        "top_k": dml_top_k,
        "error": dml_error,
        "cookbook": cookbook_info,
        "ingest": ingest_info,
        "counters": {
            "dml_get_calls_per_run": dml_get_calls,
            "dml_ingest_calls_per_run": dml_ingest_calls,
        },
    }

    start_time = time.perf_counter()
    yield _serialize(stages, goal, scenario, final="", total_ms=0, dml=dml_info)

    outputs: Dict[str, AgentResult] = {}
    failed = False
    run_id = str(uuid.uuid4())
    trace: Dict[str, Dict[str, Any]] = {
        "stages": {},
        "timings": {},
        "errors": [],
        "dml": {
            "cookbook_found": cookbook_info["found"],
            "cookbook_sources": cookbook_info["sources"],
        },
    }
    base_system_messages: List[str] = []
    if dml_enabled and cookbook_info["found"] and cookbook_info["cookbook_text"]:
        base_system_messages.append(f"DML_COOKBOOK_GUIDANCE:\n{cookbook_info['cookbook_text']}")

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
            result = call_agent(
                stage_name,
                goal,
                scenario,
                max_tokens=max_tokens,
                extra_context=extra_context,
                system_messages=base_system_messages or None,
            )
            elapsed_ms = (time.perf_counter() - stage_start) * 1000
            tokens = estimate_tokens(result.output)
            tok_s = compute_throughput(tokens, elapsed_ms)
            metrics = StageMetrics(ms=elapsed_ms, ttft_ms=elapsed_ms, tokens=tokens, tok_s=tok_s)
            outputs[stage_name] = result
            trace["stages"][stage_name] = {
                "output": result.output,
                "error": None,
                "ms": metrics.ms,
                "ttft_ms": metrics.ttft_ms,
                "tok_s": metrics.tok_s,
                "tokens": metrics.tokens,
                "extra_context": extra_context,
                "system_messages": base_system_messages,
                "max_tokens": max_tokens,
            }
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
            trace["stages"][stage_name] = {
                "output": "",
                "error": str(exc),
                "ms": elapsed_ms,
                "ttft_ms": elapsed_ms,
                "tok_s": 0.0,
                "tokens": 0,
                "extra_context": extra_context,
                "system_messages": base_system_messages,
                "max_tokens": max_tokens,
            }
            trace["errors"].append({"stage": stage_name, "error": str(exc)})
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
        trace["timings"]["total_ms"] = total_ms
        yield _serialize(stages, goal, scenario, final=final_text, total_ms=total_ms, dml=dml_info)

        if failed:
            if outputs:
                break
            return

    total_ms = (time.perf_counter() - start_time) * 1000
    final_text = outputs.get("aggregator", AgentResult("aggregator", "")).output
    if use_dml and dml_enabled:
        run_report = {
            "scenario_key": scenario_key,
            "goal": goal,
            "run_id": run_id,
            "trace": trace,
            "final": final_text,
            "success": not failed,
            "meta": {
                "scenario": scenario,
                "fast": fast,
                "cookbook_sources": cookbook_info["sources"],
            },
        }
        try:
            dml_ingest_calls += 1
            if dml_ingest_calls > 1:
                logger.error("dml_ingest_calls_per_run exceeded: %d", dml_ingest_calls)
            result = dml_http_client.ingest_run_report(run_report)
            ingest_info.update(
                {
                    "ok": result.ok,
                    "ingested_id": result.ingested_id,
                    "summary_id": result.summary_id,
                    "summary_latency_ms": result.summary_latency_ms,
                    "error": None,
                }
            )
        except dml_http_client.DMLServiceError as exc:
            ingest_info.update({"error": str(exc)})
        dml_info["counters"]["dml_get_calls_per_run"] = dml_get_calls
        dml_info["counters"]["dml_ingest_calls_per_run"] = dml_ingest_calls
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

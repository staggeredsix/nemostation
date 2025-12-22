from __future__ import annotations

import json
import logging
import re
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

from src.memory import dml_http_client
from src.playground import cluster_manager
from src.playground import manager as playground_manager

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
LONG_RUN_MARKER = "LONG_AGENT_RUN_MODE: true"
MAX_TOOL_REQUESTS_PER_STAGE = 3
MAX_CLUSTER_FIX_ITERS = 5
FIXED_PLAYGROUND_COMMANDS = {
    "coder": ["bash", "-lc", "ls -la /workspace && python --version"],
    "aggregator": ["bash", "-lc", "find /workspace -maxdepth 3 -type f | head -n 50"],
}


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
    playground: Optional[Dict] = None,
    cluster: Optional[Dict] = None,
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
        "playground": playground or {},
        "cluster": cluster or {},
        "dml": dml or {},
    }


def _is_long_run(goal: str, scenario: Optional[str]) -> bool:
    if LONG_RUN_MARKER.lower() in goal.lower():
        return True
    if scenario and LONG_RUN_MARKER.lower() in scenario.lower():
        return True
    return False


def _extract_tool_requests(text: str) -> List[Dict[str, Any]]:
    blocks = re.findall(r"```json\\s*(\\{.*?\\})\\s*```", text, flags=re.DOTALL)
    requests: List[Dict[str, Any]] = []
    for block in blocks:
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("tool"):
            requests.append(payload)
    return requests


def _format_tool_context(entry: Dict[str, Any]) -> str:
    return (
        "Playground command result:\n"
        f"$ {entry.get('cmd')}\n"
        f"Exit code: {entry.get('exit_code')}\n"
        f"Stdout:\n{entry.get('stdout')}\n"
        f"Stderr:\n{entry.get('stderr')}\n"
    )


def _format_cluster_context(entry: Dict[str, Any]) -> str:
    return (
        "Cluster command result:\n"
        f"$ {entry.get('cmd')}\n"
        f"Exit code: {entry.get('exit_code')}\n"
        f"Stdout:\n{entry.get('stdout')}\n"
        f"Stderr:\n{entry.get('stderr')}\n"
    )


def _format_validation_context(validation: Dict[str, Any], cluster_info: Dict[str, Any]) -> str:
    containers = ", ".join([c.get("name", "") for c in cluster_info.get("containers", [])]) or "none"
    status_lines = [
        f"run_id: {cluster_info.get('run_id')}",
        f"network: {cluster_info.get('network')}",
        f"containers: {containers}",
        f"api_port: {cluster_info.get('api_port')}",
        f"web_port: {cluster_info.get('web_port')}",
        f"workspace_host: {cluster_info.get('workspace_host')}",
        f"workspace_container: {cluster_info.get('workspace_container')}",
    ]
    return (
        "Cluster validation report:\n"
        f"{json.dumps(validation, indent=2)}\n\n"
        "Cluster status:\n"
        + "\n".join(status_lines)
    )


def _run_playground_command(
    playground_name: str,
    cmd: List[str],
    playground_log: List[Dict[str, Any]],
    tool_context_chunks: List[str],
    timeout_s: int = 60,
) -> Dict[str, Any]:
    entry = playground_manager.exec_cmd(playground_name, cmd, timeout_s=timeout_s)
    entry["cmd"] = " ".join(cmd)
    playground_log.append(entry)
    tool_context_chunks.append(_format_tool_context(entry))
    return entry


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
    use_playground: bool = False,
    playground_image: str = "nemotron-playground:latest",
    auto_remove_playground: bool = False,
    use_cluster: bool = False,
    cluster_image: str = "nemotron-playground:latest",
    cluster_size: int = 3,
    cluster_run_id: Optional[str] = None,
) -> Generator[Dict, None, None]:
    stages: List[StageState] = []
    stage_order = ["planner", "coder", "reviewer"]
    if not fast:
        stage_order.append("ops")
    stage_order.append("aggregator")
    stages = [StageState(name=s.title()) for s in stage_order]

    if use_cluster and isinstance(cluster_run_id, str) and cluster_run_id.strip() and "/" not in cluster_run_id and " " not in cluster_run_id:
        run_id = cluster_run_id.strip()
    else:
        run_id = str(uuid.uuid4())
    long_run_mode = _is_long_run(goal, scenario)
    playground_name = f"nemotron-playground-{run_id.split('-')[0]}"
    playground_log: List[Dict[str, Any]] = []
    playground_info: Dict[str, Any] = {
        "enabled": bool(use_playground),
        "name": playground_name if use_playground else "",
        "image": playground_image,
        "requested_image": playground_image,
        "status": "disabled" if not use_playground else "pending",
        "error": None,
        "warning": None,
        "log": playground_log,
        "auto_remove": bool(auto_remove_playground),
        "ready_for_removal": False,
        "workspace_host": "",
        "workspace_container": "",
    }
    cluster_log: List[Dict[str, Any]] = []
    tool_context_chunks: List[str] = []
    cluster_info: Dict[str, Any] = {
        "enabled": bool(use_cluster),
        "run_id": run_id,
        "size": cluster_size,
        "image": cluster_image,
        "status": "disabled" if not use_cluster else "pending",
        "network": "",
        "containers": [],
        "api_port": None,
        "web_port": None,
        "workspace_host": "",
        "workspace_container": "",
        "error": None,
        "validation": {},
        "validation_history": [],
        "iteration": 0,
        "max_iters": MAX_CLUSTER_FIX_ITERS,
        "fix_actions": [],
        "log": cluster_log,
        "ready_for_removal": False,
    }
    if use_playground:
        playground_status = playground_manager.ensure_playground(playground_image, playground_name, run_id, repo_mount=None)
        playground_info.update(
            {
                "status": playground_status.get("status", "unknown"),
                "error": playground_status.get("error"),
                "workspace_host": playground_status.get("workspace_host"),
                "workspace_container": playground_status.get("workspace_container"),
                "warning": playground_status.get("warning"),
                "image": playground_status.get("image", playground_image),
                "requested_image": playground_status.get("requested_image", playground_image),
            }
        )
    if use_cluster:
        cluster_status = cluster_manager.create_cluster(run_id, cluster_image, cluster_size, workspace_host=None)
        cluster_info.update(
            {
                "status": cluster_status.get("status", "unknown"),
                "error": cluster_status.get("error"),
                "network": cluster_status.get("network", ""),
                "containers": cluster_status.get("containers", []),
                "api_port": cluster_status.get("api_port"),
                "web_port": cluster_status.get("web_port"),
                "workspace_host": cluster_status.get("workspace_host", ""),
                "workspace_container": cluster_status.get("workspace_container", ""),
            }
        )
        if cluster_status.get("log"):
            cluster_log.extend(cluster_status.get("log", []))
            for entry in cluster_status.get("log", []):
                tool_context_chunks.append(_format_cluster_context(entry))
        if cluster_status.get("reused"):
            validation = cluster_manager.validate_cluster(run_id)
            cluster_info["validation"] = validation
            cluster_info["validation_history"].append({"iteration": 0, "validation": validation})
            entry = {
                "cmd": "cluster.validate (reuse)",
                "exit_code": 0 if validation.get("ok") else 1,
                "stdout": json.dumps(validation, indent=2),
                "stderr": "" if validation.get("ok") else validation.get("error", ""),
            }
            tool_context_chunks.append(_format_cluster_context(entry))
            cluster_log.append(entry)

    scenario_key = scenario or "general"
    if use_cluster:
        scenario_key = f"{scenario_key}-cluster-{cluster_size}"
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
    yield _serialize(stages, goal, scenario, final="", total_ms=0, playground=playground_info, cluster=cluster_info, dml=dml_info)

    outputs: Dict[str, AgentResult] = {}
    failed = False
    trace: Dict[str, Dict[str, Any]] = {
        "stages": {},
        "timings": {},
        "errors": [],
        "dml": {
            "cookbook_found": cookbook_info["found"],
            "cookbook_sources": cookbook_info["sources"],
        },
        "playground": {
            "name": playground_info.get("name"),
            "image": playground_info.get("image"),
            "workspace_host": playground_info.get("workspace_host"),
            "workspace_container": playground_info.get("workspace_container"),
            "enabled": playground_info.get("enabled"),
        },
        "cluster": {
            "run_id": run_id,
            "size": cluster_size,
            "image": cluster_image,
            "network": cluster_info.get("network"),
            "containers": cluster_info.get("containers"),
            "api_port": cluster_info.get("api_port"),
            "web_port": cluster_info.get("web_port"),
            "api_url": f"http://localhost:{cluster_info.get('api_port')}" if cluster_info.get("api_port") else None,
            "web_url": f"http://localhost:{cluster_info.get('web_port')}" if cluster_info.get("web_port") else None,
            "workspace_host": cluster_info.get("workspace_host"),
            "workspace_container": cluster_info.get("workspace_container"),
            "enabled": cluster_info.get("enabled"),
        },
    }
    base_system_messages: List[str] = []
    if dml_enabled and cookbook_info["found"] and cookbook_info["cookbook_text"]:
        base_system_messages.append(f"DML_COOKBOOK_GUIDANCE:\n{cookbook_info['cookbook_text']}")
    if use_cluster:
        container_list = ", ".join([c.get("name", "") for c in cluster_info.get("containers", [])])
        base_system_messages.append(
            "CLUSTER_TOPOLOGY:\n"
            f"- run_id: {run_id}\n"
            f"- network: {cluster_info.get('network')}\n"
            f"- containers: {container_list or 'none'}\n"
            f"- api_host_port: {cluster_info.get('api_port')}\n"
            f"- web_host_port: {cluster_info.get('web_port')}\n"
            f"- api_url: http://localhost:{cluster_info.get('api_port')}\n"
            f"- web_url: http://localhost:{cluster_info.get('web_port')}\n"
            f"- workspace_host: {cluster_info.get('workspace_host')}\n"
            f"- workspace_container: {cluster_info.get('workspace_container')}\n"
        )
        if cluster_info.get("error"):
            base_system_messages.append(f"CLUSTER_BOOTSTRAP_ERROR:\n{cluster_info.get('error')}\n")

    for stage_name in stage_order:
        _update_stage(stages, stage_name, status="running")
        yield _serialize(
            stages,
            goal,
            scenario,
            final="",
            total_ms=(time.perf_counter() - start_time) * 1000,
            playground=playground_info,
            cluster=cluster_info,
            dml=dml_info,
        )

        stage_start = time.perf_counter()
        try:
            extra_context = ""
            if stage_name != "planner":
                context_parts = [f"{k.title()} Output:\n{v.output}" for k, v in outputs.items()]
                extra_context = "\n\n".join(context_parts)
            if tool_context_chunks:
                tool_context = "\n\n".join(tool_context_chunks)
                extra_context = "\n\n".join(filter(None, [extra_context, f"Tool Command Log:\n{tool_context}"]))
            max_tokens = 384 if fast else 640
            if stage_name == "aggregator":
                max_tokens = 320 if fast else 512
            system_messages = list(base_system_messages)
            if long_run_mode and use_playground and stage_name == "coder":
                system_messages.append(
                    "All generated files must be written under /workspace inside the playground container. "
                    "Use tool steps to create files and run commands."
                )
            if long_run_mode and use_cluster and stage_name == "coder":
                system_messages.append(
                    "Cluster tools are available: use cluster.exec for container commands, cluster.logs for log collection, "
                    "and cluster.validate for validation."
                )
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
            stage_trace: Dict[str, Any] = {
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
            if long_run_mode and (use_playground or use_cluster):
                tool_requests = _extract_tool_requests(result.output)
                tool_entries: List[Dict[str, Any]] = []
                for request in tool_requests[:MAX_TOOL_REQUESTS_PER_STAGE]:
                    tool_name = request.get("tool")
                    entry: Dict[str, Any]
                    if tool_name == "playground.exec" and use_playground:
                        cmd = request.get("cmd")
                        timeout_s = int(request.get("timeout_s", 60))
                        if not isinstance(cmd, list) or not all(isinstance(arg, str) for arg in cmd):
                            entry = {
                                "cmd": str(cmd),
                                "exit_code": 125,
                                "stdout": "",
                                "stderr": "Invalid command format. Expected list[str].",
                            }
                        else:
                            entry = playground_manager.exec_cmd(playground_name, cmd, timeout_s=timeout_s)
                            entry["cmd"] = " ".join(cmd)
                        tool_context_chunks.append(_format_tool_context(entry))
                        playground_log.append(entry)
                    elif tool_name == "playground.write_file" and use_playground:
                        path = request.get("path")
                        content = request.get("content")
                        if not isinstance(path, str) or not isinstance(content, str):
                            entry = {
                                "cmd": f"write_file {path}",
                                "exit_code": 125,
                                "stdout": "",
                                "stderr": "Invalid write_file payload. Expected path/content strings.",
                            }
                        else:
                            entry = playground_manager.write_file(playground_name, path, content)
                            entry["cmd"] = f"write_file {path}"
                        tool_context_chunks.append(_format_tool_context(entry))
                        playground_log.append(entry)
                    elif tool_name == "cluster.exec" and use_cluster:
                        container = request.get("container")
                        cmd = request.get("cmd")
                        timeout_s = int(request.get("timeout_s", 60))
                        if not isinstance(container, str) or not isinstance(cmd, list) or not all(isinstance(arg, str) for arg in cmd):
                            entry = {
                                "cmd": f"{container} {cmd}",
                                "exit_code": 125,
                                "stdout": "",
                                "stderr": "Invalid cluster.exec payload. Expected container + cmd list.",
                            }
                        else:
                            entry = cluster_manager.exec_in(container, cmd, timeout_s=timeout_s)
                            entry["cmd"] = f"{container} :: {' '.join(cmd)}"
                        tool_context_chunks.append(_format_cluster_context(entry))
                        cluster_log.append(entry)
                    elif tool_name == "cluster.validate" and use_cluster:
                        validation = cluster_manager.validate_cluster(run_id)
                        cluster_info["validation"] = validation
                        entry = {
                            "cmd": "cluster.validate",
                            "exit_code": 0 if validation.get("ok") else 1,
                            "stdout": json.dumps(validation, indent=2),
                            "stderr": "" if validation.get("ok") else validation.get("error", ""),
                        }
                        tool_context_chunks.append(_format_cluster_context(entry))
                        cluster_log.append(entry)
                    elif tool_name == "cluster.logs" and use_cluster:
                        container = request.get("container")
                        if not isinstance(container, str):
                            entry = {
                                "cmd": f"{container} logs",
                                "exit_code": 125,
                                "stdout": "",
                                "stderr": "Invalid cluster.logs payload. Expected container string.",
                            }
                        else:
                            tail_value = request.get("tail", 200)
                            try:
                                tail = int(tail_value)
                            except (TypeError, ValueError):
                                tail = 200
                            entry = cluster_manager.container_logs(container, tail=tail)
                            entry["cmd"] = f"{container} :: logs (tail={tail})"
                        tool_context_chunks.append(_format_cluster_context(entry))
                        cluster_log.append(entry)
                    elif tool_name == "cluster.validate" and use_cluster:
                        validation = cluster_manager.validate_cluster(run_id)
                        entry = {
                            "cmd": "cluster.validate (fixer)",
                            "exit_code": 0 if validation.get("ok") else 1,
                            "stdout": json.dumps(validation, indent=2),
                            "stderr": "" if validation.get("ok") else validation.get("error", ""),
                        }
                        tool_context_chunks.append(_format_cluster_context(entry))
                        cluster_log.append(entry)
                    else:
                        continue
                    tool_entries.append(entry)
                if tool_entries:
                    stage_trace["tool_requests"] = tool_entries
                if use_playground:
                    fixed_command = FIXED_PLAYGROUND_COMMANDS.get(stage_name)
                    if fixed_command:
                        fixed_entry = _run_playground_command(
                            playground_name,
                            fixed_command,
                            playground_log,
                            tool_context_chunks,
                        )
                        stage_trace["playground_commands"] = [fixed_entry]
                    if stage_name == "planner":
                        skeleton_entries = []
                        skeleton_entries.append(
                            _run_playground_command(
                                playground_name,
                                ["bash", "-lc", "mkdir -p /workspace/app /workspace/tests"],
                                playground_log,
                                tool_context_chunks,
                            )
                        )
                        skeleton_entries.append(
                            _run_playground_command(
                                playground_name,
                                [
                                    "bash",
                                    "-lc",
                                    "cat <<'EOF' > /workspace/README.md\n# Workspace\n\nInitialized by long-run mode.\nEOF",
                                ],
                                playground_log,
                                tool_context_chunks,
                            )
                        )
                        stage_trace.setdefault("playground_commands", []).extend(skeleton_entries)
            trace["stages"][stage_name] = stage_trace
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
        yield _serialize(
            stages,
            goal,
            scenario,
            final=final_text,
            total_ms=total_ms,
            playground=playground_info,
            cluster=cluster_info,
            dml=dml_info,
        )

        if failed:
            if use_playground:
                playground_info["ready_for_removal"] = True
                if auto_remove_playground:
                    removal = playground_manager.remove_playground(playground_name)
                    playground_info["remove_result"] = removal
                    if removal.get("ok"):
                        playground_info["status"] = "removed"
                    else:
                        playground_info["status"] = "error"
                        playground_info["error"] = removal.get("error")
                yield _serialize(
                    stages,
                    goal,
                    scenario,
                    final=final_text,
                    total_ms=total_ms,
                    playground=playground_info,
                    cluster=cluster_info,
                    dml=dml_info,
                )
            if use_cluster:
                cluster_info["ready_for_removal"] = True
            if outputs:
                break
            return

    total_ms = (time.perf_counter() - start_time) * 1000
    final_text = outputs.get("aggregator", AgentResult("aggregator", "")).output
    if playground_log:
        trace["playground"]["log"] = playground_log
    if use_cluster:
        if long_run_mode:
            fix_iterations: List[Dict[str, Any]] = []
            last_validation: Dict[str, Any] = {}
            for iteration in range(1, MAX_CLUSTER_FIX_ITERS + 1):
                cluster_info["iteration"] = iteration
                validation = cluster_manager.validate_cluster(run_id)
                last_validation = validation
                cluster_info["validation"] = validation
                cluster_info["validation_history"].append({"iteration": iteration, "validation": validation})
                entry = {
                    "cmd": f"cluster.validate (iter {iteration}/{MAX_CLUSTER_FIX_ITERS})",
                    "exit_code": 0 if validation.get("ok") else 1,
                    "stdout": json.dumps(validation, indent=2),
                    "stderr": "" if validation.get("ok") else validation.get("error", ""),
                }
                cluster_log.append(entry)
                tool_context_chunks.append(_format_cluster_context(entry))
                trace["cluster"]["validation"] = validation
                trace["cluster"]["validation_history"] = cluster_info["validation_history"]
                if cluster_log:
                    trace["cluster"]["log"] = cluster_log
                total_ms = (time.perf_counter() - start_time) * 1000
                yield _serialize(
                    stages,
                    goal,
                    scenario,
                    final=final_text,
                    total_ms=total_ms,
                    playground=playground_info,
                    cluster=cluster_info,
                    dml=dml_info,
                )
                if validation.get("ok"):
                    break
                fixer_context = _format_validation_context(validation, cluster_info)
                fixer_system_messages = list(base_system_messages)
                fixer_system_messages.append(
                    "Cluster tools are available: use cluster.exec for container commands and cluster.logs for logs."
                )
                fixer_result = call_agent(
                    "fixer",
                    goal,
                    scenario,
                    max_tokens=384,
                    extra_context=fixer_context,
                    system_messages=fixer_system_messages,
                )
                tool_requests = _extract_tool_requests(fixer_result.output)
                tool_entries: List[Dict[str, Any]] = []
                fix_actions: List[Dict[str, Any]] = []
                for request in tool_requests[:MAX_TOOL_REQUESTS_PER_STAGE]:
                    tool_name = request.get("tool")
                    entry = {}
                    if tool_name == "playground.exec" and use_playground:
                        cmd = request.get("cmd")
                        timeout_s = int(request.get("timeout_s", 60))
                        if not isinstance(cmd, list) or not all(isinstance(arg, str) for arg in cmd):
                            entry = {
                                "cmd": str(cmd),
                                "exit_code": 125,
                                "stdout": "",
                                "stderr": "Invalid command format. Expected list[str].",
                            }
                        else:
                            entry = playground_manager.exec_cmd(playground_name, cmd, timeout_s=timeout_s)
                            entry["cmd"] = " ".join(cmd)
                        tool_context_chunks.append(_format_tool_context(entry))
                        playground_log.append(entry)
                    elif tool_name == "playground.write_file" and use_playground:
                        path = request.get("path")
                        content = request.get("content")
                        if not isinstance(path, str) or not isinstance(content, str):
                            entry = {
                                "cmd": f"write_file {path}",
                                "exit_code": 125,
                                "stdout": "",
                                "stderr": "Invalid write_file payload. Expected path/content strings.",
                            }
                        else:
                            entry = playground_manager.write_file(playground_name, path, content)
                            entry["cmd"] = f"write_file {path}"
                        tool_context_chunks.append(_format_tool_context(entry))
                        playground_log.append(entry)
                    elif tool_name == "cluster.exec" and use_cluster:
                        container = request.get("container")
                        cmd = request.get("cmd")
                        timeout_s = int(request.get("timeout_s", 60))
                        if not isinstance(container, str) or not isinstance(cmd, list) or not all(isinstance(arg, str) for arg in cmd):
                            entry = {
                                "cmd": f"{container} {cmd}",
                                "exit_code": 125,
                                "stdout": "",
                                "stderr": "Invalid cluster.exec payload. Expected container + cmd list.",
                            }
                        else:
                            entry = cluster_manager.exec_in(container, cmd, timeout_s=timeout_s)
                            entry["cmd"] = f"{container} :: {' '.join(cmd)}"
                        tool_context_chunks.append(_format_cluster_context(entry))
                        cluster_log.append(entry)
                    elif tool_name == "cluster.logs" and use_cluster:
                        container = request.get("container")
                        if not isinstance(container, str):
                            entry = {
                                "cmd": f"{container} logs",
                                "exit_code": 125,
                                "stdout": "",
                                "stderr": "Invalid cluster.logs payload. Expected container string.",
                            }
                        else:
                            tail_value = request.get("tail", 200)
                            try:
                                tail = int(tail_value)
                            except (TypeError, ValueError):
                                tail = 200
                            entry = cluster_manager.container_logs(container, tail=tail)
                            entry["cmd"] = f"{container} :: logs (tail={tail})"
                        tool_context_chunks.append(_format_cluster_context(entry))
                        cluster_log.append(entry)
                    else:
                        continue
                    tool_entries.append(entry)
                    fix_actions.append(
                        {
                            "iteration": iteration,
                            "action": entry.get("cmd", ""),
                            "exit_code": entry.get("exit_code"),
                        }
                    )
                if tool_entries:
                    fix_iterations.append(
                        {
                            "iteration": iteration,
                            "validation": validation,
                            "fixer_output": fixer_result.output,
                            "tool_requests": tool_entries,
                        }
                    )
                if fix_actions:
                    cluster_info["fix_actions"].extend(fix_actions)
                total_ms = (time.perf_counter() - start_time) * 1000
                yield _serialize(
                    stages,
                    goal,
                    scenario,
                    final=final_text,
                    total_ms=total_ms,
                    playground=playground_info,
                    cluster=cluster_info,
                    dml=dml_info,
                )
            trace["cluster"]["fix_iterations"] = fix_iterations
            if last_validation and not last_validation.get("ok"):
                failed = True
                cluster_info["error"] = f"Validation failed after {MAX_CLUSTER_FIX_ITERS} iterations."
        else:
            validation = cluster_manager.validate_cluster(run_id)
            cluster_info["validation"] = validation
            cluster_info["validation_history"].append({"iteration": 1, "validation": validation})
            cluster_log.append(
                {
                    "cmd": "cluster.validate (auto)",
                    "exit_code": 0 if validation.get("ok") else 1,
                    "stdout": json.dumps(validation, indent=2),
                    "stderr": "" if validation.get("ok") else validation.get("error", ""),
                }
            )
            trace["cluster"]["validation"] = validation
            trace["cluster"]["validation_history"] = cluster_info["validation_history"]
            if cluster_log:
                trace["cluster"]["log"] = cluster_log
    if use_dml and dml_enabled:
        run_report = {
            "scenario_key": scenario_key,
            "goal": goal,
            "run_id": run_id,
            "trace": trace,
            "final": final_text,
            "success": not failed,
            "artifacts": [{"type": "playground_command", **entry} for entry in playground_log]
            + [{"type": "cluster_command", **entry} for entry in cluster_log],
            "meta": {
                "scenario": scenario,
                "fast": fast,
                "cookbook_sources": cookbook_info["sources"],
                "cluster_topology": trace.get("cluster"),
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
    if use_playground:
        playground_info["ready_for_removal"] = True
        if auto_remove_playground:
            removal = playground_manager.remove_playground(playground_name)
            playground_info["remove_result"] = removal
            if removal.get("ok"):
                playground_info["status"] = "removed"
            else:
                playground_info["status"] = "error"
                playground_info["error"] = removal.get("error")
    if use_cluster:
        cluster_info["ready_for_removal"] = True
    yield _serialize(
        stages,
        goal,
        scenario,
        final=final_text,
        total_ms=total_ms,
        playground=playground_info,
        cluster=cluster_info,
        dml=dml_info,
    )


def run_demo(
    goal: str,
    fast: bool = False,
    scenario: Optional[str] = None,
    use_dml: bool = False,
    dml_top_k: int = 6,
    use_playground: bool = False,
    playground_image: str = "nemotron-playground:latest",
    auto_remove_playground: bool = False,
    use_cluster: bool = False,
    cluster_image: str = "nemotron-playground:latest",
    cluster_size: int = 3,
    cluster_run_id: Optional[str] = None,
) -> Dict:
    last_state = {}
    for state in run_demo_stream(
        goal,
        fast=fast,
        scenario=scenario,
        use_dml=use_dml,
        dml_top_k=dml_top_k,
        use_playground=use_playground,
        playground_image=playground_image,
        auto_remove_playground=auto_remove_playground,
        use_cluster=use_cluster,
        cluster_image=cluster_image,
        cluster_size=cluster_size,
        cluster_run_id=cluster_run_id,
    ):
        last_state = deepcopy(state)
    return last_state

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from src.memory import dml_http_client
from src.playground import cluster_manager
from src.playground import manager as playground_manager

from .agents import AgentResult, call_agent
from .metrics import StageMetrics, compute_throughput, estimate_tokens

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[2]
HUMAN_INPUT_PATH = BASE_DIR / "prompt_library" / "human_input.json"


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


DEFAULT_STAGES = ["supervisor", "planner", "coder", "reviewer", "ops", "aggregator"]
LONG_RUN_MARKER = "LONG_AGENT_RUN_MODE: true"
GIVE_UP_PHRASE = "we cannot complete the task"
HANDOFF_RE = re.compile(r"^\s*(?:NEXT_ROLE|HANDOFF_TO)\s*:\s*(\w+)\s*$", re.IGNORECASE | re.MULTILINE)
HUMAN_INPUT_REQUIRED_RE = re.compile(
    r"^\s*HUMAN_INPUT_REQUIRED\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
FIXED_PLAYGROUND_COMMANDS = {
    "coder": ["bash", "-lc", "ls -la /workspace && python --version"],
    "aggregator": ["bash", "-lc", "find /workspace -maxdepth 3 -type f | head -n 50"],
}


def _load_human_messages() -> List[str]:
    try:
        raw = HUMAN_INPUT_PATH.read_text()
    except FileNotFoundError:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    cleaned: List[str] = []
    for item in payload:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _safe_stage_name(stage_name: str) -> str:
    safe_name = re.sub(r"[^a-z0-9_-]+", "-", stage_name.lower()).strip("-")
    return safe_name or "agent"


def _should_autobuild_webserver(goal: str) -> bool:
    goal_text = (goal or "").lower()
    if "hello world" in goal_text:
        return True
    if "hello" in goal_text and "world" in goal_text:
        return any(token in goal_text for token in ("web", "server", "http", "website", "site"))
    return False


def _requires_project_scaffold(goal: str, scenario: Optional[str]) -> bool:
    text = f"{goal} {scenario or ''}".lower()
    if LONG_RUN_MARKER.lower() in text:
        return True
    return bool(
        re.search(
            r"\\b(app|service|api|server|website|web|frontend|backend|docker|compose|container|microservice|project|ui)\\b",
            text,
        )
    )


def _extract_human_input_requests(output: str) -> List[str]:
    if not output:
        return []
    return [match.strip() for match in HUMAN_INPUT_REQUIRED_RE.findall(output) if match.strip()]


def _format_access_summary(
    run_id: str,
    playground_info: Dict[str, Any],
    cluster_info: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("Access summary:")
    lines.append(f"- Run ID: {run_id}")
    lines.append(f"- Project path (host): {BASE_DIR / 'agent_projects' / run_id}")

    if playground_info.get("enabled"):
        lines.append(f"- Playground container: {playground_info.get('name') or '—'}")
        if playground_info.get("workspace_host") or playground_info.get("workspace_container"):
            lines.append(f"- Playground workspace (host): {playground_info.get('workspace_host') or '—'}")
            lines.append(f"- Playground workspace (container): {playground_info.get('workspace_container') or '—'}")
        exposed_ports = playground_info.get("exposed_ports") or []
        if exposed_ports:
            lines.append(f"- Playground exposed ports: {', '.join(str(p) for p in exposed_ports)}")
        web_port = playground_info.get("web_port")
        if web_port:
            lines.append(f"- Playground web URL: http://localhost:{web_port}")
        else:
            lines.append("- Playground web URL: not exposed (use playground.expose_port)")

    if cluster_info.get("enabled"):
        lines.append(f"- Cluster run ID: {cluster_info.get('run_id') or run_id}")
        lines.append(f"- Cluster network: {cluster_info.get('network') or '—'}")
        containers = ", ".join([c.get("name", "") for c in cluster_info.get("containers", [])]) or "—"
        lines.append(f"- Cluster containers: {containers}")
        api_port = cluster_info.get("api_port")
        web_port = cluster_info.get("web_port")
        lines.append(f"- Cluster API URL: {f'http://localhost:{api_port}' if api_port else '—'}")
        lines.append(f"- Cluster Web URL: {f'http://localhost:{web_port}' if web_port else '—'}")
        if cluster_info.get("workspace_host") or cluster_info.get("workspace_container"):
            lines.append(f"- Cluster workspace (host): {cluster_info.get('workspace_host') or '—'}")
            lines.append(f"- Cluster workspace (container): {cluster_info.get('workspace_container') or '—'}")

    return "\n".join(lines)


def _format_human_input_block(requests: List[str]) -> str:
    if not requests:
        return ""
    lines = ["HUMAN INPUT REQUIRED:"]
    lines.extend([f"- {item}" for item in requests if item])
    return "\n".join(lines)


def _write_stage_artifact(run_id: str, stage_name: str, content: str) -> Optional[str]:
    try:
        run_root = BASE_DIR / "agent_projects" / run_id / "outputs"
        run_root.mkdir(parents=True, exist_ok=True)
        try:
            run_root.chmod(0o777)
        except PermissionError:
            pass
        safe_name = _safe_stage_name(stage_name)
        path = run_root / f"{safe_name}.md"
        path.write_text(content)
        return str(path)
    except Exception:  # noqa: BLE001
        return None


def _initial_state(goal: str, scenario: Optional[str], fast: bool) -> Dict:
    stage_order = ["supervisor", "planner", "coder", "reviewer"]
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
    events: Optional[List[str]] = None,
) -> Dict:
    completed = [s for s in stages if s.status == "done"]
    total_tokens = sum(s.tokens for s in completed)
    total_ttft = sum(s.ttft_ms for s in completed)
    summed_tok_s = sum(s.tok_s for s in stages if s.status in {"done", "running"} and s.tok_s > 0)
    approx_tok_s = summed_tok_s if summed_tok_s > 0 else (compute_throughput(total_tokens, total_ms) if total_ms else 0.0)
    approx_ttft = total_ttft / len(completed) if completed else 0.0
    return {
        "goal": goal,
        "scenario": scenario,
        "stages": [stage.__dict__ for stage in stages],
        "metrics": {
            "total_ms": total_ms,
            "total_tokens": total_tokens,
            "approx_tok_s": approx_tok_s,
            "approx_ttft_ms": approx_ttft,
        },
        "final": final,
        "playground": playground or {},
        "cluster": cluster or {},
        "dml": dml or {},
        "events": events or [],
    }


def _is_long_run(goal: str, scenario: Optional[str]) -> bool:
    if LONG_RUN_MARKER.lower() in goal.lower():
        return True
    if scenario and LONG_RUN_MARKER.lower() in scenario.lower():
        return True
    return False


def _extract_tool_requests(text: str) -> List[Dict[str, Any]]:
    blocks: List[str] = []
    blocks.extend(re.findall(r"```json\\s*(\\{.*?\\})\\s*```", text, flags=re.DOTALL))
    blocks.extend(re.findall(r"```\\s*(\\{.*?\\})\\s*```", text, flags=re.DOTALL))
    blocks.extend(re.findall(r"(\\{[^{}]*\"tool\"[^{}]*\\})", text, flags=re.DOTALL))
    requests: List[Dict[str, Any]] = []
    for block in blocks:
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("tool"):
            requests.append(payload)
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and item.get("tool"):
                    requests.append(item)
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


def _parse_ops_failure(output: str) -> Optional[Dict[str, str]]:
    sections: Dict[str, str] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().upper()
        if key in {"OPS_STATUS", "OPS_ERROR", "OPS_TO_PLANNER"}:
            sections[key] = value.strip()
    if sections.get("OPS_STATUS", "").upper() != "FAIL":
        return None
    return {
        "error": sections.get("OPS_ERROR", "").strip(),
        "instruction": sections.get("OPS_TO_PLANNER", "").strip(),
        "raw": output.strip(),
    }


def _agent_gave_up(output: str) -> bool:
    return GIVE_UP_PHRASE in output.lower()


def _extract_handoff(output: str) -> Optional[str]:
    match = HANDOFF_RE.search(output or "")
    if not match:
        return None
    role = match.group(1).strip().lower()
    if role in {"supervisor", "planner", "coder", "reviewer", "ops", "aggregator"}:
        return role
    return None


def _run_playground_command(
    playground_name: str,
    cmd: List[str],
    playground_log: List[Dict[str, Any]],
    tool_context_chunks: List[str],
    timeout_s: int = 60,
    agent: Optional[str] = None,
) -> Dict[str, Any]:
    entry = playground_manager.exec_cmd(playground_name, cmd, timeout_s=timeout_s)
    entry["cmd"] = " ".join(cmd)
    if agent:
        entry["agent"] = agent
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
    parallel_agents: Optional[bool] = None,
) -> Generator[Dict, None, None]:
    stages: List[StageState] = []
    stage_order = ["supervisor", "planner", "coder", "reviewer"]
    if not fast:
        stage_order.append("ops")
    stage_order.append("aggregator")
    stage_queue = list(stage_order)
    stages = [StageState(name=s.title()) for s in stage_order]

    if use_cluster and isinstance(cluster_run_id, str) and cluster_run_id.strip() and "/" not in cluster_run_id and " " not in cluster_run_id:
        run_id = cluster_run_id.strip()
    else:
        run_id = str(uuid.uuid4())
    run_root = BASE_DIR / "agent_projects" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        run_root.chmod(0o777)
    except PermissionError:
        pass
    readme_path = run_root / "README.md"
    if not readme_path.exists():
        readme_path.write_text("Initialized by Nemotron Station run.\n")
    long_run_mode = _is_long_run(goal, scenario)
    attempt = 1
    max_attempts = int(os.getenv("AGENT_MAX_ATTEMPTS", "0"))
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
    events: List[str] = []
    events.append(f"Run ID: {run_id}")
    if max_attempts == 0:
        events.append(f"Attempt {attempt} (unlimited retries)")
    else:
        events.append(f"Attempt {attempt} of {max_attempts}")
    if not use_playground and not use_cluster:
        events.append("Playground/Cluster tools are disabled; agents can only return text output.")
    else:
        events.append("Tool commands run via docker exec; container logs may not show them.")
        if use_playground and playground_info.get("web_port"):
            events.append(f"Playground web URL: http://localhost:{playground_info.get('web_port')}")
        elif use_playground:
            events.append("No host port exposed. Use playground.expose_port to publish a port.")
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
        "max_iters": None,
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
                "web_port": playground_status.get("web_port"),
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
    parallel_enabled = (
        parallel_agents
        if parallel_agents is not None
        else os.getenv("AGENT_PARALLEL", "1").strip().lower() in {"1", "true", "yes"}
    )
    yield _serialize(
        stages,
        goal,
        scenario,
        final="",
        total_ms=0,
        playground=playground_info,
        cluster=cluster_info,
        dml=dml_info,
        events=events,
    )

    outputs: Dict[str, AgentResult] = {}
    failed = False
    ops_escalation: Optional[str] = None
    ops_fix_count = 0
    awaiting_human_input = False
    human_input_requests: List[str] = []
    final_override: Optional[str] = None
    handoff_count = 0
    handoff_max = int(os.getenv("AGENT_HANDOFF_MAX", "6"))
    trace: Dict[str, Dict[str, Any]] = {
        "stages": {},
        "timings": {},
        "errors": [],
        "ops_escalations": [],
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
        "human_input_requests": [],
        "handoffs": [],
    }
    base_system_messages: List[str] = []
    if dml_enabled and cookbook_info["found"] and cookbook_info["cookbook_text"]:
        base_system_messages.append(f"DML_COOKBOOK_GUIDANCE:\n{cookbook_info['cookbook_text']}")
    if use_playground:
        base_system_messages.append(
            "PLAYGROUND_TOOLS_AVAILABLE: Use JSON tool requests to run container commands, e.g.\n"
            "```json\n"
            "{\"tool\":\"playground.exec\",\"cmd\":[\"bash\",\"-lc\",\"ls -la /workspace\"],\"timeout_s\":60}\n"
            "```\n"
            "Expose ports to the host with:\n"
            "```json\n"
            "{\"tool\":\"playground.expose_port\",\"host_port\":18000,\"container_port\":8000}\n"
            "```\n"
            "Infrastructure tooling (if mounted): use ssh/scp/rsync for remote hosts and kubectl for clusters. "
            "If PLAYGROUND_SSH_DIR or PLAYGROUND_KUBECONFIG is mounted, you can access /root/.ssh and /root/.kube/config.\n"
            "For host Docker/Compose, use playground.docker with args like:\n"
            "```json\n"
            "{\"tool\":\"playground.docker\",\"args\":[\"compose\",\"up\",\"-d\"],\"timeout_s\":600}\n"
            "```\n"
            "Manage Docker containers (create/restart/remove) with:\n"
            "```json\n"
            "{\"tool\":\"playground.docker\",\"args\":[\"run\",\"-d\",\"--name\",\"nemotron-playground-mybox\",\"ubuntu\",\"sleep\",\"infinity\"],\"timeout_s\":60}\n"
            "```\n"
            "Or write files with:\n"
            "```json\n"
            "{\"tool\":\"playground.write_file\",\"path\":\"/workspace/README.md\",\"content\":\"...\"}\n"
            "```"
        )
    if use_cluster:
        base_system_messages.append(
            "CLUSTER_TOOLS_AVAILABLE: Use JSON tool requests, e.g.\n"
            "```json\n"
            "{\"tool\":\"cluster.exec\",\"container\":\"<container>\",\"cmd\":[\"bash\",\"-lc\",\"ls -la /workspace\"],\"timeout_s\":60}\n"
            "```\n"
            "Or validate with:\n"
            "```json\n"
            "{\"tool\":\"cluster.validate\"}\n"
            "```"
        )
    if use_playground or use_cluster:
        base_system_messages.append(
            f"AGENT_PROJECTS_DIR: /workspace/agent_projects/{run_id} (write generated project files here; this maps to repo ./agent_projects/{run_id})."
        )
        base_system_messages.append(
            "SAFE_WORKSPACE_RULE: All tool commands and file writes must stay under AGENT_PROJECTS_DIR."
        )
        base_system_messages.append(
            "REQUIRED: If tools are available, run at least one tool command and write at least one file under AGENT_PROJECTS_DIR."
        )
    base_system_messages.append(
        "HANDOFF_PROTOCOL:\n"
        "- To route work to another role, include a line: NEXT_ROLE: <supervisor|planner|coder|reviewer|ops|aggregator>\n"
        "- Use this only when more work is required; otherwise omit it.\n"
    )
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

    def _build_extra_context(stage_name: str, tool_context_snapshot: Optional[List[str]] = None) -> str:
        extra_context = ""
        if stage_name in {"supervisor", "planner"} and ops_escalation:
            extra_context = f"Ops escalation:\n{ops_escalation}"
        if outputs:
            context_parts = [f"{k.title()} Output:\n{v.output}" for k, v in outputs.items()]
            extra_context = "\n\n".join(filter(None, [extra_context, "\n\n".join(context_parts)]))
        human_messages = _load_human_messages()
        if human_messages and stage_name == "supervisor":
            human_block = "Human input (most recent last):\n" + "\n".join(f"- {msg}" for msg in human_messages)
            extra_context = "\n\n".join(filter(None, [extra_context, human_block]))
        context_chunks = tool_context_snapshot if tool_context_snapshot is not None else tool_context_chunks
        if context_chunks:
            tool_context = "\n\n".join(context_chunks)
            extra_context = "\n\n".join(filter(None, [extra_context, f"Tool Command Log:\n{tool_context}"]))
        return extra_context

    def _project_files_exist() -> bool:
        run_root = BASE_DIR / "agent_projects" / run_id
        if not run_root.exists():
            return False
        for path in run_root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(run_root)
            if rel.parts and rel.parts[0] == "outputs":
                continue
            if rel.name == "README.md":
                continue
            return True
        return False

    def _completion_check() -> tuple[bool, str]:
        if not _project_files_exist():
            return False, "No project files created under agent_projects."
        if _requires_project_scaffold(goal, scenario):
            missing: List[str] = []
            if not (run_root / "Dockerfile").exists():
                missing.append("Dockerfile")
            if not ((run_root / "docker-compose.yml").exists() or (run_root / "docker-compose.yaml").exists()):
                missing.append("docker-compose.yml")
            if missing:
                return False, f"Missing required deliverables: {', '.join(missing)}"
        if _should_autobuild_webserver(goal):
            app_path = BASE_DIR / "agent_projects" / run_id / "app.py"
            if not app_path.exists():
                return False, "Missing app.py for hello world server."
            if use_playground:
                check_cmd = _prepend_safe_dir(["bash", "-lc", "curl -sf http://localhost:8000"])
                entry = playground_manager.exec_cmd(playground_name, check_cmd, timeout_s=30)
                entry["cmd"] = " ".join(check_cmd)
                entry["tool"] = "playground.exec"
                entry["agent"] = "orchestrator"
                tool_context_chunks.append(_format_tool_context(entry))
                playground_log.append(entry)
                if entry.get("exit_code") == 0:
                    return True, ""
                start_cmd = _prepend_safe_dir(["bash", "-lc", "python3 app.py"])
                start_entry = playground_manager.exec_cmd_detached(playground_name, start_cmd)
                start_entry["cmd"] = " ".join(start_cmd)
                start_entry["tool"] = "playground.exec_detached"
                start_entry["agent"] = "orchestrator"
                tool_context_chunks.append(_format_tool_context(start_entry))
                playground_log.append(start_entry)
                if start_entry.get("exit_code") == 0:
                    events.append("Orchestrator started hello world server (retry).")
                check_retry = playground_manager.exec_cmd(playground_name, check_cmd, timeout_s=30)
                check_retry["cmd"] = " ".join(check_cmd)
                check_retry["tool"] = "playground.exec"
                check_retry["agent"] = "orchestrator"
                tool_context_chunks.append(_format_tool_context(check_retry))
                playground_log.append(check_retry)
                if check_retry.get("exit_code") == 0:
                    return True, ""
                err = check_retry.get("stderr") or check_retry.get("stdout") or "server not responding"
                return False, f"Server check failed: {err}"
            return False, "Playground disabled; cannot verify server response."
        return True, ""

    def _build_system_messages(stage_name: str) -> List[str]:
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
        return system_messages

    def _prepend_safe_dir(cmd: List[str]) -> List[str]:
        if len(cmd) >= 3 and cmd[0] == "bash" and cmd[1] == "-lc":
            safe_dir = f"/workspace/agent_projects/{run_id}"
            return ["bash", "-lc", f"cd {safe_dir} && {cmd[2]}"]
        return cmd

    def _handle_stage_result(
        stage_name: str,
        result: AgentResult,
        elapsed_ms: float,
        extra_context: str,
        system_messages: List[str],
        max_tokens: int,
    ) -> None:
        nonlocal failed, ops_escalation, ops_fix_count, handoff_count, awaiting_human_input, human_input_requests
        tokens = result.tokens or estimate_tokens(result.output)
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
            "system_messages": system_messages,
            "max_tokens": max_tokens,
        }
        if use_playground or use_cluster:
            output_text = result.output or ""
            tool_requests = _extract_tool_requests(output_text)
            tool_entries: List[Dict[str, Any]] = []
            wrote_file = False
            for request in tool_requests:
                tool_name = request.get("tool")
                entry: Dict[str, Any]
                if tool_name in {"playground.exec", "playground.exec_detached"} and use_playground:
                    cmd = request.get("cmd")
                    timeout_s = int(request.get("timeout_s", 60))
                    detach = bool(request.get("detach")) or tool_name == "playground.exec_detached"
                    if not isinstance(cmd, list) or not all(isinstance(arg, str) for arg in cmd):
                        entry = {
                            "cmd": str(cmd),
                            "exit_code": 125,
                            "stdout": "",
                            "stderr": "Invalid command format. Expected list[str].",
                        }
                    else:
                        safe_cmd = _prepend_safe_dir(cmd)
                        if detach:
                            entry = playground_manager.exec_cmd_detached(playground_name, safe_cmd)
                        else:
                            entry = playground_manager.exec_cmd(playground_name, safe_cmd, timeout_s=timeout_s)
                        entry["cmd"] = " ".join(safe_cmd)
                    entry["agent"] = stage_name
                    tool_context_chunks.append(_format_tool_context(entry))
                    playground_log.append(entry)
                    if detach and entry.get("exit_code") == 0:
                        events.append(f"{stage_name.title()} agent started detached command: {' '.join(safe_cmd)}")
                        if len(events) > 500:
                            del events[:-500]
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
                    entry["agent"] = stage_name
                    wrote_file = True
                    tool_context_chunks.append(_format_tool_context(entry))
                    playground_log.append(entry)
                elif tool_name == "playground.docker" and use_playground:
                    args = request.get("args") or request.get("cmd")
                    timeout_s = int(request.get("timeout_s", 60))
                    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
                        entry = {
                            "cmd": str(args),
                            "exit_code": 125,
                            "stdout": "",
                            "stderr": "Invalid docker args. Expected list[str].",
                        }
                    else:
                        entry = playground_manager.docker_cmd(args, timeout_s=timeout_s)
                        entry["cmd"] = f"docker {' '.join(args)}"
                    entry["agent"] = stage_name
                    tool_context_chunks.append(_format_tool_context(entry))
                    playground_log.append(entry)
                elif tool_name == "playground.expose_port" and use_playground:
                    host_port = request.get("host_port") or request.get("port")
                    container_port = request.get("container_port") or request.get("target_port") or host_port
                    try:
                        host_port = int(host_port)
                        container_port = int(container_port)
                    except (TypeError, ValueError):
                        entry = {
                            "cmd": f"expose_port {host_port}:{container_port}",
                            "exit_code": 125,
                            "stdout": "",
                            "stderr": "Invalid port values. Expected integers.",
                        }
                    else:
                        result = playground_manager.expose_port(playground_name, host_port, container_port)
                        if result.get("ok"):
                            playground_info["web_port"] = host_port
                            playground_info.setdefault("exposed_ports", [])
                            if host_port not in playground_info["exposed_ports"]:
                                playground_info["exposed_ports"].append(host_port)
                            events.append(f"Port {host_port} exposed to playground {playground_name}.")
                            entry = {
                                "cmd": f"expose_port {host_port}:{container_port}",
                                "exit_code": 0,
                                "stdout": json.dumps(result),
                                "stderr": "",
                            }
                        else:
                            entry = {
                                "cmd": f"expose_port {host_port}:{container_port}",
                                "exit_code": 1,
                                "stdout": "",
                                "stderr": result.get("error", "Failed to expose port."),
                            }
                    entry["tool"] = tool_name
                    entry["agent"] = stage_name
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
                        safe_cmd = _prepend_safe_dir(cmd)
                        entry = cluster_manager.exec_in(container, safe_cmd, timeout_s=timeout_s)
                        entry["cmd"] = f"{container} :: {' '.join(safe_cmd)}"
                    entry["agent"] = stage_name
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
                    entry["agent"] = stage_name
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
                    entry["agent"] = stage_name
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
                    entry["agent"] = stage_name
                    tool_context_chunks.append(_format_cluster_context(entry))
                    cluster_log.append(entry)
                else:
                    continue
                entry["tool"] = tool_name
                tool_entries.append(entry)
            if tool_entries:
                stage_trace["tool_requests"] = tool_entries
            if use_playground and not wrote_file:
                safe_name = _safe_stage_name(stage_name)
                container_path = f"/workspace/agent_projects/{run_id}/outputs/{safe_name}.md"
                artifact_text = f"# {stage_name.title()} Output\n\n{output_text}\n"
                entry = playground_manager.write_file(playground_name, container_path, artifact_text)
                entry["cmd"] = f"write_file {container_path}"
                entry["tool"] = "playground.write_file"
                entry["agent"] = stage_name
                tool_context_chunks.append(_format_tool_context(entry))
                playground_log.append(entry)
                if entry.get("exit_code") == 0:
                    events.append(f"{stage_name.title()} agent wrote {container_path}.")
                    if len(events) > 500:
                        del events[:-500]
                else:
                    err = entry.get("stderr") or entry.get("stdout") or "unknown error"
                    events.append(f"{stage_name.title()} agent failed to write {container_path}: {err}")
                    if len(events) > 500:
                        del events[:-500]
            if stage_name == "coder" and use_playground and not wrote_file and _should_autobuild_webserver(goal):
                app_path = f"/workspace/agent_projects/{run_id}/app.py"
                app_content = (
                    "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
                    "import os\n\n"
                    "class Handler(BaseHTTPRequestHandler):\n"
                    "    def do_GET(self):\n"
                    "        self.send_response(200)\n"
                    "        self.send_header('Content-Type', 'text/plain; charset=utf-8')\n"
                    "        self.end_headers()\n"
                    "        self.wfile.write(b'Hello world!')\n\n"
                    "def main():\n"
                    "    port = int(os.getenv('PORT', '8000'))\n"
                    "    server = HTTPServer(('', port), Handler)\n"
                    "    print(f'Serving on :{port}')\n"
                    "    server.serve_forever()\n\n"
                    "if __name__ == '__main__':\n"
                    "    main()\n"
                )
                write_entry = playground_manager.write_file(playground_name, app_path, app_content)
                write_entry["cmd"] = f"write_file {app_path}"
                write_entry["tool"] = "playground.write_file"
                write_entry["agent"] = "orchestrator"
                tool_context_chunks.append(_format_tool_context(write_entry))
                playground_log.append(write_entry)
                if write_entry.get("exit_code") == 0:
                    events.append(f"Orchestrator wrote {app_path}.")
                    if len(events) > 500:
                        del events[:-500]
                    start_cmd = _prepend_safe_dir(["bash", "-lc", "python3 app.py"])
                    start_entry = playground_manager.exec_cmd_detached(playground_name, start_cmd)
                    start_entry["cmd"] = " ".join(start_cmd)
                    start_entry["tool"] = "playground.exec_detached"
                    start_entry["agent"] = "orchestrator"
                    tool_context_chunks.append(_format_tool_context(start_entry))
                    playground_log.append(start_entry)
                    if start_entry.get("exit_code") == 0:
                        events.append("Orchestrator started hello world server (detached).")
                        if len(events) > 500:
                            del events[:-500]
                    verify_cmd = _prepend_safe_dir(["bash", "-lc", "sleep 1; curl -sf http://localhost:8000"])
                    verify_entry = playground_manager.exec_cmd(playground_name, verify_cmd, timeout_s=30)
                    verify_entry["cmd"] = " ".join(verify_cmd)
                    verify_entry["tool"] = "playground.exec"
                    verify_entry["agent"] = "orchestrator"
                    tool_context_chunks.append(_format_tool_context(verify_entry))
                    playground_log.append(verify_entry)
                    if verify_entry.get("exit_code") == 0:
                        events.append("Orchestrator verified hello world response.")
                        if len(events) > 500:
                            del events[:-500]
                else:
                    err = write_entry.get("stderr") or write_entry.get("stdout") or "unknown error"
                    events.append(f"Orchestrator failed to write {app_path}: {err}")
                    if len(events) > 500:
                        del events[:-500]
            if long_run_mode and use_playground:
                fixed_command = FIXED_PLAYGROUND_COMMANDS.get(stage_name)
                if fixed_command:
                    fixed_entry = _run_playground_command(
                        playground_name,
                        fixed_command,
                        playground_log,
                        tool_context_chunks,
                        agent=stage_name,
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
                            agent=stage_name,
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
                            agent=stage_name,
                        )
                    )
                    stage_trace.setdefault("playground_commands", []).extend(skeleton_entries)
        artifact_text = f"# {stage_name.title()} Output\n\n{result.output or ''}\n"
        artifact_path = _write_stage_artifact(run_id, stage_name, artifact_text)
        if artifact_path:
            events.append(f"{stage_name.title()} agent saved output to {artifact_path}.")
            if len(events) > 500:
                del events[:-500]

        trace["stages"][stage_name] = stage_trace
        if _agent_gave_up(result.output):
            trace["errors"].append(
                {"stage": stage_name, "error": f"Agent requested stop: {GIVE_UP_PHRASE}"}
            )
            failed = True
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
        requested = _extract_human_input_requests(result.output)
        if requested:
            awaiting_human_input = True
            human_input_requests.extend(requested)
            trace["human_input_requests"].extend(requested)
            events.append(
                f"{stage_name.title()} requested human input: " + " | ".join(requested)
            )
            if len(events) > 500:
                del events[:-500]
        if stage_name in {"supervisor", "planner"} and ops_escalation:
            ops_escalation = None
        if handoff_count < handoff_max and not awaiting_human_input:
            next_role = _extract_handoff(result.output)
            if next_role:
                stage_queue.insert(0, next_role)
                handoff_count += 1
                trace["handoffs"].append({"from": stage_name, "to": next_role, "count": handoff_count})
        return

    def _finalize_stage(stage_name: str) -> Optional[str]:
        nonlocal failed, ops_escalation, ops_fix_count, attempt, outputs, awaiting_human_input, human_input_requests, final_override
        total_ms = (time.perf_counter() - start_time) * 1000
        if awaiting_human_input:
            base_text = ""
            if outputs:
                base_text = outputs.get("aggregator", outputs.get(stage_name, AgentResult(stage_name, ""))).output
            human_block = _format_human_input_block(human_input_requests)
            summary = _format_access_summary(run_id, playground_info, cluster_info)
            final_text = "\n\n".join(filter(None, [base_text, human_block, summary])).strip()
            final_override = final_text
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
                events=events,
            )
            return final_text
        final_text = outputs.get("aggregator", AgentResult(stage_name, "")).output if outputs else ""
        if stage_name == "aggregator":
            summary = _format_access_summary(run_id, playground_info, cluster_info)
            final_text = f"{final_text}\n\n{summary}".strip()
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
            events=events,
        )

        if not failed and stage_name == "ops" and stage_name in outputs:
            ops_failure = _parse_ops_failure(outputs[stage_name].output)
            if ops_failure:
                trace["ops_escalations"].append(ops_failure)
                ops_fix_count += 1
                ops_escalation = "\n".join(
                    filter(
                        None,
                        [
                            f"Error: {ops_failure.get('error')}",
                            f"Instruction: {ops_failure.get('instruction')}",
                            f"Ops output:\n{ops_failure.get('raw')}",
                        ],
                    )
                )
                retry_stages = ["supervisor", "planner", "coder", "reviewer", "ops"]
                if "aggregator" in stage_queue:
                    insert_at = stage_queue.index("aggregator")
                    stage_queue[insert_at:insert_at] = retry_stages
                else:
                    stage_queue.extend(retry_stages)

        if not failed and stage_name == "aggregator":
            ok, reason = _completion_check()
            if not ok:
                events.append(f"Completion check failed: {reason}")
                if max_attempts == 0 or attempt < max_attempts:
                    attempt += 1
                    if max_attempts == 0:
                        events.append(f"Retrying attempt {attempt} (unlimited retries)")
                    else:
                        events.append(f"Retrying attempt {attempt} of {max_attempts}")
                    ops_escalation = f"Task incomplete: {reason}"
                    outputs.clear()
                    for stage in stages:
                        stage.status = "queued"
                        stage.ms = 0.0
                        stage.ttft_ms = 0.0
                        stage.tok_s = 0.0
                        stage.tokens = 0
                        stage.output = ""
                        stage.error = None
                    retry_stages = list(stage_order)
                    stage_queue.extend(retry_stages)
                    yield _serialize(
                        stages,
                        goal,
                        scenario,
                        final=final_text,
                        total_ms=total_ms,
                        playground=playground_info,
                        cluster=cluster_info,
                        dml=dml_info,
                        events=events,
                    )
                    return None
                if max_attempts != 0:
                    failed = True
                    trace["errors"].append({"stage": "completion", "error": reason})

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
                    events=events,
                )
            if use_cluster:
                cluster_info["ready_for_removal"] = True
            if outputs:
                return final_text
            return ""
        return None

    def _call_agent_timed(stage_name: str, extra_context: str, system_messages: List[str], max_tokens: int) -> tuple[str, AgentResult, float]:
        start = time.perf_counter()
        result = call_agent(
            stage_name,
            goal,
            scenario,
            max_tokens=max_tokens,
            extra_context=extra_context,
            system_messages=system_messages or None,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return stage_name, result, elapsed_ms

    while stage_queue:
        stage_name = stage_queue.pop(0)

        if parallel_enabled and stage_name == "planner":
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
                events=events,
            )
            extra_context = _build_extra_context(stage_name)
            max_tokens = 512 if fast else 1024
            system_messages = _build_system_messages(stage_name)
            try:
                _, result, elapsed_ms = _call_agent_timed(stage_name, extra_context, system_messages, max_tokens)
                _handle_stage_result(stage_name, result, elapsed_ms, extra_context, system_messages, max_tokens)
            except Exception as exc:  # noqa: BLE001
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                trace["stages"][stage_name] = {
                    "output": "",
                    "error": str(exc),
                    "ms": elapsed_ms,
                    "ttft_ms": elapsed_ms,
                    "tok_s": 0.0,
                    "tokens": 0,
                    "extra_context": extra_context,
                    "system_messages": system_messages,
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

            finalize_result = yield from _finalize_stage(stage_name)
            if finalize_result is not None:
                break
            continue

        if parallel_enabled and stage_name in {"coder", "reviewer", "ops"}:
            batch = [stage_name]
            while stage_queue and stage_queue[0] in {"coder", "reviewer", "ops"}:
                batch.append(stage_queue.pop(0))
            for name in batch:
                _update_stage(stages, name, status="running")
            yield _serialize(
                stages,
                goal,
                scenario,
                final="",
                total_ms=(time.perf_counter() - start_time) * 1000,
                playground=playground_info,
                cluster=cluster_info,
                dml=dml_info,
                events=events,
            )
            tool_context_snapshot = list(tool_context_chunks)
            shared_context = _build_extra_context(batch[0], tool_context_snapshot=tool_context_snapshot)
            futures = {}
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                for name in batch:
                    max_tokens = 512 if fast else 1024
                    system_messages = _build_system_messages(name)
                    futures[name] = executor.submit(_call_agent_timed, name, shared_context, system_messages, max_tokens)
            results: Dict[str, Dict[str, Any]] = {}
            for name, future in futures.items():
                try:
                    stage_key, result, elapsed_ms = future.result()
                    stage_system_messages = _build_system_messages(stage_key)
                    results[stage_key] = {
                        "result": result,
                        "elapsed_ms": elapsed_ms,
                        "system_messages": stage_system_messages,
                        "max_tokens": 512 if fast else 1024,
                        "error": None,
                    }
                except Exception as exc:  # noqa: BLE001
                    results[name] = {
                        "result": AgentResult(name, ""),
                        "elapsed_ms": 0.0,
                        "system_messages": _build_system_messages(name),
                        "max_tokens": 512 if fast else 1024,
                        "error": str(exc),
                    }

            for name in batch:
                try:
                    entry = results[name]
                    if entry.get("error"):
                        raise RuntimeError(entry["error"])
                    result = entry["result"]
                    elapsed_ms = float(entry["elapsed_ms"])
                    stage_system_messages = entry["system_messages"]
                    stage_max_tokens = int(entry["max_tokens"])
                    _handle_stage_result(
                        name,
                        result,
                        elapsed_ms,
                        shared_context,
                        stage_system_messages,
                        stage_max_tokens,
                    )
                except Exception as exc:  # noqa: BLE001
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    entry = results.get(name, {})
                    stage_system_messages = entry.get("system_messages", [])
                    stage_max_tokens = entry.get("max_tokens", 512 if fast else 1024)
                    trace["stages"][name] = {
                        "output": "",
                        "error": str(exc),
                        "ms": elapsed_ms,
                        "ttft_ms": elapsed_ms,
                        "tok_s": 0.0,
                        "tokens": 0,
                        "extra_context": shared_context,
                        "system_messages": stage_system_messages,
                        "max_tokens": stage_max_tokens,
                    }
                    trace["errors"].append({"stage": name, "error": str(exc)})
                    _update_stage(
                        stages,
                        name,
                        status="failed",
                        ms=elapsed_ms,
                        ttft_ms=elapsed_ms,
                        error=str(exc),
                    )
                    failed = True
                finalize_result = yield from _finalize_stage(name)
                if finalize_result is not None:
                    break
            if failed:
                break
            continue

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
            events=events,
        )

        stage_start = time.perf_counter()
        try:
            extra_context = _build_extra_context(stage_name)
            max_tokens = 512 if fast else 1024
            if stage_name in {"supervisor", "aggregator"}:
                max_tokens = 384 if fast else 768
            system_messages = _build_system_messages(stage_name)
            result = call_agent(
                stage_name,
                goal,
                scenario,
                max_tokens=max_tokens,
                extra_context=extra_context,
                system_messages=system_messages or None,
            )
            elapsed_ms = (time.perf_counter() - stage_start) * 1000
            _handle_stage_result(stage_name, result, elapsed_ms, extra_context, system_messages, max_tokens)
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
                "system_messages": system_messages,
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

        finalize_result = yield from _finalize_stage(stage_name)
        if finalize_result is not None:
            break

    total_ms = (time.perf_counter() - start_time) * 1000
    if final_override:
        final_text = final_override
    else:
        final_text = outputs.get("aggregator", AgentResult("aggregator", "")).output
        final_text = f"{final_text}\n\n{_format_access_summary(run_id, playground_info, cluster_info)}".strip()
    if playground_log:
        trace["playground"]["log"] = playground_log
    if use_cluster:
        if long_run_mode and not awaiting_human_input:
            fix_iterations: List[Dict[str, Any]] = []
            last_validation: Dict[str, Any] = {}
            iteration = 0
            while True:
                iteration += 1
                cluster_info["iteration"] = iteration
                validation = cluster_manager.validate_cluster(run_id)
                last_validation = validation
                cluster_info["validation"] = validation
                cluster_info["validation_history"].append({"iteration": iteration, "validation": validation})
                entry = {
                    "cmd": f"cluster.validate (iter {iteration})",
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
                    events=events,
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
                if _agent_gave_up(fixer_result.output):
                    failed = True
                    cluster_info["error"] = f"Agent requested stop: {GIVE_UP_PHRASE}"
                    break
                tool_requests = _extract_tool_requests(fixer_result.output)
                tool_entries: List[Dict[str, Any]] = []
                fix_actions: List[Dict[str, Any]] = []
                for request in tool_requests:
                    tool_name = request.get("tool")
                    entry = {}
                    if tool_name in {"playground.exec", "playground.exec_detached"} and use_playground:
                        cmd = request.get("cmd")
                        timeout_s = int(request.get("timeout_s", 60))
                        detach = bool(request.get("detach")) or tool_name == "playground.exec_detached"
                        if not isinstance(cmd, list) or not all(isinstance(arg, str) for arg in cmd):
                            entry = {
                                "cmd": str(cmd),
                                "exit_code": 125,
                                "stdout": "",
                                "stderr": "Invalid command format. Expected list[str].",
                            }
                        else:
                            if detach:
                                entry = playground_manager.exec_cmd_detached(playground_name, cmd)
                            else:
                                entry = playground_manager.exec_cmd(playground_name, cmd, timeout_s=timeout_s)
                            entry["cmd"] = " ".join(cmd)
                        entry["agent"] = "fixer"
                        tool_context_chunks.append(_format_tool_context(entry))
                        playground_log.append(entry)
                    elif tool_name == "playground.expose_port" and use_playground:
                        host_port = request.get("host_port") or request.get("port")
                        container_port = request.get("container_port") or request.get("target_port") or host_port
                        try:
                            host_port = int(host_port)
                            container_port = int(container_port)
                        except (TypeError, ValueError):
                            entry = {
                                "cmd": f"expose_port {host_port}:{container_port}",
                                "exit_code": 125,
                                "stdout": "",
                                "stderr": "Invalid port values. Expected integers.",
                            }
                        else:
                            result = playground_manager.expose_port(playground_name, host_port, container_port)
                            if result.get("ok"):
                                playground_info["web_port"] = host_port
                                playground_info.setdefault("exposed_ports", [])
                                if host_port not in playground_info["exposed_ports"]:
                                    playground_info["exposed_ports"].append(host_port)
                                events.append(f"Port {host_port} exposed to playground {playground_name}.")
                                entry = {
                                    "cmd": f"expose_port {host_port}:{container_port}",
                                    "exit_code": 0,
                                    "stdout": json.dumps(result),
                                    "stderr": "",
                                }
                            else:
                                entry = {
                                    "cmd": f"expose_port {host_port}:{container_port}",
                                    "exit_code": 1,
                                    "stdout": "",
                                    "stderr": result.get("error", "Failed to expose port."),
                                }
                        entry["agent"] = "fixer"
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
                        entry["agent"] = "fixer"
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
                        entry["agent"] = "fixer"
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
                        entry["agent"] = "fixer"
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
                    events=events,
                )
            trace["cluster"]["fix_iterations"] = fix_iterations
            if last_validation and not last_validation.get("ok") and not failed:
                failed = True
                cluster_info["error"] = f"Validation failed after {iteration} iterations."
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
        events=events,
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

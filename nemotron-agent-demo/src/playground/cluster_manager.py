from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import policy
from .manager import WORKSPACE_ROOT

CLUSTER_PREFIX = "autonomous-play-"
NETWORK_PREFIX = "autonomous-net-"
DEFAULT_API_INTERNAL_PORT = 8080
DEFAULT_WEB_INTERNAL_PORT = 7860
API_HOST_PORT_BASE = 18000
WEB_HOST_PORT_BASE = 19000
HOST_PORT_SPAN = 1000
RESOURCE_LIMITS = {
    "cpus": "2",
    "memory": "4g",
    "pids": "256",
}
REDIS_IMAGE = "redis:7-alpine"
ALLOWED_DOCKER_COMMANDS = {"inspect", "run", "exec", "rm", "network", "ps", "logs"}
ALLOWED_NETWORK_SUBCOMMANDS = {"create", "rm", "inspect"}


def _run_docker(args: List[str], timeout_s: int = 30) -> subprocess.CompletedProcess | None:
    if not args:
        return subprocess.CompletedProcess(["docker"], 125, stdout="", stderr="No docker arguments provided.")
    if args[0] not in ALLOWED_DOCKER_COMMANDS:
        return subprocess.CompletedProcess(["docker", *args], 125, stdout="", stderr="Unsupported docker command.")
    if args[0] == "network":
        if len(args) < 2 or args[1] not in ALLOWED_NETWORK_SUBCOMMANDS:
            return subprocess.CompletedProcess(["docker", *args], 125, stdout="", stderr="Unsupported network command.")
    try:
        return subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            exc.cmd,
            124,
            stdout=exc.stdout or "",
            stderr="Command timed out.",
        )
    except FileNotFoundError:
        return None


def _docker_error(result: subprocess.CompletedProcess | None, fallback: str) -> str:
    if result is None:
        return fallback
    detail = result.stderr.strip() or result.stdout.strip()
    return detail or fallback


def _validate_run_id(run_id: str) -> str | None:
    if not isinstance(run_id, str) or not run_id.strip():
        return "Run ID must be a non-empty string."
    if "/" in run_id or " " in run_id:
        return "Run ID contains invalid characters."
    return None


def _cluster_prefix(run_id: str) -> str:
    return f"{CLUSTER_PREFIX}{run_id}-"


def _network_name(run_id: str) -> str:
    return f"{NETWORK_PREFIX}{run_id}"


def _port_for_run(run_id: str, base: int) -> int:
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    offset = int(digest[:6], 16) % HOST_PORT_SPAN
    return base + offset


def _validate_container_name(name: str, run_id: str) -> str | None:
    if not name.startswith(_cluster_prefix(run_id)):
        return f"Container name must start with '{_cluster_prefix(run_id)}'."
    return None


def _extract_run_id(name: str) -> Optional[str]:
    if not name.startswith(CLUSTER_PREFIX):
        return None
    for suffix in ("-api", "-redis", "-web", "-worker"):
        if name.endswith(suffix):
            run_id = name[len(CLUSTER_PREFIX) : -len(suffix)]
            return run_id or None
    marker = "-worker-"
    if marker in name:
        run_id = name[len(CLUSTER_PREFIX) : name.rfind(marker)]
        return run_id or None
    remainder = name[len(CLUSTER_PREFIX) :]
    parts = remainder.rsplit("-", 1)
    if len(parts) != 2:
        return None
    return parts[0] or None


def _clamp(text: str) -> str:
    return policy.clamp_output(text)


def _format_log_entry(cmd: List[str], result: subprocess.CompletedProcess | None) -> Dict[str, Any]:
    if result is None:
        return {
            "cmd": " ".join(cmd),
            "exit_code": 127,
            "stdout": "",
            "stderr": "docker not available",
        }
    return {
        "cmd": " ".join(cmd),
        "exit_code": result.returncode,
        "stdout": _clamp(result.stdout),
        "stderr": _clamp(result.stderr),
    }


def _inspect_exists(name: str) -> bool:
    result = _run_docker(["inspect", name])
    return bool(result and result.returncode == 0)


def _build_worker_names(prefix: str, count: int) -> List[str]:
    if count <= 1:
        return [f"{prefix}worker"]
    names = [f"{prefix}worker"]
    for idx in range(2, count + 1):
        names.append(f"{prefix}worker-{idx}")
    return names


def _role_start_command(role: str) -> List[str]:
    if role == "api":
        launch = (
            "if [ -f /workspace/api_service/main.py ]; then "
            "python -m api_service.main; "
            "elif [ -f /workspace/api_service/app.py ]; then "
            "python -m api_service.app; "
            "else echo 'api_service entrypoint not found.' >&2; exit 1; fi"
        )
    elif role == "worker":
        launch = (
            "if [ -f /workspace/worker_service/main.py ]; then "
            "python -m worker_service.main; "
            "elif [ -f /workspace/worker_service/app.py ]; then "
            "python -m worker_service.app; "
            "else echo 'worker_service entrypoint not found.' >&2; exit 1; fi"
        )
    elif role == "web":
        launch = (
            "if [ -f /workspace/webui/app.py ]; then "
            "python -m webui.app; "
            "elif [ -f /workspace/webui/main.py ]; then "
            "python -m webui.main; "
            "else echo 'webui entrypoint not found.' >&2; exit 1; fi"
        )
    else:
        raise ValueError(f"Unsupported role: {role}")
    return ["bash", "-lc", launch]


def _list_cluster_containers(prefix: str) -> List[str]:
    result = _run_docker(["ps", "-a", "--format", "{{.Names}}", "--filter", f"name=^{prefix}"])
    if result is None or result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _container_role(name: str, prefix: str) -> Optional[str]:
    if name == f"{prefix}redis":
        return "redis"
    if name == f"{prefix}api":
        return "api"
    if name == f"{prefix}web":
        return "web"
    if name.startswith(f"{prefix}worker"):
        return "worker"
    return None


def create_cluster(run_id: str, image: str, size: int, workspace_host: Optional[str]) -> Dict[str, Any]:
    error = _validate_run_id(run_id)
    if error:
        return {"ok": False, "status": "blocked", "error": error}
    if size < 3 or size > 5:
        return {"ok": False, "status": "blocked", "error": "Cluster size must be between 3 and 5."}
    if not image:
        return {"ok": False, "status": "blocked", "error": "Cluster image must be provided."}

    prefix = _cluster_prefix(run_id)
    network = _network_name(run_id)
    workspace_root = Path(workspace_host) if workspace_host else (WORKSPACE_ROOT / f"cluster-{run_id}")
    workspace_root.mkdir(parents=True, exist_ok=True)
    api_port = _port_for_run(run_id, API_HOST_PORT_BASE)
    web_port = _port_for_run(run_id, WEB_HOST_PORT_BASE)
    include_web = size >= 4
    worker_count = max(1, size - 2 - (1 if include_web else 0))

    container_defs = [
        ("redis", f"{prefix}redis", REDIS_IMAGE),
        ("api", f"{prefix}api", image),
    ]
    for worker_name in _build_worker_names(prefix, worker_count):
        container_defs.append(("worker", worker_name, image))
    if include_web:
        container_defs.append(("web", f"{prefix}web", image))

    network_check = _run_docker(["network", "inspect", network])
    existing_containers = _list_cluster_containers(prefix)
    if network_check and network_check.returncode == 0 and existing_containers:
        containers = []
        has_web = False
        for name in existing_containers:
            role = _container_role(name, prefix)
            if role:
                containers.append({"role": role, "name": name})
                if role == "web":
                    has_web = True
        return {
            "ok": True,
            "status": "running",
            "network": network,
            "containers": containers,
            "api_port": api_port,
            "web_port": web_port if has_web else None,
            "workspace_host": str(workspace_root),
            "workspace_container": policy.WORKSPACE_ROOT,
            "log": [],
            "error": None,
            "reused": True,
        }
    if network_check and network_check.returncode == 0 and not existing_containers:
        return {
            "ok": False,
            "status": "error",
            "error": f"Network {network} already exists without containers.",
        }
    if existing_containers and not (network_check and network_check.returncode == 0):
        return {
            "ok": False,
            "status": "error",
            "error": f"Cluster containers already exist without network {network}.",
        }
    for _, name, _ in container_defs:
        if _inspect_exists(name):
            return {"ok": False, "status": "error", "error": f"Container {name} already exists."}

    network_result = _run_docker(["network", "create", network])
    if network_result is None or network_result.returncode != 0:
        return {"ok": False, "status": "error", "error": _docker_error(network_result, "Failed to create network.")}

    log: List[Dict[str, Any]] = [_format_log_entry(["network", "create", network], network_result)]
    containers: List[Dict[str, str]] = []
    redis_url = f"redis://{prefix}redis:6379/0"
    api_url = f"http://{prefix}api:{DEFAULT_API_INTERNAL_PORT}"

    for role, name, image_name in container_defs:
        name_error = _validate_container_name(name, run_id)
        if name_error:
            return {"ok": False, "status": "error", "error": name_error}
        cmd = [
            "run",
            "-d",
            "--name",
            name,
            "--network",
            network,
            "--cpus",
            RESOURCE_LIMITS["cpus"],
            "--memory",
            RESOURCE_LIMITS["memory"],
            "--pids-limit",
            RESOURCE_LIMITS["pids"],
            "--security-opt",
            "no-new-privileges",
        ]
        if role != "redis":
            cmd += [
                "-v",
                f"{workspace_root}:/workspace",
                "-w",
                policy.WORKSPACE_ROOT,
                "--env",
                f"REDIS_URL={redis_url}",
                "--env",
                f"API_URL={api_url}",
                "--env",
                f"SERVICE_ROLE={role}",
                "--user",
                policy.PLAYGROUND_USER,
            ]
        if role == "api":
            cmd += ["-p", f"{api_port}:{DEFAULT_API_INTERNAL_PORT}"]
        if role == "web":
            cmd += ["-p", f"{web_port}:{DEFAULT_WEB_INTERNAL_PORT}"]
        cmd.append(image_name)
        if role != "redis":
            cmd += _role_start_command(role)
        result = _run_docker(cmd, timeout_s=60)
        log.append(_format_log_entry(cmd, result))
        if result is None or result.returncode != 0:
            return {
                "ok": False,
                "status": "error",
                "error": _docker_error(result, f"Failed to create {role} container."),
                "log": log,
            }
        containers.append({"role": role, "name": name})

    return {
        "ok": True,
        "status": "running",
        "network": network,
        "containers": containers,
        "api_port": api_port,
        "web_port": web_port if include_web else None,
        "workspace_host": str(workspace_root),
        "workspace_container": policy.WORKSPACE_ROOT,
        "log": log,
        "error": None,
        "reused": False,
    }


def exec_in(name: str, cmd: List[str], timeout_s: int = policy.DEFAULT_COMMAND_TIMEOUT_S) -> Dict[str, Any]:
    run_id = _extract_run_id(name)
    if not run_id:
        return {"exit_code": 126, "stdout": "", "stderr": "Container name must use autonomous-play-<runid>- prefix."}
    name_error = _validate_container_name(name, run_id)
    if name_error:
        return {"exit_code": 126, "stdout": "", "stderr": name_error}
    allowed, reason = policy.validate_command(cmd)
    if not allowed:
        return {"exit_code": 126, "stdout": "", "stderr": reason or "Command rejected by policy."}
    exec_args = [
        "exec",
        "-w",
        policy.WORKSPACE_ROOT,
        name,
        "timeout",
        "--signal=KILL",
        "--preserve-status",
        f"{timeout_s}s",
        *cmd,
    ]
    result = _run_docker(exec_args, timeout_s=timeout_s + 15)
    if result is None:
        return {"exit_code": 127, "stdout": "", "stderr": "docker not available"}
    return {
        "exit_code": result.returncode,
        "stdout": _clamp(result.stdout),
        "stderr": _clamp(result.stderr),
    }


def container_logs(name: str, tail: int = 200, timeout_s: int = 15) -> Dict[str, Any]:
    run_id = _extract_run_id(name)
    if not run_id:
        return {"exit_code": 126, "stdout": "", "stderr": "Container name must use autonomous-play-<runid>- prefix."}
    name_error = _validate_container_name(name, run_id)
    if name_error:
        return {"exit_code": 126, "stdout": "", "stderr": name_error}
    if not isinstance(tail, int):
        return {"exit_code": 126, "stdout": "", "stderr": "Log tail must be an integer."}
    tail = max(1, min(tail, 2000))
    result = _run_docker(["logs", "--tail", str(tail), name], timeout_s=timeout_s)
    if result is None:
        return {"exit_code": 127, "stdout": "", "stderr": "docker not available"}
    return {
        "exit_code": result.returncode,
        "stdout": _clamp(result.stdout),
        "stderr": _clamp(result.stderr),
    }


def validate_cluster(run_id: str) -> Dict[str, Any]:
    error = _validate_run_id(run_id)
    if error:
        return {"ok": False, "error": error, "checks": []}
    prefix = _cluster_prefix(run_id)
    network = _network_name(run_id)
    checks: List[Dict[str, Any]] = []

    network_result = _run_docker(["network", "inspect", network])
    network_ok = bool(network_result and network_result.returncode == 0)
    checks.append(
        {
            "name": "network",
            "ok": network_ok,
            "detail": _docker_error(network_result, "network not found"),
        }
    )

    ps_result = _run_docker(["ps", "-a", "--format", "{{.Names}}", "--filter", f"name=^{prefix}"])
    names = []
    if ps_result and ps_result.returncode == 0:
        names = [line.strip() for line in ps_result.stdout.splitlines() if line.strip()]
    checks.append({"name": "containers_present", "ok": bool(names), "detail": ", ".join(names) or "none"})

    api_name = f"{prefix}api"
    web_name = f"{prefix}web"
    worker_name = f"{prefix}worker"

    if api_name in names:
        health = exec_in(api_name, ["curl", "-sf", f"http://localhost:{DEFAULT_API_INTERNAL_PORT}/health"], timeout_s=10)
        checks.append(
            {
                "name": "api_health",
                "ok": health["exit_code"] == 0,
                "detail": health["stdout"] or health["stderr"],
            }
        )
        redis_ping = exec_in(
            api_name,
            [
                "python",
                "-c",
                (
                    "import os,redis;"
                    "url=os.getenv('REDIS_URL','');"
                    "r=redis.Redis.from_url(url);"
                    "print(r.ping())"
                ),
            ],
            timeout_s=10,
        )
        checks.append(
            {
                "name": "api_to_redis",
                "ok": redis_ping["exit_code"] == 0 and "True" in redis_ping["stdout"],
                "detail": redis_ping["stdout"] or redis_ping["stderr"],
            }
        )
    else:
        checks.append({"name": "api_health", "ok": False, "detail": f"{api_name} missing"})
        checks.append({"name": "api_to_redis", "ok": False, "detail": f"{api_name} missing"})

    if worker_name in names:
        worker_ping = exec_in(
            worker_name,
            [
                "python",
                "-c",
                (
                    "import os,redis;"
                    "url=os.getenv('REDIS_URL','');"
                    "r=redis.Redis.from_url(url);"
                    "print(r.ping())"
                ),
            ],
            timeout_s=10,
        )
        checks.append(
            {
                "name": "worker_to_redis",
                "ok": worker_ping["exit_code"] == 0 and "True" in worker_ping["stdout"],
                "detail": worker_ping["stdout"] or worker_ping["stderr"],
            }
        )
    else:
        checks.append({"name": "worker_to_redis", "ok": False, "detail": f"{worker_name} missing"})

    if web_name in names:
        web_health = exec_in(web_name, ["curl", "-sf", f"http://{api_name}:{DEFAULT_API_INTERNAL_PORT}/health"], timeout_s=10)
        checks.append(
            {
                "name": "web_to_api",
                "ok": web_health["exit_code"] == 0,
                "detail": web_health["stdout"] or web_health["stderr"],
            }
        )
    else:
        checks.append({"name": "web_to_api", "ok": False, "detail": f"{web_name} missing"})

    end_to_end = {"name": "end_to_end_job", "ok": False, "detail": "skipped"}
    if api_name in names and web_name in names:
        job = exec_in(
            web_name,
            [
                "python",
                "-c",
                (
                    "import json,requests,time\n"
                    f"api='http://{api_name}:{DEFAULT_API_INTERNAL_PORT}'\n"
                    "resp=requests.post(f'{api}/job',json={'input':'cluster-check'})\n"
                    "resp.raise_for_status()\n"
                    "job_id=resp.json().get('job_id')\n"
                    "status={'status':'unknown'}\n"
                    "deadline=time.time()+5\n"
                    "while time.time()<deadline:\n"
                    "    r=requests.get(f'{api}/job/{job_id}')\n"
                    "    r.raise_for_status()\n"
                    "    status=r.json()\n"
                    "    if status.get('status')=='done':\n"
                    "        break\n"
                    "    time.sleep(0.5)\n"
                    "print(json.dumps(status))\n"
                ),
            ],
            timeout_s=10,
        )
        if job["exit_code"] == 0 and job["stdout"]:
            end_to_end["detail"] = job["stdout"]
            end_to_end["ok"] = '"status": "done"' in job["stdout"] or "'status': 'done'" in job["stdout"]
        else:
            end_to_end["detail"] = job["stderr"] or job["stdout"]
    checks.append(end_to_end)

    ok = all(check.get("ok") for check in checks if check["name"] != "end_to_end_job")
    if end_to_end["detail"] != "skipped":
        ok = ok and end_to_end["ok"]
    return {"ok": ok, "error": None if ok else "Validation failed", "checks": checks, "timestamp": time.time()}


def destroy_cluster(run_id: str) -> Dict[str, Any]:
    error = _validate_run_id(run_id)
    if error:
        return {"ok": False, "error": error}
    prefix = _cluster_prefix(run_id)
    network = _network_name(run_id)
    removed: List[str] = []
    errors: List[str] = []

    ps_result = _run_docker(["ps", "-a", "--format", "{{.Names}}", "--filter", f"name=^{prefix}"])
    if ps_result is None:
        return {"ok": False, "error": "docker not available"}
    if ps_result.returncode == 0:
        for name in [line.strip() for line in ps_result.stdout.splitlines() if line.strip()]:
            if not name.startswith(prefix):
                continue
            result = _run_docker(["rm", "-f", name])
            if result and result.returncode == 0:
                removed.append(name)
            else:
                errors.append(_docker_error(result, f"Failed to remove {name}."))
    else:
        errors.append(_docker_error(ps_result, "Failed to list containers."))

    if _inspect_exists(network):
        net_result = _run_docker(["network", "rm", network])
        if net_result and net_result.returncode == 0:
            removed.append(network)
        else:
            errors.append(_docker_error(net_result, f"Failed to remove network {network}."))

    return {"ok": not errors, "removed": removed, "error": "; ".join(errors) if errors else None}

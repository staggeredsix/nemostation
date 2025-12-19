from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from . import policy

BASE_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = BASE_DIR / "playground_workspace"
DOCKER_ALLOWED_COMMANDS = {"inspect", "start", "run", "exec", "rm"}
DOCKER_BLOCKED_COMMANDS = {"prune", "rmi"}
DOCKER_BLOCKED_SUBCOMMANDS = {("system", "prune"), ("volume", "rm"), ("network", "rm"), ("image", "rm")}


def _run_docker(args: List[str], timeout_s: int = 30) -> subprocess.CompletedProcess | None:
    if not args:
        return subprocess.CompletedProcess(["docker"], 125, stdout="", stderr="No docker arguments provided.")
    if args[0] in DOCKER_BLOCKED_COMMANDS:
        return subprocess.CompletedProcess(["docker", *args], 125, stdout="", stderr="Blocked destructive docker command.")
    if args[0] not in DOCKER_ALLOWED_COMMANDS and (len(args) < 2 or (args[0], args[1]) not in DOCKER_BLOCKED_SUBCOMMANDS):
        return subprocess.CompletedProcess(["docker", *args], 125, stdout="", stderr="Unsupported docker command.")
    if len(args) >= 2 and (args[0], args[1]) in DOCKER_BLOCKED_SUBCOMMANDS:
        return subprocess.CompletedProcess(["docker", *args], 125, stdout="", stderr="Blocked destructive docker subcommand.")
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


def _validate_container_name(name: str) -> str | None:
    allowed, reason = policy.validate_container_name(name)
    if not allowed:
        return reason or "Container name rejected."
    return None


def status(name: str) -> Dict:
    error = _validate_container_name(name)
    if error:
        return {"name": name, "exists": False, "running": False, "status": "blocked", "error": error}
    result = _run_docker(["inspect", name, "--format", "{{json .State}}"])
    if result is None:
        return {"name": name, "exists": False, "running": False, "status": "missing", "error": "docker not available"}
    if result.returncode != 0:
        return {"name": name, "exists": False, "running": False, "status": "missing", "error": _docker_error(result, "container not found")}
    try:
        state = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return {"name": name, "exists": True, "running": False, "status": "unknown", "error": "invalid state"}
    status_value = state.get("Status", "unknown")
    return {
        "name": name,
        "exists": True,
        "running": bool(state.get("Running", False)),
        "status": status_value,
    }


def ensure_playground(image: str, name: str, repo_mount: Optional[str] = None) -> Dict:
    error = _validate_container_name(name)
    if error:
        return {"name": name, "exists": False, "running": False, "status": "blocked", "error": error}
    current = status(name)
    if current.get("exists") and current.get("running"):
        current.update({"image": image, "workspace": str(WORKSPACE_ROOT / name)})
        return current
    if current.get("exists") and not current.get("running"):
        result = _run_docker(["start", name])
        if result is None:
            current.update({"error": "docker not available"})
            return current
        if result.returncode != 0:
            current.update({"error": _docker_error(result, "failed to start container")})
            return current
        current = status(name)
        current.update({"image": image, "workspace": str(WORKSPACE_ROOT / name)})
        return current

    workspace_host = WORKSPACE_ROOT / name
    workspace_host.mkdir(parents=True, exist_ok=True)
    volume_args = ["-v", f"{workspace_host.resolve()}:/workspace"]
    if repo_mount:
        volume_args += ["-v", f"{Path(repo_mount).resolve()}:/workspace/app"]

    result = _run_docker(
        [
            "run",
            "-d",
            "--name",
            name,
            "--cpus",
            "8",
            "--memory",
            "16g",
            "--pids-limit",
            "512",
            "--security-opt",
            "no-new-privileges",
            "--user",
            policy.PLAYGROUND_USER,
            *volume_args,
            image,
            "sleep",
            "infinity",
        ]
    )
    if result is None:
        return {
            "name": name,
            "exists": False,
            "running": False,
            "status": "missing",
            "error": "docker not available",
        }
    if result.returncode != 0:
        return {
            "name": name,
            "exists": False,
            "running": False,
            "status": "error",
            "error": _docker_error(result, "failed to start playground"),
        }
    current = status(name)
    current.update({"image": image, "workspace": str(workspace_host)})
    return current


def exec_cmd(name: str, cmd: List[str], timeout_s: int = policy.DEFAULT_COMMAND_TIMEOUT_S) -> Dict:
    name_error = _validate_container_name(name)
    if name_error:
        return {
            "exit_code": 126,
            "stdout": "",
            "stderr": name_error,
        }
    allowed, reason = policy.validate_command(cmd)
    if not allowed:
        return {
            "exit_code": 126,
            "stdout": "",
            "stderr": reason or "Command rejected by policy.",
        }

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
        return {
            "exit_code": 127,
            "stdout": "",
            "stderr": "docker not available",
        }
    stdout = policy.clamp_output(result.stdout)
    stderr = policy.clamp_output(result.stderr)
    return {
        "exit_code": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def remove_playground(name: str) -> Dict:
    error = _validate_container_name(name)
    if error:
        return {"ok": False, "error": error}
    result = _run_docker(["rm", "-f", name])
    if result is None:
        return {"ok": False, "error": "docker not available"}
    if result.returncode != 0:
        return {"ok": False, "error": _docker_error(result, "failed to remove container")}
    return {"ok": True, "error": None}

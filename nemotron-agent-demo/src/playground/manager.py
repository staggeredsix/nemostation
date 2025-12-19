from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from . import policy

BASE_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = BASE_DIR / "playground_workspace"


def _run_docker(args: List[str], timeout_s: int = 30) -> subprocess.CompletedProcess | None:
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


def status(name: str) -> Dict:
    result = _run_docker(["inspect", name, "--format", "{{json .State}}"])
    if result is None:
        return {"name": name, "exists": False, "running": False, "status": "missing", "error": "docker not available"}
    if result.returncode != 0:
        return {"name": name, "exists": False, "running": False, "status": "missing"}
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


def exec_cmd(name: str, cmd: List[str], timeout_s: int = 60) -> Dict:
    allowed, reason = policy.validate_command(cmd)
    if not allowed:
        return {
            "exit_code": 126,
            "stdout": "",
            "stderr": reason or "Command rejected by policy.",
        }

    result = _run_docker(["exec", "-w", policy.WORKSPACE_ROOT, name, *cmd], timeout_s=timeout_s)
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
    result = _run_docker(["rm", "-f", name])
    if result is None:
        return {"ok": False, "error": "docker not available"}
    if result.returncode != 0:
        return {"ok": False, "error": _docker_error(result, "failed to remove container")}
    return {"ok": True, "error": None}

from __future__ import annotations

import json
import subprocess
import base64
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

from . import policy

BASE_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = BASE_DIR / "playground_workspace"
AGENT_PROJECTS_ROOT = BASE_DIR / "agent_projects"
DOCKER_ALLOWED_COMMANDS = {"inspect", "start", "run", "exec", "rm", "build", "stop", "restart", "ps", "logs", "pull"}
DOCKER_BLOCKED_COMMANDS = {"prune", "rmi"}
DOCKER_BLOCKED_SUBCOMMANDS = {("system", "prune"), ("volume", "rm"), ("network", "rm"), ("image", "rm")}
ALLOWED_CONTAINER_PREFIXES = ("nemotron-playground-", "nemotron-play-")
DEFAULT_PLAYGROUND_IMAGE = "nemotron-playground:latest"
FALLBACK_PLAYGROUND_IMAGE = "python:3.11-slim"
MISSING_IMAGE_HINTS = ("No such image", "No such object", "not found", "pull access denied")

PLAYGROUND_DOCKERFILE = BASE_DIR / "playground" / "Dockerfile"
PLAYGROUND_BUILD_CONTEXT = BASE_DIR / "playground"


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


def _validate_docker_args(args: List[str]) -> Tuple[bool, str | None]:
    if not args:
        return False, "No docker arguments provided."
    cmd = args[0]
    if cmd not in DOCKER_ALLOWED_COMMANDS:
        return False, f"Docker command '{cmd}' is not allowed."
    if cmd == "run":
        if "--name" not in args:
            return False, "Docker run requires --name with an allowed prefix."
        name_index = args.index("--name") + 1
        if name_index >= len(args):
            return False, "Docker run requires a name after --name."
        name = args[name_index]
        if not any(name.startswith(prefix) for prefix in ALLOWED_CONTAINER_PREFIXES):
            return False, f"Container name must start with one of: {', '.join(ALLOWED_CONTAINER_PREFIXES)}"
    if cmd in {"start", "stop", "restart", "rm", "logs", "inspect", "exec"}:
        names = [arg for arg in args[1:] if arg and not arg.startswith("-")]
        for name in names:
            if name in {"--format", "--tail"}:
                continue
            if not any(name.startswith(prefix) for prefix in ALLOWED_CONTAINER_PREFIXES):
                return False, f"Container name must start with one of: {', '.join(ALLOWED_CONTAINER_PREFIXES)}"
    return True, None


def _docker_error(result: subprocess.CompletedProcess | None, fallback: str) -> str:
    if result is None:
        return fallback
    detail = result.stderr.strip() or result.stdout.strip()
    return detail or fallback


def docker_cmd(args: List[str], timeout_s: int = 60) -> Dict[str, Any]:
    ok, reason = _validate_docker_args(args)
    if not ok:
        return {"exit_code": 126, "stdout": "", "stderr": reason or "Invalid docker command."}
    result = _run_docker(args, timeout_s=timeout_s)
    if result is None:
        return {"exit_code": 127, "stdout": "", "stderr": "docker not available"}
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
    }


def _validate_container_name(name: str) -> str | None:
    allowed, reason = policy.validate_container_name(name)
    if not allowed:
        return reason or "Container name rejected."
    return None


def _resolve_playground_image(image: str) -> tuple[str, Optional[str]]:
    result = _run_docker(["inspect", "--type=image", image])
    if result is None:
        return image, None
    if result.returncode == 0:
        return image, None
    detail = (result.stderr or result.stdout).strip()
    if any(hint in detail for hint in MISSING_IMAGE_HINTS):
        if image == DEFAULT_PLAYGROUND_IMAGE and PLAYGROUND_DOCKERFILE.exists():
            build_result = _run_docker(
                [
                    "build",
                    "-t",
                    image,
                    "-f",
                    str(PLAYGROUND_DOCKERFILE),
                    str(PLAYGROUND_BUILD_CONTEXT),
                ],
                timeout_s=600,
            )
            if build_result is not None and build_result.returncode == 0:
                return image, f"Built playground image {image} locally."
            build_error = _docker_error(build_result, "failed to build playground image")
            return (
                FALLBACK_PLAYGROUND_IMAGE,
                f"Playground image {image} missing and build failed ({build_error}). "
                f"Falling back to {FALLBACK_PLAYGROUND_IMAGE}.",
            )
        return (
            FALLBACK_PLAYGROUND_IMAGE,
            f"Playground image {image} not found locally. Falling back to {FALLBACK_PLAYGROUND_IMAGE}.",
        )
    return image, None


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


def ensure_playground(image: str, name: str, run_id: str, repo_mount: Optional[str] = None) -> Dict:
    error = _validate_container_name(name)
    if error:
        return {"name": name, "exists": False, "running": False, "status": "blocked", "error": error}
    workspace_host = (WORKSPACE_ROOT / run_id).resolve()
    agent_projects_host = (AGENT_PROJECTS_ROOT / run_id).resolve()
    workspace_container = policy.WORKSPACE_ROOT
    workspace_host.mkdir(parents=True, exist_ok=True)
    agent_projects_host.mkdir(parents=True, exist_ok=True)
    current = status(name)
    if current.get("exists") and current.get("running"):
        current.update(
            {
                "image": image,
                "workspace_host": str(workspace_host),
                "workspace_container": workspace_container,
                "container": name,
            }
        )
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
        current.update(
            {
                "image": image,
                "workspace_host": str(workspace_host),
                "workspace_container": workspace_container,
                "container": name,
            }
        )
        return current

    volume_args = [
        "-v",
        f"{workspace_host}:/workspace",
        "-v",
        f"{agent_projects_host}:/workspace/agent_projects",
    ]
    if repo_mount:
        volume_args += ["-v", f"{Path(repo_mount).resolve()}:/workspace/app"]

    resolved_image, warning = _resolve_playground_image(image)
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
            "-w",
            workspace_container,
            *volume_args,
            resolved_image,
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
            "warning": warning,
            "requested_image": image,
        }
    current = status(name)
    current.update(
        {
            "image": resolved_image,
            "workspace_host": str(workspace_host),
            "workspace_container": workspace_container,
            "container": name,
            "warning": warning,
            "requested_image": image,
        }
    )
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
        policy.SAFE_WORKSPACE_ROOT,
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


def write_file(name: str, path: str, content: str) -> Dict:
    name_error = _validate_container_name(name)
    if name_error:
        return {"exit_code": 126, "stdout": "", "stderr": name_error}
    if not isinstance(path, str) or not path:
        return {"exit_code": 126, "stdout": "", "stderr": "Path must be a non-empty string."}
    if not isinstance(content, str):
        return {"exit_code": 126, "stdout": "", "stderr": "Content must be a string."}
    try:
        resolved = PurePosixPath(path)
    except ValueError:
        return {"exit_code": 126, "stdout": "", "stderr": "Invalid path format."}
    if not resolved.is_absolute():
        return {"exit_code": 126, "stdout": "", "stderr": "Path must be absolute under /workspace/agent_projects."}
    if ".." in resolved.parts:
        return {"exit_code": 126, "stdout": "", "stderr": "Parent directory traversal is not allowed."}
    workspace_root = PurePosixPath(policy.SAFE_WORKSPACE_ROOT)
    if resolved != workspace_root and workspace_root not in resolved.parents:
        return {"exit_code": 126, "stdout": "", "stderr": "Path must be under /workspace/agent_projects."}

    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    payload = json.dumps({"path": str(resolved), "data": encoded})
    cmd = [
        "python",
        "-c",
        (
            "import base64, json, pathlib;"
            f"payload=json.loads({payload!r});"
            "path=pathlib.Path(payload['path']);"
            "path.parent.mkdir(parents=True, exist_ok=True);"
            "path.write_bytes(base64.b64decode(payload['data']));"
            "print(f'Wrote {path}')"
        ),
    ]
    return exec_cmd(name, cmd)


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


def remove_workspace(path: str) -> Dict:
    if not path:
        return {"ok": False, "error": "Workspace path is required."}
    workspace_root = WORKSPACE_ROOT.resolve()
    try:
        resolved = Path(path).resolve()
    except FileNotFoundError:
        return {"ok": False, "error": "Workspace path not found."}
    try:
        resolved.relative_to(workspace_root)
    except ValueError:
        return {"ok": False, "error": "Workspace path is outside the allowed root."}
    if resolved == workspace_root:
        return {"ok": False, "error": "Refusing to delete workspace root."}
    if not resolved.exists():
        return {"ok": False, "error": "Workspace path not found."}
    import shutil

    shutil.rmtree(resolved)
    return {"ok": True, "error": None}

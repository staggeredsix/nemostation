from __future__ import annotations

import os
import re
from typing import List, Tuple

WORKSPACE_ROOT = "/workspace"
SAFE_WORKSPACE_ROOT = "/workspace/agent_projects"
MAX_OUTPUT_BYTES = 200_000
DEFAULT_COMMAND_TIMEOUT_S = 120
CONTAINER_PREFIX = "nemotron-playground-"
PLAYGROUND_USER = os.getenv("PLAYGROUND_USER", "0:0")

ALLOWLIST = {
    "bash",
    "python",
    "python3",
    "pytest",
    "pip",
    "pip3",
    "uvicorn",
    "node",
    "npm",
    "make",
    "ls",
    "pwd",
    "whoami",
    "env",
    "mkdir",
    "touch",
    "cp",
    "mv",
    "rm",
    "cat",
    "curl",
    "jq",
    "sed",
    "grep",
    "find",
    "docker",
    "ssh",
    "scp",
    "rsync",
    "kubectl",
}

DENY_PATTERNS: List[str] = []


class CommandRejectedError(ValueError):
    """Raised when a playground command violates the policy."""


def _command_string(cmd: List[str]) -> str:
    return " ".join(cmd)


def validate_command(cmd: List[str]) -> Tuple[bool, str | None]:
    if not cmd:
        return False, "Empty command list."

    command_name = cmd[0].split("/")[-1]
    if command_name not in ALLOWLIST:
        return False, f"Command '{command_name}' is not in allowlist."

    command_text = _command_string(cmd)
    for pattern in DENY_PATTERNS:
        if re.search(pattern, command_text):
            return False, f"Command matches denied pattern: {pattern}"

    if command_name != "docker":
        if ".." in command_text:
            return False, "Parent directory traversal is not allowed."
        for arg in cmd[1:]:
            if arg.startswith("/") and not arg.startswith(SAFE_WORKSPACE_ROOT):
                return False, f"Absolute paths must remain under {SAFE_WORKSPACE_ROOT}."

    return True, None


def validate_container_name(name: str) -> Tuple[bool, str | None]:
    if not name:
        return False, "Container name is required."
    if not name.startswith(CONTAINER_PREFIX):
        return False, f"Container name must start with '{CONTAINER_PREFIX}'."
    return True, None


def clamp_output(text: str, max_bytes: int = MAX_OUTPUT_BYTES) -> str:
    data = text.encode("utf-8", errors="replace")
    if len(data) <= max_bytes:
        return text
    truncated = data[:max_bytes]
    suffix = b"\n...output truncated..."
    return (truncated + suffix).decode("utf-8", errors="replace")

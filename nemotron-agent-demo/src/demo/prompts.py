from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Dict, List


BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROMPT_DIR = BASE_DIR / "prompts"
CONTEXT_PATH = BASE_DIR / "context" / "gb300.json"

PROMPT_LIBRARY_DIR = BASE_DIR / "prompt_library"
GOAL_PRESET_PATH = PROMPT_LIBRARY_DIR / "goal_presets.json"
AGENT_OVERRIDE_DIR = PROMPT_LIBRARY_DIR / "agent_overrides"

DEFAULT_GOAL_PRESETS = [
    {"name": "Meatball recipe", "content": "Give me a detailed meatball recipe with a shopping list."},
    {"name": "Fix this K8s YAML + logs", "content": "Help me debug a Kubernetes deployment using the provided YAML and logs."},
    {
        "name": "Design a GB300 inference service",
        "content": "Design a resilient inference service on GB300 with scaling guidance and observability hooks.",
    },
    {"name": "Write a one-click deploy script", "content": "Write a single-script deploy for this project with sane defaults."},
    {
        "name": "Summarize a long document",
        "content": "Summarize the attached long-form document with a bullet list and key risks.",
    },
    {"name": "Create a troubleshooting plan", "content": "Produce a stepwise troubleshooting plan with checkpoints and owners."},
    {"name": "Generate a demo script for a showroom", "content": "Draft a concise showroom demo script with beats and narration."},
    {"name": "Optimize vLLM flags for latency", "content": "Recommend vLLM flags that minimize latency for chat completions."},
    {
        "name": "Build an app end-to-end (long run)",
        "content": (
            "LONG_AGENT_RUN_MODE: true\n"
            "Build a FastAPI + SQLite Incident Tracker end-to-end. This is a long-run session: aim for a 10–20 minute "
            "walkthrough with detailed checkpoints and multi-file artifacts. Provide:\n"
            "- Backend CRUD endpoints with models, schemas, and router layout.\n"
            "- Minimal HTML UI (Jinja or static) to list/create incidents.\n"
            "- Tests (pytest) for core CRUD flows.\n"
            "- Dockerfile + docker-compose.yml.\n"
            "- Makefile with setup, test, run, docker-build.\n"
            "Checkpoints: (1) scaffold files, (2) implement backend endpoints, (3) implement UI, "
            "(4) run tests, (5) docker build/run. If playground tools are enabled, emit JSON tool "
            "requests like:\n"
            "```json\n"
            "{\"tool\":\"playground.exec\",\"cmd\":[\"bash\",\"-lc\",\"pytest -q\"],\"timeout_s\":120}\n"
            "```\n"
            "Use tool outputs to fix errors until tests pass and docker builds. End with a concise handoff summary.\n"
        ),
    },
    {
        "name": "Build & Validate a Mini Service Cluster",
        "content": (
            "LONG_AGENT_RUN_MODE: true\n"
            "Build & validate a mini service cluster inside /workspace. Generate:\n"
            "1) /workspace/api_service (FastAPI) with endpoints:\n"
            "   - GET /health\n"
            "   - POST /job {input} -> enqueue Redis, return job_id\n"
            "   - GET /job/{id} -> status/result\n"
            "2) /workspace/worker_service (Python worker) consuming Redis queue and writing results.\n"
            "3) /workspace/webui (cheap Gradio UI) that submits jobs and polls results.\n"
            "Use Redis for queue + status. Use env vars REDIS_URL and API_URL.\n"
            "If cluster tools are enabled, use JSON tool requests for:\n"
            "```json\n"
            "{\"tool\":\"cluster.exec\",\"container\":\"nemotron-play-<runid>-api\",\"cmd\":[\"bash\",\"-lc\",\"python -m api_service.main\"],\"timeout_s\":60}\n"
            "```\n"
            "Use `cluster.logs` to inspect container logs on failures.\n"
            "Also run `cluster.validate` after starting services. Verify end-to-end:\n"
            "web -> api -> redis -> worker -> api -> web. Fix failures and rerun validation.\n"
            "At the end, print host URLs for API + Web, the exact validation commands, and any fixes applied.\n"
        ),
    },
]


def _ensure_prompt_library() -> None:
    PROMPT_LIBRARY_DIR.mkdir(exist_ok=True)
    AGENT_OVERRIDE_DIR.mkdir(exist_ok=True)


def _safe_load_json(path: Path) -> List[Dict] | None:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        backup = path.with_suffix(path.suffix + f".bak-{int(time.time())}")
        shutil.move(path, backup)
        return None


def _ensure_goal_presets() -> None:
    _ensure_prompt_library()
    if GOAL_PRESET_PATH.exists():
        presets = _safe_load_json(GOAL_PRESET_PATH)
        if presets is not None:
            return
    GOAL_PRESET_PATH.write_text(json.dumps(DEFAULT_GOAL_PRESETS, indent=2))


def load_goal_presets() -> List[Dict[str, str]]:
    _ensure_goal_presets()
    try:
        return json.loads(GOAL_PRESET_PATH.read_text())
    except json.JSONDecodeError:
        # Already handled in _safe_load_json, but guard anyway
        _ensure_goal_presets()
        return DEFAULT_GOAL_PRESETS.copy()


def save_goal_presets(presets: List[Dict[str, str]]) -> None:
    _ensure_prompt_library()
    GOAL_PRESET_PATH.write_text(json.dumps(presets, indent=2))


def upsert_goal_preset(name: str, content: str) -> List[Dict[str, str]]:
    presets = load_goal_presets()
    updated = False
    for preset in presets:
        if preset["name"] == name:
            preset["content"] = content
            updated = True
            break
    if not updated:
        presets.append({"name": name, "content": content})
    save_goal_presets(presets)
    return presets


def delete_goal_preset(name: str) -> List[Dict[str, str]]:
    presets = [p for p in load_goal_presets() if p["name"] != name]
    save_goal_presets(presets)
    return presets


class PromptRegistry:
    def __init__(self, base_dir: Path = PROMPT_DIR, context_path: Path = CONTEXT_PATH):
        self.base_dir = base_dir
        self.context_path = context_path
        self._defaults: Dict[str, str] = {}
        self._overrides: Dict[str, str] = {}
        self._active_overrides: Dict[str, str] = {}
        self._context: str | None = None
        self.load_default_prompts()
        self.load_overrides()

    def load_default_prompts(self) -> Dict[str, str]:
        if not self._defaults:
            for path in self.base_dir.glob("*.txt"):
                self._defaults[path.stem] = path.read_text().strip()
        return self._defaults

    def load_overrides(self) -> Dict[str, str]:
        _ensure_prompt_library()
        self._overrides = {}
        for path in AGENT_OVERRIDE_DIR.glob("*.txt"):
            self._overrides[path.stem] = path.read_text().strip()
        self._active_overrides = dict(self._overrides)
        return self._overrides

    def get_default_prompt(self, agent_name: str) -> str:
        return self.load_default_prompts().get(agent_name, "")

    def get_override(self, agent_name: str) -> str:
        return self._overrides.get(agent_name, "")

    def get_active_prompt(self, agent_name: str) -> str:
        return self._active_overrides.get(agent_name, self.get_default_prompt(agent_name))

    def set_active_override(self, agent_name: str, content: str, persist: bool = False) -> None:
        self._active_overrides[agent_name] = content
        if persist:
            self._write_override(agent_name, content)

    def use_default(self, agent_name: str, persist: bool = False) -> None:
        if agent_name in self._active_overrides:
            self._active_overrides.pop(agent_name)
        if persist:
            path = AGENT_OVERRIDE_DIR / f"{agent_name}.txt"
            if path.exists():
                path.unlink()
            self._overrides.pop(agent_name, None)

    def _write_override(self, agent_name: str, content: str) -> None:
        _ensure_prompt_library()
        path = AGENT_OVERRIDE_DIR / f"{agent_name}.txt"
        path.write_text(content)
        self._overrides[agent_name] = content

    @property
    def context_payload(self) -> str:
        if self._context is None:
            data = json.loads(self.context_path.read_text())
            self._context = json.dumps(data, indent=2)
        return self._context


prompt_registry = PromptRegistry()


def load_default_prompts() -> Dict[str, str]:
    return prompt_registry.load_default_prompts()


def load_overrides() -> Dict[str, str]:
    return prompt_registry.load_overrides()


def get_active_prompt(agent_name: str) -> str:
    return prompt_registry.get_active_prompt(agent_name)


def set_active_override(agent_name: str, content: str, persist: bool = False) -> None:
    prompt_registry.set_active_override(agent_name, content, persist=persist)


def clear_override(agent_name: str, persist: bool = False) -> None:
    prompt_registry.use_default(agent_name, persist=persist)


def get_saved_override(agent_name: str) -> str:
    return prompt_registry.get_override(agent_name)


def get_context_payload() -> str:
    return prompt_registry.context_payload


def diff_summary(agent_name: str, active_content: str | None = None) -> str:
    default_content = prompt_registry.get_default_prompt(agent_name)
    active = active_content if active_content is not None else prompt_registry.get_active_prompt(agent_name)
    default_lines = default_content.count("\n") + 1 if default_content else 0
    active_lines = active.count("\n") + 1 if active else 0
    changed = "Yes" if active.strip() != default_content.strip() else "No"
    return f"Default lines: {default_lines} • Active lines: {active_lines} • Changed: {changed}"

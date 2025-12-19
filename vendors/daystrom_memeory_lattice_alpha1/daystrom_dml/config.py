"""Configuration loader with YAML defaults and environment overrides."""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import yaml  # type: ignore[import-untyped]

from .settings import DMLSettings

ENV_PREFIX = "DML_"
_RESERVED_ENV_KEYS = {
    f"{ENV_PREFIX}HOST",
    f"{ENV_PREFIX}PORT",
    f"{ENV_PREFIX}CONFIG",
    f"{ENV_PREFIX}CONFIG_PATH",
}
_NESTED_ROOTS = {"persistence", "rag_store", "literal", "budgets"}


def load_config(
    path: str | os.PathLike | None = None,
    *,
    overrides: Dict[str, Any] | None = None,
) -> DMLSettings:
    """Load the DML configuration and validate it."""

    config_file = _resolve_config_path(path)
    base_config = _read_yaml(config_file)
    env_file_vars = _load_env_files(config_file)
    env_overrides = _collect_env_overrides(env_file_vars)
    combined = _deep_merge(base_config, env_overrides)
    if overrides:
        combined = _deep_merge(combined, overrides)
    settings = _validate_settings(combined)
    return settings


def _resolve_config_path(path: str | os.PathLike | None) -> Path:
    if path is not None:
        return Path(path)
    env_override = os.environ.get(f"{ENV_PREFIX}CONFIG_PATH") or os.environ.get(
        f"{ENV_PREFIX}CONFIG"
    )
    if env_override:
        return Path(env_override)
    return Path(__file__).with_name("config.yaml")


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):  # pragma: no cover - defensive
        raise ValueError("Configuration root must be a mapping")
    return data


def _collect_env_overrides(
    env_values: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if env_values is None:
        items = os.environ.items()
    else:
        merged: Dict[str, str] = {}
        merged.update(env_values)
        for key, value in os.environ.items():
            merged[key] = value
        items = merged.items()
    for key, value in items:
        if not key.startswith(ENV_PREFIX):
            continue
        if key in _RESERVED_ENV_KEYS:
            continue
        path = _decode_env_path(key[len(ENV_PREFIX) :])
        if not path:
            continue
        _assign_path(overrides, path, value)
    return overrides


def _decode_env_path(raw_key: str) -> List[str]:
    if not raw_key:
        return []
    lowered = raw_key.lower()
    if "__" in raw_key:
        segments = [segment.lower() for segment in raw_key.split("__") if segment]
        return segments
    segments = [segment for segment in lowered.split("_") if segment]
    if not segments:
        return []
    root = segments[0]
    if root in _NESTED_ROOTS and len(segments) > 1:
        tail = "_".join(segments[1:])
        return [root, tail]
    return ["_".join(segments)]


def _assign_path(container: Dict[str, Any], path: Iterable[str], value: Any) -> None:
    iterator = iter(path)
    current = container
    try:
        first = next(iterator)
    except StopIteration:
        return
    key = first
    for segment in iterator:
        existing = current.get(key)
        if not isinstance(existing, dict):
            existing = {}
            current[key] = existing
        current = existing
        key = segment
    current[key] = value


def _deep_merge(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_settings(payload: Dict[str, Any]) -> DMLSettings:
    try:
        if hasattr(DMLSettings, "model_validate"):
            settings = DMLSettings.model_validate(payload)  # type: ignore[attr-defined]
        else:  # pragma: no cover - Pydantic v1 fallback
            settings = DMLSettings(**payload)
    except Exception as exc:  # pragma: no cover - validation errors bubble to caller
        raise ValueError(f"Invalid configuration: {exc}") from exc
    if hasattr(settings, "budgets"):
        with contextlib.suppress(Exception):
            settings.budgets.validate_totals()
    return settings


def _load_env_files(config_file: Path | None) -> Dict[str, str]:
    env_vars: Dict[str, str] = {}
    candidates = _discover_env_files(config_file)
    for candidate in candidates:
        try:
            file_vars = _parse_env_file(candidate)
        except OSError:  # pragma: no cover - filesystem issues bubble up elsewhere
            continue
        for key, value in file_vars.items():
            env_vars.setdefault(key, value)
    return env_vars


def _discover_env_files(config_file: Path | None) -> List[Path]:
    candidates: List[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen or not path.exists() or not path.is_file():
            return
        seen.add(resolved)
        candidates.append(path)

    _add(Path.cwd() / ".env")
    _add(Path.cwd() / ".env.local")
    if config_file is not None:
        parent = config_file.parent
        _add(parent / ".env")
        _add(parent / ".env.local")
    return candidates


def _parse_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.lower().startswith("export "):
                stripped = stripped[7:].lstrip()
            if "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            key = key.strip()
            value = raw_value.strip()
            if not key:
                continue
            if value and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            values[key] = value
    return values

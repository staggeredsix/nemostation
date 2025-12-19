"""FastAPI service exposing the Daystrom Memory Lattice."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import logging
import mimetypes
import inspect
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional, AsyncIterator, Iterable
from urllib.parse import urlparse, urlunparse

from fastapi import FastAPI, File, HTTPException, UploadFile, Request, WebSocket, Response
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader

from starlette.websockets import WebSocketState

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars
from structlog.stdlib import ProcessorFormatter

from . import utils, visualizer_bridge
from .dml_adapter import DMLAdapter
from .metrics import latest_metrics, record_tokens

try:  # httpx is required for proxying the visualizer through the API origin
    import httpx
except Exception:  # pragma: no cover - optional dependency in minimal environments
    httpx = None

try:  # websockets are needed for full Streamlit interactivity inside the iframe
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.exceptions import ConnectionClosed as WebsocketConnectionClosed
except Exception:  # pragma: no cover - optional dependency in minimal environments
    websocket_connect = None
    WebsocketConnectionClosed = Exception

try:  # requests is an optional dependency during some test scenarios
    import requests
except Exception:  # pragma: no cover - defensive fallback for minimal envs
    requests = None

WEB_DIR = Path(__file__).with_name("web")


def _configure_structlog() -> None:
    """Configure structlog so all logs share the same structured pipeline."""

    timestamper = structlog.processors.TimeStamper(fmt="iso")
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
    ]

    handler = logging.StreamHandler()
    handler.setFormatter(
        ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=shared_processors,
        )
    )

    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)

    structlog.configure(
        processors=shared_processors + [structlog.processors.JSONRenderer()],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


_configure_structlog()

LOGGER = logging.getLogger(__name__)
STRUCT_LOGGER = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PYTHON = Path(sys.executable)
VENV_DIR = REPO_ROOT / ".venv"
VISUALIZER_LOG = REPO_ROOT / "visualizer.log"

MAX_ARCHIVE_MEMBER_SIZE = int(
    os.environ.get("DML_MAX_ARCHIVE_MEMBER_SIZE", str(5 * 1024 * 1024))
)

app = FastAPI(title="Daystrom Memory Lattice")
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

ADAPTER_LOCK = Lock()
adapter = DMLAdapter(start_aging_loop=False)
SERVICE_START_TIME = time.time()


def _adapter_health() -> dict[str, Any]:
    """Return a lightweight view of the adapter state."""

    try:
        with ADAPTER_LOCK:
            current = adapter
        if current is None:
            raise RuntimeError("Adapter is not initialised")
        persistent_store = getattr(current, "persistent_rag_store", None)
        persistent_info: dict[str, Any]
        if persistent_store is not None:
            persistent_info = {
                "enabled": bool(getattr(persistent_store, "enable", False)),
                "backend": getattr(persistent_store, "backend", None),
                "index_path": str(getattr(persistent_store, "index_path", "")),
                "meta_path": str(getattr(persistent_store, "meta_path", "")),
            }
        else:
            persistent_info = {"enabled": False}
        persistence_enabled = bool(getattr(current, "_persistence_enabled", False))
        persistence_interval = int(getattr(current, "_persistence_interval", 0) or 0)
        persistence_path = str(getattr(current, "_persistence_path", ""))
        checkpoint_enabled = current.checkpoint_manager is not None
        return {
            "status": "ok",
            "memories": len(current.store.items()),
            "metrics_enabled": bool(current.metrics_enabled),
            "storage_dir": str(current.storage_dir),
            "rag_backends": current.rag_store.descriptors(),
            "persistent_rag": persistent_info,
            "persistence": {
                "enabled": persistence_enabled,
                "interval_seconds": persistence_interval,
                "path": persistence_path,
            },
            "checkpoints": {
                "enabled": checkpoint_enabled,
                "directory": str(current.checkpoint_dir),
            },
        }
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Adapter health probe failed")
        return {"status": "error", "error": str(exc)}


def _nim_health() -> dict[str, Any]:
    """Capture the current state of the managed NIM runtime."""

    try:
        runtime = _runtime_status()
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("NIM health probe failed")
        return {"status": "error", "error": str(exc)}

    payload: dict[str, Any] = {
        "status": "unconfigured",
        "configured": False,
        "runtime": runtime,
    }
    if CURRENT_NIM:
        payload["configured"] = True
        payload["selection"] = {
            "id": CURRENT_NIM.get("id"),
            "label": CURRENT_NIM.get("label"),
            "model": CURRENT_NIM.get("model_name"),
            "api_base": CURRENT_NIM.get("api_base"),
        }
        if runtime.get("running"):
            payload["status"] = "running" if runtime.get("healthy") else "starting"
        else:
            payload["status"] = "stopped"
    return payload


@app.middleware("http")
async def _request_context_middleware(
    request: Request, call_next
):  # pragma: no cover - exercised via integration tests
    """Attach a per-request identifier for structured logging and tracing."""

    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    bind_contextvars(
        request_id=request_id,
        path=request.url.path,
        method=request.method,
    )
    start = time.perf_counter()
    try:
        response = await call_next(request)
        duration_ms = int((time.perf_counter() - start) * 1000)
        response.headers.setdefault("x-request-id", request_id)
        STRUCT_LOGGER.info(
            "http_request",
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response
    except Exception:
        duration_ms = int((time.perf_counter() - start) * 1000)
        STRUCT_LOGGER.exception(
            "http_request_error",
            status_code=500,
            duration_ms=duration_ms,
        )
        raise
    finally:
        clear_contextvars()


NIM_OPTIONS = [
    {
        "id": "gpt-oss-20b",
        "label": "GPT-OSS 20B (OpenAI Compatible)",
        "image": "nvcr.io/nim/openai/gpt-oss-20b:latest",
        "model_name": "meta/llama3-70b-instruct",
        "default_api_base": "http://localhost:8000",
    },
    {
        "id": "llama3-8b",
        "label": "Llama 3 8B Instruct",
        "image": "nvcr.io/nim/openai/llama3-8b-instruct:latest",
        "model_name": "meta/llama3-8b-instruct",
        "default_api_base": "http://localhost:8000",
    },
    {
        "id": "mixtral-8x7b",
        "label": "Mixtral 8x7B Instruct",
        "image": "nvcr.io/nim/openai/mixtral-8x7b-instruct:latest",
        "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "default_api_base": "http://localhost:8000",
    },
]

DEFAULT_NIM_ID = adapter.settings.nim_default_id
VISUALIZER_URL = os.environ.get("DML_VISUALIZER_URL")
VISUALIZER_PORT = int(os.environ.get("DML_VISUALIZER_PORT", "8501"))
VISUALIZER_PATH = os.environ.get("DML_VISUALIZER_PATH", "/")
NGC_KEY_FILE = Path(
    os.environ.get(
        "NGC_KEY_FILE",
        Path(__file__).resolve().parent.parent / "ngc_api_key.txt",
    )
)

CURRENT_NIM: Optional[dict] = None
CURRENT_NIM_RUNTIME: dict = {"container_id": None, "running": False, "healthy": False}

NIM_CONTAINER_NAME = os.environ.get("NIM_CONTAINER_NAME", "daystrom-dml-nim")
NIM_DEFAULT_PORT = int(os.environ.get("NIM_PORT", "8000"))
NIM_HEALTH_TIMEOUT = int(
    os.environ.get("NIM_HEALTH_TIMEOUT", str(adapter.settings.nim_health_timeout))
)
NIM_HEALTH_INTERVAL = float(
    os.environ.get("NIM_HEALTH_INTERVAL", str(adapter.settings.nim_health_interval))
)

VISUALIZER_STATE = {"process": None, "log": None}
VISUALIZER_LOCK = Lock()
VISUALIZER_PROXY_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]


@app.on_event("startup")
def _auto_launch_visualizer() -> None:
    """Ensure the visualizer is running when the service starts."""

    if VISUALIZER_URL:
        LOGGER.info("Visualizer configured for external deployment at %s", VISUALIZER_URL)
        return

    try:
        _launch_visualizer_server()
        LOGGER.info("Visualizer startup complete on port %s", VISUALIZER_PORT)
    except HTTPException as exc:
        LOGGER.error("Visualizer failed to start during service startup: %s", exc.detail)
    except Exception:  # pragma: no cover - defensive logging
        LOGGER.exception("Unexpected error while starting visualizer on startup")


class TextPayload(BaseModel):
    text: str
    meta: Optional[dict] = None


class QueryPayload(BaseModel):
    prompt: str


class ComparePayload(BaseModel):
    prompt: str
    top_k: Optional[int] = None
    max_new_tokens: Optional[int] = 512


class NimConfigurePayload(BaseModel):
    nim_id: Optional[str] = None
    nim_image: Optional[str] = None
    api_key: str


class NimStartPayload(BaseModel):
    port: Optional[int] = None
    cache_dir: Optional[str] = None
    wait_timeout: Optional[int] = None


class NimStopPayload(BaseModel):
    timeout: Optional[int] = None


def _visualizer_health() -> dict[str, Any]:
    """Summarise the current visualizer mode and runtime state."""

    try:
        if VISUALIZER_URL:
            payload: dict[str, Any] = {
                "status": "external",
                "mode": "external",
                "running": True,
                "url": VISUALIZER_URL,
            }
            embed_path = _resolve_visualizer_embed_path()
            if embed_path:
                payload["embed_path"] = embed_path
            return payload

        process = VISUALIZER_STATE.get("process")
        running = bool(process and process.poll() is None)
        payload = {
            "status": "running" if running else "idle",
            "mode": "embedded",
            "running": running,
            "launch_on_demand": True,
        }
        if process is not None:
            payload["pid"] = getattr(process, "pid", None)
        embed_path = _resolve_visualizer_embed_path()
        if embed_path:
            payload["embed_path"] = embed_path
        if VISUALIZER_LOG.exists():
            payload["log_path"] = str(VISUALIZER_LOG)
        return payload
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Visualizer health probe failed")
        return {"status": "error", "error": str(exc)}


@app.get("/metrics")
def metrics_endpoint() -> Response:
    """Expose Prometheus metrics for the Daystrom service."""

    payload, content_type = latest_metrics()
    return Response(content=payload, media_type=content_type)


@app.get("/health")
def healthcheck() -> dict[str, Any]:
    """Aggregate coarse service health details for orchestrators."""

    components = {
        "adapter": _adapter_health(),
        "visualizer": _visualizer_health(),
        "nim": _nim_health(),
    }
    failure_states = {"error", "failed", "unavailable"}
    overall_status = "ok"
    adapter_status = components["adapter"].get("status")
    if adapter_status != "ok":
        overall_status = "degraded"
    elif any(component.get("status") in failure_states for component in components.values()):
        overall_status = "degraded"

    return {
        "status": overall_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": round(time.time() - SERVICE_START_TIME, 3),
        "components": components,
    }


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend bundle missing")
    return HTMLResponse(index.read_text(encoding="utf-8"))


def _resolve_visualizer_url(request: Request) -> str:
    """Resolve the visualizer target, adapting to remote deployments."""

    if VISUALIZER_URL:
        return VISUALIZER_URL

    headers = request.headers
    scheme = headers.get("x-forwarded-proto", request.url.scheme)
    host_header = headers.get("x-forwarded-host") or headers.get("host")
    hostname = request.url.hostname or "localhost"
    host = host_header or hostname

    forwarded_port = headers.get("x-forwarded-port")
    port = None
    request_port = request.url.port
    if request_port is None:
        if scheme == "https":
            request_port = 443
        elif scheme == "http":
            request_port = 80
    if host and ":" in host:
        host, host_port = host.rsplit(":", 1)
        try:
            port = int(host_port)
        except ValueError:
            port = None
    if forwarded_port:
        try:
            port = int(forwarded_port)
        except ValueError:
            port = None
    if port is None:
        port = VISUALIZER_PORT
    elif request_port is not None and port == request_port:
        port = VISUALIZER_PORT
    if port in {80, 443}:
        netloc = host
    else:
        netloc = f"{host}:{port}"

    path = VISUALIZER_PATH if VISUALIZER_PATH.startswith("/") else f"/{VISUALIZER_PATH}"
    return f"{scheme}://{netloc}{path}"


def _normalise_visualizer_path(path: str) -> str:
    if not path:
        return "/"
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return "/"
    return "/" + "/".join(segments)


def _visualizer_upstream_components() -> tuple[str, str, str]:
    if VISUALIZER_URL:
        parsed = urlparse(VISUALIZER_URL)
        scheme = parsed.scheme or "http"
        hostname = parsed.hostname or "localhost"
        port = parsed.port
        if port is None:
            port = 443 if scheme == "https" else 80
        if (scheme == "https" and port == 443) or (scheme == "http" and port == 80):
            netloc = hostname
        else:
            netloc = f"{hostname}:{port}"
        base_path = parsed.path
    else:
        scheme = "http"
        netloc = f"127.0.0.1:{VISUALIZER_PORT}"
        base_path = VISUALIZER_PATH
    return scheme, netloc, _normalise_visualizer_path(base_path)


def _join_visualizer_path(base_path: str, extra: str) -> str:
    base_segments = [segment for segment in base_path.split("/") if segment]
    extra_segments = [segment for segment in extra.split("/") if segment]
    segments = base_segments + extra_segments
    if not segments:
        return "/"
    return "/" + "/".join(segments)


def _resolve_visualizer_embed_path() -> Optional[str]:
    if httpx is None or websocket_connect is None:
        return None
    _, _, base_path = _visualizer_upstream_components()
    suffix = "/" if base_path == "/" else base_path
    return f"/visualizer/embed{suffix}"


def _strip_frame_ancestors(csp: str) -> Optional[str]:
    directives = []
    for directive in csp.split(";"):
        cleaned = directive.strip()
        if not cleaned:
            continue
        if cleaned.lower().startswith("frame-ancestors"):
            continue
        directives.append(cleaned)
    if not directives:
        return None
    return "; ".join(directives)


def _format_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run_command(command: list[object], *, cwd: Optional[str] = None, env: Optional[dict] = None) -> subprocess.CompletedProcess[str]:
    cmd_parts = [str(part) for part in command]
    LOGGER.info("Executing command: %s", _format_command(cmd_parts))
    try:
        result = subprocess.run(
            cmd_parts,
            cwd=cwd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive logging
        output = (exc.stderr or exc.stdout or "").strip()
        if output:
            LOGGER.error("Command failed (%s): %s", exc.returncode, output)
        else:
            LOGGER.error("Command failed with exit code %s", exc.returncode)
        detail = f"Command {_format_command(cmd_parts)} failed with exit code {exc.returncode}"
        raise HTTPException(status_code=500, detail=detail) from exc
    else:
        if result.stdout:
            LOGGER.debug("Command output: %s", result.stdout.strip())
        if result.stderr:
            LOGGER.debug("Command stderr: %s", result.stderr.strip())
        return result


def _should_include_faiss(version_info: tuple[int, int, int] | None = None) -> bool:
    """Determine if the FAISS extra should be installed in the visualizer venv."""

    if version_info is None:
        version_info = sys.version_info[:3]
    return version_info < (3, 12, 0)


def _compute_visualizer_extras(version_info: tuple[int, int, int] | None = None) -> list[str]:
    if version_info is None:
        version_info = sys.version_info[:3]
    extras = ["server", "tokenizer", "embeddings", "faiss", "mcp"]
    if not _should_include_faiss(version_info):
        LOGGER.warning(
            "Skipping 'faiss' extra for visualizer environment on Python %s.%s.%s",
            *version_info,
        )
        extras.remove("faiss")
    return extras


def _ensure_visualizer_environment() -> Path:
    if not VENV_DIR.exists():
        LOGGER.info("Creating virtual environment in %s", VENV_DIR)
        _run_command([DEFAULT_PYTHON, "-m", "venv", VENV_DIR])

    venv_python = _venv_python_path(VENV_DIR)
    if not venv_python.exists():  # pragma: no cover - corrupted environment safeguard
        raise HTTPException(
            status_code=500,
            detail=f"Unable to locate virtual environment interpreter at {venv_python}",
        )

    upgrade_cmd = [
        venv_python,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
        "setuptools",
        "wheel",
    ]
    extras = _compute_visualizer_extras()
    editable_target = ".[{}]".format(",".join(extras)) if extras else "."
    editable_cmd = [venv_python, "-m", "pip", "install", "-e", editable_target]
    visualizer_cmd = [venv_python, "-m", "pip", "install", "streamlit", "plotly"]

    _run_command(upgrade_cmd, cwd=str(REPO_ROOT))
    _run_command(editable_cmd, cwd=str(REPO_ROOT))
    _run_command(visualizer_cmd, cwd=str(REPO_ROOT))

    return venv_python


def _start_visualizer_process(venv_python: Path) -> subprocess.Popen[bytes]:
    existing = VISUALIZER_STATE.get("process")
    if existing and existing.poll() is None:
        return existing

    log_handle = VISUALIZER_STATE.get("log")
    if log_handle is not None:
        VISUALIZER_STATE["log"] = None
        try:
            log_handle.close()
        except Exception:  # pragma: no cover - best effort cleanup
            LOGGER.debug("Failed to close previous visualizer log handle", exc_info=True)
    log_file = VISUALIZER_LOG.open("ab")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"\n[{timestamp}] Starting visualizer process\n".encode("utf-8"))
    log_file.flush()

    command = [
        venv_python,
        "-m",
        "streamlit",
        "run",
        str(REPO_ROOT / "visualize_dml_live.py"),
        "--server.headless=true",
        "--server.address=0.0.0.0",
        f"--server.port={VISUALIZER_PORT}",
    ]

    env = os.environ.copy()
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    env.setdefault("PYTHONPATH", str(REPO_ROOT))

    try:
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        log_file.close()
        LOGGER.exception("Failed to start visualizer process")
        raise HTTPException(status_code=500, detail=f"Failed to start visualizer: {exc}") from exc

    VISUALIZER_STATE["process"] = process
    VISUALIZER_STATE["log"] = log_file
    return process


def _wait_for_visualizer_ready(timeout: int = 120) -> None:
    if requests is None:
        time.sleep(2)
        return

    base_url = f"http://127.0.0.1:{VISUALIZER_PORT}"
    path = VISUALIZER_PATH if VISUALIZER_PATH.startswith("/") else f"/{VISUALIZER_PATH}"
    url = f"{base_url}{path}"

    deadline = time.time() + timeout
    while time.time() < deadline:
        process = VISUALIZER_STATE.get("process")
        if process and process.poll() is not None:
            raise HTTPException(status_code=500, detail="Visualizer exited unexpectedly during startup")
        try:
            response = requests.get(url, timeout=3)
            if response.ok:
                return
        except Exception:
            pass
        time.sleep(1)

    raise HTTPException(status_code=500, detail="Visualizer failed to start within the timeout window")


def _visualizer_process_running() -> bool:
    process = VISUALIZER_STATE.get("process")
    return bool(process and process.poll() is None)


async def _ensure_visualizer_running_async() -> None:
    """Start the embedded visualizer if it's not already running."""

    if VISUALIZER_URL or _visualizer_process_running():
        return

    await asyncio.to_thread(_launch_visualizer_server)


def _launch_visualizer_server() -> None:
    started = False
    with VISUALIZER_LOCK:
        if _visualizer_process_running():
            LOGGER.info("Visualizer already running")
            return

        venv_python = _ensure_visualizer_environment()
        _start_visualizer_process(venv_python)
        started = True

    if started:
        try:
            _wait_for_visualizer_ready()
        except Exception:
            with VISUALIZER_LOCK:
                process = VISUALIZER_STATE.get("process")
                if process and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except Exception:  # pragma: no cover - best effort cleanup
                        LOGGER.debug("Visualizer process did not exit cleanly after termination")
                VISUALIZER_STATE["process"] = None
                log_handle = VISUALIZER_STATE.get("log")
                if log_handle is not None:
                    try:
                        log_handle.flush()
                    except Exception:
                        LOGGER.debug("Unable to flush visualizer log after failure", exc_info=True)
                    try:
                        log_handle.close()
                    except Exception:
                        LOGGER.debug("Unable to close visualizer log after failure", exc_info=True)
                VISUALIZER_STATE["log"] = None
            raise


@app.api_route("/visualizer/embed/{path:path}", methods=VISUALIZER_PROXY_METHODS)
async def visualizer_embed_http(path: str, request: Request) -> StreamingResponse:
    if httpx is None:
        raise HTTPException(
            status_code=503,
            detail="Visualizer proxy is unavailable because the httpx dependency is missing.",
        )

    await _ensure_visualizer_running_async()

    scheme, netloc, base_path = _visualizer_upstream_components()
    upstream_path = _join_visualizer_path(base_path, path)

    forward_headers = {}
    for key, value in request.headers.items():
        header = key.lower()
        if header in {"host", "content-length", "connection", "keep-alive"}:
            continue
        if header == "origin":
            continue
        forward_headers[key] = value
    forward_headers["host"] = netloc
    forward_headers.setdefault("origin", f"{scheme}://{netloc}")

    body = await request.body()
    client = httpx.AsyncClient(
        base_url=f"{scheme}://{netloc}",
        follow_redirects=False,
        timeout=httpx.Timeout(None),
    )
    try:
        upstream_request = client.build_request(
            request.method,
            upstream_path or "/",
            headers=forward_headers,
            content=body if body else None,
            params=list(request.query_params.multi_items()),
        )
        upstream_response = await client.send(upstream_request, stream=True)
    except Exception as exc:
        await client.aclose()
        LOGGER.exception("Visualizer proxy request failed")
        raise HTTPException(status_code=502, detail=f"Visualizer proxy request failed: {exc}") from exc

    async def stream_response() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream_response.aiter_raw():
                yield chunk
        finally:
            await upstream_response.aclose()
            await client.aclose()

    headers: list[tuple[str, str]] = []
    for name, value in upstream_response.headers.items():
        lower = name.lower()
        if lower in {"transfer-encoding", "connection", "keep-alive", "x-frame-options"}:
            continue
        if lower == "content-security-policy":
            sanitised = _strip_frame_ancestors(value)
            if sanitised:
                headers.append((name, sanitised))
            continue
        headers.append((name, value))
    headers.append(("x-frame-options", "ALLOWALL"))

    response = StreamingResponse(
        stream_response(),
        status_code=upstream_response.status_code,
    )
    for name, value in headers:
        response.headers.append(name, value)
    return response


@app.websocket("/visualizer/embed/{path:path}")
async def visualizer_embed_websocket(path: str, websocket: WebSocket) -> None:
    if websocket_connect is None:
        await websocket.close(code=1011, reason="Visualizer proxy unavailable (websockets dependency missing)")
        return

    try:
        await _ensure_visualizer_running_async()
    except HTTPException as exc:
        LOGGER.error("Visualizer unavailable for websocket proxy: %s", exc.detail)
        await websocket.close(code=1011, reason="Visualizer proxy unavailable")
        return
    except Exception:  # pragma: no cover - defensive logging
        LOGGER.exception("Unexpected error while ensuring visualizer availability")
        await websocket.close(code=1011, reason="Visualizer proxy unavailable")
        return

    scheme, netloc, base_path = _visualizer_upstream_components()
    upstream_path = _join_visualizer_path(base_path, path)
    ws_scheme = "wss" if scheme == "https" else "ws"
    query = websocket.scope.get("query_string", b"").decode("latin-1")
    target = f"{ws_scheme}://{netloc}{upstream_path}"
    if query:
        target = f"{target}?{query}"

    client_origin = None
    extra_headers: list[tuple[str, str]] = []
    for name, value in websocket.scope.get("headers", []):
        header_name = name.decode("latin-1")
        lower = header_name.lower()
        if lower in {"host", "origin"}:
            if lower == "origin" and client_origin is None:
                client_origin = value.decode("latin-1")
            continue
        if lower.startswith("sec-websocket"):
            continue
        extra_headers.append((header_name, value.decode("latin-1")))
    origin_header = f"{scheme}://{netloc}"
    extra_headers.append(("origin", origin_header))
    if client_origin and client_origin != origin_header:
        extra_headers.append(("x-forwarded-origin", client_origin))

    subprotocols = websocket.scope.get("subprotocols") or None
    if not subprotocols:
        for name, value in websocket.scope.get("headers", []):
            if name.decode("latin-1").lower() == "sec-websocket-protocol":
                parsed = [
                    token.strip()
                    for token in value.decode("latin-1").split(",")
                    if token.strip()
                ]
                if parsed:
                    subprotocols = parsed
                break

    header_items = list(extra_headers)
    origin_value = None
    other_headers: list[tuple[str, str]] = []
    for name, value in header_items:
        if name.lower() == "origin" and origin_value is None:
            origin_value = value
            continue
        other_headers.append((name, value))

    connect_kwargs: dict[str, object] = {
        "subprotocols": subprotocols,
        "open_timeout": None,
        "close_timeout": None,
    }

    try:
        signature = inspect.signature(websocket_connect)
        parameters = signature.parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        parameters = {}

    def _supports(name: str) -> bool:
        param = parameters.get(name)
        if param is None:
            return False
        return param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)

    forwarded_headers: list[tuple[str, str]] | None = header_items if header_items else None

    if header_items:
        if _supports("extra_headers"):
            connect_kwargs["extra_headers"] = header_items
        elif _supports("additional_headers"):
            connect_kwargs["additional_headers"] = header_items
        elif _supports("headers"):
            connect_kwargs["headers"] = header_items
        else:
            if origin_value and _supports("origin"):
                connect_kwargs["origin"] = origin_value
            elif origin_value:
                other_headers.append(("origin", origin_value))
            if other_headers:
                dropped = ", ".join(sorted({name for name, _ in other_headers}))
                LOGGER.debug(
                    "websocket_connect() cannot forward websocket headers; dropping: %s",
                    dropped,
                )
            forwarded_headers = None

    if forwarded_headers is None and origin_value and _supports("origin") and "origin" not in connect_kwargs:
        connect_kwargs["origin"] = origin_value

    try:
        upstream = await websocket_connect(
            target,
            **connect_kwargs,
        )
    except Exception:  # pragma: no cover - network exceptions
        LOGGER.exception("Visualizer websocket proxy connection failed")
        await websocket.close(code=1011, reason="Visualizer proxy connection failed")
        return

    await websocket.accept(subprotocol=getattr(upstream, "subprotocol", None))

    async def client_to_server() -> None:
        try:
            while True:
                message = await websocket.receive()
                message_type = message.get("type")
                if message_type == "websocket.receive":
                    text = message.get("text")
                    data = message.get("bytes")
                    if text is not None:
                        await upstream.send(text)
                    elif data is not None:
                        await upstream.send(data)
                elif message_type == "websocket.disconnect":
                    await upstream.close()
                    break
        except Exception:  # pragma: no cover - defensive cleanup
            with contextlib.suppress(Exception):
                await upstream.close()

    async def server_to_client() -> None:
        try:
            while True:
                try:
                    payload = await upstream.recv()
                except WebsocketConnectionClosed:
                    break
                if isinstance(payload, str):
                    await websocket.send_text(payload)
                else:
                    await websocket.send_bytes(payload)
        finally:
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.close()

    try:
        await asyncio.gather(client_to_server(), server_to_client())
    finally:
        with contextlib.suppress(Exception):
            await upstream.close()
        if websocket.application_state == WebSocketState.CONNECTED:
            with contextlib.suppress(Exception):
                await websocket.close()

@app.get("/visualizer", response_class=HTMLResponse)
def visualizer_page() -> HTMLResponse:
    """Serve the embedded visualizer page."""

    page = WEB_DIR / "visualizer.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Visualizer page missing from bundle")
    return HTMLResponse(page.read_text(encoding="utf-8"))


def _translate_visualizer_failure(exc: HTTPException) -> HTTPException:
    """Normalise visualizer startup failures for client consumption."""

    if exc.status_code >= 500:
        detail = exc.detail or "Visualizer is unavailable"
        return HTTPException(status_code=503, detail=f"Visualizer unavailable: {detail}")
    return exc


@app.get("/visualizer/redirect")
def visualizer_redirect(request: Request) -> RedirectResponse:
    """Open the external visualizer in a new tab."""
    if not VISUALIZER_URL:
        try:
            _launch_visualizer_server()
        except HTTPException as exc:
            raise _translate_visualizer_failure(exc) from exc
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception("Unexpected error while preparing visualizer redirect")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    target = _resolve_visualizer_url(request)
    query_string = str(request.query_params) if request.query_params else ""
    if query_string:
        separator = "&" if "?" in target else "?"
        target = f"{target}{separator}{query_string}"
    return RedirectResponse(url=target)


@app.get("/visualizer/url")
def visualizer_url(request: Request) -> dict:
    """Expose the configured visualizer target for the frontend."""

    payload = {"url": _resolve_visualizer_url(request)}
    embed_path = _resolve_visualizer_embed_path()
    if embed_path:
        payload["embed_url"] = embed_path
    return payload


@app.get("/visualizer/state")
def visualizer_state() -> dict:
    """Expose the most recent prompt consumed by the visualizer."""

    payload = visualizer_bridge.latest_prompt()
    if not payload:
        return {"available": False, "payload": None}
    return {"available": True, "payload": payload}


@app.post("/visualizer/launch")
def launch_visualizer(request: Request) -> dict:
    """Ensure the Streamlit visualizer is ready and return its URL."""

    target_url = _resolve_visualizer_url(request)
    embed_path = _resolve_visualizer_embed_path()
    if VISUALIZER_URL:
        # External deployments are assumed to be managed separately.
        LOGGER.info("External visualizer configured, skipping local launch")
        payload = {"status": "external", "url": target_url}
        if embed_path:
            payload["embed_url"] = embed_path
        return payload

    try:
        _launch_visualizer_server()
    except HTTPException as exc:
        raise _translate_visualizer_failure(exc) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Unexpected error while launching visualizer")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload = {"status": "ready", "url": target_url}
    if embed_path:
        payload["embed_url"] = embed_path
    return payload


@app.post("/ingest")
def ingest(payload: TextPayload) -> dict:
    adapter.ingest(payload.text, meta=payload.meta)
    STRUCT_LOGGER.info(
        "ingest_completed",
        text_length=len(payload.text or ""),
        meta_keys=sorted((payload.meta or {}).keys()),
    )
    return {"status": "ok"}


@app.post("/upload")
async def upload(
    files: list[UploadFile] | None = File(default=None),
    file: UploadFile | None = File(default=None),
) -> dict:
    """Accept one or more uploaded files and ingest supported documents."""

    uploads: list[UploadFile] = []
    if files:
        uploads.extend(files)
    if file is not None:
        uploads.append(file)
    if not uploads:
        raise HTTPException(status_code=400, detail="No files were provided for upload.")

    total_chunks = 0
    total_tokens = 0
    document_count = 0
    processed_files = 0
    skipped_files = 0
    errors: list[str] = []

    for upload_file in uploads:
        if upload_file is None:
            continue
        filename = upload_file.filename or "uploaded file"
        try:
            contents = await upload_file.read()
        except Exception as exc:  # pragma: no cover - depends on Starlette internals
            errors.append(f"{filename}: failed to read upload ({exc})")
            skipped_files += 1
            continue

        try:
            documents = list(
                _iter_ingest_documents(
                    filename,
                    contents,
                    upload_file.content_type,
                )
            )
        except HTTPException as exc:
            errors.append(f"{filename}: {exc.detail}")
            skipped_files += 1
            continue

        if not documents:
            errors.append(f"{filename}: no supported text documents detected.")
            skipped_files += 1
            continue

        processed_files += 1
        for text, meta in documents:
            if not text.strip():
                continue
            chunks = utils.chunk_text(text)
            if not chunks:
                errors.append(
                    f"{meta.get('doc_path', filename)}: document produced no ingestible chunks."
                )
                continue
            document_count += 1
            total_chunks += len(chunks)
            for chunk in chunks:
                tokens = utils.estimate_tokens(chunk)
                total_tokens += tokens
                adapter.ingest(chunk, meta=meta)

    if total_chunks == 0:
        message = errors[0] if errors else "Document produced no ingestible chunks."
        raise HTTPException(status_code=400, detail=message)

    status = "ok" if not errors else "partial"
    payload: dict[str, object] = {
        "status": status,
        "files_ingested": processed_files,
        "documents": document_count,
        "chunks": total_chunks,
        "tokens": total_tokens,
        "skipped_files": skipped_files,
    }
    if errors:
        payload["errors"] = errors
    return payload


@app.post("/reinforce")
def reinforce(payload: TextPayload) -> dict:
    adapter.reinforce("", payload.text, meta=payload.meta)
    return {"status": "ok"}


@app.post("/query")
def query(payload: QueryPayload) -> dict:
    retrieval = adapter.query_database(payload.prompt)
    context = (retrieval.get("context") or "").strip()
    if context:
        augmented = f"{context}\n\n{payload.prompt}"
    else:
        augmented = payload.prompt
    response = adapter.runner.generate(augmented)
    adapter.reinforce(payload.prompt, response)
    prompt_tokens = utils.estimate_tokens(payload.prompt)
    context_tokens = int(retrieval.get("tokens") or 0)
    tokens_consumed = prompt_tokens + context_tokens
    tokens_saved = max(0, prompt_tokens - context_tokens)
    if adapter.metrics_enabled:
        record_tokens(tokens_consumed, tokens_saved)
    STRUCT_LOGGER.info(
        "query_completed",
        mode=retrieval.get("mode", "unknown"),
        tokens_consumed=tokens_consumed,
        tokens_saved=tokens_saved,
        context_tokens=context_tokens,
        latency_ms=retrieval.get("latency_ms", 0),
    )
    return {
        "mode": retrieval.get("mode"),
        "context": context,
        "response": response,
        "stats": adapter.stats(),
    }


@app.post("/rag/retrieve")
def rag_retrieve(payload: QueryPayload) -> dict:
    rag_top_k = adapter.config.get("top_k", 6)
    rag_reports = adapter.rag_store.report_all(payload.prompt, top_k=rag_top_k)
    dml_report = adapter.retrieval_report(payload.prompt)
    visualizer_bridge.queue_prompt(
        payload.prompt,
        top_k=rag_top_k,
        mode="auto",
        metadata={"source": "rag_retrieve"},
    )
    return {
        "prompt": payload.prompt,
        "rag_backends": rag_reports,
        "dml": dml_report,
    }


@app.post("/rag/compare")
def rag_compare(payload: ComparePayload) -> dict:
    try:
        result = adapter.compare_responses(
            payload.prompt,
            top_k=payload.top_k,
            max_new_tokens=payload.max_new_tokens or 512,
        )
    except Exception as exc:
        if requests and isinstance(exc, requests.RequestException):
            raise HTTPException(status_code=503, detail="NIM backend is unreachable. Start the container and try again.")
        raise
    prompt_tokens = utils.estimate_tokens(payload.prompt)
    visualizer_bridge.queue_prompt(
        payload.prompt,
        top_k=payload.top_k or adapter.config.get("top_k", 6),
        mode="auto",
        metadata={"source": "rag_compare"},
    )
    return {
        **result,
        "prompt_tokens_est": prompt_tokens,
    }


@app.get("/stats")
def stats() -> dict:
    return adapter.stats()


@app.get("/knowledge")
def knowledge() -> dict:
    """Expose summaries of the documents stored in RAG and the DML lattice."""

    return adapter.knowledge_report()


@app.get("/nim/options")
def nim_options() -> dict:
    """Expose the curated list of NVIDIA NIM container options."""

    return {
        "options": NIM_OPTIONS,
        "current": CURRENT_NIM,
        "default": _nim_summary(_nim_option(DEFAULT_NIM_ID)),
        "runtime": _runtime_status(),
    }


@app.post("/nim/configure")
def nim_configure(payload: NimConfigurePayload) -> dict:
    """Pull a NIM container image and reconfigure the adapter."""

    if not payload.api_key.strip():
        raise HTTPException(status_code=400, detail="NGC API key is required")
    nim_id = (payload.nim_id or "").strip()
    nim_image = (payload.nim_image or "").strip()
    if not nim_id and not nim_image:
        nim_id = DEFAULT_NIM_ID
    option = None
    if nim_id:
        option = _nim_option(nim_id)
    elif nim_image:
        option = _nim_option_by_image(nim_image)
    if not option:
        identifier = nim_id or nim_image or ""
        raise HTTPException(status_code=404, detail=f"Unknown NIM selection provided: {identifier}")
    try:
        pull_status, pull_logs = _pull_nim_image(option["image"], payload.api_key.strip())
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    _apply_nim_configuration(option, payload.api_key.strip())
    summary = _nim_summary(option)
    global CURRENT_NIM
    CURRENT_NIM = summary
    CURRENT_NIM_RUNTIME.update({"running": False, "healthy": False, "container_id": None})
    return {
        "status": "ok",
        "nim": summary,
        "pull_status": pull_status,
        "logs": pull_logs,
        "runtime": _runtime_status(),
    }


@app.post("/nim/start")
def nim_start(payload: NimStartPayload | None = None) -> dict:
    """Start the configured NIM container and wait for it to become healthy."""

    if CURRENT_NIM is None:
        raise HTTPException(status_code=400, detail="Configure a NIM before attempting to start it.")
    docker_bin = shutil.which("docker")
    runtime = _runtime_status()
    if not docker_bin:
        return {
            "status": "skipped",
            "message": "Docker binary not available on server; cannot start NIM.",
            "runtime": runtime,
        }
    api_key = os.environ.get("NIM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("NGC_API_KEY")
    port = NIM_DEFAULT_PORT
    if payload and payload.port:
        port = int(payload.port)
    cache_dir = None
    if payload and payload.cache_dir:
        cache_dir = Path(payload.cache_dir).expanduser()
    else:
        cache_env = os.environ.get("LOCAL_NIM_CACHE")
        if cache_env:
            cache_dir = Path(cache_env).expanduser()
    if cache_dir:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:  # pragma: no cover - best effort for unusual permissions
            cache_dir = None
    if runtime.get("running"):
        healthy, reason = _nim_healthcheck(CURRENT_NIM["api_base"], api_key)
        CURRENT_NIM_RUNTIME.update({"healthy": healthy})
        runtime = _runtime_status()
        message = "NIM container already running and healthy." if healthy else (
            "NIM container is running but not responding yet." + (f" Reason: {reason}" if reason else "")
        )
        return {
            "status": "running" if healthy else "starting",
            "message": message,
            "runtime": runtime,
        }
    api_base = _configure_runtime_api_base(port)
    run_cmd = [
        docker_bin,
        "run",
        "-d",
        "--rm",
        "--gpus=all",
        "--name",
        NIM_CONTAINER_NAME,
        "-p",
        f"{port}:8000",
    ]
    extra_opts = os.environ.get("NIM_DOCKER_RUN_OPTS")
    if extra_opts:
        run_cmd.extend(shlex.split(extra_opts))
    if cache_dir:
        run_cmd.extend(["-v", f"{cache_dir}:/opt/nim/.cache"])
    if api_key:
        run_cmd.extend(["-e", f"NGC_API_KEY={api_key}"])
    run_cmd.append(CURRENT_NIM["image"])
    logs: list[str] = []
    try:
        run_proc = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except Exception as exc:  # pragma: no cover - subprocess errors are environment dependent
        raise HTTPException(status_code=500, detail=f"Failed to launch NIM container: {exc}") from exc
    if run_proc.stdout:
        logs.append(run_proc.stdout.strip())
    if run_proc.stderr:
        logs.append(run_proc.stderr.strip())
    if run_proc.returncode != 0:
        runtime = _runtime_status()
        return {
            "status": "error",
            "message": "Docker failed to start the NIM container.",
            "logs": logs,
            "runtime": runtime,
        }
    container_id = run_proc.stdout.strip()
    CURRENT_NIM_RUNTIME.update({"container_id": container_id or None, "running": True, "healthy": False})
    wait_timeout = NIM_HEALTH_TIMEOUT
    if payload and payload.wait_timeout:
        wait_timeout = int(payload.wait_timeout)
    healthy, health_logs = _wait_for_nim_health(
        api_base,
        api_key,
        timeout=wait_timeout,
    )
    CURRENT_NIM_RUNTIME.update({"healthy": healthy})
    runtime = _runtime_status()
    logs.extend(health_logs)
    status = "running" if healthy else "starting"
    message = "NIM container is ready." if healthy else "NIM container launched but health check timed out."
    return {
        "status": status,
        "message": message,
        "logs": logs,
        "runtime": runtime,
    }


@app.post("/nim/stop")
def nim_stop(payload: NimStopPayload | None = None) -> dict:
    """Stop the managed NIM container."""

    docker_bin = shutil.which("docker")
    runtime = _runtime_status()
    if not docker_bin:
        return {
            "status": "skipped",
            "message": "Docker binary not available on server; cannot stop NIM.",
            "runtime": runtime,
        }
    if not runtime.get("running"):
        return {
            "status": "not-running",
            "message": "No running NIM container detected.",
            "runtime": runtime,
        }
    timeout = 60
    if payload and payload.timeout:
        timeout = int(payload.timeout)
    stop_cmd = [docker_bin, "stop", NIM_CONTAINER_NAME]
    logs: list[str] = []
    stop_proc = subprocess.run(
        stop_cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if stop_proc.stdout:
        logs.append(stop_proc.stdout.strip())
    if stop_proc.stderr:
        logs.append(stop_proc.stderr.strip())
    if stop_proc.returncode != 0:
        runtime = _runtime_status()
        return {
            "status": "error",
            "message": "Docker failed to stop the NIM container.",
            "logs": logs,
            "runtime": runtime,
        }
    CURRENT_NIM_RUNTIME.update({"running": False, "healthy": False, "container_id": None})
    runtime = _runtime_status()
    return {
        "status": "stopped",
        "message": "NIM container stopped.",
        "logs": logs,
        "runtime": runtime,
    }


def _iter_ingest_documents(
    filename: str,
    contents: bytes,
    content_type: str | None,
    *,
    parent: str | None = None,
) -> Iterable[tuple[str, dict[str, str]]]:
    """Yield ``(text, meta)`` tuples extracted from ``contents``.

    The helper is resilient to archives and nested directory structures.  Each
    yielded metadata dictionary contains a ``doc_path`` entry used to surface
    the original source in the UI.
    """

    source_name = _compose_source_path(parent, filename)
    if _is_archive_like(filename, content_type):
        try:
            with zipfile.ZipFile(io.BytesIO(contents)) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    inner_type, _ = mimetypes.guess_type(member.filename)
                    if _should_skip_binary(member.filename, inner_type):
                        LOGGER.debug(
                            "Skipping binary archive member %s inside %s",
                            member.filename,
                            source_name,
                        )
                        continue
                    if member.file_size and member.file_size > MAX_ARCHIVE_MEMBER_SIZE:
                        LOGGER.debug(
                            "Skipping oversized archive member %s inside %s (%s bytes)",
                            member.filename,
                            source_name,
                            member.file_size,
                        )
                        continue
                    try:
                        payload = archive.read(member)
                    except Exception as exc:  # pragma: no cover - depends on zipfile internals
                        LOGGER.debug(
                            "Skipping %s inside %s due to read failure: %s",
                            member.filename,
                            source_name,
                            exc,
                        )
                        continue
                    yield from _iter_ingest_documents(
                        member.filename,
                        payload,
                        inner_type,
                        parent=source_name,
                    )
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail=f"Failed to open archive: {exc}") from exc
        return

    if _should_skip_binary(filename, content_type):
        LOGGER.debug("Skipping non-text upload: %s", source_name)
        return

    text = _extract_text(filename, contents, content_type)
    if not text.strip():
        return
    if not _looks_like_text(text):
        LOGGER.debug("Skipping file that does not appear to contain text: %s", source_name)
        return
    yield text, {"doc_path": source_name}


def _compose_source_path(parent: str | None, name: str) -> str:
    cleaned = (name or "uploaded document").replace("\\", "/").strip()
    cleaned = cleaned.lstrip("./") or "uploaded document"
    if parent:
        joiner = "::" if parent.lower().endswith(".zip") and "::" not in parent else "/"
        parent_clean = parent.rstrip("/")
        cleaned = cleaned.lstrip("/")
        return f"{parent_clean}{joiner}{cleaned}" if cleaned else parent_clean
    return cleaned


def _is_archive_like(filename: str, content_type: str | None) -> bool:
    suffix = (filename or "").lower()
    if suffix.endswith(".zip"):
        return True
    if content_type and "zip" in content_type:
        return True
    return False


def _should_skip_binary(filename: str, content_type: str | None) -> bool:
    suffix = (filename or "").lower()
    binary_exts = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".svg",
        ".mp3",
        ".wav",
        ".flac",
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".gz",
        ".tar",
        ".tgz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".iso",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".xls",
        ".xlsx",
    }
    if suffix and any(suffix.endswith(ext) for ext in binary_exts):
        return True
    if content_type:
        lower = content_type.lower()
        if lower.startswith(("image/", "audio/", "video/")):
            return True
        if lower in {"application/octet-stream", "application/x-msdownload"} and suffix not in {".txt", ".csv"}:
            return True
    return False


def _looks_like_text(value: str) -> bool:
    if not value:
        return False
    sample = value[:2000]
    if not sample:
        return False
    control_chars = sum(1 for ch in sample if ord(ch) < 9 or 13 < ord(ch) < 32)
    ratio = control_chars / max(1, len(sample))
    return ratio < 0.02


def _extract_text(filename: str, contents: bytes, content_type: str | None) -> str:
    suffix = (filename or "").lower()
    if suffix.endswith(".pdf") or (content_type and "pdf" in content_type):
        try:
            reader = PdfReader(io.BytesIO(contents))
        except Exception as exc:  # pragma: no cover - depends on external lib
            raise HTTPException(status_code=400, detail=f"Failed to read PDF: {exc}") from exc
        pages = []
        for page in reader.pages:
            try:
                extracted = page.extract_text() or ""
            except Exception:  # pragma: no cover - best effort for malformed PDFs
                extracted = ""
            pages.append(extracted)
        return "\n\n".join(pages)
    try:
        return contents.decode("utf-8")
    except UnicodeDecodeError:
        return contents.decode("latin-1", errors="ignore")


def _nim_option(nim_id: str) -> dict:
    for option in NIM_OPTIONS:
        if option["id"] == nim_id:
            return option
    raise HTTPException(status_code=404, detail=f"Unknown NIM identifier: {nim_id}")


def _nim_option_by_image(image: str) -> Optional[dict]:
    for option in NIM_OPTIONS:
        if option["image"] == image:
            return option
    return None


def _nim_summary(option: dict) -> dict:
    return {
        "id": option["id"],
        "label": option["label"],
        "model_name": option["model_name"],
        "api_base": option["default_api_base"],
        "image": option["image"],
    }


def _pull_nim_image(image: str, api_key: str) -> tuple[str, list[str]]:
    """Attempt to pull the requested NIM image via Docker."""

    docker_bin = shutil.which("docker")
    if not docker_bin:
        return "skipped", ["Docker binary not available on server; skipping image pull."]
    logs: list[str] = []
    login_cmd = [
        docker_bin,
        "login",
        "nvcr.io",
        "--username",
        "$oauthtoken",
        "--password-stdin",
    ]
    login_proc = subprocess.run(
        login_cmd,
        input=f"{api_key}\n",
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if login_proc.stdout:
        logs.append(login_proc.stdout.strip())
    if login_proc.stderr:
        logs.append(login_proc.stderr.strip())
    if login_proc.returncode != 0:
        raise RuntimeError("Docker login failed; verify the provided NGC API key is valid.")
    pull_proc = subprocess.run(
        [docker_bin, "pull", image],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    if pull_proc.stdout:
        logs.append(pull_proc.stdout.strip())
    if pull_proc.stderr:
        logs.append(pull_proc.stderr.strip())
    if pull_proc.returncode != 0:
        raise RuntimeError(f"Docker pull failed for image {image}.")
    return "ok", logs


def _apply_nim_configuration(option: dict, api_key: str) -> None:
    """Set environment variables and reload the adapter for the selected NIM."""

    os.environ["NIM_API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["NIM_API_BASE"] = option["default_api_base"]
    os.environ["OPENAI_API_BASE"] = option["default_api_base"]
    _save_ngc_key(api_key)
    _reload_adapter(config_overrides={"model_name": option["model_name"]})


def _save_ngc_key(api_key: str) -> None:
    """Persist the provided NGC API key for convenience."""

    try:
        NGC_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        NGC_KEY_FILE.write_text(api_key.strip() + "\n", encoding="utf-8")
    except Exception as exc:  # pragma: no cover - filesystem permissions vary
        LOGGER.warning("Failed to persist NGC API key: %s", exc)


def _reload_adapter(*, config_overrides: Optional[dict] = None) -> None:
    """Recreate the global adapter with the provided overrides."""

    global adapter
    with ADAPTER_LOCK:
        previous = adapter
        try:
            adapter = DMLAdapter(
                start_aging_loop=False,
                config_overrides=config_overrides,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            adapter = previous
            raise HTTPException(status_code=500, detail=f"Failed to initialise adapter: {exc}") from exc
        try:
            previous.close()
        except Exception:
            pass


def _runtime_status() -> dict:
    """Return the current runtime view of the managed NIM container."""

    docker_bin = shutil.which("docker")
    running = CURRENT_NIM_RUNTIME.get("running", False)
    healthy = CURRENT_NIM_RUNTIME.get("healthy", False)
    container_id = CURRENT_NIM_RUNTIME.get("container_id")
    if docker_bin:
        ps_proc = subprocess.run(
            [docker_bin, "ps", "-q", "--filter", f"name={NIM_CONTAINER_NAME}"],
            capture_output=True,
            text=True,
            check=False,
        )
        listed = ps_proc.stdout.strip().splitlines()
        if listed:
            container_id = listed[0]
            running = True
        else:
            running = False
            healthy = False
            container_id = None
    else:
        healthy = healthy if running else False
    CURRENT_NIM_RUNTIME.update(
        {
            "running": running,
            "healthy": healthy if running else False,
            "container_id": container_id,
        }
    )
    return {
        "running": CURRENT_NIM_RUNTIME["running"],
        "healthy": CURRENT_NIM_RUNTIME["healthy"],
        "container_id": CURRENT_NIM_RUNTIME["container_id"],
        "container_name": NIM_CONTAINER_NAME,
        "docker_available": docker_bin is not None,
    }


def _nim_healthcheck(api_base: str, api_key: Optional[str]) -> tuple[bool, Optional[str]]:
    """Perform a lightweight request to verify the NIM endpoint is responsive."""

    if not requests:
        return False, "Requests library unavailable; cannot perform health check."
    if not api_base:
        return False, "NIM API base URL is not configured."
    url = f"{api_base.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": CURRENT_NIM["model_name"] if CURRENT_NIM else "model",
        "messages": [{"role": "user", "content": "Are you alive?"}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        return False, str(exc)
    if response.status_code in {200, 401, 403}:
        return True, None
    text = response.text[:200] if response.text else f"status {response.status_code}"
    return False, text


def _wait_for_nim_health(
    api_base: str,
    api_key: Optional[str],
    *,
    timeout: int,
) -> tuple[bool, list[str]]:
    """Poll the NIM endpoint until it responds or the timeout elapses."""

    deadline = time.time() + max(timeout, 1)
    attempts: list[str] = []
    if not requests:
        attempts.append("Requests library unavailable; skipping health polling.")
        return False, attempts
    while time.time() < deadline:
        healthy, reason = _nim_healthcheck(api_base, api_key)
        if healthy:
            return True, attempts
        attempts.append(f"Health check failed: {reason or 'unknown error'}")
        time.sleep(NIM_HEALTH_INTERVAL)
    return False, attempts


def _configure_runtime_api_base(port: int) -> str:
    """Derive and apply the runtime API base for the configured NIM port."""

    if CURRENT_NIM is None:
        return f"http://localhost:{port}"
    existing_base = CURRENT_NIM.get("api_base") or f"http://localhost:{port}"
    updated_base = _nim_api_base_with_port(existing_base, port)
    CURRENT_NIM["api_base"] = updated_base
    os.environ["NIM_API_BASE"] = updated_base
    os.environ["OPENAI_API_BASE"] = updated_base
    os.environ["NIM_PORT"] = str(port)
    runner_backend = getattr(getattr(adapter, "runner", None), "_backend", None)
    if hasattr(runner_backend, "base_url"):
        runner_backend.base_url = updated_base.rstrip("/")
    return updated_base


def _nim_api_base_with_port(api_base: str, port: int) -> str:
    """Return the API base with the provided port applied to the netloc."""

    if not api_base:
        return f"http://localhost:{port}"
    parsed = urlparse(api_base)
    scheme = parsed.scheme or "http"
    if not parsed.netloc:
        return f"{scheme}://localhost:{port}"
    host = parsed.hostname or "localhost"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    userinfo = ""
    if parsed.username:
        userinfo = parsed.username
        if parsed.password:
            userinfo += f":{parsed.password}"
        userinfo += "@"
    netloc = f"{userinfo}{host}:{port}"
    rebuilt = parsed._replace(netloc=netloc)
    return urlunparse(rebuilt)


def main(argv: Optional[list[str]] = None) -> None:
    """Run the Daystrom Memory Lattice API server via ``uvicorn``."""

    parser = argparse.ArgumentParser(description="Run the Daystrom Memory Lattice API server.")
    parser.add_argument(
        "--host",
        default=os.environ.get("DML_HOST", "0.0.0.0"),
        help="Host interface for uvicorn to bind (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("DML_PORT", "8000")),
        help="Port for uvicorn to expose (default: 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn autoreload (useful for local development).",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("DML_LOG_LEVEL", "info"),
        help="Log level passed to uvicorn (default: info).",
    )

    args = parser.parse_args(argv)

    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - optional dependency import guard
        raise RuntimeError(
            "uvicorn is required to run the DML server. Install the 'server' extra "
            "with `pip install .[server]`."
        ) from exc

    uvicorn.run(
        "daystrom_dml.server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

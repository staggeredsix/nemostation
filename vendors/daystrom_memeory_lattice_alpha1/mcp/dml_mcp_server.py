"""Expose the Daystrom Memory Lattice via the Model Context Protocol."""
from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from typing import Any, Iterable, List, TYPE_CHECKING

from daystrom_dml.dml_adapter import DMLAdapter

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from mcp.server.fastmcp import FastMCP as FastMCPType
else:  # pragma: no cover - runtime import guard
    FastMCPType = Any

try:  # pragma: no cover - optional dependency
    from mcp.server.fastmcp import FastMCP as _FastMCP
except Exception:  # pragma: no cover - best effort import guard
    _FastMCP = None
    MCP_AVAILABLE = False
else:
    MCP_AVAILABLE = True


def _build_adapter(
    config_path: Path | None,
    storage_dir: Path | None,
) -> DMLAdapter:
    overrides: dict[str, Any] = {}
    if storage_dir:
        overrides["storage_dir"] = str(storage_dir)
    return DMLAdapter(
        config_path=str(config_path) if config_path else None,
        config_overrides=overrides or None,
        start_aging_loop=False,
    )


def create_server(
    *,
    config_path: Path | None = None,
    storage_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> FastMCPType:
    """Instantiate a FastMCP server bound to a DML adapter."""

    if not MCP_AVAILABLE:  # pragma: no cover - import guard
        raise RuntimeError("mcp extras are not installed; install with '.[mcp]'")

    adapter = _build_adapter(config_path, storage_dir)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastMCPType) -> Any:
        try:
            yield
        finally:
            adapter.close()

    if _FastMCP is None:  # pragma: no cover - defensive import guard
        raise RuntimeError("FastMCP runtime is unavailable")

    server = _FastMCP(
        name="daystrom-dml",
        instructions="Augment prompts with Daystrom Memory Lattice context",
        host=host,
        port=port,
        lifespan=lifespan,
    )

    @server.tool(name="ingest", description="Ingest a file or directory into the lattice")
    async def ingest(path: str) -> dict[str, Any]:
        target = Path(path).expanduser()
        if not target.exists():
            raise ValueError(f"Path does not exist: {target}")
        files = list(_iter_ingest_targets(target))
        if not files:
            raise ValueError(f"No ingestible files found in {target}")
        count = 0
        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            adapter.ingest(text, meta={"doc_path": str(file_path)})
            count += 1
        return {"files": count, "target": str(target)}

    @server.tool(name="query", description="Query the lattice and receive structured context")
    async def query(prompt: str, mode: str = "auto") -> dict[str, Any]:
        report = adapter.query_database(prompt, mode=mode or "auto")
        return {
            "mode": report["mode"],
            "context": report["context"],
            "tokens": int(report.get("tokens", 0)),
            "latency_ms": int(report.get("latency_ms", 0)),
            "sources": report.get("source_docs", []),
        }

    @server.tool(name="stats", description="Return basic adapter statistics")
    async def stats() -> dict[str, Any]:
        return adapter.stats()

    return server


def _iter_ingest_targets(root: Path) -> Iterable[Path]:
    if root.is_file():
        return [root]
    files: List[Path] = []
    for candidate in root.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in {".txt", ".md", ".log", ".json"}:
            files.append(candidate)
    return files


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the DML MCP server")
    parser.add_argument("--config", type=Path, default=None, help="Optional config file override")
    parser.add_argument("--storage", type=Path, default=None, help="Persistent storage directory override")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport to expose",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for HTTP transports")
    parser.add_argument("--port", type=int, default=8000, help="Bind port for HTTP transports")
    args = parser.parse_args(argv)

    server = create_server(
        config_path=args.config,
        storage_dir=args.storage,
        host=args.host,
        port=args.port,
    )
    server.run(transport=args.transport)  # type: ignore[union-attr]


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()

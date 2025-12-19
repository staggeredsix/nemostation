"""Model Context Protocol (MCP) server exposing CMA operations."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Optional

try:  # pragma: no cover - optional dependency
    from mcp.server.fastmcp import FastMCP
except Exception:  # pragma: no cover
    FastMCP = None  # type: ignore

from .adapter import CMAAdapter


DEFAULT_STORAGE_PATH = Path("cma_store.json")


def create_mcp_server(
    *,
    name: str = "concept-memory-adapter",
    storage_path: str | Path | None = None,
) -> "FastMCP":
    """Build an MCP server exposing CMA primitives.

    Parameters
    ----------
    name:
        Name reported to the MCP client.
    storage_path:
        Optional path for persisting the concept memory store. Falls back to
        ``cma_store.json`` in the current working directory.

    Returns
    -------
    FastMCP
        Configured server instance ready to run.
    """

    if FastMCP is None:  # pragma: no cover - requires optional dependency
        raise RuntimeError(
            "The 'mcp' package is required to run the CMA MCP server. Install "
            "the 'mcp' extra with `pip install cma[mcp]`."
        )

    adapter = CMAAdapter(storage_path=Path(storage_path or DEFAULT_STORAGE_PATH))
    server = FastMCP(name)

    @server.tool()
    async def ingest(text: str, meta: Optional[dict[str, Any]] = None) -> dict[str, str]:
        """Add a new memory snippet to the lattice."""

        adapter.ingest(text, meta)
        return {"status": "ok"}

    @server.tool()
    async def augment(
        prompt: str,
        *,
        top_k: Optional[int] = None,
        token_budget: Optional[int] = None,
    ) -> dict[str, str]:
        """Return a context preamble for the supplied prompt."""

        preamble = adapter.augment_prompt(
            prompt, top_k=top_k, token_budget=token_budget
        )
        return {"preamble": preamble}

    @server.tool()
    async def reinforce(text: str) -> dict[str, str]:
        """Reinforce salient information after a generation."""

        adapter.reinforce(text)
        return {"status": "ok"}

    @server.tool()
    async def stats() -> dict[str, Any]:
        """Return summary statistics for the concept memory."""

        return adapter.stats()

    return server


def run(name: str = "concept-memory-adapter", storage_path: str | Path | None = None) -> None:
    """Run the MCP server using stdio transport."""

    server = create_mcp_server(name=name, storage_path=storage_path)
    server.run()


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point used by the ``dml-mcp-server`` console script."""

    parser = argparse.ArgumentParser(description="Run the Daystrom MCP server.")
    parser.add_argument(
        "--name",
        default=os.environ.get("CMA_MCP_NAME", "concept-memory-adapter"),
        help="Service name reported to MCP clients.",
    )
    parser.add_argument(
        "--storage-path",
        default=os.environ.get("CMA_STORAGE_PATH"),
        help=(
            "Optional path to persist the concept memory store. When omitted, "
            "a local cma_store.json file is used."
        ),
    )

    args = parser.parse_args(argv)
    run(name=args.name, storage_path=args.storage_path)


__all__ = ["create_mcp_server", "run", "main"]


if __name__ == "__main__":  # pragma: no cover - module executable
    main()


"""Command line interface for the Daystrom Memory Lattice."""
from __future__ import annotations

import contextlib
import json
import logging
from typing import Iterator, Optional

import typer

from .dml_adapter import DMLAdapter

app = typer.Typer(help="Daystrom Memory Lattice control interface")


def _build_adapter(model: Optional[str]) -> DMLAdapter:
    overrides = {"model_name": model} if model else None
    return DMLAdapter(config_overrides=overrides, start_aging_loop=False)


@contextlib.contextmanager
def _adapter_scope(model: Optional[str]) -> Iterator[DMLAdapter]:
    adapter = _build_adapter(model)
    try:
        yield adapter
    finally:
        with contextlib.suppress(Exception):
            adapter.close()


@app.command()
def ingest(text: str) -> None:
    """Store a new memory fragment."""

    with _adapter_scope(None) as adapter:
        adapter.ingest(text)
    typer.echo("Ingested snippet.")


@app.command()
def query(prompt: str, model: Optional[str] = typer.Option(None)) -> None:
    """Retrieve context for a prompt."""

    with _adapter_scope(model) as adapter:
        context = adapter.build_preamble(prompt)
    typer.echo(context)


@app.command()
def reinforce(text: str) -> None:
    """Reinforce a conclusion."""

    with _adapter_scope(None) as adapter:
        adapter.reinforce("", text)
    typer.echo("Reinforced memory.")


@app.command()
def run(prompt: str, model: Optional[str] = typer.Option(None)) -> None:
    """Run an augmented generation round-trip."""

    with _adapter_scope(model) as adapter:
        response = adapter.run_generation(prompt)
    typer.echo(response)


@app.command()
def stats() -> None:
    """Print diagnostic information about the current lattice."""

    with _adapter_scope(None) as adapter:
        payload = adapter.stats()
    typer.echo(json.dumps(payload, indent=2))


@app.command()
def checkpoint() -> None:
    """Create an immediate persistence checkpoint."""

    with _adapter_scope(None) as adapter:
        path = adapter.create_checkpoint()
    typer.echo(f"Checkpoint written to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app()

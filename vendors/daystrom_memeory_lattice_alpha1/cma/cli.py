"""Typer CLI for CMA."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .adapter import CMAAdapter

app = typer.Typer(add_completion=False)


def _resolve_store(path: Optional[str]) -> Path:
    return Path(path or "cma_store.json")


@app.command()
def init(store: Optional[str] = typer.Option(None, help="Path to store file")) -> None:
    """Initialise an empty CMA store."""
    path = _resolve_store(store)
    path.write_text(json.dumps([], indent=2))
    typer.echo(f"Initialised store at {path}")


@app.command()
def ingest(text: str = typer.Option(..., prompt=True), store: Optional[str] = None) -> None:
    """Ingest a new memory snippet."""
    adapter = CMAAdapter(storage_path=_resolve_store(store))
    adapter.ingest(text)
    typer.echo("Memory stored.")


@app.command()
def query(
    prompt: str = typer.Option(..., prompt=True),
    top_k: int = typer.Option(None, help="Number of memories to include"),
    budget: int = typer.Option(None, help="Token budget for the preamble"),
    store: Optional[str] = None,
) -> None:
    """Build a prompt preamble from the memory bank."""
    adapter = CMAAdapter(storage_path=_resolve_store(store))
    preamble = adapter.augment_prompt(prompt, top_k=top_k, token_budget=budget)
    typer.echo(preamble)


@app.command()
def reinforce(text: str = typer.Option(..., prompt=True), store: Optional[str] = None) -> None:
    """Reinforce memory items using generated text."""
    adapter = CMAAdapter(storage_path=_resolve_store(store))
    adapter.reinforce(text)
    typer.echo("Reinforcement applied.")


@app.command()
def stats(store: Optional[str] = None) -> None:
    """Display memory statistics."""
    adapter = CMAAdapter(storage_path=_resolve_store(store))
    data = adapter.stats()
    typer.echo(json.dumps(data, indent=2))

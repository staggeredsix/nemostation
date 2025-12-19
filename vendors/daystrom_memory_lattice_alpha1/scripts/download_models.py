#!/usr/bin/env python3
"""Download recommended LLM and embedding models before launching the playground."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_LLM_MODELS: Sequence[str] = (
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
)

DEFAULT_EMBEDDING_MODELS: Sequence[str] = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
)


def _download_llm(model_name: str, cache_dir: Path | None, token: str | None) -> None:
    print(f"Downloading LLM model: {model_name}")
    AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=token)
    AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, token=token)


def _download_embedding(model_name: str, cache_dir: Path | None) -> None:
    print(f"Downloading embedding model: {model_name}")
    SentenceTransformer(model_name, cache_folder=str(cache_dir) if cache_dir else None)


def _resolve_models(values: Iterable[str] | None, defaults: Sequence[str]) -> list[str]:
    resolved = list(values or [])
    return resolved or list(defaults)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm",
        action="append",
        help="LLM model names to pre-download (defaults to a curated set if omitted).",
    )
    parser.add_argument(
        "--embedding",
        action="append",
        help="Embedding model names to pre-download (defaults to a curated set if omitted).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory; defaults to the Hugging Face cache location.",
    )
    parser.add_argument(
        "--hf-token",
        dest="hf_token",
        default=None,
        help="Hugging Face token for gated models (if required).",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    llm_models = _resolve_models(args.llm, DEFAULT_LLM_MODELS)
    embedding_models = _resolve_models(args.embedding, DEFAULT_EMBEDDING_MODELS)

    for model_name in llm_models:
        _download_llm(model_name, cache_dir, args.hf_token)

    for model_name in embedding_models:
        _download_embedding(model_name, cache_dir)

    print("All requested models are available locally.")


if __name__ == "__main__":
    main()

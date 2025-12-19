from __future__ import annotations

import os
import shutil
import sys
import sysconfig
from pathlib import Path
from typing import List

import numpy as np
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

try:
    import pybind11
except Exception:  # pragma: no cover - handled when CUDA build is requested
    pybind11 = None  # type: ignore


class CUDABuildExt(build_ext):
    """Custom build_ext that optionally compiles the CUDA extension."""

    def build_extensions(self) -> None:  # pragma: no cover - build-time hook
        cuda_extensions = [ext for ext in self.extensions if getattr(ext, "is_cuda", False)]
        non_cuda_extensions = [ext for ext in self.extensions if not getattr(ext, "is_cuda", False)]
        build_cuda = os.environ.get("DML_BUILD_CUDA", "auto").lower() != "0"
        nvcc = shutil.which("nvcc")
        if not build_cuda or nvcc is None or pybind11 is None:
            # Skip CUDA extensions when explicitly disabled or tooling is unavailable.
            self.extensions = non_cuda_extensions
            super().build_extensions()
            return
        for ext in cuda_extensions:
            self.build_cuda_extension(ext, nvcc)
        if non_cuda_extensions:
            self.extensions = non_cuda_extensions
            super().build_extensions()

    def build_cuda_extension(self, ext: Extension, nvcc_path: str) -> None:
        sources = [str(Path(src)) for src in ext.sources]
        if not sources:
            return
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        include_dirs: List[str] = [
            pybind11.get_include(),
            np.get_include(),
            sysconfig.get_paths()["include"],
        ]
        include_flags = [flag for inc in include_dirs for flag in ("-I", inc)]

        ext_path = Path(self.get_ext_fullpath(ext.name))
        ext_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            nvcc_path,
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "-Xcompiler",
            "-fPIC",
            "-shared",
            *include_flags,
            "-o",
            str(ext_path),
            *sources,
            "-lcudart",
        ]
        self.spawn(cmd)


cuda_extension = Extension(
    "daystrom_dml._cuda_backend",
    sources=["daystrom_dml/_cuda_backend.cu"],
)
setattr(cuda_extension, "is_cuda", True)


setup(
    name="cma",
    version="0.1.0",
    description="Concept Memory Adapter providing lossy associative memory for LLMs",
    author="CMA Authors",
    author_email="cma@example.com",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(exclude=["nim*", "tests*", "daystrom_dml.tests*"]),
    install_requires=[
        "typer>=0.9",
        "fastapi>=0.100",
        "requests>=2.31",
        "numpy>=1.24",
        "PyYAML>=6.0",
        "pydantic>=1.10",
        "pydantic-settings>=2.0",
        "prometheus_client>=0.16",
        "structlog>=23.2",
        "httpx>=0.26",
        "pypdf>=4.0",
        "python-multipart>=0.0.6",
    ],
    extras_require={
        "server": [
            "fastapi>=0.100",
            "uvicorn>=0.23",
            "python-multipart>=0.0.6",
            "pypdf>=4.0",
            "httpx>=0.26",
            "websockets>=11.0",
        ],
        "tokenizer": ["tiktoken>=0.4"],
        "embeddings": ["sentence-transformers>=2.2"],
        "faiss": ["faiss-cpu>=1.7"],
        "multiplex_rag": ["chromadb>=0.4", "faiss-cpu>=1.7"],
        "mcp": ["mcp>=0.1.0"],
        "playground": ["streamlit>=1.39", "plotly>=5.22"],
        "dev": ["pytest>=7.4", "ruff>=0.1.9", "mypy>=1.6.0"],
        "cuda": ["pybind11>=2.10"],
    },
    package_data={"daystrom_dml": ["web/*", "web/**/*"]},
    ext_modules=[cuda_extension],
    cmdclass={"build_ext": CUDABuildExt},
)

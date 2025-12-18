from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROMPT_DIR = BASE_DIR / "prompts"
CONTEXT_PATH = BASE_DIR / "context" / "gb300.json"


class PromptLoader:
    def __init__(self, base_dir: Path = PROMPT_DIR, context_path: Path = CONTEXT_PATH):
        self.base_dir = base_dir
        self.context_path = context_path
        self._cache: Dict[str, str] = {}
        self._context: str | None = None

    def load(self, name: str) -> str:
        if name not in self._cache:
            path = self.base_dir / f"{name}.txt"
            self._cache[name] = path.read_text().strip()
        return self._cache[name]

    @property
    def context_payload(self) -> str:
        if self._context is None:
            data = json.loads(self.context_path.read_text())
            self._context = json.dumps(data, indent=2)
        return self._context


prompt_loader = PromptLoader()

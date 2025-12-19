"""High level adapter orchestrating the Daystrom Memory Lattice."""
from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from threading import Event, RLock
from typing import Any, Dict, List, Optional

import numpy as np
from .config import load_config
from .checkpoint import CheckpointManager
from .embeddings import Embedder, create_embedder
from .gpt_runner import GPTRunner
from .memory_store import MemoryItem, MemoryStore
from .metrics import record_retrieval, update_memory_gauge
from .multi_rag import DEFAULT_BACKENDS, MultiRAGStore, RAGBackendDescriptor
from .persistent_index import PersistentVectorBackend
from .persistence import load_state as load_persisted_memories
from .persistence import save_state as save_persisted_memories
from .summarizer import DummySummarizer, LLMSummarizer, Summarizer
from .retrievers import LiteralRetriever
from .router import decide_mode
from .rag_store import PersistentRAGStore
from . import utils

LOGGER = logging.getLogger(__name__)

DEFAULT_DML_TOP_K = 8
MAX_RETRIEVAL_TOP_K = 10

# Limit the amount of data returned by the knowledge endpoint to keep the
# payload responsive even when the lattice contains thousands of memories.
KNOWLEDGE_MAX_ENTRIES = 200
KNOWLEDGE_ENTRY_PREVIEW_CHARS = 320

STARFLEET_BANNER = "\n".join(
    [
        "Initializing Daystrom Memory Lattice v1.0",
        "Semantic coherence field stabilized.",
        "Cognitive resonance online.",
    ]
)


class DMLAdapter:
    """Facade used by the CLI and service to interact with the DML."""

    def __init__(
        self,
        config_path: str | os.PathLike | None = None,
        *,
        config_overrides: Optional[Dict] = None,
        embedder: Optional[Embedder] = None,
        summarizer: Optional[Summarizer] = None,
        runner: Optional[GPTRunner] = None,
        start_aging_loop: bool = True,
    ) -> None:
        overrides = dict(config_overrides or {})
        self.settings = load_config(config_path, overrides=overrides)
        self.config = self.settings.as_dict()
        self.config.setdefault("dml_top_k", DEFAULT_DML_TOP_K)
        self.enable_workflow_cache = bool(
            self.config.get("enable_workflow_cache", False)
        )
        self.enable_quality_on_retrieval = bool(
            self.config.get("enable_quality_on_retrieval", False)
        )
        self.metrics_enabled = bool(self.settings.metrics_enabled)
        self.runner = runner or GPTRunner(self.config["model_name"])
        self.embedder = embedder or create_embedder(
            self.config.get("embedding_model"),
            device=self.config.get("embedding_device"),
        )
        if summarizer is not None:
            self.summarizer = summarizer
        elif self.runner.is_dummy:
            self.summarizer = DummySummarizer()
        else:
            self.summarizer = LLMSummarizer(self.runner)
        storage_dir = self.settings.storage_dir.expanduser()
        if not storage_dir.is_absolute():
            storage_dir = Path.cwd() / storage_dir
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        persistence_settings = getattr(self.settings, "persistence", None)
        persistence_path = getattr(persistence_settings, "path", None) if persistence_settings else None
        if persistence_path:
            persistence_path = Path(persistence_path).expanduser()
        else:
            persistence_path = Path("dml_state.jsonl")
        if not persistence_path.is_absolute():
            relative_path = persistence_path
            if relative_path.parts and relative_path.parts[0] == self.storage_dir.name:
                if len(relative_path.parts) > 1:
                    relative_path = Path(*relative_path.parts[1:])
                else:
                    relative_path = Path(relative_path.name)
            persistence_path = (self.storage_dir / relative_path).expanduser()
        self._persistence_path = persistence_path
        interval_value = getattr(persistence_settings, "interval_sec", 0) if persistence_settings else 0
        try:
            self._persistence_interval = max(0, int(interval_value))
        except (TypeError, ValueError):
            self._persistence_interval = 0
        self._persistence_enabled = bool(persistence_settings and getattr(persistence_settings, "enable", False))
        self._persistence_stop_event = Event()
        self._persistence_thread: Optional[threading.Thread] = None
        self.dml_state_path = self.storage_dir / "dml_store.json"
        self.rag_state_path = self.storage_dir / "rag_store.json"
        self.checkpoint_dir = self.storage_dir / "checkpoints"
        self._persist_lock = RLock()
        literal_cfg = getattr(self.settings, "literal", None)
        literal_tokens = 160
        literal_snippets = 8
        if literal_cfg is not None:
            literal_tokens = int(getattr(literal_cfg, "max_snippet_tokens", literal_tokens))
            literal_snippets = int(getattr(literal_cfg, "max_snippets", literal_snippets))
        self.literal_snippet_cap = max(1, literal_snippets)
        self.literal_token_cap = max(16, literal_tokens)
        char_window = max(64, int(self.literal_token_cap * 4))
        self.literal_retriever = LiteralRetriever(
            self.embedder,
            self.summarizer,
            context_window=int(self.config.get("literal_context", 1)),
            max_snippet_chars=char_window,
        )
        rag_settings = getattr(self.settings, "rag_store", None)
        self.persistent_rag_store: Optional[PersistentRAGStore] = None
        if rag_settings and getattr(rag_settings, "enable", False):
            index_path = Path(rag_settings.path).expanduser()
            meta_path = Path(rag_settings.meta_path).expanduser()
            if not index_path.is_absolute():
                index_path = self._resolve_storage_path(index_path)
            if not meta_path.is_absolute():
                meta_path = self._resolve_storage_path(meta_path)
            try:
                self.persistent_rag_store = PersistentRAGStore(
                    enable=True,
                    index_path=index_path,
                    meta_path=meta_path,
                    dim=int(rag_settings.dim),
                    backend=str(rag_settings.backend),
                )
            except Exception:
                LOGGER.exception("Failed to initialise persistent RAG store.")
                self.persistent_rag_store = None
            else:
                with contextlib.suppress(Exception):
                    self.persistent_rag_store.load()
        else:
            self.persistent_rag_store = None
        persistent_backend = RAGBackendDescriptor(
            identifier="persistent",
            label="Persistent Index",
            description="Disk-backed cosine similarity index",
            factory=lambda: PersistentVectorBackend(
                self.storage_dir / self.settings.vector_index_file
            ),
        )
        backends = [persistent_backend]
        backends.extend(DEFAULT_BACKENDS)
        self.rag_store = MultiRAGStore(self.embedder, backends=backends)
        self.store = MemoryStore(
            self.summarizer,
            beta_a=float(self.config["beta_a"]),
            beta_r=float(self.config["beta_r"]),
            eta=float(self.config["eta"]),
            gamma=float(self.config["gamma"]),
            kappa=float(self.config["kappa"]),
            tau_s=float(self.config["tau_s"]),
            theta_merge=float(self.config["theta_merge"]),
            K=int(self.config["K"]),
            capacity=int(self.config["capacity"]),
            start_aging_loop=start_aging_loop,
            enable_quality_on_retrieval=self.enable_quality_on_retrieval,
            similarity_threshold=float(self.config.get("similarity_threshold", 0.0)),
        )
        self.checkpoint_manager: Optional[CheckpointManager] = None
        if int(self.settings.checkpoint_interval_seconds) > 0:
            self.checkpoint_manager = CheckpointManager(
                self.checkpoint_dir,
                self._gather_checkpoint_state,
                interval_seconds=int(self.settings.checkpoint_interval_seconds),
                retention=int(self.settings.checkpoint_retention),
            )
        self._load_persisted_state()
        if self._persistence_enabled and self._persistence_interval > 0:
            self._start_persistence_loop()
        if self.metrics_enabled:
            update_memory_gauge(len(self.store.items()))
        LOGGER.info("Daystrom Memory Lattice initialised with %d capacity", self.store.capacity)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._persist_all()
        self._stop_persistence_loop()
        if self.checkpoint_manager:
            self.checkpoint_manager.close()
        if self.metrics_enabled:
            update_memory_gauge(len(self.store.items()))
        self.store.close()

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------
    def ingest(self, text: str, meta: Optional[Dict] = None) -> None:
        if not text:
            return
        embedding = self.embedder.embed(text)
        salience = self._estimate_salience(text)
        item, merged = self.store.ingest(text, embedding, salience=salience, meta=meta)
        rag_text = item.text if merged else text
        rag_embedding = item.embedding if merged else embedding
        rag_meta: Dict[str, Any] = dict(meta or {})
        rag_meta.setdefault("memory_id", item.id)
        if merged:
            rag_meta["memory_merges"] = int(item.meta.get("merges", 0))
        if self.persistent_rag_store is not None:
            with contextlib.suppress(Exception):
                self.persistent_rag_store.add(rag_text, rag_embedding, meta=rag_meta)
        self.rag_store.add_document(rag_text, meta=rag_meta)
        self._persist_all()
        if self.metrics_enabled:
            update_memory_gauge(len(self.store.items()))

    def build_preamble(self, prompt: str, top_k: Optional[int] = None) -> str:
        items = self._retrieve_items(prompt, top_k)
        _, preamble, _ = self._prepare_context(prompt, items)
        return preamble

    def reinforce(self, prompt: str, response: str, meta: Optional[Dict] = None) -> None:
        prompt_text = (prompt or "").strip()
        response_text = (response or "").strip()
        if not response_text:
            return
        response_summary = self.summarizer.summarize(response_text, max_len=220).strip()
        if not response_summary:
            response_summary = response_text[:220].strip()
        lines: List[str] = []
        if prompt_text:
            lines.append(f"Prompt: {prompt_text}")
        else:
            lines.append("Prompt: (empty)")
        lines.append(f"Answer summary: {response_summary}")
        memory_text = "\n".join(lines)
        embedding = self.embedder.embed(memory_text)
        salience = self._estimate_salience(response_summary) + 0.1
        memory_meta = dict(meta or {})
        if prompt_text:
            memory_meta.setdefault("prompt", prompt_text)
        if response_text:
            memory_meta.setdefault("response_excerpt", response_text[:500])
        self.store.ingest(memory_text, embedding, salience=salience, meta=memory_meta)
        self._persist_dml_state()
        if self.metrics_enabled:
            update_memory_gauge(len(self.store.items()))

    def run_generation(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        context = self.build_preamble(prompt)
        augmented_prompt = f"{context}\n\n{prompt}"
        response = self.runner.generate(augmented_prompt, max_new_tokens=max_new_tokens)
        self.reinforce(prompt, response)
        return response

    def retrieval_report(self, prompt: str, *, top_k: Optional[int] = None) -> Dict:
        start = time.perf_counter()
        items = self._retrieve_items(prompt, top_k)
        entries, preamble, tokens_used = self._prepare_context(prompt, items)
        fidelities = [entry["fidelity"] for entry in entries]
        avg_fidelity = float(np.mean(fidelities) if fidelities else 0.0)
        latency_ms = int((time.perf_counter() - start) * 1000.0)
        return {
            "entries": entries,
            "preamble": preamble,
            "tokens": tokens_used,
            "avg_fidelity": avg_fidelity,
            "latency_ms": latency_ms,
        }

    def compare_responses(
        self,
        prompt: str,
        *,
        top_k: Optional[int] = None,
        max_new_tokens: int = 512,
    ) -> Dict:
        step_counter = 0
        pipeline_trace: List[Dict[str, Any]] = []

        base_response, base_usage, base_latency = self._generate_with_metrics(
            prompt, max_new_tokens=max_new_tokens
        )
        step_counter += 1
        base_sequence = step_counter
        pipeline_trace.append(
            {
                "step": step_counter,
                "stage": "base",
                "label": "Base model",
            }
        )

        dml_report = self.retrieval_report(prompt, top_k=top_k)
        dml_context = self._format_dml_context(dml_report["entries"])
        dml_prompt = self._compose_prompt(prompt, dml_context)
        dml_response, dml_usage, dml_latency = self._generate_with_metrics(
            dml_prompt, max_new_tokens=max_new_tokens
        )
        self.reinforce(prompt, dml_response)
        step_counter += 1
        dml_sequence = step_counter
        pipeline_trace.append(
            {
                "step": step_counter,
                "stage": "dml",
                "label": "Daystrom memory lattice",
            }
        )

        dml_reference = dml_response or ""
        evaluations: List[Dict[str, Any]] = []

        reference_available = bool(dml_reference.strip())
        if reference_available:
            dml_grade = {
                "score": 1.0,
                "grade": self._score_to_grade(1.0),
                "explanation": "Reference response for grading.",
            }
            evaluations.append(
                {
                    "backend_id": "dml",
                    "score": 1.0,
                    "grade": dml_grade["grade"],
                }
            )
        else:
            dml_grade = {
                "score": 0.0,
                "grade": "N/A",
                "explanation": "DML response unavailable for comparison.",
            }
            evaluations.append(
                {
                    "backend_id": "dml",
                    "score": 0.0,
                    "grade": dml_grade["grade"],
                }
            )

        rag_top_k = self.config.get("top_k", 6) if top_k is None else top_k
        rag_reports = self.rag_store.report_all(prompt, top_k=rag_top_k)

        # Ensure FAISS results are surfaced before Chroma when both are available.
        preferred_order = {"faiss": 0, "chroma": 1}
        indexed_reports = list(enumerate(rag_reports))
        indexed_reports.sort(
            key=lambda item: (
                preferred_order.get(item[1].get("id"), len(preferred_order) + item[0]),
                item[0],
            )
        )

        rag_results: List[Dict[str, Any]] = []
        for original_index, report in indexed_reports:
            if not report.get("available", True):
                entry = {
                    **report,
                    "response": "",
                    "usage": None,
                    "context_tokens": report.get("tokens", 0),
                    "sequence": None,
                    "generation_latency_ms": 0,
                    "retrieval_latency_ms": report.get("latency_ms", 0),
                }
                if reference_available:
                    entry["grade"] = {
                        "score": 0.0,
                        "grade": "N/A",
                        "explanation": report.get("error") or "Backend unavailable",
                    }
                else:
                    entry["grade"] = {
                        "score": 0.0,
                        "grade": "N/A",
                        "explanation": "DML response unavailable for comparison.",
                    }
                rag_results.append(entry)
                continue

            rag_context = self._format_rag_context(report.get("context") or "")
            rag_prompt = self._compose_prompt(prompt, rag_context)
            rag_response, rag_usage, rag_latency = self._generate_with_metrics(
                rag_prompt, max_new_tokens=max_new_tokens
            )
            step_counter += 1
            entry = {
                "id": report.get("id"),
                "label": report.get("label"),
                "strategy": report.get("strategy"),
                "response": rag_response,
                "usage": rag_usage,
                "context": rag_context,
                "context_tokens": report.get("tokens"),
                "documents": report.get("documents"),
                "sequence": step_counter,
                "retrieval_latency_ms": report.get("latency_ms", 0),
                "generation_latency_ms": rag_latency,
                "available": True,
                "error": report.get("error"),
            }

            if reference_available:
                grade = self._grade_response(rag_response, dml_reference)
                entry["grade"] = grade
                evaluations.append(
                    {
                        "backend_id": entry.get("id"),
                        "score": grade.get("score", 0.0),
                        "grade": grade.get("grade"),
                    }
                )
            else:
                entry["grade"] = {
                    "score": 0.0,
                    "grade": "N/A",
                    "explanation": "DML response unavailable for comparison.",
                }

            rag_results.append(entry)
            pipeline_trace.append(
                {
                    "step": step_counter,
                    "stage": "rag",
                    "id": entry["id"],
                    "label": entry.get("label"),
                }
            )

        return {
            "prompt": prompt,
            "base": {
                "response": base_response,
                "usage": base_usage,
                "sequence": base_sequence,
                "generation_latency_ms": base_latency,
            },
            "rag_backends": rag_results,
            "dml": {
                "response": dml_response,
                "usage": dml_usage,
                "context": dml_context,
                "context_tokens": utils.estimate_tokens(dml_context),
                "avg_fidelity": dml_report["avg_fidelity"],
                "entries": dml_report["entries"],
                "sequence": dml_sequence,
                "retrieval_latency_ms": dml_report.get("latency_ms", 0),
                "generation_latency_ms": dml_latency,
                "grade": dml_grade,
            },
            "rag_token_breakdown": [
                {
                    "id": entry.get("id"),
                    "label": entry.get("label"),
                    "strategy": entry.get("strategy"),
                    "tokens": entry.get("context_tokens", 0),
                    "sequence": entry.get("sequence"),
                    "retrieval_latency_ms": entry.get("retrieval_latency_ms", 0),
                }
                for entry in rag_results
                if entry.get("available")
            ],
            "pipeline_trace": pipeline_trace,
            "evaluations": evaluations,
        }

    def knowledge_report(self) -> Dict:
        """Expose summaries of the RAG corpus and DML memory lattice."""

        rag_summary = self.rag_store.catalog_summary()

        dml_items = []
        dml_total_tokens = 0
        store_items = self.store.items()
        truncated = len(store_items) > KNOWLEDGE_MAX_ENTRIES
        for index, item in enumerate(store_items):
            text = (item.text or "").strip()
            tokens = utils.estimate_tokens(text)
            dml_total_tokens += tokens
            if index < KNOWLEDGE_MAX_ENTRIES:
                dml_items.append(
                    {
                        "id": item.id,
                        "level": item.level,
                        "fidelity": item.fidelity,
                        "tokens": tokens,
                        "summary": self._trim_summary(text),
                        "meta": item.meta or {},
                    }
                )

        return {
            "rag": rag_summary,
            "dml": {
                "entries": dml_items,
                "total_tokens": dml_total_tokens,
                "count": len(store_items),
                "truncated": truncated,
                "display_limit": KNOWLEDGE_MAX_ENTRIES,
            },
        }

    def _generate_with_metrics(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
    ) -> tuple[str, Optional[dict], int]:
        start = time.perf_counter()
        response = self.runner.generate(prompt, max_new_tokens=max_new_tokens)
        usage = self.runner.last_usage
        latency_ms = int((time.perf_counter() - start) * 1000.0)
        return response, usage, latency_ms

    def _grade_response(self, candidate: str, reference: str) -> Dict[str, Any]:
        candidate = (candidate or "").strip()
        reference = (reference or "").strip()
        if not candidate:
            return {
                "score": 0.0,
                "grade": "F",
                "explanation": "No response generated by the backend.",
            }
        if not reference:
            return {
                "score": 0.0,
                "grade": "N/A",
                "explanation": "Reference answer missing for comparison.",
            }
        try:
            cand_vec = np.asarray(self.embedder.embed(candidate), dtype=np.float32)
            ref_vec = np.asarray(self.embedder.embed(reference), dtype=np.float32)
            score = float(utils.cosine_similarity(cand_vec, ref_vec))
        except Exception:
            return {
                "score": 0.0,
                "grade": "N/A",
                "explanation": "Failed to compute similarity for grading.",
            }
        grade = self._score_to_grade(score)
        return {
            "score": score,
            "grade": grade,
            "explanation": f"Cosine similarity to DML response: {score:.2f}",
        }

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= 0.9:
            return "A"
        if score >= 0.75:
            return "B"
        if score >= 0.6:
            return "C"
        if score >= 0.45:
            return "D"
        return "F"

    def _trim_summary(self, text: str) -> str:
        if not text:
            return ""
        if len(text) <= KNOWLEDGE_ENTRY_PREVIEW_CHARS:
            return text
        truncated = text[: KNOWLEDGE_ENTRY_PREVIEW_CHARS - 1].rstrip()
        return f"{truncated}…"

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load_persisted_state(self) -> None:
        state_loaded = False
        if self._persistence_enabled:
            try:
                items = load_persisted_memories(self._persistence_path)
            except FileNotFoundError:
                pass
            except Exception:
                LOGGER.exception(
                    "Failed to load durable DML state from %s", self._persistence_path
                )
            else:
                payload = {"items": [item.to_dict() for item in items]}
                self._ensure_embedding_compatibility(payload)
                self.store.import_state(payload)
                state_loaded = True
        if not state_loaded:
            with contextlib.suppress(Exception):
                if self.dml_state_path.exists():
                    data = json.loads(self.dml_state_path.read_text(encoding="utf-8"))
                    self._ensure_embedding_compatibility(data)
                    self.store.import_state(data)
        with contextlib.suppress(Exception):
            if self.rag_state_path.exists():
                data = json.loads(self.rag_state_path.read_text(encoding="utf-8"))
                self.rag_store.import_state(data)

    def _persist_all(self) -> None:
        self._persist_dml_state()
        self._persist_rag_state()

    def _start_persistence_loop(self) -> None:
        if self._persistence_thread and self._persistence_thread.is_alive():
            return
        if self._persistence_interval <= 0:
            return
        self._persistence_stop_event.clear()
        self._persistence_thread = threading.Thread(
            target=self._persistence_loop,
            name="dml-persistence",
            daemon=True,
        )
        self._persistence_thread.start()

    def _stop_persistence_loop(self) -> None:
        self._persistence_stop_event.set()
        thread = self._persistence_thread
        if thread and thread.is_alive():
            thread.join(timeout=max(2.0, float(self._persistence_interval)))
        self._persistence_thread = None

    def _persistence_loop(self) -> None:
        while not self._persistence_stop_event.wait(self._persistence_interval):
            try:
                self._persist_dml_state()
            except Exception:
                LOGGER.exception("Failed to persist DML state during background save.")

    def _gather_checkpoint_state(self) -> Dict[str, Any]:
        """Collect a combined state payload for checkpointing."""

        return {
            "timestamp": time.time(),
            "dml": self.store.export_state(),
            "rag": self.rag_store.export_state(),
            "stats": self.stats(),
        }

    def _persist_dml_state(self) -> None:
        if self._persistence_enabled:
            with self._persist_lock:
                items = self.store.items()
                try:
                    save_persisted_memories(items, self._persistence_path)
                except Exception:
                    LOGGER.exception(
                        "Failed to persist DML state to %s", self._persistence_path
                    )
            return
        with self._persist_lock:
            data = self.store.export_state()
            tmp = self.dml_state_path.with_suffix(".tmp")
            try:
                tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                tmp.replace(self.dml_state_path)
            except Exception:
                LOGGER.exception("Failed to persist DML state to %s", self.dml_state_path)

    def _persist_rag_state(self) -> None:
        with self._persist_lock:
            if self.persistent_rag_store is not None:
                try:
                    self.persistent_rag_store.persist()
                except Exception:
                    LOGGER.exception("Failed to persist persistent RAG index.")
            data = self.rag_store.export_state()
            tmp = self.rag_state_path.with_suffix(".tmp")
            try:
                tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                tmp.replace(self.rag_state_path)
            except Exception:
                LOGGER.exception("Failed to persist RAG state to %s", self.rag_state_path)

    def _ensure_embedding_compatibility(self, payload: Dict) -> None:
        items = payload.get("items") if isinstance(payload, dict) else None
        if not items:
            return
        first = items[0] or {}
        stored_embedding = first.get("embedding")
        if stored_embedding is None:
            return
        try:
            stored_dim = int(np.asarray(stored_embedding, dtype=np.float32).size)
        except Exception:
            return
        try:
            probe = self.embedder.embed("Daystrom persistence probe")
            current_dim = int(np.asarray(probe, dtype=np.float32).size)
        except Exception:
            LOGGER.debug("Unable to determine embedder dimensions for persistence compatibility check.")
            return
        if stored_dim == current_dim or current_dim == 0:
            return
        LOGGER.warning(
            "Embedding dimension changed from %s to %s; re-embedding persisted memories.",
            stored_dim,
            current_dim,
        )
        for entry in items:
            text = entry.get("text") or ""
            try:
                new_embedding = self.embedder.embed(text)
                entry["embedding"] = utils.ensure_serializable(new_embedding)
            except Exception:
                entry["embedding"] = []

    def query_database(self, prompt: str, mode: str = "auto") -> Dict:
        """Retrieve context-aware snippets from the external corpus."""

        if mode not in {"semantic", "literal", "hybrid", "auto"}:
            raise ValueError(f"Unsupported mode: {mode}")
        selected_mode = mode if mode != "auto" else decide_mode(prompt)
        start = time.perf_counter()
        query_embedding = self.embedder.embed(prompt)
        items = self.store.items()
        dml_limit = self._resolve_dml_top_k(None)
        top_k = min(len(items), dml_limit)
        literal_results: List[Dict[str, Any]] = []
        semantic_results: List[Dict[str, Any]] = []
        if selected_mode in {"literal", "hybrid"}:
            literal_results = self._literal_retrieve(
                prompt, items, query_embedding, top_k=top_k
            )
        if selected_mode in {"semantic", "hybrid"}:
            semantic_results = self._semantic_retrieve(query_embedding, top_k=top_k)
        if selected_mode == "literal" and not literal_results:
            # fall back to semantic snippets if no literal hits
            semantic_results = self._semantic_retrieve(query_embedding, top_k=top_k)
        alpha = self._alpha_for_mode(selected_mode)
        combined = self._blend_results(literal_results, semantic_results, alpha, top_k=top_k)
        context_blocks = []
        sources: List[str] = []
        for entry in combined:
            source = entry.get("source") or "unknown"
            if source not in sources and entry.get("source"):
                sources.append(source)
            block_lines = [f"Source: {source}"]
            for segment in entry.get("context", []):
                block_lines.append(segment)
            context_blocks.append("\n".join(block_lines))
        context = "\n\n".join(context_blocks).strip()
        latency = time.perf_counter() - start
        latency_ms = latency * 1000.0
        if self.metrics_enabled:
            record_retrieval(selected_mode, latency_ms)
        token_count = utils.estimate_tokens(context)
        return {
            "mode": selected_mode,
            "context": context,
            "source_docs": sources,
            "tokens": token_count,
            "latency_ms": int(latency_ms),
        }

    def create_checkpoint(self) -> Path:
        """Persist a combined snapshot of the lattice and RAG stores."""

        if self.checkpoint_manager:
            return self.checkpoint_manager.checkpoint()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        manager = CheckpointManager(
            self.checkpoint_dir,
            self._gather_checkpoint_state,
            interval_seconds=0,
            retention=int(self.settings.checkpoint_retention),
            start=False,
        )
        try:
            return manager.checkpoint()
        finally:
            manager.close()

    def stats(self) -> Dict:
        items = self.store.items()
        return {
            "count": len(items),
            "levels": {level: sum(1 for it in items if it.level == level) for level in range(self.store.K + 1)},
            "avg_fidelity": float(np.mean([it.fidelity for it in items]) if items else 0.0),
        }

    def run_maintenance(self, sample_ratio: float = 0.1) -> None:
        """Run a maintenance pass to assess quality without slowing retrieval."""

        self.store.maintenance_pass(sample_ratio=sample_ratio)

    # ------------------------------------------------------------------
    # Multi-tenant helpers used by the DML memory service
    # ------------------------------------------------------------------
    def ingest_memory(
        self,
        text: str,
        *,
        tenant_id: str,
        client_id: Optional[str] = None,
        session_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        kind: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        enriched_meta: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "client_id": client_id,
            "session_id": session_id,
            "instance_id": instance_id,
            "kind": kind or "memory",
        }
        if meta:
            enriched_meta.update(meta)
        embedding = self.embedder.embed(text)
        return self.store.add(text=text, embedding=embedding, meta=enriched_meta)

    def retrieve_context(
        self,
        query: str,
        *,
        tenant_id: str,
        client_id: Optional[str] = None,
        session_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        kinds: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            limit = int(top_k) if top_k is not None else int(self.config.get("dml_top_k", DEFAULT_DML_TOP_K))
        except (TypeError, ValueError):
            limit = int(self.config.get("dml_top_k", DEFAULT_DML_TOP_K))
        limit = max(1, min(MAX_RETRIEVAL_TOP_K, limit))
        query_embedding = self.embedder.embed(query)
        candidates = self.store.retrieve_filtered(
            query_embedding,
            tenant_id=tenant_id,
            client_id=client_id,
            session_id=session_id,
            instance_id=instance_id,
            kinds=kinds,
            top_k=limit,
        )

        budget = int(self.config.get("token_budget", 600))
        consumed = 0
        lines: List[str] = [STARFLEET_BANNER, "=== Daystrom Memory Lattice ==="]
        entries: List[Dict[str, Any]] = []
        for item in candidates:
            summary = item.cached_summary(max_len=180)
            tokens = utils.estimate_tokens(summary)
            if consumed + tokens > budget:
                break
            consumed += tokens
            lines.append(f"- L{item.level} (f={item.fidelity:.2f}): {summary}")
            entries.append(
                {
                    "id": str(item.id),
                    "summary": summary,
                    "meta": item.meta or {},
                    "level": item.level,
                    "fidelity": item.fidelity,
                    "tokens": tokens,
                }
            )

        raw_context = "\n".join(lines)
        latency_ms = int((time.perf_counter() - start) * 1000.0)
        return {
            "entries": entries,
            "context_tokens": consumed,
            "raw_context": raw_context,
            "latency_ms": latency_ms,
        }

    def collect_instance_scratch(
        self,
        tenant_id: str,
        client_id: Optional[str],
        session_id: Optional[str],
        instance_id: Optional[str],
    ) -> List[MemoryStore.MemoryItem]:
        return self.store.list_scratch(
            tenant_id=tenant_id,
            client_id=client_id,
            session_id=session_id,
            instance_id=instance_id,
        )

    def record_agent_workflow(
        self, task_description: str, steps: List[str], outcome: str
    ) -> Optional[str]:
        """Optionally store a successful agent workflow as a reusable template."""

        if not self.enable_workflow_cache:
            return None

        text = self._format_workflow_text(task_description, steps, outcome)
        embedding = self.embedder.embed(text)
        item = self.store.add(
            text=text,
            embedding=embedding,
            meta={
                "kind": "workflow",
                "task_description": task_description,
                "steps_count": len(steps),
                "outcome": outcome,
            },
        )
        return str(item.id)

    def suggest_workflows_for_task(
        self, task_description: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve reusable workflow templates related to a new task."""

        if not self.enable_workflow_cache:
            return []
        try:
            limit = max(1, int(top_k))
        except (TypeError, ValueError):
            limit = 3

        query_embedding = self.embedder.embed(task_description)
        candidates = self.store.retrieve_by_kind(
            query_embedding=query_embedding, kind="workflow", top_k=limit
        )

        results: List[Dict[str, Any]] = []
        for item in candidates:
            results.append(
                {
                    "id": item.id,
                    "summary": item.cached_summary(max_len=200),
                    "task_description": (item.meta or {}).get("task_description", ""),
                    "outcome": (item.meta or {}).get("outcome", ""),
                }
            )
        return results

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _format_rag_context(self, context: str) -> str:
        if not context:
            return ""
        return "=== RAG Retrieval ===\n" + context.strip()

    def _format_dml_context(self, entries: List[Dict]) -> str:
        if not entries:
            return ""
        lines = [STARFLEET_BANNER, "=== Daystrom Memory Lattice ==="]
        for entry in entries:
            lines.append(
                f"- L{entry['level']} (f={entry['fidelity']:.2f}): {entry['summary']}"
            )
        return "\n".join(lines)

    def _format_workflow_text(
        self, task_description: str, steps: List[str], outcome: str
    ) -> str:
        lines = [
            f"Task: {task_description.strip()}",
            "Steps:",
        ]
        for idx, step in enumerate(steps, start=1):
            lines.append(f"{idx}. {step}")
        lines.append(f"Outcome: {outcome.strip()}")
        return "\n".join(lines)

    def _compose_prompt(self, prompt: str, context: str) -> str:
        blocks: List[str] = []
        if context:
            blocks.append(context.strip())
        blocks.append("=== User Prompt ===")
        blocks.append(prompt.strip())
        return "\n\n".join(blocks)

    def _resolve_storage_path(self, candidate: Path) -> Path:
        path = candidate.expanduser()
        if path.is_absolute():
            return path
        relative = path
        if relative.parts and relative.parts[0] == self.storage_dir.name:
            if len(relative.parts) > 1:
                relative = Path(*relative.parts[1:])
            else:
                relative = Path(relative.name)
        return (self.storage_dir / relative).expanduser()

    def _estimate_salience(self, text: str) -> float:
        tokens = utils.estimate_tokens(text)
        return float(max(0.1, min(1.0, tokens / 200.0)))

    def _fallback_truncate(self, text: str, *, max_len: int) -> str:
        cleaned = str(text or "").strip()
        if len(cleaned) <= max_len:
            return cleaned
        return cleaned[: max_len - 3].rstrip() + "..."

    def _summary_for_item(self, item: MemoryStore.MemoryItem, *, max_len: int) -> str:
        summary = ""
        if item.meta:
            summary = str(item.meta.get("summary") or "").strip()
        if summary:
            return self._fallback_truncate(summary, max_len=max_len)
        return self._fallback_truncate(item.text, max_len=max_len)

    def _retrieve_items(self, prompt: str, top_k: Optional[int]) -> List[MemoryStore.MemoryItem]:
        limit = self._resolve_dml_top_k(top_k)
        prompt_embedding = self.embedder.embed(prompt)
        return self.store.retrieve(prompt_embedding, top_k=limit)

    def _resolve_dml_top_k(self, requested: Optional[int]) -> int:
        """Resolve a safe retrieval cap.

        Non-positive or invalid values fall back to the configured default to
        avoid unbounded scans of the lattice during retrieval.
        """

        if requested is None:
            raw_value = self.config.get("dml_top_k", DEFAULT_DML_TOP_K)
        else:
            raw_value = requested
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            LOGGER.warning(
                "Invalid dml_top_k value %r; defaulting to %d.",
                raw_value,
                DEFAULT_DML_TOP_K,
            )
            return DEFAULT_DML_TOP_K
        if parsed <= 0:
            return DEFAULT_DML_TOP_K
        return parsed

    def _prepare_context(
        self, prompt: str, items: List[MemoryStore.MemoryItem]
    ) -> tuple[List[Dict], str, int]:
        budget = int(self.config.get("token_budget", 600))
        consumed = 0
        lines: List[str] = [STARFLEET_BANNER, "=== Daystrom Memory Lattice ==="]
        entries: List[Dict] = []
        for item in items:
            summary = item.cached_summary(max_len=180)
            tokens = utils.estimate_tokens(summary)
            if consumed + tokens > budget:
                break
            consumed += tokens
            lines.append(f"- L{item.level} (f={item.fidelity:.2f}): {summary}")
            entries.append(
                {
                    "id": item.id,
                    "summary": summary,
                    "level": item.level,
                    "fidelity": float(item.fidelity),
                    "salience": float(item.salience),
                    "meta": item.meta or {},
                    "tokens": tokens,
                }
            )
        lines.append("=== User Prompt ===")
        lines.append(prompt)
        preamble = "\n".join(lines)
        return entries, preamble, consumed

    def _classify_mode(self, prompt: str) -> str:
        """Backward compatible wrapper around the intent router."""
        return decide_mode(prompt)

    def _alpha_for_mode(self, mode: str) -> float:
        if mode == "semantic":
            return 0.8
        if mode == "literal":
            return 0.2
        if mode == "hybrid":
            return 0.5
        return 0.5

    def _semantic_retrieve(self, query_embedding: np.ndarray, *, top_k: int) -> List[Dict]:
        items = self.store.retrieve(query_embedding, top_k=top_k)
        results: List[Dict] = []
        for item in items:
            summary = item.cached_summary(max_len=220)
            source = item.meta.get("doc_path") if item.meta else None
            similarity = utils.cosine_similarity(item.embedding, query_embedding)
            results.append(
                {
                    "text": summary,
                    "context": [summary],
                    "semantic_score": similarity,
                    "literal_score": 0.0,
                    "source": source,
                    "origin": "semantic",
                }
            )
        return results

    def _literal_retrieve(
        self,
        prompt: str,
        items: List[Any],
        query_embedding: np.ndarray,
        *,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        store_matches: List[Dict[str, Any]] = []
        if self.persistent_rag_store is not None:
            with contextlib.suppress(Exception):
                store_matches = self.persistent_rag_store.search(query_embedding, top_k=top_k)
        if store_matches:
            return self._format_rag_matches(store_matches)
        fallback = self.literal_retriever.retrieve(prompt, items, query_embedding, top_k=top_k)
        formatted: List[Dict[str, Any]] = []
        for result in fallback:
            meta = result.item.meta if getattr(result, "item", None) else {}
            formatted.append(
                {
                    "context": list(result.context),
                    "semantic_score": float(result.semantic_score),
                    "literal_score": float(result.literal_score),
                    "source": result.source,
                    "meta": dict(meta or {}),
                    "text": result.snippet,
                    "origin": "literal",
                }
            )
        return formatted

    def _format_rag_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        max_snippet_chars = getattr(self.literal_retriever, "max_snippet_chars", 320)

        def _clip_segment(value: Any) -> str:
            segment = str(value or "").strip()
            if not segment:
                return ""
            if len(segment) <= max_snippet_chars:
                return segment
            return segment[: max_snippet_chars - 3] + "..."

        for match in matches:
            meta = dict(match.get("meta") or {})
            context_segments: List[str] = []
            raw_context = meta.get("context")
            if isinstance(raw_context, list):
                context_segments.extend(
                    _clip_segment(segment)
                    for segment in raw_context
                    if isinstance(segment, str) and segment.strip()
                )
            for key in ("context_before", "preceding"):
                value = meta.get(key)
                if isinstance(value, str) and value.strip():
                    clipped = _clip_segment(value)
                    if clipped:
                        context_segments.append(clipped)
            text = _clip_segment(match.get("text"))
            if text:
                context_segments.append(text)
            for key in ("context_after", "following"):
                value = meta.get(key)
                if isinstance(value, str) and value.strip():
                    clipped = _clip_segment(value)
                    if clipped:
                        context_segments.append(clipped)
            deduped: List[str] = []
            seen: set[str] = set()
            for segment in context_segments:
                if not segment or segment in seen:
                    continue
                seen.add(segment)
                deduped.append(segment)
            if not deduped and text:
                deduped = [text]
            source = self.literal_retriever._resolve_source(meta) if meta else None
            formatted.append(
                {
                    "id": match.get("id"),
                    "text": text,
                    "meta": meta,
                    "literal_score": float(match.get("score", 0.0)),
                    "semantic_score": 0.0,
                    "source": source,
                    "context": deduped,
                    "origin": "literal",
                }
            )
        return formatted

    def _blend_results(
        self,
        literal_results: List[Dict[str, Any]],
        semantic_results: List[Dict[str, Any]],
        alpha: float,
        *,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        def _normalise(entry: Dict[str, Any]) -> Dict[str, Any]:
            data = dict(entry)
            context_value = data.get("context") or []
            if isinstance(context_value, str):
                context_list = [context_value]
            else:
                context_list = [
                    str(segment).strip()
                    for segment in context_value
                    if isinstance(segment, str) and segment.strip()
                ]
            data["context"] = context_list
            data["semantic_score"] = float(data.get("semantic_score", 0.0))
            data["literal_score"] = float(data.get("literal_score", 0.0))
            origin = str(data.get("origin") or "semantic")
            data["origin"] = origin
            if origin == "literal":
                data["context"] = self._clip_literal_context(data["context"])
            return data

        blended: List[Dict[str, Any]] = []
        blended.extend(_normalise(res) for res in literal_results)
        blended.extend(_normalise(res) for res in semantic_results)
        for entry in blended:
            semantic_score = entry.get("semantic_score", 0.0)
            literal_score = entry.get("literal_score", 0.0)
            entry["final_score"] = alpha * semantic_score + (1 - alpha) * literal_score
        blended.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        seen: set[str] = set()
        deduped: List[Dict[str, Any]] = []
        for entry in blended:
            key = "|".join(entry.get("context", []))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entry)
            if len(deduped) >= top_k:
                break
        constrained = self._apply_token_budgets(deduped)
        return constrained

    def _clip_literal_context(self, segments: List[str]) -> List[str]:
        if not segments:
            return []
        tokens_used = 0
        clipped: List[str] = []
        for segment in segments[: self.literal_snippet_cap]:
            text = str(segment or "").strip()
            if not text:
                continue
            tokens = max(1, utils.estimate_tokens(text))
            if tokens_used + tokens > self.literal_token_cap:
                remaining = self.literal_token_cap - tokens_used
                if remaining <= 0:
                    break
                approx_chars = max(32, remaining * 4)
                truncated = text[:approx_chars]
                if len(text) > approx_chars:
                    truncated = truncated.rstrip() + "..."
                clipped.append(truncated)
                tokens_used = self.literal_token_cap
                break
            clipped.append(text)
            tokens_used += tokens
        return clipped

    def _apply_token_budgets(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not entries:
            return []
        total_budget = int(self.config.get("token_budget", 600))
        total_budget = max(1, total_budget)
        budgets = self.config.get("budgets", {}) or {}

        def _pct(key: str, default: float) -> float:
            raw = budgets.get(key, default)
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default

        semantic_limit = max(0, int(total_budget * _pct("semantic_pct", 0.7)))
        literal_limit = max(0, int(total_budget * _pct("literal_pct", 0.2)))
        free_limit = max(0, total_budget - semantic_limit - literal_limit)

        consumed = {"semantic": 0, "literal": 0}
        free_used = 0
        total_used = 0
        allowed: List[Dict[str, Any]] = []
        for entry in entries:
            origin = str(entry.get("origin") or "semantic")
            if origin not in consumed:
                origin = "semantic"
            context_segments = entry.get("context") or []
            if not isinstance(context_segments, list):
                context_segments = [str(context_segments)]
            tokens = entry.get("tokens")
            if not isinstance(tokens, (int, float)):
                tokens = sum(max(1, utils.estimate_tokens(seg)) for seg in context_segments)
            tokens = int(max(1, tokens))
            if total_used + tokens > total_budget:
                continue
            limit = semantic_limit if origin == "semantic" else literal_limit
            if consumed[origin] + tokens <= limit:
                consumed[origin] += tokens
                total_used += tokens
                entry["tokens"] = tokens
                allowed.append(entry)
                continue
            if free_used + tokens <= free_limit:
                free_used += tokens
                total_used += tokens
                entry["tokens"] = tokens
                allowed.append(entry)

        if not allowed and entries:
            first = entries[0]
            first_tokens = sum(
                max(1, utils.estimate_tokens(seg)) for seg in first.get("context", [])
            )
            first["tokens"] = int(max(1, first_tokens))
            return [first]
        return allowed

    def __del__(self) -> None:  # pragma: no cover - destructor best effort
        try:
            self.close()
        except Exception:
            pass


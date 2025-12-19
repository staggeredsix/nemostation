"""Streamlit playground for live DML retrieval visualisation."""
from __future__ import annotations

import os
import shutil
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

try:  # PDF ingestion is optional but preferred when available
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

from daystrom_dml.dml_adapter import DMLAdapter
from app.benchmarks import BENCHMARK_SUITES, BenchmarkSuite

st.set_page_config(page_title="Daystrom Playground", layout="wide")

# Pricing defaults — roughly aligns with blended GPT-4o prompt/completion rates.
DEFAULT_BASELINE_TOKENS = 8192
DEFAULT_PRICE_PER_1K = 0.01

LLM_MODEL_OPTIONS = [
    {"label": "Mistral 7B Instruct (default)", "value": "mistralai/Mistral-7B-Instruct-v0.2"},
    {"label": "Llama 3 8B Instruct", "value": "meta/llama3-8b-instruct"},
    {"label": "Llama 3 70B Instruct", "value": "meta/llama3-70b-instruct"},
    {"label": "Mixtral 8x7B Instruct", "value": "mistralai/Mixtral-8x7B-Instruct-v0.1"},
]

EMBEDDING_MODEL_OPTIONS = [
    {
        "label": "MiniLM-L6 v2 (default)",
        "value": "sentence-transformers/all-MiniLM-L6-v2",
    },
    {"label": "BGE Base EN v1.5", "value": "BAAI/bge-base-en-v1.5"},
    {"label": "BGE Large EN v1.5", "value": "BAAI/bge-large-en-v1.5"},
]


AGENTIC_SCENARIOS = {
    "project_worker": {
        "id": "project_worker",
        "name": "Multi-Day Project Worker",
        "description": "Tracks evolving project decisions over several days and tests if the system surfaces the latest, correct decisions.",
        "steps": [
            {"role": "memory", "text": "We are planning a product launch. Track every decision starting today."},
            {"role": "memory", "text": "We decided the launch date will be October 12."},
            {"role": "memory", "text": "We chose the codename: Project Aurora."},
            {"role": "memory", "text": "Change of plans — the launch date moves to October 20."},
            {"role": "memory", "text": "The codename stays Aurora."},
            {"role": "memory", "text": "We added a stretch goal: Slack integration."},
            {"role": "memory", "text": "Remove the old idea about Teams integration. Not happening."},
            {
                "role": "query",
                "text": "Give me an authoritative, up-to-date summary of the project decisions we’ve made so far.",
            },
        ],
    },
    "tool_troubleshooting": {
        "id": "tool_troubleshooting",
        "name": "Tool-Using Troubleshooting Agent",
        "description": "Simulates a diagnostic agent running tools, handling failures, and converging on a stable system state.",
        "steps": [
            {"role": "memory", "text": "Run diagnostic A. It failed with code 43."},
            {"role": "memory", "text": "Retry diagnostic A with safe mode enabled."},
            {"role": "memory", "text": "Diagnostic A succeeded this time."},
            {"role": "memory", "text": "Move on to Diagnostic B. Diagnostic B passed on the first try."},
            {"role": "memory", "text": "Ignore the earlier code 43 failure now that the retry succeeded."},
            {"role": "memory", "text": "We only store the final successful state for diagnostics when possible."},
            {
                "role": "query",
                "text": "Summarize what diagnostics were run, which ones failed, how they were resolved, and the final system health state.",
            },
        ],
    },
    "writing_prefs": {
        "id": "writing_prefs",
        "name": "Knowledge Worker Writing Preferences",
        "description": "Tests updating and overriding user preferences over time and whether the system surfaces the latest rules only.",
        "steps": [
            {"role": "memory", "text": "When writing emails for me, always use a formal, professional style."},
            {"role": "memory", "text": "Keep messages at around 150 words."},
            {"role": "memory", "text": "New rule: use a friendly, casual tone instead of a formal one."},
            {"role": "memory", "text": "Ignore the old formality rule."},
            {"role": "memory", "text": "Preferred email length is now around 75 words."},
            {
                "role": "query",
                "text": "State my current writing style preferences clearly and ignore outdated rules.",
            },
        ],
    },
}


# Benchmark suites -----------------------------------------------------


def _token_overlap_score(reference: str, candidate: str) -> float:
    """Simple lexical overlap score between reference and candidate answers."""

    ref_tokens = {token.strip(".,!?;:") for token in reference.lower().split() if token.strip()}
    cand_tokens = {token.strip(".,!?;:") for token in candidate.lower().split() if token.strip()}
    if not ref_tokens or not cand_tokens:
        return 0.0
    overlap = ref_tokens & cand_tokens
    union = ref_tokens | cand_tokens
    return float(len(overlap) / len(union))


def llm_judge_score(question: str, context: str, answer: str, reference: str) -> float:
    """Optional: hook for an LLM-judged score.

    Currently returns ``0.0`` as a stub so the UI can present "N/A" when the
    checkbox is disabled. Replace with an evaluator prompt if needed.
    """

    return 0.0


def _normalise_storage_dir(path: Path) -> Path:
    """Expand ``path`` and ensure it is rooted on the local filesystem."""

    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return Path.cwd() / expanded


def _model_overrides(storage_dir: Path) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {"storage_dir": str(_normalise_storage_dir(storage_dir))}
    llm_model = (st.session_state.get("selected_llm_model") or "").strip()
    embedding_model = (st.session_state.get("selected_embedding_model") or "").strip()
    if llm_model:
        overrides["model_name"] = llm_model
    if embedding_model:
        overrides["embedding_model"] = embedding_model
    return overrides


def _scenario_choices() -> tuple[list[str], dict[str, str]]:
    names = [scenario["name"] for scenario in AGENTIC_SCENARIOS.values()]
    choices = ["None", *names]
    mapping = {scenario["name"]: scenario_id for scenario_id, scenario in AGENTIC_SCENARIOS.items()}
    return choices, mapping


def _benchmark_doc_path(suite_id: str, doc_id: str) -> str:
    return f"benchmark:{suite_id}:{doc_id}"


def _ensure_benchmark_corpus(adapter: DMLAdapter, suite: BenchmarkSuite) -> None:
    """Ingest benchmark documents into the lattice and RAG store once per session."""

    ingested_key = f"bench_ingested_{suite['id']}_{adapter.storage_dir}"
    if st.session_state.get(ingested_key):
        return

    for document in suite.get("documents", []):
        doc_id = document.get("id") or "doc"
        meta = dict(document.get("meta") or {})
        meta.setdefault("doc_path", _benchmark_doc_path(suite["id"], doc_id))
        meta.setdefault("benchmark_suite", suite["id"])
        meta.setdefault("benchmark_doc_id", doc_id)
        adapter.ingest(document.get("text") or "", meta=meta)

    st.session_state[ingested_key] = True


def _extract_id_from_document(document: Any) -> str:
    if isinstance(document, dict):
        for key in ("id", "source", "label", "doc_path"):
            value = document.get(key)
            if value:
                return str(value)
    return str(document)


def _collect_retrieved_ids(documents: list[Any]) -> list[str]:
    ids: list[str] = []
    for doc in documents or []:
        ids.append(_extract_id_from_document(doc))
    return ids


def _compute_retrieval_scores(retrieved: list[str], ground_truth: list[str], *, k: int) -> dict[str, float]:
    if not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "hit": 0.0}
    retrieved_set = set(retrieved[:k])
    truth_set = set(ground_truth)
    overlap = retrieved_set & truth_set
    precision = float(len(overlap) / max(k, 1))
    recall = float(len(overlap) / max(len(truth_set), 1))
    hit = 1.0 if overlap else 0.0
    return {"precision": precision, "recall": recall, "hit": hit}


def _rag_retrieval(adapter: DMLAdapter, prompt: str, *, top_k: int) -> Optional[Dict[str, Any]]:
    reports = adapter.rag_store.report_all(prompt, top_k=top_k)
    if not reports:
        return None
    report = next((entry for entry in reports if entry.get("available", True)), reports[0])
    documents = report.get("documents") or []
    context = adapter._format_rag_context(report.get("context") or "")
    return {
        "report": report,
        "context": context,
        "retrieved_ids": _collect_retrieved_ids(documents),
        "tokens": report.get("tokens", 0),
        "latency_ms": report.get("latency_ms", 0),
    }


def _dml_retrieval(adapter: DMLAdapter, prompt: str, *, top_k: int) -> Dict[str, Any]:
    report = adapter.retrieval_report(prompt, top_k=top_k)
    dml_ids = []
    for entry in report.get("entries", []) or []:
        meta = entry.get("meta") or {}
        doc_path = meta.get("doc_path") or meta.get("memory_id")
        dml_ids.append(str(doc_path or entry.get("id")))
    context = adapter._format_dml_context(report.get("entries", []))
    return {
        "report": report,
        "context": context,
        "retrieved_ids": dml_ids,
        "tokens": report.get("tokens", 0),
        "latency_ms": report.get("latency_ms", 0),
    }


def _rag_generate(adapter: DMLAdapter, prompt: str, retrieval: Dict[str, Any], *, max_new_tokens: int) -> Dict[str, Any]:
    context = retrieval.get("context") or ""
    augmented_prompt = adapter._compose_prompt(prompt, context)
    response, usage, latency = adapter._generate_with_metrics(augmented_prompt, max_new_tokens=max_new_tokens)
    return {
        "response": response,
        "usage": usage,
        "context_tokens": retrieval.get("tokens", 0),
        "prompt": augmented_prompt,
        "prompt_tokens": usage.get("prompt_tokens") if usage else None,
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "generation_latency_ms": latency,
    }


def _dml_generate(
    adapter: DMLAdapter,
    prompt: str,
    retrieval: Dict[str, Any],
    *,
    max_new_tokens: int,
    reinforce: bool = True,
) -> Dict[str, Any]:
    context = retrieval.get("context") or ""
    augmented_prompt = adapter._compose_prompt(prompt, context)
    response, usage, latency = adapter._generate_with_metrics(augmented_prompt, max_new_tokens=max_new_tokens)
    if reinforce:
        adapter.reinforce(prompt, response)
    return {
        "response": response,
        "usage": usage,
        "context_tokens": retrieval.get("tokens", 0),
        "prompt": augmented_prompt,
        "prompt_tokens": usage.get("prompt_tokens") if usage else None,
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "generation_latency_ms": latency,
    }


def _needle_answer_hit(expected: Optional[str], answer: str) -> bool:
    if not expected:
        return False
    return expected.lower() in (answer or "").lower()


def _run_agentic_scenario(
    adapter: DMLAdapter, scenario: Dict[str, Any], *, max_new_tokens: int
) -> tuple[Dict[str, Any] | None, list[str]]:
    logs: list[str] = []
    result: Dict[str, Any] | None = None

    for index, step in enumerate(scenario.get("steps", [])):
        role = (step.get("role") or "").lower()
        text = step.get("text") or ""
        prefix = "MEMORY" if role == "memory" else "QUERY"
        logs.append(f"[{prefix}] {text}")

        if role == "memory":
            adapter.ingest(
                text,
                meta={
                    "doc_path": f"scenario:{scenario['id']}",
                    "scenario_id": scenario["id"],
                    "scenario_step": index,
                    "scenario_role": role,
                },
            )
            continue

        if role == "query":
            result = adapter.compare_responses(text, max_new_tokens=max_new_tokens)
            break

    return result, logs


def _resolve_default_storage() -> Path:
    """Return the default storage directory for the playground."""

    env_override = os.environ.get("DML_PLAYGROUND_STORAGE") or os.environ.get(
        "DML_STORAGE_DIR"
    )
    if env_override:
        return _normalise_storage_dir(Path(env_override))
    return _normalise_storage_dir(Path("~/.dml/playground"))


DEFAULT_STORAGE_DIR = _resolve_default_storage()


def _create_adapter(storage_dir: Path) -> DMLAdapter:
    """Create a Daystrom adapter rooted at ``storage_dir``."""

    target = _normalise_storage_dir(storage_dir)
    overrides = _model_overrides(target)
    try:
        target.mkdir(parents=True, exist_ok=True)
        adapter = DMLAdapter(config_overrides=overrides, start_aging_loop=False)
    except Exception as exc:  # pragma: no cover - surfaced in the UI
        raise RuntimeError(f"Failed to initialise Daystrom adapter: {exc}") from exc
    return adapter


def _store_adapter(adapter: DMLAdapter, storage_dir: Path) -> None:
    st.session_state["adapter"] = adapter
    st.session_state["storage_dir"] = str(storage_dir)
    st.session_state.setdefault("selected_llm_model", adapter.config.get("model_name"))
    st.session_state.setdefault(
        "selected_embedding_model", adapter.config.get("embedding_model")
    )


def _close_adapter() -> None:
    adapter: DMLAdapter | None = st.session_state.pop("adapter", None)
    if adapter is not None:
        adapter.close()


def _replace_adapter(storage_dir: Path) -> DMLAdapter:
    _close_adapter()
    adapter = _create_adapter(storage_dir)
    _store_adapter(adapter, _normalise_storage_dir(storage_dir))
    st.session_state.pop("last_result", None)
    st.session_state.pop("advanced_result", None)
    st.session_state.pop("simple_result", None)
    return adapter


def _resolve_storage_dir() -> Path:
    raw = st.session_state.get("storage_dir")
    if raw:
        return _normalise_storage_dir(Path(str(raw)))
    st.session_state["storage_dir"] = str(DEFAULT_STORAGE_DIR)
    return DEFAULT_STORAGE_DIR


def _clear_storage(storage_dir: Path) -> None:
    target = _normalise_storage_dir(storage_dir)
    if not target.exists():
        return
    for child in target.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError as exc:  # pragma: no cover - surfaced via Streamlit
            raise RuntimeError(f"Failed to clear storage: {exc}") from exc


def _vector_to_coords(vector: np.ndarray) -> Tuple[float, float, float]:
    if vector.size == 0:
        return (0.0, 0.0, 0.0)
    arr = np.asarray(vector, dtype=np.float32)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size))
    return float(arr[0]), float(arr[1]), float(arr[2])


def _build_lattice_plot(items: List, highlighted: set[int]) -> go.Figure:
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    colors: List[int] = []
    texts: List[str] = []
    for item in items:
        x, y, z = _vector_to_coords(item.embedding)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        colors.append(item.level)
        summary = item.text[:120].replace("\n", " ")
        texts.append(f"L{item.level} • {summary}")

    base = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        name="Memories",
        marker=dict(size=4, color=colors, colorscale="Viridis", opacity=0.5),
        text=texts,
        hoverinfo="text",
    )

    highlighted_points = []
    if highlighted:
        hx: List[float] = []
        hy: List[float] = []
        hz: List[float] = []
        htext: List[str] = []
        for item in items:
            if item.id not in highlighted:
                continue
            x, y, z = _vector_to_coords(item.embedding)
            hx.append(x)
            hy.append(y)
            hz.append(z)
            htext.append(f"Hit • L{item.level}")
        highlighted_points.append(
            go.Scatter3d(
                x=hx,
                y=hy,
                z=hz,
                mode="markers",
                name="Retrieved",
                marker=dict(size=8, color="#FF3366", opacity=0.95),
                text=htext,
                hoverinfo="text",
            )
        )

    fig = go.Figure(data=[base, *highlighted_points])
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=30),
        height=520,
    )
    return fig


def _render_cost_savings(tokens_used: int, *, key_prefix: str) -> None:
    """Render a calculator estimating savings versus a naive baseline.

    ``tokens_used`` should reflect the tokens consumed by the latest request. The
    calculator compares that to an adjustable "no DML" baseline and a blended
    average generation price.
    """

    st.markdown("#### Cost impact")
    st.caption(
        "We compare the tokens you actually sent to the model against a configurable"
        " 'no-DML' baseline and multiply by the per-1K token rate."
    )
    input_cols = st.columns(2)
    baseline_key = f"{key_prefix}_baseline_tokens"
    price_key = f"{key_prefix}_price_per_1k"
    baseline_default = int(st.session_state.get(baseline_key, DEFAULT_BASELINE_TOKENS))
    price_default = float(st.session_state.get(price_key, DEFAULT_PRICE_PER_1K))

    baseline_tokens = input_cols[0].number_input(
        "Baseline tokens without DML",
        min_value=1,
        value=baseline_default,
        step=512,
        help="Rough size of a naive context window you would have sent to the model.",
    )
    price_per_1k = input_cols[1].number_input(
        "Avg generation price per 1K tokens (USD)",
        min_value=0.0,
        value=price_default,
        step=0.001,
        format="%.4f",
        help="Use your provider's blended prompt + completion rate; defaults to GPT-4o averages.",
    )

    st.session_state[baseline_key] = int(baseline_tokens)
    st.session_state[price_key] = float(price_per_1k)

    adjusted_tokens_used = max(tokens_used, 0)
    baseline_cost = (baseline_tokens / 1000) * price_per_1k
    actual_cost = (adjusted_tokens_used / 1000) * price_per_1k
    token_delta = max(baseline_tokens - adjusted_tokens_used, 0)
    savings = max(baseline_cost - actual_cost, 0.0)
    savings_pct = (savings / baseline_cost * 100) if baseline_cost else 0.0

    metric_cols = st.columns(3)
    metric_cols[0].metric("Baseline spend (est.)", f"${baseline_cost:,.4f}")
    metric_cols[1].metric("DML spend", f"${actual_cost:,.4f}", f"{token_delta} tokens saved")
    metric_cols[2].metric("Savings", f"${savings:,.4f}", f"{savings_pct:.1f}% vs baseline")

    with st.expander("How this is calculated", expanded=False):
        st.markdown(
            textwrap.dedent(
                f"""
                - **Baseline spend:** `(baseline tokens / 1000) × price per 1K` → `${baseline_cost:,.4f}`.
                - **DML spend:** `(actual tokens / 1000) × price per 1K` → `${actual_cost:,.4f}`.
                - **Savings:** `baseline spend - DML spend` → `${savings:,.4f}` ({savings_pct:.1f}% better).
                - **Tokens avoided:** `{token_delta}` fewer tokens sent compared to the baseline window.
                """
            ).strip()
        )


def _run_retrieval(adapter: DMLAdapter, prompt: str, *, mode: str) -> Dict[str, Any] | None:
    cleaned = prompt.strip()
    if not cleaned:
        return None
    try:
        result = adapter.query_database(cleaned, mode=mode)
    except Exception as exc:
        st.error(f"Retrieval failed: {exc}")
        return None
    st.session_state["last_result"] = result
    st.session_state["last_prompt"] = cleaned
    st.session_state["last_mode"] = mode
    return result


def _try_ingest_text(adapter: DMLAdapter, text: str, source_label: str) -> tuple[str, str]:
    snippet = (text or "").strip()
    if not snippet:
        return "warning", "Provide some text before ingesting."
    try:
        adapter.ingest(snippet, meta={"doc_path": source_label})
    except Exception as exc:
        return "error", f"Failed to ingest {source_label}: {exc}"
    return "success", f"Ingested {source_label}"


def _extract_text_from_file(upload) -> str:
    suffix = Path(upload.name).suffix.lower()
    if suffix == ".pdf" and PdfReader is not None:
        try:
            reader = PdfReader(upload)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""
    try:
        return upload.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _ingest_uploads(adapter: DMLAdapter, uploads: list, label_prefix: str = "upload") -> None:
    for index, upload in enumerate(uploads):
        text = _extract_text_from_file(upload)
        label = f"{label_prefix}:{upload.name or index}"
        status, message = _try_ingest_text(adapter, text, label)
        if status == "success":
            st.success(message)
        elif status == "warning":
            st.warning(message)
        else:
            st.error(message)


def _render_sources(sources: List[str]) -> None:
    cleaned = [Path(src).name if src else "unknown" for src in sources]
    if cleaned:
        st.markdown("**Sources**")
        st.markdown("\n".join(f"- {src}" for src in cleaned))
    else:
        st.caption("Sources: none")


def _render_budget_controls(adapter: DMLAdapter) -> None:
    config = adapter.config
    budgets = config.get("budgets", {}) or {}
    total_budget_default = int(config.get("token_budget", 600))

    control_cols = st.columns([1, 1, 1])
    total_budget = control_cols[0].number_input(
        "Total token budget",
        min_value=64,
        max_value=4096,
        value=total_budget_default,
        step=32,
        key="token_budget_total",
    )
    semantic_default = float(budgets.get("semantic_pct", 0.7))
    literal_default = float(budgets.get("literal_pct", 0.2))
    semantic_pct = control_cols[1].slider(
        "Semantic %",
        min_value=0.0,
        max_value=1.0,
        value=min(semantic_default, 1.0),
        step=0.05,
        key="semantic_budget_pct",
    )
    literal_pct = control_cols[2].slider(
        "Literal %",
        min_value=0.0,
        max_value=max(0.0, 1.0 - semantic_pct),
        value=min(literal_default, max(0.0, 1.0 - semantic_pct)),
        step=0.05,
        key="literal_budget_pct",
    )

    free_pct = max(0.0, 1.0 - semantic_pct - literal_pct)
    st.caption(f"Free % (calculated): {free_pct:.2f}")
    budget_cols = st.columns(3)
    budget_cols[0].progress(min(1.0, semantic_pct), text="Semantic")
    budget_cols[1].progress(min(1.0, literal_pct), text="Literal")
    budget_cols[2].progress(min(1.0, free_pct), text="Free")

    config["token_budget"] = int(total_budget)
    config["budgets"] = {
        "semantic_pct": float(semantic_pct),
        "literal_pct": float(literal_pct),
        "free_pct": float(free_pct),
    }


def _run_benchmark_suite(
    adapter: DMLAdapter,
    suite: BenchmarkSuite,
    *,
    run_retrieval: bool,
    run_generation: bool,
    use_llm_judge: bool,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Run retrieval and generation benchmarks for the selected suite."""

    _ensure_benchmark_corpus(adapter, suite)
    k = int(suite.get("k", 3))

    per_query: list[Dict[str, Any]] = []
    retrieval_totals = {"rag": {"precision": 0.0, "recall": 0.0, "hit": 0.0, "count": 0}, "dml": {"precision": 0.0, "recall": 0.0, "hit": 0.0, "count": 0}}
    generation_totals = {"rag": {"ref": 0.0, "judge": 0.0, "context_tokens": 0, "count": 0}, "dml": {"ref": 0.0, "judge": 0.0, "context_tokens": 0, "count": 0}}
    needle_correct = {"rag": 0, "dml": 0, "count": 0}

    for query in suite.get("queries", []):
        question = query.get("question") or ""
        reference_answer = query.get("answer") or ""
        ground_truth = query.get("ground_truth_ids") or []
        query_row: Dict[str, Any] = {"id": query.get("id"), "question": question}

        rag_retrieval = _rag_retrieval(adapter, question, top_k=k)
        dml_retrieval = _dml_retrieval(adapter, question, top_k=k)

        query_row["rag_retrieval_ids"] = (rag_retrieval or {}).get("retrieved_ids", [])
        query_row["dml_retrieval_ids"] = dml_retrieval.get("retrieved_ids", [])

        if run_retrieval and ground_truth:
            if rag_retrieval:
                rag_scores = _compute_retrieval_scores(query_row["rag_retrieval_ids"], ground_truth, k=k)
                for metric, value in rag_scores.items():
                    retrieval_totals["rag"][metric] += value
                retrieval_totals["rag"]["count"] += 1
                query_row["rag_retrieval_hit"] = bool(rag_scores.get("hit"))
            if dml_retrieval:
                dml_scores = _compute_retrieval_scores(query_row["dml_retrieval_ids"], ground_truth, k=k)
                for metric, value in dml_scores.items():
                    retrieval_totals["dml"][metric] += value
                retrieval_totals["dml"]["count"] += 1
                query_row["dml_retrieval_hit"] = bool(dml_scores.get("hit"))

        if run_generation and reference_answer:
            rag_generation: Dict[str, Any] | None = None
            if rag_retrieval:
                rag_generation = _rag_generate(adapter, question, rag_retrieval, max_new_tokens=max_new_tokens)
                reference_score = _token_overlap_score(reference_answer, rag_generation.get("response") or "")
                rag_judge = None
                if use_llm_judge:
                    rag_judge = llm_judge_score(question, rag_retrieval.get("context", ""), rag_generation.get("response") or "", reference_answer)
                generation_totals["rag"]["ref"] += reference_score
                if rag_judge is not None:
                    generation_totals["rag"]["judge"] += rag_judge
                generation_totals["rag"]["context_tokens"] += int(rag_generation.get("context_tokens") or 0)
                generation_totals["rag"]["count"] += 1
                query_row.update(
                    {
                        "rag_answer": rag_generation.get("response"),
                        "rag_reference_score": reference_score,
                        "rag_judge_score": rag_judge,
                        "rag_context_tokens": rag_generation.get("context_tokens"),
                    }
                )

            dml_generation = _dml_generate(
                adapter,
                question,
                dml_retrieval,
                max_new_tokens=max_new_tokens,
                reinforce=False,
            )
            dml_reference = _token_overlap_score(reference_answer, dml_generation.get("response") or "")
            dml_judge = None
            if use_llm_judge:
                dml_judge = llm_judge_score(question, dml_retrieval.get("context", ""), dml_generation.get("response") or "", reference_answer)
            generation_totals["dml"]["ref"] += dml_reference
            if dml_judge is not None:
                generation_totals["dml"]["judge"] += dml_judge
            generation_totals["dml"]["context_tokens"] += int(dml_generation.get("context_tokens") or 0)
            generation_totals["dml"]["count"] += 1
            query_row.update(
                {
                    "dml_answer": dml_generation.get("response"),
                    "dml_reference_score": dml_reference,
                    "dml_judge_score": dml_judge,
                    "dml_context_tokens": dml_generation.get("context_tokens"),
                }
            )

            needle_expected = query.get("meta", {}).get("needle_fact") if isinstance(query.get("meta"), dict) else None
            if needle_expected:
                needle_correct["count"] += 1
                if rag_generation and _needle_answer_hit(needle_expected, rag_generation.get("response") or ""):
                    needle_correct["rag"] += 1
                if _needle_answer_hit(needle_expected, dml_generation.get("response") or ""):
                    needle_correct["dml"] += 1

        per_query.append(query_row)

    def _average(metrics: Dict[str, float]) -> Dict[str, float]:
        count = float(metrics.pop("count", 0))
        if count <= 0:
            count = 0.0
        base = {key: value for key, value in metrics.items()}
        if count:
            base = {key: value / count for key, value in base.items()}
        else:
            base = {key: 0.0 for key in base.keys()}
        return base

    retrieval_summary = {
        "rag": _average(dict(retrieval_totals["rag"])),
        "dml": _average(dict(retrieval_totals["dml"])),
    }
    generation_summary = {
        "rag": _average(dict(generation_totals["rag"])),
        "dml": _average(dict(generation_totals["dml"])),
    }

    needle_summary = None
    if needle_correct["count"]:
        needle_summary = {
            "rag": needle_correct["rag"] / max(needle_correct["count"], 1),
            "dml": needle_correct["dml"] / max(needle_correct["count"], 1),
        }

    return {
        "suite": suite,
        "k": k,
        "queries": per_query,
        "retrieval": retrieval_summary,
        "generation": generation_summary,
        "needle": needle_summary,
    }


def _render_simple_mode(adapter: DMLAdapter, storage_dir: Path) -> None:
    if st.session_state.pop("simple_manual_reset", False):
        st.session_state["simple_manual_text"] = ""

    st.caption("Quickstart mode — minimal controls with safe defaults.")
    st.info(
        textwrap.dedent(
            """
            1. Upload a `.txt`, `.md`, `.json`, or `.log` file to populate the lattice.
            2. Paste snippets for quick notes when a file upload is overkill.
            3. Ask a question — retrieval automatically balances semantic and literal search.
            4. Switch to **Advanced** for storage management, token budgets, and the 3D lattice view.
            """
        ).strip()
    )

    st.subheader("Add knowledge")
    uploaded = st.file_uploader(
        "Upload text, markdown, JSON, logs, or PDFs",
        type=["txt", "md", "log", "json", "pdf"],
        accept_multiple_files=True,
        help="Each file is chunked and stored as individual memories.",
        key="simple_upload",
    )
    if uploaded:
        _ingest_uploads(adapter, uploaded, label_prefix="simple-upload")

    with st.form("simple_manual_ingest"):
        manual_text = st.text_area(
            "Paste text to ingest",
            key="simple_manual_text",
            placeholder="Drop snippets here to create memories without uploading a file.",
            height=160,
        )
        submit_manual = st.form_submit_button(
            "Ingest text snippet", use_container_width=True
        )
    if submit_manual:
        status, message = _try_ingest_text(adapter, manual_text, "manual-entry")
        if status == "success":
            st.success("Manual snippet ingested.")
            st.session_state["simple_manual_reset"] = True
            st.rerun()
        elif status == "warning":
            st.warning(message)
        else:
            st.error(message)

    items = list(adapter.store.items())
    st.markdown("### Lattice health")
    col_count, col_storage = st.columns(2)
    col_count.metric("Stored memories", len(items))
    col_storage.metric("Storage directory", str(storage_dir.resolve()))

    st.markdown("---")
    st.subheader("Ask the lattice")
    with st.form("simple_query_form"):
        st.text_area(
            "Prompt",
            key="simple_prompt",
            placeholder="Ask a question about your ingested data",
        )
        run_simple = st.form_submit_button("Ask", type="primary")
    if run_simple:
        prompt_value = st.session_state.get("simple_prompt", "")
        if not prompt_value.strip():
            st.warning("Enter a prompt before running retrieval.")
        else:
            result = _run_retrieval(adapter, prompt_value, mode="auto")
            if result:
                st.session_state["simple_result"] = result

    result = st.session_state.get("simple_result")
    if result:
        st.markdown(f"**Mode:** {result['mode']}")
        st.write(result.get("context") or "No context produced.")
        sources = result.get("source_docs", []) or []
        _render_sources(sources)
        metrics = st.columns(2)
        metrics[0].metric("Tokens", int(result.get("tokens", 0)))
        metrics[1].metric("Latency (ms)", int(result.get("latency_ms", 0)))

        _render_cost_savings(int(result.get("tokens", 0)), key_prefix="simple")

    st.markdown("---")
    st.caption(
        "Ready for power features? Switch to **Advanced** to manage storage, inspect memory salience, and explore the 3D lattice."
    )


def _render_advanced_mode(adapter: DMLAdapter, storage_dir: Path) -> None:
    if st.session_state.pop("advanced_manual_reset", False):
        st.session_state["advanced_manual_text"] = ""

    st.caption("Advanced mode — full control for power users and enterprise tuning.")

    prompt = st.text_area(
        "Prompt",
        key="advanced_prompt",
        placeholder="Ask a question about your ingested data",
    )
    mode = st.selectbox(
        "Retrieval mode",
        options=["auto", "semantic", "literal", "hybrid"],
        index=0,
        key="advanced_retrieval_mode",
    )
    run_query = st.button("Run retrieval", type="primary", key="advanced_run")
    if run_query:
        if not prompt.strip():
            st.warning("Enter a prompt before running retrieval.")
        else:
            result = _run_retrieval(adapter, prompt, mode=mode)
            if result:
                st.session_state["advanced_result"] = result

    result: Dict[str, Any] | None = st.session_state.get("advanced_result")
    if result is None:
        result = st.session_state.get("last_result")

    items = list(adapter.store.items())
    highlighted_ids: set[int] = set()

    if result:
        col_context, col_stats = st.columns([3, 1])
        with col_context:
            st.subheader(f"Mode: {result['mode']}")
            st.write(result.get("context") or "No context produced.")
            sources = result.get("source_docs", []) or []
            _render_sources(sources)
        with col_stats:
            st.metric("Tokens", int(result.get("tokens", 0)))
            st.metric("Latency (ms)", int(result.get("latency_ms", 0)))

        _render_cost_savings(int(result.get("tokens", 0)), key_prefix="advanced")

        retrieved_texts = [
            segment for segment in (result.get("context") or "").split("\n") if segment
        ]
        source_docs = {doc for doc in result.get("source_docs", []) if doc}
        for item in items:
            snippet = item.text
            meta = getattr(item, "meta", {}) or {}
            doc_path = str(meta.get("doc_path") or "")
            if doc_path and doc_path in source_docs:
                highlighted_ids.add(item.id)
                continue
            if any(fragment in snippet for fragment in retrieved_texts):
                highlighted_ids.add(item.id)

    st.markdown("### Token budget")
    _render_budget_controls(adapter)

    fig = _build_lattice_plot(items, highlighted_ids)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Memory catalogue"):
        if items:
            table_rows = []
            for item in items:
                meta = getattr(item, "meta", {}) or {}
                doc_path = meta.get("doc_path") or "—"
                table_rows.append(
                    {
                        "ID": item.id,
                        "Level": item.level,
                        "Salience": f"{item.salience:.2f}",
                        "Fidelity": f"{item.fidelity:.2f}",
                        "Updated": datetime.fromtimestamp(item.timestamp).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "Source": doc_path,
                        "Preview": item.text[:100].replace("\n", " "),
                    }
                )
            st.dataframe(table_rows, hide_index=True, use_container_width=True)
        else:
            st.info("No memories ingested yet.")

    with st.expander("Adapter stats"):
        st.json(adapter.stats())

    st.caption(f"Storage directory: {storage_dir.resolve()}")


def _render_interactive_benchmark(adapter: DMLAdapter, storage_dir: Path) -> None:
    st.caption(
        "Benchmark mode — compare RAG vs DML pipelines with real LLM calls and token usage."
    )

    scenario_choices, scenario_mapping = _scenario_choices()
    scenario_name = st.selectbox(
        "Agentic scenario",
        options=scenario_choices,
        key="agentic_scenario_selection",
        help="Select a scripted workflow to ingest and compare automatically.",
    )
    scenario_id = scenario_mapping.get(scenario_name)
    scenario_meta = AGENTIC_SCENARIOS.get(scenario_id) if scenario_id else None
    if scenario_meta:
        st.caption(scenario_meta.get("description") or "")

    run_scenario = st.button(
        "Load scenario into memory and run comparison",
        use_container_width=True,
        key="benchmark_run_scenario",
    )

    uploaded = st.file_uploader(
        "Upload data for benchmarking (multi-file, PDFs supported)",
        type=["txt", "md", "log", "json", "pdf"],
        accept_multiple_files=True,
        help="Files are ingested into both the RAG index and the DML lattice.",
        key="benchmark_upload",
    )
    if uploaded:
        _ingest_uploads(adapter, uploaded, label_prefix="benchmark-upload")

    with st.form("benchmark_prompt_form"):
        prompt = st.text_area(
            "Prompt for benchmarking",
            key="benchmark_prompt",
            placeholder="Ask a question to benchmark DML vs RAG",
        )
        max_tokens = st.slider(
            "Max new tokens for the LLM",
            min_value=64,
            max_value=1024,
            step=32,
            value=256,
            key="benchmark_max_tokens",
        )
        run_benchmark = st.form_submit_button("Submit for Benchmark", type="primary")

    if run_benchmark:
        if not prompt.strip():
            st.warning("Enter a prompt before benchmarking.")
        else:
            with st.spinner("Running benchmark across RAG and DML pipelines..."):
                result = adapter.compare_responses(prompt, max_new_tokens=max_tokens)
            st.session_state["benchmark_result"] = result
            st.session_state["last_agentic_scenario"] = None
            st.session_state.pop("agentic_trace", None)

    if run_scenario:
        if not scenario_meta:
            st.warning("Select a scenario to run.")
        else:
            with st.spinner("Ingesting scenario steps and running comparisons..."):
                scenario_result, logs = _run_agentic_scenario(
                    adapter,
                    scenario_meta,
                    max_new_tokens=int(st.session_state.get("benchmark_max_tokens", 256)),
                )
            if scenario_result:
                st.session_state["benchmark_result"] = scenario_result
                st.session_state["agentic_trace"] = logs
                st.session_state["last_agentic_scenario"] = scenario_meta["id"]
                st.success("Scenario completed. Review the comparison below.")
            else:
                st.warning("Scenario did not include a query step to run.")

    result = st.session_state.get("benchmark_result")
    if not result:
        st.info("Upload content and submit a prompt to see benchmark results.")
        return

    active_scenario_id = st.session_state.get("last_agentic_scenario")
    if active_scenario_id and active_scenario_id in AGENTIC_SCENARIOS:
        active_scenario = AGENTIC_SCENARIOS[active_scenario_id]
        st.markdown(
            f"**Scenario:** {active_scenario['name']} — {active_scenario.get('description') or ''}"
        )
        with st.expander("Show scenario steps", expanded=False):
            logs = st.session_state.get("agentic_trace") or []
            if logs:
                st.markdown("\n".join(f"- {entry}" for entry in logs))
            else:
                st.caption("No steps recorded for this scenario run.")

    rag_backends = result.get("rag_backends", []) or []
    rag_choice = next((entry for entry in rag_backends if entry.get("available")), None)
    if rag_choice is None and rag_backends:
        rag_choice = rag_backends[0]

    dml_result = result.get("dml") or {}

    base_result = result.get("base") or {}

    st.markdown("### Baseline model (no retrieval)")
    st.write(base_result.get("response") or "No baseline response generated.")

    rag_col, dml_col = st.columns(2)
    with rag_col:
        st.subheader("RAG pipeline")
        if rag_choice:
            rag_latency = int(rag_choice.get("retrieval_latency_ms", 0)) + int(
                rag_choice.get("generation_latency_ms", 0)
            )
            rag_tokens = int(rag_choice.get("context_tokens", 0))
            st.metric("Pipeline latency (ms)", rag_latency)
            st.metric("Context tokens sent", rag_tokens)
            if rag_choice.get("usage"):
                usage = rag_choice["usage"]
                prompt_tokens = usage.get("prompt_tokens") or usage.get("prompt")
                completion_tokens = usage.get("completion_tokens") or usage.get(
                    "completion"
                )
                st.caption(
                    f"LLM tokens — prompt: {prompt_tokens}, completion: {completion_tokens}"
                )
            with st.expander("View RAG output"):
                st.markdown("**LLM answer**")
                st.write(rag_choice.get("response") or "No response generated.")
                sources = []
                documents = rag_choice.get("documents") or []
                for doc in documents:
                    if isinstance(doc, dict):
                        label = doc.get("source") or doc.get("id") or doc.get("label")
                        if label:
                            sources.append(str(label))
                    elif isinstance(doc, str):
                        sources.append(doc)
                _render_sources(sources)
                st.markdown("**Retrieved context**")
                st.write(rag_choice.get("context") or "No context returned.")
        else:
            st.warning("No RAG backend results available.")

    with dml_col:
        st.subheader("DML pipeline")
        dml_latency = int(dml_result.get("retrieval_latency_ms", 0)) + int(
            dml_result.get("generation_latency_ms", 0)
        )
        dml_tokens = int(dml_result.get("context_tokens", 0))
        st.metric("Pipeline latency (ms)", dml_latency)
        st.metric("Context tokens sent", dml_tokens)
        usage = dml_result.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens") or usage.get("prompt")
        completion_tokens = usage.get("completion_tokens") or usage.get("completion")
        if prompt_tokens is not None or completion_tokens is not None:
            st.caption(
                f"LLM tokens — prompt: {prompt_tokens}, completion: {completion_tokens}"
            )
        with st.expander("View DML output"):
            st.markdown("**LLM answer**")
            st.write(dml_result.get("response") or "No response generated.")
            sources = []
            for entry in dml_result.get("entries", []) or []:
                meta = entry.get("meta") or {}
                doc_path = meta.get("doc_path") or meta.get("source") or meta.get("memory_id")
                if doc_path:
                    sources.append(str(doc_path))
            _render_sources(sources)
            st.markdown("**Retrieved context**")
            st.write(dml_result.get("context") or "No context returned.")

    st.markdown("### Pipeline trace")
    trace_rows = []
    for step in result.get("pipeline_trace", []) or []:
        trace_rows.append(
            {
                "Step": step.get("step"),
                "Stage": step.get("stage"),
                "Label": step.get("label") or step.get("id"),
            }
        )
    if trace_rows:
        st.table(trace_rows)
    else:
        st.caption("No trace data recorded.")

    st.caption(f"Storage directory: {storage_dir.resolve()}")


def _render_benchmark_suite_tab(adapter: DMLAdapter) -> None:
    st.caption("Benchmarks (DML vs RAG) — run offline suites with retrieval and generation metrics.")

    with st.expander("How the agentic tests are scored", expanded=False):
        st.markdown(
            "\n".join(
                [
                    "- **Agentic suites** replay scripted multi-step workflows (decision changes, tool retries, user preference updates) before issuing a query. They check whether the system keeps the freshest facts and discards superseded ones.",
                    "- **Precision@k** measures how many of the top-*k* retrieved memories are actually relevant. Higher precision means less noisy context injected into the LLM.",
                    "- **Recall@k** tracks how many of the ground-truth memories were recovered. Higher recall means the agent is less likely to omit critical facts when answering.",
                    "- **HitRate@k** is a simple yes/no on whether at least one correct memory was returned. It highlights catastrophic misses.",
                    "- **Reference similarity** compares generated answers to the suite's reference response using lexical overlap; better scores mean closer alignment to expected outcomes.",
                    "- **LLM judge score** (optional) lets a model grade answers for faithfulness and completeness when lexical overlap is too strict.",
                    "- **Avg context tokens** shows how much retrieved text was fed into the LLM; excessive tokens can bloat cost and dilute focus.",
                    "- **Needle-in-a-Haystack accuracy** counts how often the system surfaces planted facts hidden in distractor text—an indicator of robustness under clutter.",
                ]
            )
        )

    suite_options = {suite["name"]: suite for suite in BENCHMARK_SUITES}
    selection = st.selectbox("Benchmark suite", options=list(suite_options.keys()), key="benchmark_suite_selection")
    suite = suite_options.get(selection)

    if not suite:
        st.warning("No benchmark suite selected.")
        return

    st.caption(suite.get("description") or "")

    run_retrieval = st.checkbox("Run retrieval evaluation", value=True, key="benchmark_run_retrieval")
    run_generation = st.checkbox("Run generation evaluation", value=True, key="benchmark_run_generation")
    use_judge = st.checkbox("Use LLM-as-judge (optional)", value=False, key="benchmark_use_judge")
    max_tokens = st.slider(
        "Max new tokens per answer",
        min_value=64,
        max_value=1024,
        value=256,
        step=32,
        key="benchmark_suite_max_tokens",
    )

    if st.button("Run benchmark", type="primary", use_container_width=True, key="run_benchmark_suite"):
        with st.spinner("Running benchmark suite across RAG and DML..."):
            st.session_state["offline_benchmark_result"] = _run_benchmark_suite(
                adapter,
                suite,
                run_retrieval=run_retrieval,
                run_generation=run_generation,
                use_llm_judge=use_judge,
                max_new_tokens=max_tokens,
            )

    result = st.session_state.get("offline_benchmark_result")
    if not result:
        st.info("Select a suite and run to view metrics.")
        return

    st.markdown(f"### Results — {suite['name']}")
    if run_retrieval:
        st.markdown("#### Retrieval metrics")
        retrieval_rows = [
            {
                "Metric": "Precision@k",
                "RAG": f"{result['retrieval']['rag'].get('precision', 0.0):.3f}",
                "DML": f"{result['retrieval']['dml'].get('precision', 0.0):.3f}",
            },
            {
                "Metric": "Recall@k",
                "RAG": f"{result['retrieval']['rag'].get('recall', 0.0):.3f}",
                "DML": f"{result['retrieval']['dml'].get('recall', 0.0):.3f}",
            },
            {
                "Metric": "HitRate@k",
                "RAG": f"{result['retrieval']['rag'].get('hit', 0.0):.3f}",
                "DML": f"{result['retrieval']['dml'].get('hit', 0.0):.3f}",
            },
        ]
        st.table(retrieval_rows)

    if run_generation:
        st.markdown("#### Generation metrics")
        judge_label = "LLM judge score" + (" (enabled)" if use_judge else " (N/A)")
        gen_rows = [
            {
                "Metric": "Reference similarity",
                "RAG": f"{result['generation']['rag'].get('ref', 0.0):.3f}",
                "DML": f"{result['generation']['dml'].get('ref', 0.0):.3f}",
            },
            {
                "Metric": judge_label,
                "RAG": "N/A" if not use_judge else f"{result['generation']['rag'].get('judge', 0.0):.3f}",
                "DML": "N/A" if not use_judge else f"{result['generation']['dml'].get('judge', 0.0):.3f}",
            },
            {
                "Metric": "Avg context tokens",
                "RAG": f"{result['generation']['rag'].get('context_tokens', 0.0):.1f}",
                "DML": f"{result['generation']['dml'].get('context_tokens', 0.0):.1f}",
            },
        ]
        st.table(gen_rows)

    if result.get("needle"):
        st.markdown("#### Needle-in-a-Haystack Accuracy")
        accuracy_cols = st.columns(2)
        accuracy_cols[0].metric("RAG", f"{result['needle'].get('rag', 0.0)*100:.1f}%")
        accuracy_cols[1].metric("DML", f"{result['needle'].get('dml', 0.0)*100:.1f}%")

    st.markdown("#### Per-query breakdown")
    table_rows = []
    for entry in result.get("queries", []):
        table_rows.append(
            {
                "ID": entry.get("id"),
                "Question": entry.get("question"),
                "RAG IDs": ", ".join(entry.get("rag_retrieval_ids", [])),
                "DML IDs": ", ".join(entry.get("dml_retrieval_ids", [])),
                "RAG ref": f"{entry.get('rag_reference_score', 0.0):.3f}" if entry.get("rag_reference_score") is not None else "-",
                "DML ref": f"{entry.get('dml_reference_score', 0.0):.3f}" if entry.get("dml_reference_score") is not None else "-",
                "RAG ctx tokens": entry.get("rag_context_tokens") or "-",
                "DML ctx tokens": entry.get("dml_context_tokens") or "-",
                "RAG hit": "✅" if entry.get("rag_retrieval_hit") else "",
                "DML hit": "✅" if entry.get("dml_retrieval_hit") else "",
            }
        )
    if table_rows:
        st.dataframe(table_rows, hide_index=True, use_container_width=True)
    else:
        st.info("No per-query data to display yet.")


def _render_benchmark_mode(adapter: DMLAdapter, storage_dir: Path) -> None:
    tabs = st.tabs(["Interactive comparison", "Benchmarks (DML vs RAG)"])
    with tabs[0]:
        _render_interactive_benchmark(adapter, storage_dir)
    with tabs[1]:
        _render_benchmark_suite_tab(adapter)


def _render_real_world_mode(adapter: DMLAdapter, storage_dir: Path) -> None:
    st.caption(
        "Real World Test — run the full DML retrieval + LLM generation pipeline with context injection."
    )

    uploaded = st.file_uploader(
        "Upload supporting files",
        type=["txt", "md", "log", "json", "pdf"],
        accept_multiple_files=True,
        help="Files are ingested and made available to the DML retriever.",
        key="real_world_upload",
    )
    if uploaded:
        _ingest_uploads(adapter, uploaded, label_prefix="real-world-upload")

    prompt = st.text_area(
        "Ask the LLM",
        key="real_world_prompt",
        placeholder="Ask a grounded question that should use retrieved memories.",
    )
    max_tokens = st.slider(
        "Max new tokens",
        min_value=64,
        max_value=1024,
        value=256,
        step=32,
        key="real_world_max_tokens",
    )
    run_test = st.button("Run real-world pipeline", type="primary")

    if run_test:
        if not prompt.strip():
            st.warning("Enter a prompt before running the real-world test.")
        else:
            with st.spinner("Retrieving context and generating with the live LLM..."):
                report = adapter.retrieval_report(prompt)
                context = adapter._format_dml_context(report.get("entries", []))
                augmented_prompt = adapter._compose_prompt(prompt, context)
                response, usage, generation_latency = adapter._generate_with_metrics(
                    augmented_prompt, max_new_tokens=max_tokens
                )
            adapter.reinforce(prompt, response)
            st.session_state["real_world_result"] = {
                "prompt": prompt,
                "context": context,
                "response": response,
                "retrieval_latency_ms": report.get("latency_ms", 0),
                "context_tokens": report.get("tokens", 0),
                "usage": usage,
                "generation_latency_ms": generation_latency,
                "entries": report.get("entries", []),
            }

    result = st.session_state.get("real_world_result")
    if not result:
        st.info("Upload files and run the pipeline to see grounded answers.")
        return

    total_latency = int(result.get("retrieval_latency_ms", 0)) + int(
        result.get("generation_latency_ms", 0)
    )
    st.subheader("Live LLM answer")
    st.write(result.get("response") or "No response generated.")
    metrics = st.columns(3)
    metrics[0].metric("Pipeline latency (ms)", total_latency)
    metrics[1].metric("Context tokens", int(result.get("context_tokens", 0)))
    usage = result.get("usage") or {}
    completion_tokens = usage.get("completion_tokens") or usage.get("completion")
    metrics[2].metric("Completion tokens", completion_tokens or 0)

    sources = []
    for entry in result.get("entries", []) or []:
        meta = entry.get("meta") or {}
        doc_path = meta.get("doc_path") or meta.get("source") or meta.get("memory_id")
        if doc_path:
            sources.append(str(doc_path))
    _render_sources(sources)

    with st.expander("Show retrieved context (DML)"):
        st.write(result.get("context") or "No context returned.")

    st.caption(f"Storage directory: {storage_dir.resolve()}")


if hasattr(st, "on_session_end"):
    st.on_session_end(_close_adapter)

current_storage = _resolve_storage_dir()
if "storage_dir_input" not in st.session_state:
    st.session_state["storage_dir_input"] = str(current_storage)

adapter: DMLAdapter | None = st.session_state.get("adapter")
if adapter is None:
    try:
        adapter = _replace_adapter(current_storage)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

assert adapter is not None

interface_options = ["Simple", "Advanced", "Benchmark", "Real World Test"]
if "ui_mode" in st.session_state:
    try:
        default_index = interface_options.index(st.session_state["ui_mode"])
    except ValueError:
        default_index = 0
else:
    default_index = 0
ui_mode = st.sidebar.radio(
    "Interface mode",
    options=interface_options,
    index=default_index,
    key="ui_mode",
)

if ui_mode in {"Advanced", "Benchmark", "Real World Test"}:
    st.sidebar.header("Storage")
    storage_input = st.sidebar.text_input(
        "Storage directory",
        value=st.session_state.get("storage_dir_input", str(current_storage)),
        key="storage_dir_input",
    )
    if st.sidebar.button("Use storage directory", use_container_width=True):
        requested_raw = (storage_input or "").strip() or str(DEFAULT_STORAGE_DIR)
        requested_dir = _normalise_storage_dir(Path(requested_raw))
        if requested_dir != current_storage:
            try:
                adapter = _replace_adapter(requested_dir)
            except RuntimeError as exc:
                st.sidebar.error(str(exc))
                st.stop()
            current_storage = requested_dir
            st.sidebar.success(f"Switched storage to {requested_dir}")
        else:
            st.sidebar.info("Storage directory unchanged.")

    if st.sidebar.button("Reset lattice", use_container_width=True):
        _close_adapter()
        try:
            _clear_storage(current_storage)
        except RuntimeError as exc:
            st.sidebar.error(str(exc))
            st.stop()
        try:
            adapter = _replace_adapter(current_storage)
        except RuntimeError as exc:
            st.sidebar.error(str(exc))
            st.stop()
        st.sidebar.success("Cleared stored memories.")

    st.sidebar.divider()
    st.sidebar.subheader("Models")

    def _label_for_value(value: str, options: list[dict[str, str]]) -> str:
        return next((entry["label"] for entry in options if entry["value"] == value), value)

    llm_values = [entry["value"] for entry in LLM_MODEL_OPTIONS]
    llm_default_value = st.session_state.get("selected_llm_model") or llm_values[0]
    llm_default_index = llm_values.index(llm_default_value) if llm_default_value in llm_values else 0
    llm_selection = st.sidebar.selectbox(
        "LLM presets (fills the field below)",
        options=llm_values,
        index=llm_default_index,
        format_func=lambda value: _label_for_value(value, LLM_MODEL_OPTIONS),
        key="llm_preset_selection",
    )
    if st.session_state.get("last_llm_preset") != llm_selection:
        st.session_state["llm_model_input"] = llm_selection
        st.session_state["last_llm_preset"] = llm_selection
    if "llm_model_input" not in st.session_state:
        st.session_state["llm_model_input"] = llm_default_value
    llm_text = st.sidebar.text_input(
        "LLM model name",
        value=st.session_state.get("llm_model_input", llm_selection),
        key="llm_model_input",
        help="Used for summarisation, generation, and RAG comparisons.",
    )

    embedding_values = [entry["value"] for entry in EMBEDDING_MODEL_OPTIONS]
    embedding_default_value = (
        st.session_state.get("selected_embedding_model") or embedding_values[0]
    )
    embedding_default_index = (
        embedding_values.index(embedding_default_value)
        if embedding_default_value in embedding_values
        else 0
    )
    embedding_selection = st.sidebar.selectbox(
        "Embedding presets",
        options=embedding_values,
        index=embedding_default_index,
        format_func=lambda value: _label_for_value(value, EMBEDDING_MODEL_OPTIONS),
        key="embedding_preset_selection",
    )
    if st.session_state.get("last_embedding_preset") != embedding_selection:
        st.session_state["embedding_model_input"] = embedding_selection
        st.session_state["last_embedding_preset"] = embedding_selection
    if "embedding_model_input" not in st.session_state:
        st.session_state["embedding_model_input"] = embedding_default_value
    embedding_text = st.sidebar.text_input(
        "Embedding model name",
        value=st.session_state.get("embedding_model_input", embedding_selection),
        key="embedding_model_input",
        help="Vectoriser used across the lattice and RAG store.",
    )

    if st.sidebar.button(
        "Apply model choices", use_container_width=True, key="apply_model_choices"
    ):
        st.session_state["selected_llm_model"] = llm_text or llm_selection
        st.session_state["selected_embedding_model"] = embedding_text or embedding_selection
        requested_embedding = st.session_state["selected_embedding_model"]
        previous_embedding = adapter.config.get("embedding_model")
        embedding_changed = requested_embedding != previous_embedding
        try:
            if embedding_changed:
                _clear_storage(current_storage)
            adapter = _replace_adapter(current_storage)
        except RuntimeError as exc:
            st.sidebar.error(str(exc))
            st.stop()
        st.sidebar.success("Updated models and restarted the adapter.")
        if embedding_changed:
            st.sidebar.info("Cleared stored memories due to embedding model change.")

    st.sidebar.divider()
    st.sidebar.subheader("Ingestion")

    if st.session_state.pop("advanced_manual_reset", False):
        st.session_state["advanced_manual_text"] = ""

    uploaded = st.sidebar.file_uploader(
        "Upload text, markdown, JSON, logs, or PDFs",
        type=["txt", "md", "log", "json", "pdf"],
        accept_multiple_files=True,
        help="Each file is chunked and stored as individual memories.",
        key="advanced_upload",
    )
    if uploaded:
        _ingest_uploads(adapter, uploaded, label_prefix="advanced-upload")

    manual_text = st.sidebar.text_area(
        "Paste text to ingest",
        key="advanced_manual_text",
        placeholder="Drop snippets here to create memories without uploading a file.",
        height=160,
    )
    if st.sidebar.button("Ingest text snippet", use_container_width=True):
        status, message = _try_ingest_text(adapter, manual_text, "manual-entry")
        if status == "success":
            st.sidebar.success("Manual snippet ingested.")
            st.session_state["advanced_manual_reset"] = True
            st.rerun()
        elif status == "warning":
            st.sidebar.warning(message)
        else:
            st.sidebar.error(message)
else:
    st.sidebar.info(
        "Advanced storage and ingestion controls are available when Advanced mode is selected."
    )

current_storage = _resolve_storage_dir()
st.sidebar.caption(f"Storage directory: {current_storage.resolve()}")

st.title("Daystrom Memory Lattice Playground")

if ui_mode == "Simple":
    _render_simple_mode(adapter, current_storage)
elif ui_mode == "Advanced":
    _render_advanced_mode(adapter, current_storage)
elif ui_mode == "Benchmark":
    _render_benchmark_mode(adapter, current_storage)
else:
    _render_real_world_mode(adapter, current_storage)

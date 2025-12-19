# 🧠 Daystrom Memory Lattice (DML)
*A hierarchical, self-compressing memory substrate for intelligent retrieval and generation pipelines.*

---

## Table of contents
- [Overview](#overview)
- [Core concepts](#core-concepts)
- [System architecture](#system-architecture)
- [Installation and setup](#installation-and-setup)
- [Running the stack](#running-the-stack)
- [Feature reference](#feature-reference)
- [Configuration guide](#configuration-guide)
- [Integration cookbook](#integration-cookbook)
- [Benchmarks and load testing](#benchmarks-and-load-testing)
- [DML vs traditional RAG](#dml-vs-traditional-rag)
- [Summary](#summary)

---

## Overview
The **Daystrom Memory Lattice (DML)** compresses, abstracts, and retrieves large knowledge bases on GPU hardware. It is purpose-built for long-horizon assistants that must *remember*, *reason*, and *explain* instead of simply vector-searching.

Key ideas:
- **Hierarchical memory** – lattice levels L0–Lk range from verbatim fragments to progressively distilled abstractions.
- **Adaptive routing** – the router can choose semantic, literal, or hybrid retrieval based on the prompt.
- **Self-maintenance** – salience decay, reinforcement, and summarisation continuously rebalance the store.
- **OpenAI-compatible generation** – the lattice can drive NVIDIA NIMs or any OpenAI-compatible endpoint.
- **Multi-RAG fanout** – a single ingest feeds FAISS, Chroma, and the persistent lattice simultaneously.

---

## Core concepts
### Memory node
```
M_i = (e_i, s_i, f_i, t_i)
```
- **eᵢ** – embedding vector
- **sᵢ** – salience score
- **fᵢ** – fidelity (quality/confidence)
- **tᵢ** – timestamp

### Retrieval scoring
```
score_i = cos(e_i, q) + η * r_i + γ * s_i + κ * f_i
```
- **rᵢ = 1 / (1 + ageᵢ)** captures recency
- **η, γ, κ** control recency, salience, and fidelity weighting

### Decay and abstraction
```
λ* = σ(β_r * r_i − β_a * age_i)
```
Older memories gradually lose fidelity and merge into higher-level summaries.

### Token budgeting
```
while Σ(tokens(S_i)) < B,  S_i ∈ top_k
```
A greedy knapsack packs the highest information-density memories within budget **B**.

---

## System architecture
```
        ┌──────────────┐
        │ Data Sources │ ← PDFs, code, logs, archives
        └──────┬───────┘
               │
      ┌────────▼────────┐
      │ Embedding Model │ → GPU accelerated
      └────────┬────────┘
               │
     ┌─────────▼──────────┐
     │ Memory Lattice     │
     │  • Decay / Merge   │
     │  • Summarisation   │
     │  • Persistence     │
     └────────┬───────────┘
               │
      ┌────────▼──────────┐
      │ Retrieval Router  │ → literal / semantic / hybrid
      └────────┬──────────┘
               │
       ┌───────▼───────┐
       │  Query Engine │ → OpenAI-compatible LLM / MCP / custom
       └───────────────┘
```

---

## Installation and setup
### Requirements
- Python 3.10+
- NVIDIA GPU (optional but recommended for embeddings and summarisation)
- CUDA-compatible drivers if running GPU workloads

### Install from source
```bash
pip install .[server]
```
Optional extras:
- `pip install .[embeddings]` – GPU/CPU embedding backends
- `pip install .[faiss]` – FAISS vector index acceleration
- `pip install .[multiplex_rag]` – combined FAISS + Chroma fanout
- `pip install .[playground]` – 3D Streamlit visualiser
- `pip install .[mcp]` – MCP server adapter

### Repository layout
- `daystrom_dml/` – core lattice, APIs, adapters, and web assets
- `app/` – Streamlit visualiser
- `bench/` – synthetic benchmarking utilities
- `scripts/` – helper automation

---

## Running the stack
### Local execution (uvicorn)
```bash
pip install .[server]
dml-server --host 0.0.0.0 --port 8000
```
Use `--reload` during development for hot reloading. The server honours `DML_HOST` and `DML_PORT` when set.

### Docker
```bash
docker build -t daystrom-dml .
docker run --gpus all \
  -p 8000:8000 \
  -e DML_PORT=8000 \
  -v "$(pwd)/data:/opt/dml/data" \
  daystrom-dml
```
Mounting `./data` preserves the lattice and vector indexes across restarts. Provide a custom configuration via `-e DML_CONFIG_PATH=/opt/dml/config.yaml`.

### Docker Compose
```bash
docker compose up -d
```
The compose stack builds the CUDA image, exposes `8000:8000`, and mounts `./data` into `/opt/dml/data`. Tear down with `docker compose down`.

### GPU and NIM environment hints
- `NIM_KVCACHE_PERCENT`, `NIM_ENABLE_KV_CACHE_REUSE`, `NIM_ENABLE_KV_CACHE_HOST_OFFLOAD`, and `NIM_KV_CACHE_HOST_MEM_FRACTION` tune NVIDIA NIM memory behaviour.
- `DML_GPU_ACCELERATION=1` ensures GPU-optimised paths are enabled when available.
- `DML_EMBEDDING_DEVICE=cuda` (or `cuda:1`, `mps`, etc.) pins the SentenceTransformer embedder to a specific accelerator and skips CPU fallback.

### Streamlit playground
**Simple mode (zero-config)**

1. Install the playground extra:
   ```bash
   pip install .[playground]
   ```
2. Launch Streamlit:
   ```bash
   streamlit run app/playground.py
   ```

The UI boots into **Simple** mode with a CPU-friendly embedder and stores data in `~/.dml/playground` (override via `DML_PLAYGROUND_STORAGE` or `DML_STORAGE_DIR`). Upload snippets, ask a question, and you’re done.

**Advanced mode (GPU + enterprise controls)**

1. Install the GPU-capable extras:
   ```bash
   pip install .[playground,embeddings]
   ```
2. Pin the embedder to your accelerator before launching Streamlit:
   ```bash
   export DML_EMBEDDING_DEVICE=cuda  # or cuda:0 / mps
   ```
3. (Optional) Point the lattice at a dedicated storage root:
   ```bash
   export DML_STORAGE_DIR=./data/playground
   ```
4. Launch the playground:
   ```bash
   streamlit run app/playground.py
   ```

Switch the in-app mode selector to **Advanced** for storage management, manual ingestion, token budgets, and the 3D lattice visualiser. The adapter initialises once, reports the chosen device, and subsequent ingestion/retrieval runs remain on GPU without the tqdm “Batches” spam.

---

## Feature reference
### 1. Memory ingestion
**CLI:**
Run the Daystrom CLI as a module (no standalone console script is published yet, e.g. `python -m daystrom_dml.cli --help`).
```bash
python -m daystrom_dml.cli ingest "Investigate warp-drive telemetry anomalies."
```
**HTTP API:** `POST /ingest` with JSON `{ "text": "...", "meta": {...} }`.

**Bulk uploads:** `POST /upload` accepts multiple files or zipped archives, extracts supported text (PDF, `.txt`, `.md`, `.py`, etc.), chunks them, and streams each chunk into the lattice while preserving `doc_path` metadata. Unsupported or binary files are skipped gracefully.

### 2. Querying & generation
- `python -m daystrom_dml.cli query "Why did the telemetry fail?"` returns the DML preamble for inspection.
- `python -m daystrom_dml.cli run "Summarise the latest warp-drive postmortem."` performs retrieval + generation and reinforces the answer.
- `POST /query` triggers adaptive retrieval, appends the resulting context to the prompt, sends it to the configured LLM, and emits usage metrics.

Literal versus semantic routing is automatically selected, but can be forced via `mode` on advanced APIs such as `DMLAdapter.query_database()`.

### 3. Reinforcement learning loop
- `python -m daystrom_dml.cli reinforce "Drive realignment succeeded after recalibration."`
- `POST /reinforce` stores summarised outcomes (prompt + answer digest) with slightly higher salience to bias future retrievals.
- Automatic reinforcement happens after every `/query` or `python -m daystrom_dml.cli run` round-trip.

### 4. Retrieval analytics & knowledge surfaces
- `POST /rag/retrieve` compares the lattice with each RAG backend, returning context, latency, and token usage per backend.
- `GET /stats` summarises lattice size, fidelity averages, and distribution across hierarchy levels.
- `GET /knowledge` produces a combined catalogue (capped to 200 entries) containing lattice summaries and multi-RAG inventory counts.

### 5. Multi-RAG fanout & comparisons
- Every ingest fans out to FAISS, Chroma, and the disk-backed persistent index (when enabled).
- `POST /rag/compare` runs: baseline model → DML-augmented model → each RAG backend, then grades their outputs, traces pipeline order, and records token budgets.

### 6. Persistence & checkpoints
- Background persistence writes JSONL snapshots or full-state dumps (including RAG) on the configured interval.
- `python -m daystrom_dml.cli checkpoint` forces an immediate checkpoint with retention controls.
- Storage defaults to `./data` but can be redirected via `storage_dir` or `DML_STORAGE_DIR`.

### 7. Metrics & observability
- `GET /metrics` exposes Prometheus metrics (ingest counts, retrieval latency histograms, token savings).
- Token consumption/savings per query are recorded when metrics are enabled.
- Structured logs ship with request IDs and JSON formatting for easy ingestion.

### 8. Streamlit visualiser
- `POST /visualizer/launch` launches or connects to the 3D lattice explorer.
- `/visualizer/state` mirrors the latest prompt for synchronising dashboards.
- `/visualizer/embed/...` proxies the Streamlit UI through the FastAPI origin for iframe embedding.

### 9. CLI quick reference
| Command | Description |
|---------|-------------|
| `python -m daystrom_dml.cli ingest <text>` | Store a new memory fragment |
| `python -m daystrom_dml.cli query <prompt>` | Print retrieval preamble |
| `python -m daystrom_dml.cli run <prompt>` | Retrieve + generate + reinforce |
| `python -m daystrom_dml.cli reinforce <text>` | Inject outcome summaries |
| `python -m daystrom_dml.cli stats` | Print lattice statistics |
| `python -m daystrom_dml.cli checkpoint` | Persist a snapshot immediately |

---

## Configuration guide
The canonical configuration lives at `daystrom_dml/config.yaml`. Key sections:

| Setting | Description |
|---------|-------------|
| `model_name` | Default LLM (used locally or for remote OpenAI-compatible calls) |
| `embedding_model` | Embedding backend identifier |
| `token_budget` | Maximum tokens reserved for DML context |
| `similarity_threshold` | Minimum cosine similarity required for a memory to be eligible for retrieval |
| `persistence.enable` + `interval_sec` | Enable JSONL checkpoints and set cadence |
| `rag_store.enable`/`backend` | Persist FAISS index to disk |
| `literal.max_snippet_tokens` & `max_snippets` | Literal retriever window sizes |
| `budgets.semantic_pct/literal_pct/free_pct` | Token allocation ratios |

### Environment overrides
- Any environment variable prefixed with `DML_` overrides configuration keys (`DML_MODEL_NAME`, `DML_STORAGE_DIR`, `DML_BUDGETS_SEMANTIC_PCT`, etc.).
- Nested keys use underscores: `DML_PERSISTENCE_ENABLE=1`, `DML_LITERAL_MAX_SNIPPET_TOKENS=256`.
- `.env` and `.env.local` files (current working directory and configuration directory) are loaded automatically.

---

## Integration cookbook
### Python client (requests-based)
```python
from daystrom_dml import DMLClient

with DMLClient("http://localhost:8000") as client:
    client.ingest("Warp-drive postmortem: capacitor failure at T+42s", meta={"source": "logs/warp.txt"})
    answer = client.query("What triggered the capacitor failure?")
    print(answer["response"])
```
Use `client.stats()` and `client.knowledge()` for observability dashboards.

### Embedding the adapter in custom agents
```python
from daystrom_dml.dml_adapter import DMLAdapter

adapter = DMLAdapter()
context = adapter.build_preamble("Summarise warp-drive failure mitigations")
print(context)
response = adapter.run_generation("Draft a remediation plan for the next launch window.")
```
`run_generation` executes retrieval → LLM call → reinforcement in one step. Use `adapter.query_database(..., mode="literal")` to force literal snippets for structured lookups.

### NVIDIA NIM control plane
1. Call `POST /nim/options` to discover curated container images and defaults.
2. `POST /nim/configure` with `{"nim_id": "llama3-8b", "api_key": "<NGC_TOKEN>"}` to pull the image, update the adapter model, and seed environment variables.
3. `POST /nim/start` to launch the container (honours `NIM_PORT`, optional cache mounts, and waits for health checks).
4. Point the UI or your agents at the running DML server—its GPTRunner automatically uses the NIM endpoint via the exported OpenAI-compatible API base.
5. `POST /nim/stop` gracefully shuts down the managed container.

### OpenAI-compatible endpoints (Ollama, vLLM, LM Studio, Azure, OpenAI, etc.)
Set the following environment variables before starting `dml-server` or invoking the CLI:
```bash
export DML_API_BASE=http://localhost:11434      # Ollama / vLLM / LM Studio
export DML_API_KEY=your-token-if-required
export DML_MODEL_NAME=meta/llama3-8b-instruct   # Model identifier understood by the endpoint
```
`GPTRunner` detects `DML_API_BASE`, `OPENAI_API_BASE`, or `NIM_API_BASE` automatically and routes completions through the provided endpoint. Token usage metadata is captured when the remote server returns OpenAI-compatible usage objects.

### Custom orchestration
- Wrap `/rag/compare` in automated evaluations to benchmark retrieval strategies as you iterate on prompt templates.
- Combine `/upload` with CI artefacts (docs, release notes, logs) to pre-warm the lattice before deployments.
- Consume `/metrics` from Prometheus/Grafana and `/visualizer/state` from custom dashboards to correlate live prompts with retrieval topology.

---

## Benchmarks and load testing
Run synthetic comparisons against baseline RAG pipelines:
```bash
python bench/bench_dml_vs_rag.py --corpus-size 120 --queries 12
```
Make targets are provided for convenience:
- `make bench-small`
- `make bench-large`

Each run emits CSV reports (latency, token usage, cost projections) under `bench/` for analysis.

---

## DML vs traditional RAG
| Capability | Traditional RAG | Daystrom Memory Lattice |
|------------|-----------------|-------------------------|
| Retrieval granularity | Flat top-K chunks | Hierarchical (verbatim → summary → abstraction) |
| Context optimisation | Fixed, redundant | Dynamic token budgeting |
| Compression | Minimal | Continuous semantic + vector compression |
| Decay / reinforcement | Usually absent | Mathematical fidelity decay + reinforcement |
| Exact lookups | Difficult | Dedicated literal retriever |
| Compute profile | Linear with corpus size | GPU-accelerated lattice with bounded merges |
| Output quality | Redundant snippets | Dense, contextual, citation-ready |

> **In short:** RAG *searches*. DML *remembers*.

---

## Summary
**DML = Hierarchical Memory + Semantic Compression + GPU Efficiency.**

Deploy it as a persistent memory layer between your databases and LLMs, orchestrate NVIDIA NIMs or any OpenAI-compatible endpoint, and gain precise observability into what your assistant recalls, summarises, and reinforces over time.

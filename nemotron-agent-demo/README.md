# Station Autonomous Agent Testing

Agentic demo that drives a single LLM through planner/coder/reviewer/ops/aggregator roles and visualizes progress live.

## Features
- vLLM OpenAI-compatible endpoint at `http://host.docker.internal:8000/v1` (configurable via `VLLM_BASE_URL`).
- Gradio UI with animated status badges, live metrics (approx TTFT, tokens/sec), and progressive timeline updates.
- CLI demo that streams stage states in the terminal.
- Optional Daystrom Memory Lattice (DML) layer for persistent memory + transparent retrievals.
- Offline-friendly after first model download.

## Containerized quickstart (recommended)

### Prerequisites
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- An NVIDIA GPU

### Run everything in one command
```bash
./run_all.sh
```
Or directly:
```bash
docker compose up --build
```
This builds the Gradio UI image (`Dockerfile.ui`) and launches the UI + DML services via `docker compose`:
- **Gradio UI** (`nemotron-ui`): served at `http://localhost:7860`
  - Note: the UI expects a vLLM server running on the host at `http://host.docker.internal:8000/v1` (override with `VLLM_BASE_URL`).
  - Linux: ensure `host.docker.internal` is mapped via `extra_hosts: ["host.docker.internal:host-gateway"]` (already included in the compose files).

### vLLM health check
```bash
curl http://localhost:8000/v1/models
```
UI: `http://localhost:7860`

### Rebuild the server image
If the UI image changes, rebuild with no cache:
```bash
docker compose build --no-cache nemotron-ui
docker compose up -d --force-recreate nemotron-ui
docker exec nemotron-ui docker ps
```

### vLLM via host (recommended for GB300 / Grace)
Terminal 1:
```bash
./run_kimi_vllm_host.sh
```
Terminal 2:
```bash
docker compose -f docker-compose.kimik2-nvfp4.yml up --build
```
Verify services:
```bash
curl http://localhost:8000/v1/models
curl http://localhost:9001/health
```
Note: the model must already exist at `/mnt/raid/kimik2/hf/hub/models--nvidia--Kimi-K2-Thinking-NVFP4`.

### Build the playground image
The playground container image is a local-only dev image used by the UI when running command tools. Build it once:
```bash
docker compose --profile playground build nemotron-playground-image
```

### Useful endpoints and volumes
- Check vLLM: `curl http://localhost:8000/v1/models`
- Open the UI: `http://localhost:7860`
- Prompt edits persist in `./prompt_library` (mounted to `/app/prompt_library`)
- Default prompts and context are bind-mounted read-only from `./prompts` and `./context`

### Environment overrides
- `VLLM_BASE_URL` (default: `http://host.docker.internal:8000/v1`)
- `VLLM_MODEL_ID` (default: `mistral/Mistral-Large-3-675B-Instruct-2512-NVFP4/`)
- `VLLM_TIMEOUT_S` (default: `120`)

### Model requirements
- vLLM must expose the OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints.
- Toggle reasoning via `chat_template_kwargs.enable_thinking` (or `extra_body.chat_template_kwargs.enable_thinking`) in `/v1/chat/completions` requests.

### Troubleshooting
- **GPU access**: verify `nvidia-smi` works on the host and Docker can see GPUs.
- **Model download/auth**: set `HF_TOKEN` and ensure `./_hf_cache` has space.
- **OOM or slow load**: lower `--max-model-len` or `--gpu-memory-utilization` in `docker-compose.yml`.
- **Port conflicts**: adjust port mappings in `docker-compose.yml` (e.g., `8001:8000`, `7861:7860`).

## Running locally without containers (legacy)
```bash
cd nemotron-agent-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
VLLM_BASE_URL=http://localhost:8000/v1 ./run_ui.sh
```
Container-first is preferred; local runs expect a vLLM server running on the host.

## Managing prompts from the UI
- Open the **Prompts** tab to manage both goal presets and agent prompts without touching the CLI.
- **Goal Presets** sub-tab:
  - Pick any preset, edit the text, and click **Push to Goal Input** to move it into the main Goal box. The "Prompt source" badge switches to `preset:<name>` so you can see the origin.
  - Use **Save Preset** to update the selected entry or **Save as New** to create another preset (both persist to `prompt_library/goal_presets.json`).
  - **Delete Preset** requires the confirmation checkbox to avoid accidental removal.
  - An "Unsaved changes" indicator appears if the editor differs from the stored version.
- **Agent Prompts** sub-tab:
  - Choose an agent (system/planner/coder/reviewer/ops/aggregator) to view the default prompt alongside the active prompt.
  - Click **Save Override** to persist an edited prompt to `prompt_library/agent_overrides/<agent>.txt`, or **Push Override Live** for a session-only override.
  - **Use Default** clears the override file and flips the badge back to DEFAULT. A diff summary shows when an override diverges from the default template.

## Daystrom Memory Lattice (DML)
The DML layer adds **persistent memory** across runs and a **transparent retrieval report** per stage so you can see exactly which memories were used. It runs as a separate `dml-service` container with FAISS-backed storage and real embeddings.

### Enable/disable
- In the UI, toggle **DML Memory ON / OFF**.
- When enabled, adjust **DML top_k** to control how many memories are retrieved per stage.

### Storage location
- DML data is stored in `./data/dml` (persisted across runs).
- Start services with `docker compose up --build` to launch `dml-service` alongside the UI.

### Reset memory
- Stop the UI, then delete the directory: `rm -rf ./data/dml`

### Verify the service
- Inside the Compose network: `curl http://dml-service:9001/health`

## Run the CLI demo
```bash
./run_demo_cli.sh "Build a resilient offline LLM demo" --scenario "Ship a resilient offline demo"
```
Streams the same stages as the UI using the orchestrator.

## Preset scenarios
- Optimize inference for latency
- Ship a resilient offline demo
- Benchmark throughput on GB300

## Troubleshooting
- **Server not ready**: `python -m src.server.healthcheck --host localhost --port 8000`
- **OOM or slow load**: lower `--max-model-len` (e.g., 12288) or reduce `--gpu-memory-utilization` in `run_server.sh`.
- **HF cache issues**: set `HF_HOME` to a fast disk and ensure the model is fully downloaded before going offline.
- **Port conflicts**: adjust `PORT` env var for server/UI scripts.

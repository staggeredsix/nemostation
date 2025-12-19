# Nemotron-3 Nano Agentic Demo (Transformers server)

Brain-melty local demo that drives a single Nemotron-3 Nano model through planner/coder/reviewer/ops/aggregator roles and visualizes progress live.

## Features
- Hugging Face weights only (default `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`).
- TensorRT-LLM AutoDeploy server at `http://localhost:8000/v1`.
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
This builds the Gradio UI image (`Dockerfile.ui`) and launches both containers via `docker compose`:
- **TRT-LLM server** (`nemotron-trtllm`): nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc5, served at `http://localhost:8000/v1`
- **Gradio UI** (`nemotron-ui`): served at `http://localhost:7860`
  - Note: the server image must include CUDA + Python 3.11 so `mamba-ssm` installs from wheels.

### TRT-LLM Server (AutoDeploy)
```bash
docker compose up --build
```
```bash
curl http://localhost:8000/v1/models
```
UI: `http://localhost:7860`

On first run, TensorRT-LLM may compile/build artifacts and download weights. Subsequent runs reuse `./_trtllm_cache` and `./_hf_cache` for faster startup.

### Rebuild the server image
If the UI image changes, rebuild with no cache:
```bash
docker compose build --no-cache nemotron-ui
docker compose up -d --force-recreate nemotron-ui
docker exec nemotron-ui docker ps
```

### Build the playground image
The playground container image is a local-only dev image used by the UI when running command tools. Build it once:
```bash
docker compose --profile playground build nemotron-playground-image
```

### Useful endpoints and volumes
- Check the server: `curl http://localhost:8000/v1/models`
- Open the UI: `http://localhost:7860`
- On first run the server container warms the model cache (full weights download) before reporting ready; this can take a while depending on your network and disk.
- Tail the server logs while it downloads: `docker logs -f nemotron-trtllm`
- Verify cached weights: `ls ./_hf_cache/hub`
- HF cache is persisted to `./_hf_cache` (mounted to `/root/.cache/huggingface`)
- Prompt edits persist in `./prompt_library` (mounted to `/app/prompt_library`)
- Default prompts and context are bind-mounted read-only from `./prompts` and `./context`

### Environment overrides
- `MODEL_ID` (default: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`)
- `HF_TOKEN` and `HF_HOME` for Hugging Face access/cache location (set `HF_TOKEN` for gated models)
- The UI calls the server at `OPENAI_BASE_URL` (set in compose to `http://nemotron-trtllm:8000/v1`)

### Model requirements
- Nemotron-3 Nano runs via TensorRT-LLM AutoDeploy with `trust_remote_code=True`.
- Plan for a long first startup: the model weights and tokenizer files are downloaded into `./_hf_cache` and reused on subsequent runs.
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
./run_server.sh &
./run_ui.sh
```
Container-first is preferred; local runs still expect a GPU and will download weights into `~/.cache/huggingface`.

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

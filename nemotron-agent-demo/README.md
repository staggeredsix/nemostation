# Autonomous Agents Stress Testing

An agentic demo that runs planner/coder/reviewer/ops/aggregator roles against an OpenAI-compatible vLLM endpoint and visualizes progress live. Built for fast model swaps and stress testing.

## Features
- vLLM OpenAI-compatible endpoint support (defaults to `http://host.docker.internal:8000/v1`).
- Model selection via env vars, UI dropdown, or CLI flag (auto-detects via `/models`).
- Gradio UI with animated status badges, live metrics (approx TTFT, tokens/sec), and progressive timeline updates.
- CLI demo that streams stage states in the terminal.
- Simple stress test harness for concurrent requests.
- Optional Daystrom Memory Lattice (DML) layer for persistent memory + transparent retrievals.
- Offline-friendly after first model download (if running locally).

## Quickstart (vLLM)

### Known-good model IDs
- `deepseek/DeepSeek-V3.1-NVFP4/`
- `mistral/Mistral-Large-3-675B-Instruct-2512-NVFP4/`

### Proof-of-life check
```bash
curl http://host.docker.internal:8000/v1/models
```

### Run the UI against vLLM
```bash
export VLLM_BASE_URL=http://host.docker.internal:8000/v1
export VLLM_MODEL_ID=deepseek/DeepSeek-V3.1-NVFP4/
./run_ui.sh
```
The UI will auto-detect models from `/models` and default to the detected list if available.

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
- **TRT-LLM server** (`autonomous-trtllm`): nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc5, served at `http://localhost:8000/v1`
- **Gradio UI** (`autonomous-ui`): served at `http://localhost:7860`
  - Note: the server image must include CUDA + Python 3.11 so `mamba-ssm` installs from wheels.

### Linux-only note
If you use `host.docker.internal` on Linux, add the mapping:
```yaml
extra_hosts: ["host.docker.internal:host-gateway"]
```
(Already present in `docker-compose.kimik2-nvfp4.yml`.)

### TRT-LLM Server (AutoDeploy)
```bash
docker compose up --build
```
```bash
curl http://localhost:8000/v1/models
```
UI: `http://localhost:7860`

On first run, TensorRT-LLM may compile/build artifacts and download weights. Subsequent runs reuse `./_trtllm_cache` and `./_hf_cache` for faster startup.

### Rebuild the UI image
If the UI image changes, rebuild with no cache:
```bash
docker compose build --no-cache autonomous-ui
docker compose up -d --force-recreate autonomous-ui
docker exec autonomous-ui docker ps
```

### Kimi-K2 NVFP4 via host vLLM (recommended for GB300 / Grace)
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
docker compose --profile playground build autonomous-playground-image
```

### Useful endpoints and volumes
- Check the server: `curl http://localhost:8000/v1/models`
- Open the UI: `http://localhost:7860`
- On first run the server container warms the model cache (full weights download) before reporting ready; this can take a while depending on your network and disk.
- Tail the server logs while it downloads: `docker logs -f autonomous-trtllm`
- Verify cached weights: `ls ./_hf_cache/hub`
- HF cache is persisted to `./_hf_cache` (mounted to `/root/.cache/huggingface`)
- Prompt edits persist in `./prompt_library` (mounted to `/app/prompt_library`)
- Default prompts and context are bind-mounted read-only from `./prompts` and `./context`

### Environment overrides
- `VLLM_BASE_URL` (default: `http://host.docker.internal:8000/v1`)
- `VLLM_MODEL_ID` (default: `deepseek/DeepSeek-V3.1-NVFP4/`)
- `VLLM_REQUEST_TIMEOUT_S` (default: `120`)
- `HF_TOKEN` and `HF_HOME` for Hugging Face access/cache location (set `HF_TOKEN` for gated models)
- The UI calls the server at `VLLM_BASE_URL` (set in compose to `http://autonomous-trtllm:8000/v1`)

### Model requirements
- TRT-LLM mode runs via AutoDeploy with `trust_remote_code=True`.
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

## Run the CLI demo
```bash
./run_demo_cli.sh "Build a resilient offline LLM demo" --scenario "Ship a resilient offline demo"
```
Streams the same stages as the UI using the orchestrator.

## Run the stress test harness
```bash
./run_stress_test.sh --concurrency 8 --num-requests 50 --max-tokens 256 --prompt "Explain vLLM batching."
```
Use `--endpoint completions` to test `/v1/completions` instead of chat.

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

## Preset scenarios
- Optimize inference for latency
- Ship a resilient offline demo
- Benchmark throughput on GB300

## Troubleshooting
- **Server not ready**: `python -m src.server.healthcheck --host localhost --port 8000`
- **OOM or slow load**: lower `--max-model-len` (e.g., 12288) or reduce `--gpu-memory-utilization` in `run_server.sh`.
- **HF cache issues**: set `HF_HOME` to a fast disk and ensure the model is fully downloaded before going offline.
- **Port conflicts**: adjust `PORT` env var for server/UI scripts.

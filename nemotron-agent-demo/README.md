# Nemotron Station Agent Demo

Agentic demo that drives a single LLM through planner/coder/reviewer/ops/aggregator roles and visualizes progress live.

Default model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` via vLLM.

**Quickstart (Docker Compose, recommended)**
1. `export HF_TOKEN=...` if the model is gated for your account.
2. `./autostart_nemotron3.sh`
3. Open the UI at `http://localhost:7860`.

Containers are started with `restart: unless-stopped`, so they will auto-restart when the Docker daemon starts.
Model downloads are cached locally in `../_hf_cache` (repo root, created on first run).

**What This Starts**
- `nemotron-vllm` (vLLM OpenAI-compatible API) on `http://localhost:8000/v1`
- `nemotron-ui` (Gradio UI) on `http://localhost:7860`
- `dml-service` (Daystrom Memory Lattice) on `http://localhost:9001/health`

**Compose Files**
- `docker-compose.yml`: UI + DML only (expects external vLLM at `VLLM_BASE_URL`).
- `docker-compose.nemotron3.yml`: adds the Nemotron 3 NVFP4 vLLM service and points UI + DML to it.
- `docker-compose.kimik2-nvfp4.yml`: legacy helper for a Kimi K2 host vLLM.

**Run In Foreground**
- Default (includes Nemotron vLLM): `./run_all.sh`
- Use an external vLLM: `VLLM_MODE=host ./run_all.sh`

**Environment Overrides**
- `HF_TOKEN` for gated model access.
- `HF_HOME` or `HUGGINGFACE_HUB_CACHE` for an alternate cache directory (defaults to `../_hf_cache` for the Nemotron compose stack).
- `VLLM_BASE_URL` (default: `http://host.docker.internal:8000/v1`).
- `VLLM_MODEL_ID` (default: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`).
- `VLLM_TIMEOUT_S` (default: `120`).

**Health Checks**
- vLLM: `curl http://localhost:8000/v1/models`
- UI: `http://localhost:7860`
- DML: `curl http://localhost:9001/health`

**Kimi Host (Legacy / Optional)**
Terminal 1:
`./run_kimi_vllm_host.sh`

Terminal 2:
`docker compose -f docker-compose.kimik2-nvfp4.yml up --build`

If needed, set `VLLM_MODEL_ID=kimi-k2-nvfp4`.
The Kimi host script now uses repo-local caches: `../_hf_cache`, `../_vllm_cache`, and `../_tmp`.

**Local Run Without Containers (Legacy)**
```bash
cd nemotron-agent-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
VLLM_BASE_URL=http://localhost:8000/v1 ./run_ui.sh
```

**Build The Playground Image**
```bash
docker compose --profile playground build nemotron-playground-image
```

**Managing Prompts From The UI**
- Open the Prompts tab to manage goal presets and agent prompts.
- Goal Presets: edit, save, or create entries in `prompt_library/goal_presets.json`.
- Agent Prompts: save overrides to `prompt_library/agent_overrides/<agent>.txt`.

**Daystrom Memory Lattice (DML)**
- Toggle DML in the UI to enable persistent memory + retrieval reports.
- Storage: `../_dml` (repo root, persisted across runs).
- Reset: stop the UI and remove `../_dml`.

**Run The CLI Demo**
```bash
./run_demo_cli.sh "Build a resilient offline LLM demo" --scenario "Ship a resilient offline demo"
```

**Troubleshooting**
- GPU access: verify `nvidia-smi` works on the host and Docker can see GPUs.
- OOM or slow load: lower `--max-model-len` or `--gpu-memory-utilization` for vLLM.
- Port conflicts: adjust port mappings in the compose files.

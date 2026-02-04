# Nemotron Station Agent Demo

Agentic demo that drives a single LLM through planner/coder/reviewer/ops/aggregator roles and visualizes progress live.

Default model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` via NVIDIA NIM.

**Quickstart (Docker Compose, recommended)**
1. `./ngc_login.sh`
2. `docker compose --env-file creds.env -f docker-compose.yml -f docker-compose.nemotron3-nim.yml up -d`
3. Open the UI at `http://localhost:7860`.

Containers are started with `restart: unless-stopped`, so they will auto-restart when the Docker daemon starts.

**What This Starts**
- `nemotron-nim` (NIM OpenAI-compatible API) on `http://localhost:8000/v1`
- `nemotron-ui` (Gradio UI) on `http://localhost:7860`
- `dml-service` (Daystrom Memory Lattice) on `http://localhost:9001/health`

**Compose Files**
- `docker-compose.yml`: UI + DML only (expects external vLLM at `VLLM_BASE_URL`).
- `docker-compose.nemotron3-nim.yml`: adds the Nemotron 3 Nano NIM service and points UI + DML to it.
- `docker-compose.nemotron3-nim-multi.yml`: runs separate NIM containers per agent role.

**Multi-NIM (Per-Role Agents)**
```bash
./ngc_login.sh
docker compose --env-file creds.env -f docker-compose.yml -f docker-compose.nemotron3-nim-multi.yml up -d
```

**Environment Overrides**
- `NGC_API_KEY` for NGC registry auth (stored locally in `creds.env`).
- `VLLM_BASE_URL` (default: `http://host.docker.internal:8000/v1`).
- `VLLM_MODEL_ID` (default: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`).
- `VLLM_TIMEOUT_S` (default: `120`).

**Health Checks**
- NIM: `curl http://localhost:8000/v1/models`
- UI: `http://localhost:7860`
- DML: `curl http://localhost:9001/health`

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
- Port conflicts: adjust port mappings in the compose files.

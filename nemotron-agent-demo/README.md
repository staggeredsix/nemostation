# Nemotron-3 Nano Agentic Demo (HF-only)

Brain-melty local demo that drives a single Nemotron-3 Nano model through planner/coder/reviewer/ops/aggregator roles and visualizes progress live.

## Features
- Hugging Face weights only (default `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`).
- vLLM OpenAI-compatible server at `http://localhost:8000/v1` with conservative defaults.
- Gradio UI with animated status badges, live metrics (approx TTFT, tokens/sec), and progressive timeline updates.
- CLI demo that streams stage states in the terminal.
- Offline-friendly after first model download.

## Setup
```bash
cd nemotron-agent-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Start the server (one command)
```bash
MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 ./run_server.sh
```
This launches vLLM with `--max-model-len 16384` and `--gpu-memory-utilization 0.90`, waits for `/v1/models`, then prints `SERVER READY`.

## Start the UI (one command)
```bash
./run_ui.sh
```
Opens Gradio on `http://localhost:7860` with the live timeline, animated badges, and metrics panel. The UI pings `/v1/models` on load and blocks runs if the server is down.

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

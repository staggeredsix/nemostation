# DML Memory Service Demos

This folder hosts a lightweight FastAPI service and proof-of-life demos for using the Daystrom Memory Lattice (DML) as a shared memory substrate across multiple clients.

## Running the service

The service can be built with the CUDA-ready `Dockerfile.dml` or launched directly:

```bash
python -m app.dml_service
```

By default the API listens on `http://localhost:8000` and exposes endpoints under `/v1/...`.

## docker-compose prototype

A prototype compose file (`docker-compose.yml`) starts the DML memory service plus mock/demo clients. Run:

```bash
docker compose up --build
```

## Demo entrypoints

Each demo uses the shared DML service (set `DML_SERVICE_URL` if not on localhost):

- **NIM/OpenAI-style LLM**: `python -m app.nim_demo`
- **ACE-like agent with instance memory**: `python -m app.ace_agent_demo`
- **Isaac robotics logging**: `python -m app.isaac_demo`

Set `OPENAI_API_BASE` and `OPENAI_API_KEY` to call a real OpenAI-compatible endpoint. Without them, the demos fall back to stubbed text so the memory flows can still be exercised.

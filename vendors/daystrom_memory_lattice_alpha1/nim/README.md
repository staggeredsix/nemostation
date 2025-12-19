# Running Daystrom Memory Lattice with NVIDIA NIM

The repository ships with a lightweight FastAPI service and frontend that can be
paired with NVIDIA NIM deployments exposing an OpenAI-compatible API. The
instructions below target the `nvcr.io/nim/openai/gpt-oss-20b:latest` container
but can be adapted to other NIM models.

## 1. Prepare environment

Create an `.env` file (or export the variables directly) containing your NGC
API key. The same key is used as the bearer token for authenticated requests
against the NIM endpoint.

```bash
export NGC_API_KEY=<PASTE_API_KEY_HERE>
export NIM_API_KEY="$NGC_API_KEY"
export LOCAL_NIM_CACHE=${LOCAL_NIM_CACHE:-$HOME/.cache/nim}
mkdir -p "$LOCAL_NIM_CACHE"
```

## 2. Start the NIM container

```bash
docker run -it --rm \
    --gpus all \
    --shm-size=16GB \
    -e NGC_API_KEY \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    -u $(id -u) \
    -p 8000:8000 \
    nvcr.io/nim/openai/gpt-oss-20b:latest
```

When the container is running, the OpenAI-compatible API is available at
`http://localhost:8000/v1/chat/completions`.

## 3. Launch the DML service

Run the DML server locally (or via Docker) and point it at the NIM endpoint by
setting `NIM_API_BASE` or `OPENAI_API_BASE`.

```bash
export NIM_API_BASE="http://localhost:8000"
uvicorn daystrom_dml.server:app --host 0.0.0.0 --port 9000
```

Alternatively, build the provided Docker image and run the service as a
container alongside NIM:

```bash
docker build -t daystrom-dml .
docker run --rm -p 9000:9000 \
    -e NIM_API_BASE="http://host.docker.internal:8000" \
    -e NIM_API_KEY="$NGC_API_KEY" \
    daystrom-dml
```

## 4. Access the web playground

Navigate to `http://localhost:9000/` to use the built-in playground. Upload PDF
or text files, issue prompts, and compare the base model output to the RAG
augmented response. Retrieval fidelity, token estimates, and raw context entries
are exposed directly in the UI.

## 5. Troubleshooting

- Ensure the host has a GPU compatible with the NVIDIA container runtime.
- The frontend relies on the FastAPI server bundling static assets under
  `daystrom_dml/web`. Verify the directory exists if you are running from source.
- Any OpenAI-compatible deployment can be used by setting `OPENAI_API_BASE` and
  `OPENAI_API_KEY` instead of the `NIM_*` variables.

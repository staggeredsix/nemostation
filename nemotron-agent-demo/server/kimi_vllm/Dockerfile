FROM nvcr.io/nvidia/vllm:25.11-py3

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

RUN uv pip install vllm --pre \
    --extra-index-url https://wheels.vllm.ai/1d495c2f92c7e75f580c2f4b465823f2fb688abe/cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match

RUN python -m pip install --no-cache-dir model-hosting-container-standards

RUN python -c "import model_hosting_container_standards.sagemaker; print('mhcs ok')"

RUN python - <<'PY'
import vllm
import torch

print("vLLM", vllm.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

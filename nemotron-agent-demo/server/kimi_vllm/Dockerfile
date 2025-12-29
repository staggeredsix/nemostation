FROM nvcr.io/nvidia/vllm:25.11-py3

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip

RUN curl -fL https://github.com/vllm-project/vllm/releases/download/v0.13.0/vllm-0.13.0+cu130-cp38-abi3-manylinux_2_35_aarch64.whl \
      -o /tmp/vllm.whl
RUN python -m pip uninstall -y vllm || true
RUN python -m pip install --no-cache-dir /tmp/vllm.whl
RUN python -c "import vllm; print('vLLM version:', vllm.__version__)"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

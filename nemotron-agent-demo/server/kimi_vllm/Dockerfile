FROM nvcr.io/nvidia/vllm:25.11-py3

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        curl \
        git \
        build-essential \
        ninja-build \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip

RUN python -m pip uninstall -y vllm || true

RUN git clone https://github.com/vllm-project/vllm.git /opt/vllm-src \
    && cd /opt/vllm-src \
    && git checkout v0.13.0 \
    && python -m pip install -U pip setuptools wheel packaging \
    && python -m pip install -e . \
    && python -c "import vllm; print('vLLM version:', vllm.__version__)"

RUN python -m pip install --no-cache-dir model-hosting-container-standards

RUN python -c "import model_hosting_container_standards.sagemaker; print('mhcs ok')"

RUN python - <<'PY'
import os
import torch

libdir = os.path.join(os.path.dirname(torch.__file__), "lib")
libtorch = os.path.join(libdir, "libtorch_cuda.so")
if not os.path.exists(libtorch):
    raise FileNotFoundError(f"Missing {libtorch}")
with open("/etc/ld.so.conf.d/torch.conf", "w", encoding="utf-8") as handle:
    handle.write(f"{libdir}\n")
PY

RUN ldconfig

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

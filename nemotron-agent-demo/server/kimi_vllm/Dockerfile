FROM nvcr.io/nvidia/vllm:25.11-py3

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip

RUN WHEEL=/tmp/vllm-0.13.0+cu130-cp38-abi3-manylinux_2_35_aarch64.whl \
    && curl -fL https://github.com/vllm-project/vllm/releases/download/v0.13.0/vllm-0.13.0+cu130-cp38-abi3-manylinux_2_35_aarch64.whl -o "$WHEEL" \
    && python -m pip uninstall -y vllm || true

RUN WHEEL=/tmp/vllm-0.13.0+cu130-cp38-abi3-manylinux_2_35_aarch64.whl \
    && python -m pip install --no-cache-dir --no-deps "$WHEEL" \
    && python -c "import vllm; print('vLLM version:', vllm.__version__)"

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

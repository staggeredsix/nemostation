FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        jq \
        git \
        openssh-client \
        rsync \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://dl.k8s.io/release/stable.txt -o /tmp/k8s-stable.txt \
    && K8S_VERSION="$(cat /tmp/k8s-stable.txt)" \
    && ARCH="$(uname -m)" \
    && case "${ARCH}" in \
        x86_64|amd64) K8S_ARCH="amd64" ;; \
        aarch64|arm64) K8S_ARCH="arm64" ;; \
        *) echo "Unsupported arch: ${ARCH}" >&2; exit 1 ;; \
      esac \
    && curl -fsSL "https://dl.k8s.io/release/${K8S_VERSION}/bin/linux/${K8S_ARCH}/kubectl" -o /usr/local/bin/kubectl \
    && chmod +x /usr/local/bin/kubectl \
    && kubectl version --client=true --output=yaml >/tmp/kubectl-version.yml

RUN pip install --no-cache-dir \
        fastapi \
        gradio \
        pydantic \
        pytest \
        redis \
        requests \
        uvicorn

RUN python -c "import requests, redis; print('deps ok')"

RUN useradd -ms /bin/bash playground

USER playground
WORKDIR /workspace

FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        jq \
        git \
    && rm -rf /var/lib/apt/lists/*

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

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
        redis \
        requests \
        uvicorn

RUN useradd -ms /bin/bash playground

USER playground
WORKDIR /workspace

FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash playground

USER playground
WORKDIR /workspace

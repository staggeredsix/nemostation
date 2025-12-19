FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \ 
    && apt-get install --no-install-recommends -y python3 python3-venv python3-pip build-essential curl \ 
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

WORKDIR /opt/dml

COPY pyproject.toml README.md .
COPY daystrom_dml ./daystrom_dml
COPY tests ./tests
COPY bench ./bench
COPY scripts ./scripts

RUN pip install --no-cache-dir --upgrade pip \ 
    && pip install --no-cache-dir .[server,tokenizer,embeddings,faiss]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD curl -f http://localhost:8000/health || exit 1

CMD ["dml-server"]

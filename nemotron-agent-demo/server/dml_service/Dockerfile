FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY nemotron-agent-demo/server/dml_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY nemotron-agent-demo/server/dml_service ./dml_service
COPY vendors/daystrom_memory_lattice_alpha1 /app/nemostation/vendors/daystrom_memory_lattice_alpha1

ENV PYTHONPATH=/app:/app/nemostation/vendors/daystrom_memory_lattice_alpha1 \
    DML_STORAGE_DIR=/data/dml

EXPOSE 9001

CMD ["uvicorn", "dml_service.app:app", "--host", "0.0.0.0", "--port", "9001"]

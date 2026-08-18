FROM python:3.12.8-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=300
RUN useradd --create-home --uid 10001 rag
WORKDIR /app
COPY pyproject.toml README.md ./

FROM base AS api
RUN mkdir -p src/rag_platform && touch src/rag_platform/__init__.py && pip install .
COPY alembic.ini ./
COPY alembic ./alembic
COPY src ./src
RUN pip install --no-deps .
USER rag
CMD ["uvicorn", "rag_platform.main:app", "--host", "0.0.0.0", "--port", "8100"]

FROM base AS worker
RUN mkdir -p src/rag_platform && touch src/rag_platform/__init__.py \
    && pip install ".[worker]"
COPY src ./src
RUN pip install --no-deps .
RUN mkdir -p /home/rag/.cache/huggingface && chown -R rag:rag /home/rag/.cache
USER rag
CMD ["celery", "-A", "rag_platform.worker.celery_app", "worker", "--beat", "--schedule=/tmp/celerybeat-schedule", "--queues=indexing,search", "--loglevel=INFO", "--concurrency=1"]

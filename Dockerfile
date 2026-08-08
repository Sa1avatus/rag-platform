FROM python:3.12.8-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=300
RUN useradd --create-home --uid 10001 rag
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

FROM base AS api
RUN pip install .
USER rag
CMD ["uvicorn", "rag_platform.main:app", "--host", "0.0.0.0", "--port", "8100"]

FROM base AS worker
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3" \
    && pip install ".[worker]"
USER rag
CMD ["celery", "-A", "rag_platform.worker.celery_app", "worker", "--beat", "--schedule=/tmp/celerybeat-schedule", "--queues=indexing,search", "--loglevel=INFO", "--concurrency=2"]

FROM python:3.12.8-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd --create-home --uid 10001 rag
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .
USER rag
CMD ["uvicorn", "rag_platform.main:app", "--host", "0.0.0.0", "--port", "8100"]

import json
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from celery.signals import worker_process_init
from redis import Redis
from transformers import AutoTokenizer

from rag_platform.core.config import get_settings
from rag_platform.core.metrics import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DURATION,
    EMBEDDING_FAILURES,
    EMBEDDING_REQUESTS,
)
from rag_platform.services.embedding_contract import validate_embedding_dimension
from rag_platform.services.readiness import MODEL_READY_KEY

_LOCAL_ONNX_DIR = "/app/onnx-model"


def _resolve_device(requested: str) -> str:
    """Resolve embedding device.  ``auto`` picks CUDA if *nvidia-smi* reports
    ≥1 GB free memory, otherwise falls back to CPU."""
    if requested != "auto":
        return requested
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            free_mb = int(result.stdout.strip().split("\n")[0])
            if free_mb >= 1024:
                return "cuda"
    except Exception:
        pass
    return "cpu"


def _onnx_provider(device: str) -> str:
    """Map a resolved device string to an ONNX Runtime execution provider."""
    if device == "cuda":
        return "CUDAExecutionProvider"
    return "CPUExecutionProvider"


def _find_onnx_model(model_dir: str) -> str:
    """Locate the ``.onnx`` model file inside *model_dir*."""
    candidates = sorted(Path(model_dir).glob("*.onnx"))
    if not candidates:
        raise FileNotFoundError(f"No .onnx file found in {model_dir}")
    return str(candidates[0])


@lru_cache(maxsize=1)
def _load() -> tuple:
    """Load ONNX session + tokenizer once per worker process.

    If a pre-exported ONNX model is baked into the Docker image at
    ``/app/onnx-model`` it is used directly; otherwise the model path
    from settings is expected to contain ``.onnx`` weights.
    """
    settings = get_settings()
    device = _resolve_device(settings.embedding_device)
    provider = _onnx_provider(device)

    model_path = _LOCAL_ONNX_DIR if os.path.isdir(_LOCAL_ONNX_DIR) else settings.embedding_model

    onnx_file = _find_onnx_model(model_path)
    session = ort.InferenceSession(
        onnx_file,
        providers=[provider],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return session, tokenizer, device


def model() -> ort.InferenceSession:
    """Return the cached ONNX Runtime session (for introspection / testing)."""
    return _load()[0]


def embed(texts: list[str]) -> list[list[float]]:
    EMBEDDING_REQUESTS.inc()
    EMBEDDING_BATCH_SIZE.observe(len(texts))
    started = time.perf_counter()
    try:
        session, tokenizer, _ = _load()

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        # Build the feed dict for ONNX Runtime using actual input names.
        valid = {i.name for i in session.get_inputs()}
        ort_inputs = {k: v for k, v in inputs.items() if k in valid}
        (token_embeddings,) = session.run(None, ort_inputs)

        # Mean-pool over token embeddings, respecting the attention mask.
        attention_mask: np.ndarray = inputs["attention_mask"]
        mask = np.expand_dims(attention_mask, axis=-1)
        summed = np.sum(token_embeddings * mask, axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = summed / counts

        # L2-normalize.
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        embeddings = pooled / norms
    except Exception:
        EMBEDDING_FAILURES.inc()
        raise
    finally:
        EMBEDDING_DURATION.observe(time.perf_counter() - started)
    return embeddings.tolist()


def dimension() -> int:
    """Return the embedding dimension by probing the ONNX model's output shape."""
    session, tokenizer, _ = _load()
    dummy = tokenizer("test", return_tensors="np")
    valid = {i.name for i in session.get_inputs()}
    ort_inputs = {k: v for k, v in dummy.items() if k in valid}
    (out,) = session.run(None, ort_inputs)
    return int(out.shape[-1])


@worker_process_init.connect
def validate_model_contract(**kwargs: object) -> None:
    detected = dimension()
    settings = get_settings()
    validate_embedding_dimension(detected, settings.embedding_dimension)
    device = _resolve_device(settings.embedding_device)
    cache = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        cache.set(
            MODEL_READY_KEY,
            json.dumps(
                {
                    "model": settings.embedding_model,
                    "dimension": detected,
                    "device": device,
                }
            ),
            ex=60,
        )
    finally:
        cache.close()

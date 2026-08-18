import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from celery.signals import worker_process_init
from huggingface_hub import snapshot_download
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

_ONNX_SUBFOLDER = "onnx"


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


@lru_cache(maxsize=1)
def _load() -> tuple:
    """Load ONNX session + tokenizer once per worker process.

    Downloads the ``onnx/`` subfolder from the HuggingFace model repo so
    that external data files (``model.onnx_data``) are co-located with
    ``model.onnx``.  The folder is cached by *huggingface_hub*.
    """
    settings = get_settings()
    device = _resolve_device(settings.embedding_device)
    provider = _onnx_provider(device)

    # Download the entire onnx/ subfolder (model.onnx + model.onnx_data + tokenizer).
    local_dir = snapshot_download(
        settings.embedding_model,
        allow_patterns=[f"{_ONNX_SUBFOLDER}/*"],
    )
    onnx_dir = Path(local_dir) / _ONNX_SUBFOLDER

    onnx_model_path = str(onnx_dir / "model.onnx")
    tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))

    available = set(ort.get_available_providers())
    use_provider = provider if provider in available else "CPUExecutionProvider"

    session = ort.InferenceSession(onnx_model_path, providers=[use_provider])
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
    outputs = session.run(None, ort_inputs)
    return int(outputs[0].shape[-1])


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

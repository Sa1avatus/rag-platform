"""Load BGE-M3 directly from cache, no Hub download."""

import os
import sys
import uuid

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

sys.path.insert(0, "/usr/local/lib/python3.12/site-packages")
import psycopg2

from rag_platform.core.config import get_settings
from rag_platform.core.embedding_registry import MAX_VECTOR_DIMENSION, get_model_by_id

CACHE_DIR = "/home/rag/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots"


def find_onnx_dir():
    for s in os.listdir(CACHE_DIR):
        d = os.path.join(CACHE_DIR, s, "onnx")
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "model.onnx")):
            return d
    raise FileNotFoundError("No ONNX model found in cache")


def main():
    cfg = get_model_by_id("bge-m3")
    print(f"Model: {cfg.model_name} (dim={cfg.dimension})")

    onnx_dir = find_onnx_dir()
    print(f"Loading from cache: {onnx_dir}")

    onnx_file = os.path.join(onnx_dir, "model.onnx")
    session = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
    print(f"Model loaded. Providers={session.get_providers()}")

    settings = get_settings()
    conn = psycopg2.connect(settings.database_url.replace("+asyncpg", ""))
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("""
        SELECT c.id, c.content
        FROM chunks c
        LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.id AND ce.model = %s
        WHERE ce.id IS NULL
        ORDER BY c.id
    """, (cfg.model_name,))
    missing = cur.fetchall()
    print(f"Chunks missing BGE embedding: {len(missing)}")

    if not missing:
        print("Nothing to do.")
        return

    batch_size = 8
    inserted = 0

    for i in range(0, len(missing), batch_size):
        batch = missing[i:i + batch_size]
        chunk_ids = [row[0] for row in batch]
        texts = [row[1] for row in batch]

        inputs = tokenizer(
            texts, padding=True, truncation=True,
            max_length=cfg.max_input_tokens, return_tensors="np",
        )
        valid = {inp.name for inp in session.get_inputs()}
        ort_inputs = {k: v for k, v in inputs.items() if k in valid}
        if "token_type_ids" in valid and "token_type_ids" not in ort_inputs:
            ort_inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
        outputs = session.run(None, ort_inputs)
        token_embeddings = outputs[0]

        attention_mask = inputs["attention_mask"]
        mask = np.expand_dims(attention_mask, axis=-1)
        summed = np.sum(token_embeddings * mask, axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = summed / counts
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        embeddings = pooled / norms
        raw_vecs = embeddings.tolist()
        padded_vecs = [v + [0.0] * (MAX_VECTOR_DIMENSION - len(v)) for v in raw_vecs]

        for chunk_id, vec in zip(chunk_ids, padded_vecs, strict=False):
            emb_id = uuid.uuid4()
            vec_str = "[" + ",".join(str(x) for x in vec) + "]"
            cur.execute("""
                INSERT INTO chunk_embeddings
                    (id, chunk_id, model, model_revision, backend, normalization,
                     embedding_dimension, embedding, created_at, updated_at)
                VALUES
                    (%s, %s, %s, 'default', 'onnxruntime', %s, %s, %s::vector, now(), now())
                ON CONFLICT (chunk_id, model, model_revision) DO NOTHING
            """, (str(emb_id), str(chunk_id), cfg.model_name, cfg.normalization,
                  cfg.dimension, vec_str))
            inserted += 1

        if (i // batch_size) % 10 == 0:
            print(f"  {inserted}/{len(missing)}")

    print(f"\nDone. Inserted={inserted}")
    cur.execute("SELECT count(DISTINCT chunk_id) FROM chunk_embeddings WHERE model = %s",
                (cfg.model_name,))
    print(f"Final BGE embeddings: {cur.fetchone()[0]}")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()

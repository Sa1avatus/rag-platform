"""One-off script: embed missing chunks for a specific model.

Run inside the worker container:
  python /tmp/embed_missing.py <model_id>

model_id: 'multilingual-e5-small' or 'bge-m3'
"""

import sys
import uuid

import numpy as np

# Add the project to path
sys.path.insert(0, "/usr/local/lib/python3.12/site-packages")

import psycopg2

from rag_platform.core.config import get_settings
from rag_platform.core.embedding_registry import get_model_by_id
from rag_platform.worker.embeddings import _load_model


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "multilingual-e5-small"
    cfg = get_model_by_id(model_id)
    print(f"Model: {cfg.model_name} (dim={cfg.dimension})")
    print("Loading ONNX model...")
    session, tokenizer, device = _load_model(cfg)
    print(f"Model loaded. Device={device}, providers={session.get_providers()}")

    settings = get_settings()
    conn = psycopg2.connect(settings.database_url.replace("+asyncpg", ""))
    conn.autocommit = True
    cur = conn.cursor()

    # Find chunks missing this model's embedding
    cur.execute("""
        SELECT c.id, c.content
        FROM chunks c
        LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.id AND ce.model = %s
        WHERE ce.id IS NULL
        ORDER BY c.id
    """, (cfg.model_name,))
    missing = cur.fetchall()
    print(f"Chunks missing {cfg.model_name} embedding: {len(missing)}")

    if not missing:
        print("Nothing to do.")
        return

    batch_size = 32
    inserted = 0
    failed = 0

    for i in range(0, len(missing), batch_size):
        batch = missing[i:i + batch_size]
        chunk_ids = [row[0] for row in batch]
        texts = [row[1] for row in batch]

        try:
            # Apply passage prefix if needed
            if cfg.passage_prefix:
                texts = [cfg.passage_prefix + t for t in texts]

            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=cfg.max_input_tokens,
                return_tensors="np",
            )
            valid = {inp.name for inp in session.get_inputs()}
            ort_inputs = {k: v for k, v in inputs.items() if k in valid}
            if "token_type_ids" in valid and "token_type_ids" not in ort_inputs:
                ort_inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
            outputs = session.run(None, ort_inputs)
            token_embeddings = outputs[0]

            # Mean-pool
            attention_mask = inputs["attention_mask"]
            mask = np.expand_dims(attention_mask, axis=-1)
            summed = np.sum(token_embeddings * mask, axis=1)
            counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
            pooled = summed / counts

            # L2-normalize
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            embeddings = pooled / norms
            raw_vecs = embeddings.tolist()

            # Pad to MAX_VECTOR_DIMENSION
            padded_vecs = [cfg.pad_vector(v) for v in raw_vecs]

            # Insert embeddings
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

            print(f"  Batch {i // batch_size + 1}: {len(batch)} chunks embedded and inserted "
                  f"(total: {inserted}/{len(missing)})")

        except Exception as e:
            failed += len(batch)
            print(f"  Batch {i // batch_size + 1} FAILED: {e}")

    print(f"\nDone. Inserted={inserted}, Failed={failed}, Total missing={len(missing)}")

    # Verify
    cur.execute("""
        SELECT count(DISTINCT chunk_id)
        FROM chunk_embeddings
        WHERE model = %s
    """, (cfg.model_name,))
    final_count = cur.fetchone()[0]
    print(f"Final {cfg.model_name} embeddings: {final_count}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()

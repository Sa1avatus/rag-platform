"""Embedding model A/B benchmark: E5-small vs BGE-M3.

Runs pure vector retrieval (no BM25, no reranker) on the same corpus
for both models, measures latency, and computes inter-model agreement.

Run inside the worker container:
  python /tmp/benchmark_embeddings.py
"""

import json
import sys
import time
from statistics import mean, median, stdev

import numpy as np
import psycopg2

sys.path.insert(0, "/usr/local/lib/python3.12/site-packages")

from rag_platform.core.config import get_settings
from rag_platform.core.embedding_registry import get_model_by_id
from rag_platform.worker.embeddings import _load_model

# ── Configuration ──────────────────────────────────────────────────────

PROJECT_ID = "9f6f0f20-24e3-4f8c-83a6-d97d3c62dc00"
OWNER_USER_ID = "bad20767-dfd5-4b81-9c13-fe92034a32e1"
COLLECTIONS = ["vacancies", "resumes", "profiles"]

TOP_K_VALUES = [5, 10, 20]
VECTOR_TOP_K = 30  # retrieve this many for comparison

# Test queries — representative job search queries
TEST_QUERIES = [
    "Python machine learning engineer with RAG experience",
    "Senior backend developer Go microservices",
    "Data scientist natural language processing",
    "DevOps engineer Kubernetes cloud infrastructure",
    "Frontend React TypeScript developer",
    "Full stack JavaScript Node.js MongoDB",
    "QA automation engineer Selenium Cypress",
    "Product manager fintech startup",
    "iOS developer Swift mobile applications",
    "Java Spring Boot enterprise applications",
    "AI engineer LLM prompt engineering",
    "System architect distributed systems",
    "UX designer user research",
    "Security engineer penetration testing",
    "Data engineer Apache Spark ETL pipeline",
    "Разработчик Python Django PostgreSQL",
    "Machine learning engineer computer vision",
    "Cloud architect AWS Azure",
    "Embedded systems C++ RTOS",
    "Blockchain developer Solidity Web3",
]

# ── Helpers ────────────────────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-9))


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    gains = [1.0 if item in relevant else 0.0 for item in retrieved[:k]]
    dcg = sum(g / np.log2(rank + 1) for rank, g in enumerate(gains, 1))
    ideal = sorted([1.0] * min(len(relevant), k), reverse=True)
    idcg = sum(g / np.log2(rank + 1) for rank, g in enumerate(ideal, 1))
    return dcg / idcg if idcg > 0 else 0.0


def overlap_at_k(list_a: list[str], list_b: list[str], k: int) -> float:
    set_a = set(list_a[:k])
    set_b = set(list_b[:k])
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ── Main benchmark ────────────────────────────────────────────────────

def main():
    settings = get_settings()
    conn = psycopg2.connect(settings.database_url.replace("+asyncpg", ""))
    conn.autocommit = True
    cur = conn.cursor()

    # Load both models
    models = {}
    for model_id in ["multilingual-e5-small", "bge-m3"]:
        cfg = get_model_by_id(model_id)
        print(f"Loading {cfg.model_name} (dim={cfg.dimension})...")
        session, tokenizer, device = _load_model(cfg)
        models[model_id] = {
            "cfg": cfg,
            "session": session,
            "tokenizer": tokenizer,
            "device": device,
        }
        print(f"  Loaded. Device={device}")

    # Verify corpus state
    for _model_id, m in models.items():
        cur.execute(
            "SELECT count(DISTINCT chunk_id) FROM chunk_embeddings WHERE model = %s",
            (m["cfg"].model_name,),
        )
        count = cur.fetchone()[0]
        print(f"  {m['cfg'].model_name}: {count} embeddings")

    # ── Benchmark results storage ──────────────────────────────────────
    results = {}
    for model_id in models:
        results[model_id] = {
            "query_latencies_ms": [],
            "embedding_latencies_ms": [],
            "search_latencies_ms": [],
            "retrieved_ids": [],  # per-query list of chunk IDs
            "scores": [],         # per-query list of scores
        }

    # ── Run benchmark ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"Running benchmark: {len(TEST_QUERIES)} queries × {len(models)} models")
    print(f"{'='*80}\n")

    for qi, query in enumerate(TEST_QUERIES):
        print(f"[{qi+1}/{len(TEST_QUERIES)}] \"{query[:60]}...\"")

        for model_id, m in models.items():
            cfg = m["cfg"]
            ort_session = m["session"]
            tokenizer = m["tokenizer"]

            # Step 1: Embed query
            t0 = time.perf_counter()
            text = (cfg.query_prefix + query) if cfg.query_prefix else query
            inputs = tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=cfg.max_input_tokens,
                return_tensors="np",
            )
            valid = {inp.name for inp in ort_session.get_inputs()}
            ort_inputs = {k: v for k, v in inputs.items() if k in valid}
            if "token_type_ids" in valid and "token_type_ids" not in ort_inputs:
                ort_inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
            outputs = ort_session.run(None, ort_inputs)
            token_embeddings = outputs[0]
            attention_mask = inputs["attention_mask"]
            mask = np.expand_dims(attention_mask, axis=-1)
            summed = np.sum(token_embeddings * mask, axis=1)
            counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
            pooled = summed / counts
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            embedding = (pooled / norms)[0].tolist()
            query_vec = cfg.pad_vector(embedding)
            t_embed = time.perf_counter()

            # Step 2: Vector search via pgvector
            vec_str = "[" + ",".join(str(x) for x in query_vec) + "]"
            t1 = time.perf_counter()
            cur.execute("""
                SELECT c.id::text, (1.0 - (ce.embedding <=> %s::vector)) AS score
                FROM chunk_embeddings ce
                JOIN chunks c ON c.id = ce.chunk_id
                WHERE ce.model = %s
                  AND c.project_id = %s
                  AND c.owner_user_id = %s
                  AND c.collection = ANY(%s)
                ORDER BY ce.embedding <=> %s::vector
                LIMIT %s
            """, (vec_str, cfg.model_name, PROJECT_ID, OWNER_USER_ID,
                  COLLECTIONS, vec_str, VECTOR_TOP_K))
            rows = cur.fetchall()
            t_search = time.perf_counter()

            chunk_ids = [r[0] for r in rows]
            scores = [float(r[1]) for r in rows]

            embed_ms = (t_embed - t0) * 1000
            search_ms = (t_search - t1) * 1000
            total_ms = (t_search - t0) * 1000

            results[model_id]["embedding_latencies_ms"].append(embed_ms)
            results[model_id]["search_latencies_ms"].append(search_ms)
            results[model_id]["query_latencies_ms"].append(total_ms)
            results[model_id]["retrieved_ids"].append(chunk_ids)
            results[model_id]["scores"].append(scores)

            top5_preview = chunk_ids[:3]
            print(f"  {cfg.id}: embed={embed_ms:.1f}ms search={search_ms:.1f}ms "
                  f"total={total_ms:.1f}ms top3={top5_preview}")

    # ── Compute inter-model metrics ────────────────────────────────────
    list(models.keys())
    e5_id = "multilingual-e5-small"
    bge_id = "bge-m3"

    print(f"\n{'='*80}")
    print("INTER-MODEL AGREEMENT (treat each model's top-K as 'relevant')")
    print(f"{'='*80}\n")

    for k in TOP_K_VALUES:
        agreements = []
        for qi in range(len(TEST_QUERIES)):
            e5_ids = results[e5_id]["retrieved_ids"][qi]
            bge_ids = results[bge_id]["retrieved_ids"][qi]

            e5_as_relevant = set(e5_ids[:k])
            bge_as_relevant = set(bge_ids[:k])

            # How well does BGE retrieve what E5 considers top-K?
            bge_recall_vs_e5 = recall_at_k(bge_ids, e5_as_relevant, k)
            # How well does E5 retrieve what BGE considers top-K?
            e5_recall_vs_bge = recall_at_k(e5_ids, bge_as_relevant, k)
            # Jaccard overlap
            jaccard = overlap_at_k(e5_ids, bge_ids, k)

            agreements.append({
                "bge_recall_vs_e5": bge_recall_vs_e5,
                "e5_recall_vs_bge": e5_recall_vs_bge,
                "jaccard": jaccard,
            })

        avg_bge_r = mean(a["bge_recall_vs_e5"] for a in agreements)
        avg_e5_r = mean(a["e5_recall_vs_bge"] for a in agreements)
        avg_jaccard = mean(a["jaccard"] for a in agreements)

        print(f"  K={k:2d}: BGE_recall_vs_E5={avg_bge_r:.3f}  "
              f"E5_recall_vs_BGE={avg_e5_r:.3f}  "
              f"Jaccard_overlap={avg_jaccard:.3f}")

    # ── Latency comparison ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("LATENCY COMPARISON (milliseconds)")
    print(f"{'='*80}\n")

    header = f"{'Metric':<30} {'E5-small':>12} {'BGE-M3':>12} {'Ratio':>10}"
    print(header)
    print("-" * len(header))

    for metric_name, metric_key in [
        ("Query embedding", "embedding_latencies_ms"),
        ("Vector search", "search_latencies_ms"),
        ("Total (embed+search)", "query_latencies_ms"),
    ]:
        e5_vals = results[e5_id][metric_key]
        bge_vals = results[bge_id][metric_key]

        e5_mean = mean(e5_vals)
        bge_mean = mean(bge_vals)
        ratio = bge_mean / e5_mean if e5_mean > 0 else float("inf")

        print(f"  {metric_name:<28} {e5_mean:>10.2f}ms {bge_mean:>10.2f}ms {ratio:>8.2f}x")

    for metric_name, metric_key in [
        ("Query embedding (p50)", "embedding_latencies_ms"),
        ("Query embedding (p95)", "embedding_latencies_ms"),
        ("Vector search (p50)", "search_latencies_ms"),
        ("Vector search (p95)", "search_latencies_ms"),
    ]:
        e5_vals = sorted(results[e5_id][metric_key])
        bge_vals = sorted(results[bge_id][metric_key])

        if "p95" in metric_name:
            idx_e5 = max(0, int(len(e5_vals) * 0.95) - 1)
            idx_bge = max(0, int(len(bge_vals) * 0.95) - 1)
            e5_v = e5_vals[idx_e5]
            bge_v = bge_vals[idx_bge]
        else:
            e5_v = median(e5_vals)
            bge_v = median(bge_vals)

        ratio = bge_v / e5_v if e5_v > 0 else float("inf")
        print(f"  {metric_name:<28} {e5_v:>10.2f}ms {bge_v:>10.2f}ms {ratio:>8.2f}x")

    # ── Throughput estimate ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("THROUGHPUT ESTIMATE")
    print(f"{'='*80}\n")

    for model_id in [e5_id, bge_id]:
        m = models[model_id]
        cfg = m["cfg"]
        ort_session = m["session"]
        tokenizer = m["tokenizer"]

        # Batch embedding throughput
        sample_texts = TEST_QUERIES[:10]
        if cfg.passage_prefix:
            sample_texts = [cfg.passage_prefix + t for t in sample_texts]

        t0 = time.perf_counter()
        for _ in range(3):  # 3 iterations for stability
            inputs = tokenizer(
                sample_texts,
                padding=True,
                truncation=True,
                max_length=cfg.max_input_tokens,
                return_tensors="np",
            )
            valid = {inp.name for inp in ort_session.get_inputs()}
            ort_inputs = {k: v for k, v in inputs.items() if k in valid}
            if "token_type_ids" in valid and "token_type_ids" not in ort_inputs:
                ort_inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
            ort_session.run(None, ort_inputs)
        t1 = time.perf_counter()

        batch_time = (t1 - t0) / 3
        queries_per_sec = len(sample_texts) / batch_time
        ms_per_query = batch_time / len(sample_texts) * 1000

        print(f"  {cfg.id}:")
        print(f"    Batch ({len(sample_texts)} queries): {batch_time*1000:.1f}ms")
        print(f"    Throughput: {queries_per_sec:.1f} queries/sec")
        print(f"    Per-query: {ms_per_query:.1f}ms/query")

    # ── Score distribution comparison ──────────────────────────────────
    print(f"\n{'='*80}")
    print("SCORE DISTRIBUTION (cosine similarity)")
    print(f"{'='*80}\n")

    for model_id in [e5_id, bge_id]:
        cfg = models[model_id]["cfg"]
        all_scores = [s for scores in results[model_id]["scores"] for s in scores]
        if all_scores:
            print(f"  {cfg.id}:")
            print(f"    Mean:   {mean(all_scores):.4f}")
            print(f"    Median: {median(all_scores):.4f}")
            print(f"    Min:    {min(all_scores):.4f}")
            print(f"    Max:    {max(all_scores):.4f}")
            if len(all_scores) > 1:
                print(f"    Stdev:  {stdev(all_scores):.4f}")

    # ── Per-query detailed results ─────────────────────────────────────
    print(f"\n{'='*80}")
    print("PER-QUERY TOP-5 RESULTS COMPARISON")
    print(f"{'='*80}\n")

    for qi, query in enumerate(TEST_QUERIES):
        print(f"  Query: \"{query}\"")
        e5_ids = results[e5_id]["retrieved_ids"][qi][:5]
        bge_ids = results[bge_id]["retrieved_ids"][qi][:5]
        e5_scores = results[e5_id]["scores"][qi][:5]
        bge_scores = results[bge_id]["scores"][qi][:5]

        e5_top = list(zip(e5_ids[:3], [f"{s:.3f}" for s in e5_scores[:3]], strict=False))
        bge_top = list(zip(bge_ids[:3], [f"{s:.3f}" for s in bge_scores[:3]], strict=False))
        print(f"    E5  top5: {e5_top}")
        print(f"    BGE top5: {bge_top}")

        overlap = len(set(e5_ids) & set(bge_ids))
        print(f"    Overlap@5: {overlap}/5")
        print()

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    e5_embed_mean = mean(results[e5_id]["embedding_latencies_ms"])
    bge_embed_mean = mean(results[bge_id]["embedding_latencies_ms"])
    e5_search_mean = mean(results[e5_id]["search_latencies_ms"])
    bge_search_mean = mean(results[bge_id]["search_latencies_ms"])
    e5_total_mean = mean(results[e5_id]["query_latencies_ms"])
    bge_total_mean = mean(results[bge_id]["query_latencies_ms"])

    print("  E5-small (384d):")
    print(f"    Embed latency:  {e5_embed_mean:.2f}ms")
    print(f"    Search latency: {e5_search_mean:.2f}ms")
    print(f"    Total latency:  {e5_total_mean:.2f}ms")
    e5_qps = len(TEST_QUERIES) / (sum(results[e5_id]["query_latencies_ms"]) / 1000)
    print(f"    Throughput:     {e5_qps:.1f} queries/sec")
    print()
    print("  BGE-M3 (1024d):")
    print(f"    Embed latency:  {bge_embed_mean:.2f}ms")
    print(f"    Search latency: {bge_search_mean:.2f}ms")
    print(f"    Total latency:  {bge_total_mean:.2f}ms")
    bge_qps = len(TEST_QUERIES) / (sum(results[bge_id]["query_latencies_ms"]) / 1000)
    print(f"    Throughput:     {bge_qps:.1f} queries/sec")
    print()
    print("  Speed ratio (BGE/E5):")
    print(f"    Embedding: {bge_embed_mean/e5_embed_mean:.2f}x")
    print(f"    Search:    {bge_search_mean/e5_search_mean:.2f}x")
    print(f"    Total:     {bge_total_mean/e5_total_mean:.2f}x")

    # Save results to JSON
    output = {
        "corpus": {
            "total_chunks": 2030,
            "project_id": PROJECT_ID,
            "collections": COLLECTIONS,
        },
        "models": {
            e5_id: {"dimension": 384, "model_name": "intfloat/multilingual-e5-small"},
            bge_id: {"dimension": 1024, "model_name": "BAAI/bge-m3"},
        },
        "queries": TEST_QUERIES,
        "latency": {
            e5_id: {
                "embedding_ms_mean": e5_embed_mean,
                "search_ms_mean": e5_search_mean,
                "total_ms_mean": e5_total_mean,
            },
            bge_id: {
                "embedding_ms_mean": bge_embed_mean,
                "search_ms_mean": bge_search_mean,
                "total_ms_mean": bge_total_mean,
            },
        },
    }

    output_path = "/tmp/benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()

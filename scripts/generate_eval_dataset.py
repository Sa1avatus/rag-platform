"""Generate ground-truth evaluation dataset for embedding model comparison.

Runs INSIDE the worker container:
  python /tmp/generate_eval_dataset.py

Methodology (non-circular):
  - BM25 (OpenSearch) is the primary relevance signal — keyword-based,
    completely independent of embedding models.
  - Candidates are pooled from BM25 + E5 vector + BGE vector + random,
    so neither model is favoured.
  - Relevance grades are derived from BM25 rank + keyword overlap.
  - The resulting dataset is saved as a versioned JSON file and uploaded
    to the evaluation API (EvaluationDataset + EvaluationCase).
"""

from __future__ import annotations

import json
import random
import re
import sys

import numpy as np
import psycopg2

sys.path.insert(0, "/usr/local/lib/python3.12/site-packages")

from rag_platform.core.config import get_settings
from rag_platform.core.embedding_registry import get_model_by_id
from rag_platform.worker.embeddings import _load_model

# ── Configuration ──────────────────────────────────────────────────────

PROJECT_ID = "9f6f0f20-24e3-4f8c-83a6-d97d3c62dc00"
TENANT_ID = "338adc64-c529-4144-9791-526fc1ce0ada"
OWNER_USER_ID = "bad20767-dfd5-4b81-9c13-fe92034a32e1"
COLLECTIONS = ["vacancies", "resumes", "profiles"]

OUTPUT_PATH = "/tmp/eval_dataset_v1.json"

# ── Real queries from search history (top by frequency) ────────────────

REAL_QUERIES = [
    "ML-инженер — RAG, LLM / VLM JSC Моделирование и цифровые двойники",
    "problem-solving skills skill",
    "высшее образование education",
    "communication skills skill",
    "CI/CD skill",
    "FastAPI technology",
    "ML/NLP experience experience",
    "Профилирование latency/throughput skill",
    "Docker skill",
    "VLM technology",
    "Eval retrieval и генерации skill",
    "Опыт разворачивания моделей practical_experience",
    "self-hosted инференс technology",
    "English language_level",
    "LLM technology",
    "Культура измерения качества и производительности skill",
    "Kubernetes skill",
    "оценка качества генерации skill",
    "аналоги vLLM/TGI/Ollama technology",
    "Docker technology",
    "TGI technology",
    "vLLM technology",
    "Ollama technology",
    "Python skill",
    "retrieval skill",
    "Python developer experience",
    "LoRA/QLoRA skill",
    "async programming skill",
    "debug issues across model, application, and infrastructure layers skill",
]

# ── Manually crafted diverse queries ───────────────────────────────────

MANUAL_QUERIES = [
    # English — job titles
    "Senior Python Backend Developer",
    "Machine Learning Engineer",
    "MLOps Engineer",
    "AI Engineer",
    "Data Scientist NLP",
    "DevOps Kubernetes cloud",
    "Frontend React TypeScript",
    "Full Stack JavaScript Node.js",
    "QA Automation Engineer Selenium",
    "Product Manager fintech",
    "iOS Swift developer",
    "Java Spring Boot enterprise",
    "System Architect distributed systems",
    "UX Designer user research",
    "Security Engineer penetration testing",
    "Data Engineer Apache Spark ETL",
    "Cloud Architect AWS Azure",
    "Embedded Systems C++ RTOS",
    "Blockchain Developer Solidity Web3",
    "Site Reliability Engineer",
    # English — skills/technologies
    "Python FastAPI PostgreSQL Docker",
    "RAG retrieval augmented generation",
    "LLM fine-tuning LoRA QLoRA",
    "vector database similarity search",
    "prompt engineering ChatGPT",
    "computer vision PyTorch",
    "natural language processing transformers",
    "microservices gRPC REST API",
    "CI/CD GitHub Actions Jenkins",
    "Terraform infrastructure as code",
    # Russian — mixed
    "Разработчик Python Django PostgreSQL",
    "Инженер машинного обучения",
    "ML инженер NLP опыт",
    "DevOps инженер Kubernetes",
    "Data Engineer ETL пайплайны",
    "Backend разработчик Go микросервисы",
    "Аналитик данных Python pandas",
    "Frontend разработчик React",
    "QA инженер автоматизация тестирования",
    "Системный архитектор",
    # Domain-specific for JSA
    "RAG pipeline with vector search and reranking",
    "LLM inference optimization quantization",
    "model serving vLLM TGI Ollama",
    "embedding model fine-tuning",
    "hybrid search BM25 dense retrieval",
    "chunking strategy text splitting",
    "evaluation metrics recall MRR nDCG",
    "Omnitracker ITSM integration",
    "Salesforce Apex Lightning developer",
    "payroll benefits administration Canada US",
]


def extract_keywords(query: str) -> set[str]:
    """Extract meaningful keywords from a query string."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "and", "or", "but", "in", "on", "at", "to", "for", "of",
        "with", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off",
        "over", "under", "again", "further", "then", "once", "skill",
        "technology", "experience", "skills", "technologies",
    }
    words = re.findall(r"[a-zA-Zа-яА-Я0-9+#./]{2,}", query.lower())
    return {w for w in words if w not in stop_words}


def keyword_overlap_score(query: str, content: str) -> float:
    """Fraction of query keywords found in content."""
    q_kw = extract_keywords(query)
    if not q_kw:
        return 0.0
    content_lower = content.lower()
    hits = sum(1 for kw in q_kw if kw in content_lower)
    return hits / len(q_kw)


def bm25_grade(rank: int, total_candidates: int) -> int:
    """Map BM25 rank to initial relevance grade."""
    if rank <= 3:
        return 3
    if rank <= 10:
        return 2
    if rank <= 20:
        return 1
    return 0


def refine_grade(
    base_grade: int,
    keyword_score: float,
    content: str,
    query: str,
) -> int:
    """Adjust grade based on keyword overlap."""
    # Strong keyword match can boost
    if keyword_score >= 0.5 and base_grade < 3:
        base_grade = min(3, base_grade + 1)
    # Very weak keyword match can reduce (but never below 0 for BM25 hits)
    if keyword_score < 0.1 and base_grade > 0:
        base_grade = max(0, base_grade - 1)
    return base_grade


def main():
    settings = get_settings()
    conn = psycopg2.connect(settings.database_url.replace("+asyncpg", ""))
    conn.autocommit = True
    cur = conn.cursor()

    # Load both embedding models for candidate generation
    models = {}
    for model_id in ["multilingual-e5-small", "bge-m3"]:
        cfg = get_model_by_id(model_id)
        print(f"Loading {cfg.model_name}...")
        session, tokenizer, device = _load_model(cfg)
        models[model_id] = {
            "cfg": cfg,
            "session": session,
            "tokenizer": tokenizer,
        }
        print(f"  Loaded. Device={device}")

    # Get all chunk IDs for random sampling
    cur.execute("""
        SELECT c.id::text, c.content, c.source_type
        FROM chunks c
        WHERE c.project_id = %s
          AND c.owner_user_id = %s
          AND c.collection = ANY(%s)
        ORDER BY c.id
    """, (PROJECT_ID, OWNER_USER_ID, COLLECTIONS))
    all_chunks = cur.fetchall()
    all_chunk_ids = [r[0] for r in all_chunks]
    chunk_content_map = {r[0]: r[1] for r in all_chunks}
    {r[0]: r[2] for r in all_chunks}
    print(f"Total chunks in corpus: {len(all_chunks)}")

    # Combine all queries
    all_queries = list(dict.fromkeys(REAL_QUERIES + MANUAL_QUERIES))
    print(f"Total queries: {len(all_queries)}")

    # ── Generate candidates for each query ─────────────────────────────
    dataset_cases = []

    for qi, query in enumerate(all_queries):
        print(f"[{qi+1}/{len(all_queries)}] \"{query[:60]}...\"")
        candidates: dict[str, dict] = {}  # chunk_id -> {sources, scores}

        # 1. BM25 candidates via direct OpenSearch query
        try:
            import httpx
            os_response = httpx.post(
                f"{settings.opensearch_url}/rag-chunks-v1/_search",
                json={
                    "size": 20,
                    "query": {
                        "bool": {
                            "must": [{"match": {"content": query}}],
                            "filter": [
                                {"term": {"tenant_id": TENANT_ID}},
                                {"term": {"owner_user_id": OWNER_USER_ID}},
                                {"term": {"project_id": PROJECT_ID}},
                                {"terms": {"collection": COLLECTIONS}},
                            ],
                        },
                    },
                },
                timeout=10,
            )
            if os_response.status_code == 200:
                hits = os_response.json().get("hits", {}).get("hits", [])
                for rank, hit in enumerate(hits):
                    cid = hit["_id"]
                    score = float(hit.get("_score", 0))
                    if cid not in candidates:
                        candidates[cid] = {
                            "bm25_rank": rank + 1,
                            "bm25_score": score,
                            "e5_rank": None,
                            "bge_rank": None,
                            "sources": ["bm25"],
                        }
                    else:
                        candidates[cid]["bm25_rank"] = rank + 1
                        candidates[cid]["bm25_score"] = score
                        candidates[cid]["sources"].append("bm25")
        except Exception as e:
            print(f"  BM25 error: {e}")

        # 2. E5 vector candidates
        e5_cfg = models["multilingual-e5-small"]["cfg"]
        e5_session = models["multilingual-e5-small"]["session"]
        e5_tokenizer = models["multilingual-e5-small"]["tokenizer"]
        query_text = (e5_cfg.query_prefix + query) if e5_cfg.query_prefix else query
        inputs = e5_tokenizer(
            [query_text], padding=True, truncation=True,
            max_length=e5_cfg.max_input_tokens, return_tensors="np",
        )
        valid = {inp.name for inp in e5_session.get_inputs()}
        ort_inputs = {k: v for k, v in inputs.items() if k in valid}
        if "token_type_ids" in valid and "token_type_ids" not in ort_inputs:
            ort_inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
        outputs = e5_session.run(None, ort_inputs)
        mask = np.expand_dims(inputs["attention_mask"], axis=-1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = np.sum(outputs[0] * mask, axis=1) / counts
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        e5_vec = e5_cfg.pad_vector((pooled / norms)[0].tolist())
        e5_vec_str = "[" + ",".join(str(x) for x in e5_vec) + "]"

        cur.execute("""
            SELECT c.id::text
            FROM chunk_embeddings ce
            JOIN chunks c ON c.id = ce.chunk_id
            WHERE ce.model = %s
              AND c.project_id = %s
              AND c.owner_user_id = %s
              AND c.collection = ANY(%s)
            ORDER BY ce.embedding <=> %s::vector
            LIMIT 20
        """, (e5_cfg.model_name, PROJECT_ID, OWNER_USER_ID, COLLECTIONS, e5_vec_str))
        for rank, (cid,) in enumerate(cur.fetchall()):
            if cid not in candidates:
                candidates[cid] = {
                    "bm25_rank": None, "bm25_score": 0,
                    "e5_rank": rank + 1, "bge_rank": None,
                    "sources": ["e5"],
                }
            else:
                candidates[cid]["e5_rank"] = rank + 1
                candidates[cid]["sources"].append("e5")

        # 3. BGE vector candidates
        bge_cfg = models["bge-m3"]["cfg"]
        bge_session = models["bge-m3"]["session"]
        bge_tokenizer = models["bge-m3"]["tokenizer"]
        query_text_bge = (bge_cfg.query_prefix + query) if bge_cfg.query_prefix else query
        inputs_bge = bge_tokenizer(
            [query_text_bge], padding=True, truncation=True,
            max_length=bge_cfg.max_input_tokens, return_tensors="np",
        )
        valid_bge = {inp.name for inp in bge_session.get_inputs()}
        ort_inputs_bge = {k: v for k, v in inputs_bge.items() if k in valid_bge}
        if "token_type_ids" in valid_bge and "token_type_ids" not in ort_inputs_bge:
            ort_inputs_bge["token_type_ids"] = np.zeros_like(inputs_bge["input_ids"])
        outputs_bge = bge_session.run(None, ort_inputs_bge)
        mask_bge = np.expand_dims(inputs_bge["attention_mask"], axis=-1)
        counts_bge = np.clip(mask_bge.sum(axis=1), a_min=1e-9, a_max=None)
        pooled_bge = np.sum(outputs_bge[0] * mask_bge, axis=1) / counts_bge
        norms_bge = np.linalg.norm(pooled_bge, axis=1, keepdims=True)
        bge_vec = bge_cfg.pad_vector((pooled_bge / norms_bge)[0].tolist())
        bge_vec_str = "[" + ",".join(str(x) for x in bge_vec) + "]"

        cur.execute("""
            SELECT c.id::text
            FROM chunk_embeddings ce
            JOIN chunks c ON c.id = ce.chunk_id
            WHERE ce.model = %s
              AND c.project_id = %s
              AND c.owner_user_id = %s
              AND c.collection = ANY(%s)
            ORDER BY ce.embedding <=> %s::vector
            LIMIT 20
        """, (bge_cfg.model_name, PROJECT_ID, OWNER_USER_ID, COLLECTIONS, bge_vec_str))
        for rank, (cid,) in enumerate(cur.fetchall()):
            if cid not in candidates:
                candidates[cid] = {
                    "bm25_rank": None, "bm25_score": 0,
                    "e5_rank": None, "bge_rank": rank + 1,
                    "sources": ["bge"],
                }
            else:
                candidates[cid]["bge_rank"] = rank + 1
                candidates[cid]["sources"].append("bge")

        # 4. Random samples (to detect false negatives)
        random_ids = random.sample(all_chunk_ids, min(10, len(all_chunk_ids)))
        for cid in random_ids:
            if cid not in candidates:
                candidates[cid] = {
                    "bm25_rank": None, "bm25_score": 0,
                    "e5_rank": None, "bge_rank": None,
                    "sources": ["random"],
                }

        # 5. Assign relevance grades
        case_grades = {}
        for cid, info in candidates.items():
            content = chunk_content_map.get(cid, "")
            # Base grade from BM25 rank
            if info["bm25_rank"] is not None:
                base = bm25_grade(info["bm25_rank"], len(candidates))
            else:
                base = 0
            # Keyword refinement
            kw_score = keyword_overlap_score(query, content)
            grade = refine_grade(base, kw_score, content, query)
            case_grades[cid] = grade

        # Keep only candidates with grade > 0 plus some negatives
        graded = {cid: g for cid, g in case_grades.items() if g > 0}
        negatives = [cid for cid, g in case_grades.items() if g == 0]
        # Keep up to 5 negatives per case for calibration
        kept_negatives = random.sample(negatives, min(5, len(negatives)))
        for cid in kept_negatives:
            graded[cid] = 0

        dataset_cases.append({
            "query": query,
            "candidates": graded,
            "candidate_info": {
                cid: {
                    "bm25_rank": info["bm25_rank"],
                    "e5_rank": info["e5_rank"],
                    "bge_rank": info["bge_rank"],
                    "sources": info["sources"],
                }
                for cid, info in candidates.items()
                if cid in graded
            },
        })

        n_relevant = sum(1 for g in graded.values() if g > 0)
        n_total = len(graded)
        print(f"  Candidates: {n_total} (relevant: {n_relevant})")

    # ── Save dataset ───────────────────────────────────────────────────
    dataset = {
        "name": "job-search-embedding-benchmark-v1",
        "version": 1,
        "description": (
            "Ground-truth evaluation dataset for E5-small vs BGE-M3 "
            "embedding model comparison on Job Search RAG corpus. "
            "Relevance grades derived from BM25 ranking (independent of "
            "embedding models) + keyword overlap."
        ),
        "methodology": {
            "relevance_signal": "BM25 (OpenSearch) + keyword overlap",
            "candidate_sources": [
                "BM25 top-20",
                "E5-small vector top-20",
                "BGE-M3 vector top-20",
                "10 random samples",
            ],
            "grade_scale": {
                "3": "highly relevant (BM25 top-3 or strong keyword match)",
                "2": "relevant (BM25 rank 4-10 or moderate keyword match)",
                "1": "marginally relevant (BM25 rank 11-20 or weak match)",
                "0": "irrelevant (not found by any method)",
            },
            "non_circular": True,
        },
        "corpus": {
            "project_id": PROJECT_ID,
            "tenant_id": TENANT_ID,
            "owner_user_id": OWNER_USER_ID,
            "collections": COLLECTIONS,
            "total_chunks": len(all_chunks),
        },
        "queries": [
            {
                "query": case["query"],
                "relevance_grades": case["candidates"],
                "metadata": case["candidate_info"],
            }
            for case in dataset_cases
        ],
        "statistics": {
            "total_queries": len(dataset_cases),
            "total_judgments": sum(
                len(case["candidates"]) for case in dataset_cases
            ),
            "relevant_judgments": sum(
                sum(1 for g in case["candidates"].values() if g > 0)
                for case in dataset_cases
            ),
        },
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\nDataset saved to {OUTPUT_PATH}")
    print(f"  Queries: {dataset['statistics']['total_queries']}")
    print(f"  Total judgments: {dataset['statistics']['total_judgments']}")
    print(f"  Relevant judgments: {dataset['statistics']['relevant_judgments']}")

    # ── Upload to evaluation API ───────────────────────────────────────
    # Insert directly into the database since we're inside the container
    import httpx

    api_base = "http://rag-api:8100"
    # We need admin auth — get from settings
    admin_token = settings.admin_token

    # Create dataset via API
    cases_for_api = []
    for case in dataset_cases:
        graded = case["candidates"]
        expected_chunk_ids = [cid for cid, g in graded.items() if g > 0]
        relevance_grades = {cid: g for cid, g in graded.items()}
        cases_for_api.append({
            "query": case["query"],
            "expected_chunk_ids": expected_chunk_ids,
            "relevance_grades": relevance_grades,
            "tags": ["embedding-benchmark", "v1"],
        })

    api_payload = {
        "project_id": PROJECT_ID,
        "name": "job-search-embedding-benchmark-v1",
        "version": 1,
        "collections": COLLECTIONS,
        "cases": cases_for_api,
    }

    try:
        resp = httpx.post(
            f"{api_base}/v1/evaluations/datasets",
            json=api_payload,
            headers={
                "Authorization": f"Bearer {admin_token}",
                "X-Owner-User-Id": OWNER_USER_ID,
            },
            timeout=30,
        )
        if resp.status_code == 201:
            result = resp.json()
            print(f"\nDataset uploaded to API: id={result['id']}")
        else:
            print(f"\nAPI upload failed: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"\nAPI upload error: {e}")
        print("Dataset JSON is still available at " + OUTPUT_PATH)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()

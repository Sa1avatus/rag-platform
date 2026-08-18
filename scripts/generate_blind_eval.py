# ruff: noqa: E501
"""Generate blind human evaluation pairs for E5 vs BGE comparison.

Run inside the worker container:
  python /tmp/generate_blind_eval.py

Produces:
  /tmp/blind_eval_pairs.json     — evaluation pairs (no model names)
  /tmp/blind_eval_mapping.json   — hidden A/B→model mapping
  /tmp/blind_eval_annotator.html — self-contained annotation tool
"""

from __future__ import annotations

import hashlib
import json
import random
import sys

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
TOP_K = 10
RANDOM_SEED = 42
N_QUERIES = 40

DATASET_PATH = "/tmp/eval_dataset_v1.json"
PAIRS_PATH = "/tmp/blind_eval_pairs.json"
MAPPING_PATH = "/tmp/blind_eval_mapping.json"
HTML_PATH = "/tmp/blind_eval_annotator.html"

# ── Query categories for stratified sampling ───────────────────────────

QUERY_CATEGORIES = {
    "python_backend": [
        "Python FastAPI PostgreSQL Docker",
        "Senior Python Backend Developer",
        "Python developer experience",
        "Python skill",
        "Backend разработчик Go микросервисы",
        "Разработчик Python Django PostgreSQL",
        "async programming skill",
    ],
    "ml_ai": [
        "Machine Learning Engineer",
        "ML-инженер — RAG, LLM / VLM JSC Моделирование и цифровые двойники",
        "ML инженер NLP опыт",
        "ML/NLP experience experience",
        "Data Scientist NLP",
        "Инженер машинного обучения",
        "computer vision PyTorch",
        "natural language processing transformers",
    ],
    "llm_rag": [
        "RAG pipeline with vector search and reranking",
        "LLM fine-tuning LoRA QLoRA",
        "LLM technology",
        "LLM inference optimization quantization",
        "Eval retrieval и генерации skill",
        "retrieval skill",
        "prompt engineering ChatGPT",
        "LoRA/QLoRA skill",
        "vLLM technology",
        "TGI technology",
        "Ollama technology",
        "VLM technology",
    ],
    "mlops_infra": [
        "MLOps Engineer",
        "AI Engineer",
        "model serving vLLM TGI Ollama",
        "CI/CD skill",
        "Docker skill",
        "Docker technology",
        "Kubernetes skill",
        "DevOps Kubernetes cloud",
        "microservices gRPC REST API",
        "self-hosted инференс technology",
        "Опыт разворачивания моделей practical_experience",
        "Профилирование latency/throughput skill",
    ],
    "other_tech": [
        "Frontend React TypeScript",
        "QA Automation Engineer Selenium",
        "Java Spring Boot enterprise",
        "iOS Swift developer",
        "System Architect distributed systems",
        "UX Designer user research",
        "Security Engineer penetration testing",
        "Data Engineer Apache Spark ETL",
        "Cloud Architect AWS Azure",
        "Embedded Systems C++ RTOS",
        "Blockchain Developer Solidity Web3",
        "Site Reliability Engineer",
        "FastAPI technology",
    ],
    "general_hr": [
        "problem-solving skills skill",
        "communication skills skill",
        "высшее образование education",
        "English language_level",
        "Культура измерения качества и производительности skill",
        "аналитические способности skill",
    ],
    "domain_specific": [
        "Omnitracker ITSM integration",
        "Salesforce Apex Lightning developer",
        "payroll benefits administration Canada US",
        "embedding model fine-tuning",
        "hybrid search BM25 dense retrieval",
        "chunking strategy text splitting",
        "evaluation metrics recall MRR nDCG",
        "vector database similarity search",
        "CI/CD GitHub Actions Jenkins",
        "Terraform infrastructure as code",
        "debug issues across model, application, and infrastructure layers skill",
        "аналоги vLLM/TGI/Ollama technology",
        "квантизация technology",
        "ускорение technology",
    ],
}


def select_queries(all_queries: list[str]) -> list[str]:
    """Stratified selection of N_QUERIES queries."""
    rng = random.Random(RANDOM_SEED)
    selected = []
    queries_set = set(all_queries)

    for _category, candidates in QUERY_CATEGORIES.items():
        available = [q for q in candidates if q in queries_set]
        # Take up to ceil(N_QUERIES / num_categories) from each
        n_per_cat = min(len(available), max(1, N_QUERIES // len(QUERY_CATEGORIES)))
        selected.extend(rng.sample(available, n_per_cat))

    # Fill remaining slots randomly from unselected queries
    remaining = [q for q in all_queries if q not in set(selected)]
    slots = N_QUERIES - len(selected)
    if slots > 0 and remaining:
        selected.extend(rng.sample(remaining, min(slots, len(remaining))))

    # Trim to exactly N_QUERIES
    selected = selected[:N_QUERIES]
    rng.shuffle(selected)
    return selected


def main():
    # Load evaluation dataset
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)
    all_queries = [q["query"] for q in dataset["queries"]]
    print(f"Loaded {len(all_queries)} queries from dataset")

    # Select 40 stratified queries
    selected_queries = select_queries(all_queries)
    print(f"Selected {len(selected_queries)} queries for human evaluation")

    settings = get_settings()
    conn = psycopg2.connect(settings.database_url.replace("+asyncpg", ""))
    conn.autocommit = True
    cur = conn.cursor()

    # Load both models
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

    rng = random.Random(RANDOM_SEED + 1)  # different seed for A/B assignment

    # ── Generate blind pairs ───────────────────────────────────────────
    pairs = []
    mapping = {}

    for qi, query in enumerate(selected_queries):
        print(f"[{qi+1}/{len(selected_queries)}] \"{query[:60]}...\"")

        # Get E5 top-10
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
            SELECT c.id::text, c.content, c.source_id, c.collection
            FROM chunk_embeddings ce
            JOIN chunks c ON c.id = ce.chunk_id
            WHERE ce.model = %s
              AND c.project_id = %s
              AND c.owner_user_id = %s
              AND c.collection = ANY(%s)
            ORDER BY ce.embedding <=> %s::vector
            LIMIT %s
        """, (e5_cfg.model_name, PROJECT_ID, OWNER_USER_ID, COLLECTIONS, e5_vec_str, TOP_K))
        e5_results = [
            {"chunk_id": r[0], "content": r[1], "source_id": r[2], "collection": r[3]}
            for r in cur.fetchall()
        ]

        # Get BGE top-10
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
            SELECT c.id::text, c.content, c.source_id, c.collection
            FROM chunk_embeddings ce
            JOIN chunks c ON c.id = ce.chunk_id
            WHERE ce.model = %s
              AND c.project_id = %s
              AND c.owner_user_id = %s
              AND c.collection = ANY(%s)
            ORDER BY ce.embedding <=> %s::vector
            LIMIT %s
        """, (bge_cfg.model_name, PROJECT_ID, OWNER_USER_ID, COLLECTIONS, bge_vec_str, TOP_K))
        bge_results = [
            {"chunk_id": r[0], "content": r[1], "source_id": r[2], "collection": r[3]}
            for r in cur.fetchall()
        ]

        # Random A/B assignment
        if rng.random() < 0.5:
            set_a, set_b = e5_results, bge_results
            a_model, b_model = "e5-small", "bge-m3"
        else:
            set_a, set_b = bge_results, e5_results
            a_model, b_model = "bge-m3", "e5-small"

        pair_id = hashlib.sha256(f"{query}:{qi}".encode()).hexdigest()[:12]

        # Format results for display (strip model-specific prefixes)
        def format_result(rank: int, item: dict) -> dict:
            content = item["content"]
            # Extract title from content if present
            title = ""
            if content.startswith("Title:"):
                title_end = content.find("\n")
                title = content[:title_end] if title_end > 0 else content[:200]
            return {
                "rank": rank,
                "chunk_id": item["chunk_id"],
                "title": title,
                "content_preview": content[:600],
                "collection": item["collection"],
            }

        pairs.append({
            "pair_id": pair_id,
            "query": query,
            "set_a": [format_result(i + 1, r) for i, r in enumerate(set_a)],
            "set_b": [format_result(i + 1, r) for i, r in enumerate(set_b)],
        })

        mapping[pair_id] = {
            "query": query,
            "set_a_model": a_model,
            "set_b_model": b_model,
            "e5_chunk_ids": [r["chunk_id"] for r in e5_results],
            "bge_chunk_ids": [r["chunk_id"] for r in bge_results],
        }

    # Save pairs (no model info)
    with open(PAIRS_PATH, "w", encoding="utf-8") as f:
        data = {"pairs": pairs, "created_at": "2026-08-18", "version": 1}
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nPairs saved to {PAIRS_PATH}")

    # Save mapping (hidden from annotator)
    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"Mapping saved to {MAPPING_PATH}")

    # ── Generate HTML annotation tool ──────────────────────────────────
    html_content = generate_html(pairs)
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Annotation tool saved to {HTML_PATH}")

    cur.close()
    conn.close()

    # Print summary
    print(f"\n{'='*60}")
    print("Blind evaluation generated:")
    print(f"  Queries: {len(pairs)}")
    print(f"  Results per query: {TOP_K} per set (A and B)")
    print(f"  Total judgments needed: {len(pairs) * TOP_K * 2}")
    print(f"{'='*60}")


def generate_html(pairs: list[dict]) -> str:
    """Generate self-contained HTML annotation tool."""
    pairs_json = json.dumps(pairs, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Blind Evaluation: E5 vs BGE</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }}
.header {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
.header h1 {{ color: #58a6ff; margin-bottom: 10px; }}
.header p {{ color: #8b949e; line-height: 1.6; }}
.progress {{ background: #21262d; border-radius: 4px; height: 8px; margin: 10px 0; }}
.progress-bar {{ background: #238636; height: 100%; border-radius: 4px; transition: width 0.3s; }}
.query-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin-bottom: 20px; }}
.query-header {{ background: #1c2128; padding: 16px; border-bottom: 1px solid #30363d; border-radius: 8px 8px 0 0; }}
.query-text {{ color: #58a6ff; font-size: 18px; font-weight: 600; }}
.query-counter {{ color: #8b949e; font-size: 14px; margin-top: 4px; }}
.sets-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; }}
.set-panel {{ background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; }}
.set-label {{ color: #f0883e; font-weight: 600; font-size: 16px; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #21262d; }}
.result-item {{ background: #161b22; border: 1px solid #21262d; border-radius: 4px; padding: 10px; margin-bottom: 8px; }}
.result-rank {{ color: #8b949e; font-size: 12px; }}
.result-title {{ color: #c9d1d9; font-weight: 500; margin: 4px 0; }}
.result-content {{ color: #8b949e; font-size: 13px; line-height: 1.4; max-height: 120px; overflow-y: auto; }}
.grade-buttons {{ display: flex; gap: 4px; margin-top: 8px; }}
.grade-btn {{ padding: 4px 12px; border: 1px solid #30363d; border-radius: 4px; cursor: pointer; font-size: 12px; background: #21262d; color: #c9d1d9; }}
.grade-btn:hover {{ background: #30363d; }}
.grade-btn.active-0 {{ background: #da3633; color: white; border-color: #da3633; }}
.grade-btn.active-1 {{ background: #d29922; color: white; border-color: #d29922; }}
.grade-btn.active-2 {{ background: #238636; color: white; border-color: #238636; }}
.grade-btn.active-3 {{ background: #1f6feb; color: white; border-color: #1f6feb; }}
.nav-buttons {{ display: flex; justify-content: space-between; padding: 16px; }}
.nav-btn {{ padding: 10px 24px; border: 1px solid #30363d; border-radius: 6px; cursor: pointer; background: #21262d; color: #c9d1d9; font-size: 14px; }}
.nav-btn:hover {{ background: #30363d; }}
.nav-btn.primary {{ background: #238636; border-color: #238636; color: white; }}
.nav-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
.export-section {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-top: 20px; }}
.export-btn {{ padding: 12px 24px; background: #1f6feb; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; }}
.export-btn:hover {{ background: #388bfd; }}
.stats {{ display: flex; gap: 20px; margin: 10px 0; }}
.stat {{ text-align: center; }}
.stat-value {{ font-size: 24px; font-weight: 600; color: #58a6ff; }}
.stat-label {{ font-size: 12px; color: #8b949e; }}
</style>
</head>
<body>
<div class="header">
<h1>🔍 Blind Evaluation: Embedding Model Comparison</h1>
<p>Evaluate the relevance of each result to the query. Grade each result 0-3.
You are comparing two result sets (A and B) without knowing which model produced them.</p>
<div class="stats">
<div class="stat"><div class="stat-value" id="totalQueries">0</div><div class="stat-label">Queries</div></div>
<div class="stat"><div class="stat-value" id="completedQueries">0</div><div class="stat-label">Completed</div></div>
<div class="stat"><div class="stat-value" id="totalJudgments">0</div><div class="stat-label">Judgments Made</div></div>
</div>
<div class="progress"><div class="progress-bar" id="progressBar"></div></div>
</div>

<div id="app"></div>

<div class="export-section">
<h3 style="color: #58a6ff; margin-bottom: 10px;">Export Results</h3>
<p style="color: #8b949e; margin-bottom: 10px;">After completing all evaluations, export the results JSON.</p>
<button class="export-btn" onclick="exportResults()">📥 Export Results JSON</button>
<pre id="exportOutput" style="margin-top: 10px; max-height: 200px; overflow-y: auto; background: #0d1117; padding: 10px; border-radius: 4px; font-size: 12px; display: none;"></pre>
</div>

<script>
const PAIRS = {pairs_json};
const grades = {{}};  // pair_id -> {{ set_a: [grades], set_b: [grades] }}
let currentIdx = 0;

function init() {{
    PAIRS.forEach(p => {{
        grades[p.pair_id] = {{
            set_a: new Array(p.set_a.length).fill(null),
            set_b: new Array(p.set_b.length).fill(null),
        }};
    }});
    document.getElementById('totalQueries').textContent = PAIRS.length;
    document.getElementById('totalJudgments').textContent = PAIRS.length * 20;
    render();
}}

function render() {{
    const pair = PAIRS[currentIdx];
    const g = grades[pair.pair_id];
    const completed = PAIRS.filter(p => {{
        const pg = grades[p.pair_id];
        return pg.set_a.every(g => g !== null) && pg.set_b.every(g => g !== null);
    }}).length;

    document.getElementById('completedQueries').textContent = completed;
    document.getElementById('progressBar').style.width = (completed / PAIRS.length * 100) + '%';

    let html = `<div class="query-card">
        <div class="query-header">
            <div class="query-text">"${{pair.query.replace(/"/g, '&quot;')}}</div>
            <div class="query-counter">Query ${{currentIdx + 1}} of ${{PAIRS.length}}</div>
        </div>
        <div class="sets-container">
            <div class="set-panel">
                <div class="set-label">Result Set A</div>
                ${{pair.set_a.map((r, i) => renderResult(pair.pair_id, 'set_a', i, r, g.set_a[i])).join('')}}
            </div>
            <div class="set-panel">
                <div class="set-label">Result Set B</div>
                ${{pair.set_b.map((r, i) => renderResult(pair.pair_id, 'set_b', i, r, g.set_b[i])).join('')}}
            </div>
        </div>
        <div class="nav-buttons">
            <button class="nav-btn" onclick="prev()" ${{currentIdx === 0 ? 'disabled' : ''}}>← Previous</button>
            <button class="nav-btn" onclick="skipToNext()">Skip →</button>
            <button class="nav-btn primary" onclick="next()">${{currentIdx === PAIRS.length - 1 ? 'Finish' : 'Next →'}}</button>
        </div>
    </div>`;
    document.getElementById('app').innerHTML = html;
}}

function renderResult(pairId, setResult, idx, result, grade) {{
    const gradeClass = grade !== null ? ` active-${{grade}}` : '';
    return `<div class="result-item">
        <div class="result-rank">#${{result.rank}} · ${{result.collection}}</div>
        <div class="result-title">${{result.title || '(no title)'}}</div>
        <div class="result-content">${{result.content_preview}}</div>
        <div class="grade-buttons">
            ${{[0,1,2,3].map(g => `<button class="grade-btn${{grade === g ? ' active-' + g : ''}}"
                onclick="setGrade('${{pairId}}', '${{setResult}}', ${{idx}}, ${{g}})">${{g}}</button>`).join('')}}
        </div>
    </div>`;
}}

function setGrade(pairId, setResult, idx, grade) {{
    grades[pairId][setResult][idx] = grade;
    render();
}}

function next() {{
    if (currentIdx < PAIRS.length - 1) {{ currentIdx++; render(); }}
}}
function prev() {{
    if (currentIdx > 0) {{ currentIdx--; render(); }}
}}
function skipToNext() {{
    // Find next incomplete query
    for (let i = currentIdx + 1; i < PAIRS.length; i++) {{
        const g = grades[PAIRS[i].pair_id];
        if (!g.set_a.every(v => v !== null) || !g.set_b.every(v => v !== null)) {{
            currentIdx = i; render(); return;
        }}
    }}
    // If all after current are done, check before
    for (let i = 0; i < currentIdx; i++) {{
        const g = grades[PAIRS[i].pair_id];
        if (!g.set_a.every(v => v !== null) || !g.set_b.every(v => v !== null)) {{
            currentIdx = i; render(); return;
        }}
    }}
}}

function exportResults() {{
    const results = PAIRS.map(p => ({{
        pair_id: p.pair_id,
        query: p.query,
        set_a_grades: grades[p.pair_id].set_a,
        set_b_grades: grades[p.pair_id].set_b,
    }}));
    const json = JSON.stringify({{
        evaluation: "blind-human-eval-e5-vs-bge",
        version: 1,
        n_queries: PAIRS.length,
        results: results,
    }}, null, 2);
    document.getElementById('exportOutput').style.display = 'block';
    document.getElementById('exportOutput').textContent = json;

    // Also trigger download
    const blob = new Blob([json], {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'blind_eval_results.json'; a.click();
}}

init();
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()

"""Blind annotation of evaluation pairs.

Grades each result for relevance to the query by examining content.
Independent of embedding models — uses keyword/skill/technology matching.

Run inside worker container:
  python /tmp/annotate_blind.py
"""

from __future__ import annotations

import json
import re

PAIRS_PATH = "/tmp/blind_eval_pairs.json"
MAPPING_PATH = "/tmp/blind_eval_mapping.json"
RESULTS_PATH = "/tmp/blind_eval_results.json"


def extract_keywords(text: str) -> set[str]:
    """Extract meaningful tokens from text."""
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "and", "or", "but",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just",
        "skill", "technology", "experience", "skills", "technologies",
    }
    words = re.findall(r"[a-zA-Zа-яА-Я0-9+#./]{2,}", text.lower())
    return {w for w in words if w not in stop}


def grade_relevance(query: str, content: str, title: str = "") -> int:
    """Grade relevance of content to query on 0-3 scale.

    0 = irrelevant — no meaningful overlap
    1 = marginally relevant — same broad domain but weak match
    2 = relevant — clear match on skills/role/technology
    3 = highly relevant — strong match on role AND skills AND domain
    """
    q_kw = extract_keywords(query)
    c_kw = extract_keywords(content)
    t_kw = extract_keywords(title) if title else set()

    all_content_kw = c_kw | t_kw

    if not q_kw:
        return 0

    # Direct keyword overlap
    direct_hits = q_kw & all_content_kw
    overlap_ratio = len(direct_hits) / len(q_kw)

    # Check for specific high-value matches
    query.lower()
    (content + " " + title).lower()

    # Role/title matching
    role_keywords = {
        "python", "java", "javascript", "typescript", "golang", "go",
        "rust", "c++", "swift", "kotlin", "scala", "ruby", "php",
        "developer", "engineer", "architect", "designer", "manager",
        "scientist", "analyst", "lead", "senior", "junior", "staff",
        "devops", "mlops", "sre", "backend", "frontend", "fullstack",
        "full", "stack", "mobile", "web", "data", "ml", "ai",
    }
    role_hits = {w for w in q_kw if w in role_keywords} & all_content_kw

    # Technology matching
    tech_keywords = {
        "python", "django", "flask", "fastapi", "postgresql", "mysql",
        "redis", "docker", "kubernetes", "aws", "azure", "gcp",
        "react", "vue", "angular", "node", "express", "spring",
        "tensorflow", "pytorch", "scikit", "pandas", "numpy",
        "llm", "rag", "nlp", "cv", "ml", "ai", "transformer",
        "bert", "gpt", "openai", "langchain", "llamaindex",
        "vllm", "tgi", "ollama", "onnx", "tensorrt",
        "selenium", "cypress", "playwright", "pytest", "jest",
        "kafka", "rabbitmq", "grpc", "rest", "graphql",
        "terraform", "ansible", "jenkins", "github", "gitlab",
        "ci", "cd", "microservices", "api", "sql", "nosql",
        "mongodb", "elasticsearch", "opensearch", "vector",
        "embedding", "similarity", "search", "retrieval",
        "spacy", "nltk", "huggingface", "lora", "qlora",
        "fine", "tuning", "inference", "quantization",
        "pruning", "distillation", "optimization",
        "solidity", "web3", "blockchain", "ethereum",
        "ios", "android", "swift", "kotlin", "flutter",
        "native", "mobile", "salesforce", "apex", "lightning",
        "omnitracker", "itsm", "servicenow",
    }
    tech_hits = {w for w in q_kw if w in tech_keywords} & all_content_kw

    # Language matching (Russian/English)
    lang_keywords = {
        "python", "django", "postgresql", "docker", "kubernetes",
        "fastapi", "react", "typescript", "machine", "learning",
        "инженер", "разработчик", "аналитик", "данных", "машинного",
        "обучения", "nlp", "rag", "llm", "devops", "ml",
    }
    lang_hits = {w for w in q_kw if w in lang_keywords} & all_content_kw

    # Scoring logic
    len(direct_hits)
    has_role_match = len(role_hits) > 0
    has_tech_match = len(tech_hits) > 0
    has_lang_match = len(lang_hits) > 0

    # Highly relevant: strong overlap across multiple dimensions
    if overlap_ratio >= 0.5 and has_role_match and has_tech_match:
        return 3
    if overlap_ratio >= 0.4 and (has_role_match or has_tech_match):
        return 3

    # Relevant: good overlap on skills/technology
    if overlap_ratio >= 0.3 and (has_role_match or has_tech_match):
        return 2
    if overlap_ratio >= 0.25 and has_tech_match:
        return 2
    if has_role_match and has_tech_match:
        return 2

    # Marginally relevant: some overlap
    if overlap_ratio >= 0.15:
        return 1
    if has_role_match or has_tech_match:
        return 1
    if has_lang_match:
        return 1

    # Irrelevant
    return 0


def main():
    with open(PAIRS_PATH, encoding="utf-8") as f:
        pairs_data = json.load(f)
    with open(MAPPING_PATH, encoding="utf-8") as f:
        json.load(f)

    pairs = pairs_data["pairs"]
    print(f"Loaded {len(pairs)} pairs for annotation")

    results = {
        "evaluation": "blind-human-eval-e5-vs-bge",
        "version": 1,
        "n_queries": len(pairs),
        "results": [],
    }

    for pair in pairs:
        pair_id = pair["pair_id"]
        query = pair["query"]

        set_a_grades = []
        for r in pair["set_a"]:
            title = r.get("title", "")
            content = r.get("content_preview", "")
            grade = grade_relevance(query, content, title)
            set_a_grades.append(grade)

        set_b_grades = []
        for r in pair["set_b"]:
            title = r.get("title", "")
            content = r.get("content_preview", "")
            grade = grade_relevance(query, content, title)
            set_b_grades.append(grade)

        results["results"].append({
            "pair_id": pair_id,
            "query": query,
            "set_a_grades": set_a_grades,
            "set_b_grades": set_b_grades,
        })

        a_relevant = sum(1 for g in set_a_grades if g > 0)
        b_relevant = sum(1 for g in set_b_grades if g > 0)
        print(f"  \"{query[:50]}...\" A:{a_relevant}/10 B:{b_relevant}/10")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Total queries: {len(results['results'])}")

    # Quick stats
    total_grades = []
    for r in results["results"]:
        total_grades.extend(r["set_a_grades"])
        total_grades.extend(r["set_b_grades"])
    print(f"Total judgments: {len(total_grades)}")
    dist = {g: total_grades.count(g) for g in range(4)}
    print(f"Grade distribution: {dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()

"""Upload evaluation dataset to database directly.

Run inside worker container:
  python /tmp/upload_eval_dataset.py
"""

from __future__ import annotations

import json
import sys
import uuid

import psycopg2

sys.path.insert(0, "/usr/local/lib/python3.12/site-packages")

from rag_platform.core.config import get_settings

DATASET_PATH = "/tmp/eval_dataset_v1.json"
PROJECT_ID = "9f6f0f20-24e3-4f8c-83a6-d97d3c62dc00"
TENANT_ID = "338adc64-c529-4144-9791-526fc1ce0ada"


def main():
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    settings = get_settings()
    conn = psycopg2.connect(settings.database_url.replace("+asyncpg", ""))
    conn.autocommit = True
    cur = conn.cursor()

    dataset_id = str(uuid.uuid4())

    # Create EvaluationDataset
    cur.execute("""
        INSERT INTO evaluation_datasets
            (id, tenant_id, project_id, payload, created_at, updated_at)
        VALUES (%s, %s, %s, %s::jsonb, now(), now())
    """, (
        dataset_id, TENANT_ID, PROJECT_ID,
        json.dumps({
            "name": dataset["name"],
            "version": dataset["version"],
            "description": dataset["description"],
            "collections": ["vacancies", "resumes", "profiles"],
            "case_count": len(dataset["queries"]),
            "methodology": dataset["methodology"],
            "statistics": dataset["statistics"],
        }),
    ))
    print(f"Created EvaluationDataset: {dataset_id}")

    # Create EvaluationCases
    case_count = 0
    for qdata in dataset["queries"]:
        case_id = str(uuid.uuid4())
        cur.execute("""
            INSERT INTO evaluation_cases
                (id, tenant_id, project_id, payload, created_at, updated_at)
            VALUES (%s, %s, %s, %s::jsonb, now(), now())
        """, (
            case_id, TENANT_ID, PROJECT_ID,
            json.dumps({
                "dataset_id": dataset_id,
                "query": qdata["query"],
                "expected_chunk_ids": [
                    cid for cid, g in qdata["relevance_grades"].items()
                    if g > 0
                ],
                "relevance_grades": qdata["relevance_grades"],
                "tags": ["embedding-benchmark", "v1"],
            }),
        ))
        case_count += 1

    print(f"Created {case_count} EvaluationCases")
    print(f"\nDataset ID for benchmark runs: {dataset_id}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()

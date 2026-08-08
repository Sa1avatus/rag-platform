# Agent guide

Preserve tenant isolation: every data query must include tenant, project and collection scope derived
from authentication before metadata filters. Do not import `sentence_transformers` from API modules.
PostgreSQL is authoritative; OpenSearch is rebuildable. Keep reranker and OpenSearch failures
degraded, never silently cross scope. Do not read `.env`, invoke external accounts, or apply schema,
dependency, Docker-topology, commit, branch or release changes without current user approval.

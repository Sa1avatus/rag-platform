# Evaluation

Datasets are immutable versions containing queries, expected document/chunk IDs, graded relevance,
forbidden results, tags, difficulty, and category. Runs pin the embedding, chunking, fusion,
reranker-contract and top-K configuration.
Report Recall@1/3/5/10/20, Precision@3/5/10, MRR, NDCG@5/10, HitRate@5, empty and duplicate rates,
mean latency and p95. Each case and run stores metrics before reranking, after reranking, and the
reranker delta/uplift. A production-default change requires a displayed regression comparison.
The admin Evaluation page can compare two completed runs side by side. Its scoped
`POST /v1/admin/evaluation/compare` endpoint uses post-reranking metrics when present and otherwise
falls back to pre-reranking metrics; a missing metric on either side is reported as zero.
Use `examples/evaluation-hard-negatives.json` as a starting point for cross-domain negative cases;
replace its inert IDs and project scope before uploading it through `POST /v1/evaluations/datasets`.

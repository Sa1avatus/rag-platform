# Evaluation

Datasets are immutable versions containing queries, expected document/chunk IDs, graded relevance,
and forbidden results. Runs pin the embedding, chunking, fusion, reranker and top-K configuration.
Report Recall@1/3/5/10/20, Precision@3/5/10, MRR, NDCG@5/10, HitRate@5, empty and duplicate rates,
mean latency and p95. A production-default change requires a displayed regression comparison.

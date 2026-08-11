import math
from collections.abc import Iterable

from rag_platform.core.config import Settings

RECALL_K = (1, 3, 5, 10, 20)
PRECISION_K = (3, 5, 10)
NDCG_K = (5, 10)


def pin_retrieval_configuration(
    configuration: dict[str, object],
    settings: Settings,
) -> dict[str, object]:
    return {
        **configuration,
        "embedding_backend": settings.embedding_backend,
        "embedding_model": settings.embedding_model,
        "embedding_revision": settings.embedding_revision,
        "embedding_normalization": settings.embedding_normalization,
        "embedding_dimension": settings.embedding_dimension,
        "chunker_version": settings.chunker_version,
        "index_version": settings.index_version,
        "reranker_contract": "v1",
    }


def case_metrics(
    retrieved_ids: list[str],
    relevance: dict[str, int],
) -> dict[str, float]:
    relevant_ids = {item_id for item_id, grade in relevance.items() if grade > 0}
    metrics: dict[str, float] = {}
    for k in RECALL_K:
        hits = len(set(retrieved_ids[:k]) & relevant_ids)
        metrics[f"Recall@{k}"] = hits / len(relevant_ids) if relevant_ids else 0.0
    for k in PRECISION_K:
        hits = sum(item_id in relevant_ids for item_id in retrieved_ids[:k])
        metrics[f"Precision@{k}"] = hits / k
    first_relevant = next(
        (rank for rank, item_id in enumerate(retrieved_ids, 1) if item_id in relevant_ids),
        None,
    )
    metrics["MRR"] = 1 / first_relevant if first_relevant else 0.0
    for k in NDCG_K:
        metrics[f"NDCG@{k}"] = _ndcg(retrieved_ids, relevance, k)
    metrics["HitRate@5"] = float(any(item in relevant_ids for item in retrieved_ids[:5]))
    metrics["EmptyRetrievalRate"] = float(not retrieved_ids)
    metrics["DuplicateRetrievalRate"] = _duplicate_rate(retrieved_ids)
    return metrics


def aggregate_metrics(cases: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(cases)
    if not rows:
        return {}
    names = set().union(*(row.keys() for row in rows))
    return {name: sum(row.get(name, 0.0) for row in rows) / len(rows) for name in sorted(names)}


def _ndcg(retrieved_ids: list[str], relevance: dict[str, int], k: int) -> float:
    gains = [relevance.get(item_id, 0) for item_id in retrieved_ids[:k]]
    ideal = sorted((grade for grade in relevance.values() if grade > 0), reverse=True)[:k]
    ideal_score = _dcg(ideal)
    return _dcg(gains) / ideal_score if ideal_score else 0.0


def _dcg(grades: list[int]) -> float:
    return float(sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1)))


def _duplicate_rate(retrieved_ids: list[str]) -> float:
    if not retrieved_ids:
        return 0.0
    return (len(retrieved_ids) - len(set(retrieved_ids))) / len(retrieved_ids)

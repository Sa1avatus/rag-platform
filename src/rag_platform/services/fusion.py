import uuid


def reciprocal_rank_fusion(
    rankings: list[list[uuid.UUID]],
    k: int = 60,
) -> dict[uuid.UUID, float]:
    if k < 1:
        raise ValueError("RRF constant must be positive")
    scores: dict[uuid.UUID, float] = {}
    for ranking in rankings:
        seen: set[uuid.UUID] = set()
        for rank, item in enumerate(ranking, 1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    return scores

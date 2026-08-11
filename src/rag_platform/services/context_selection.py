import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextSelection:
    items: list[dict[str, Any]]
    estimated_tokens: int
    duplicates_suppressed: int
    document_limit_suppressed: int
    token_budget_suppressed: int


def select_context(
    results: list[dict[str, Any]],
    *,
    max_chunks: int,
    max_estimated_tokens: int,
    per_document_limit: int,
    near_duplicate_threshold: float = 0.9,
) -> ContextSelection:
    selected: list[dict[str, Any]] = []
    selected_word_sets: list[set[str]] = []
    per_document: Counter[str] = Counter()
    estimated_tokens = 0
    duplicates_suppressed = 0
    document_limit_suppressed = 0
    token_budget_suppressed = 0

    for result in results:
        if len(selected) >= max_chunks:
            break
        document_id = str(result.get("document_id", ""))
        if per_document[document_id] >= per_document_limit:
            document_limit_suppressed += 1
            continue
        content = str(result.get("content", ""))
        words = set(re.findall(r"\w+", content.casefold()))
        if any(
            _jaccard(words, existing) >= near_duplicate_threshold for existing in selected_word_sets
        ):
            duplicates_suppressed += 1
            continue
        token_estimate = max(1, math.ceil(len(content) / 4))
        if estimated_tokens + token_estimate > max_estimated_tokens:
            token_budget_suppressed += 1
            continue
        item = {**result, "estimated_tokens": token_estimate}
        selected.append(item)
        selected_word_sets.append(words)
        per_document[document_id] += 1
        estimated_tokens += token_estimate

    return ContextSelection(
        items=selected,
        estimated_tokens=estimated_tokens,
        duplicates_suppressed=duplicates_suppressed,
        document_limit_suppressed=document_limit_suppressed,
        token_budget_suppressed=token_budget_suppressed,
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0

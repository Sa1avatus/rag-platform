"""Tests for embedding benchmark metrics and reconciliation."""

from __future__ import annotations

import json
import math

import pytest

# ── Metric implementations (standalone for testing) ────────────────────


def recall_at_k(
    retrieved: list[str], relevant: set[str], k: int,
) -> float:
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for rank, item in enumerate(retrieved[:k], 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: list[str],
    relevance_grades: dict[str, int],
    k: int,
) -> float:
    gains = [relevance_grades.get(item, 0) for item in retrieved[:k]]
    dcg = sum(
        (2 ** grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(gains, 1)
    )
    ideal = sorted(
        (g for g in relevance_grades.values() if g > 0), reverse=True,
    )[:k]
    idcg = sum(
        (2 ** grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal, 1)
    )
    return dcg / idcg if idcg > 0 else 0.0


# ── Recall@K tests ────────────────────────────────────────────────────


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, 5) == 1.0

    def test_zero_recall(self) -> None:
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_recall(self) -> None:
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)

    def test_recall_at_different_k(self) -> None:
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "e"}
        assert recall_at_k(retrieved, relevant, 1) == pytest.approx(1 / 3)
        assert recall_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)
        assert recall_at_k(retrieved, relevant, 5) == pytest.approx(1.0)

    def test_empty_relevant(self) -> None:
        assert recall_at_k(["a", "b"], set(), 2) == 0.0

    def test_empty_retrieved(self) -> None:
        assert recall_at_k([], {"a"}, 5) == 0.0


# ── MRR@K tests ───────────────────────────────────────────────────────


class TestMRRAtK:
    def test_first_rank(self) -> None:
        assert mrr_at_k(["a", "b", "c"], {"a"}, 5) == 1.0

    def test_second_rank(self) -> None:
        assert mrr_at_k(["x", "a", "b"], {"a"}, 5) == pytest.approx(0.5)

    def test_fifth_rank(self) -> None:
        assert mrr_at_k(
            ["x", "y", "z", "w", "a"], {"a"}, 5,
        ) == pytest.approx(0.2)

    def test_not_in_top_k(self) -> None:
        assert mrr_at_k(["x", "y", "z"], {"a"}, 3) == 0.0

    def test_mrr_at_different_k(self) -> None:
        retrieved = ["x", "y", "a", "b", "c"]
        relevant = {"a"}
        assert mrr_at_k(retrieved, relevant, 2) == 0.0
        assert mrr_at_k(retrieved, relevant, 3) == pytest.approx(1 / 3)
        assert mrr_at_k(retrieved, relevant, 5) == pytest.approx(1 / 3)

    def test_multiple_relevant(self) -> None:
        # MRR cares about the FIRST relevant
        retrieved = ["x", "a", "b"]
        relevant = {"a", "b"}
        assert mrr_at_k(retrieved, relevant, 3) == pytest.approx(0.5)


# ── nDCG@K tests ──────────────────────────────────────────────────────


class TestNDCGAtK:
    def test_perfect_ranking(self) -> None:
        retrieved = ["a", "b", "c"]
        grades = {"a": 3, "b": 2, "c": 1}
        assert ndcg_at_k(retrieved, grades, 3) == pytest.approx(1.0)

    def test_worst_ranking(self) -> None:
        retrieved = ["c", "b", "a"]
        grades = {"a": 3, "b": 2, "c": 1}
        # DCG = (2^1-1)/log2(2) + (2^2-1)/log2(3) + (2^3-1)/log2(4)
        # IDCG = (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4)
        assert ndcg_at_k(retrieved, grades, 3) < 1.0

    def test_zero_grades(self) -> None:
        retrieved = ["a", "b"]
        grades = {"a": 0, "b": 0}
        assert ndcg_at_k(retrieved, grades, 2) == 0.0

    def test_empty_retrieved(self) -> None:
        assert ndcg_at_k([], {"a": 3}, 5) == 0.0

    def test_ndcg_at_different_k(self) -> None:
        retrieved = ["a", "b", "c", "d", "e"]
        grades = {"a": 3, "b": 2, "c": 1, "d": 0, "e": 0}
        ndcg1 = ndcg_at_k(retrieved, grades, 1)
        ndcg3 = ndcg_at_k(retrieved, grades, 3)
        ndcg5 = ndcg_at_k(retrieved, grades, 5)
        # Perfect at all K since sorted by grade
        assert ndcg1 == pytest.approx(1.0)
        assert ndcg3 == pytest.approx(1.0)
        assert ndcg5 == pytest.approx(1.0)

    def test_binary_relevance(self) -> None:
        retrieved = ["a", "x", "b"]
        grades = {"a": 1, "b": 1, "x": 0}
        ndcg = ndcg_at_k(retrieved, grades, 3)
        assert 0.0 < ndcg < 1.0  # imperfect because x is between a and b


# ── Bootstrap CI tests ────────────────────────────────────────────────


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    import numpy as np
    arr = np.array(values)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(42)
    boot_means = np.array([
        arr[rng.integers(0, n, size=n)].mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, alpha * 100))
    hi = float(np.percentile(boot_means, (1 - alpha) * 100))
    return float(arr.mean()), lo, hi


class TestBootstrapCI:
    def test_constant_values(self) -> None:
        mean, lo, hi = bootstrap_ci([1.0] * 20)
        assert mean == pytest.approx(1.0)
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(1.0)

    def test_ci_contains_mean(self) -> None:
        values = [0.5, 0.6, 0.7, 0.4, 0.8, 0.5, 0.6, 0.7, 0.5, 0.6]
        mean, lo, hi = bootstrap_ci(values)
        assert lo <= mean <= hi

    def test_ci_narrows_with_more_data(self) -> None:
        import numpy as np
        rng = np.random.default_rng(42)
        small = rng.normal(0.5, 0.1, 10).tolist()
        large = rng.normal(0.5, 0.1, 100).tolist()
        _, lo_s, hi_s = bootstrap_ci(small)
        _, lo_l, hi_l = bootstrap_ci(large)
        assert (hi_l - lo_l) < (hi_s - lo_s)

    def test_empty(self) -> None:
        mean, lo, hi = bootstrap_ci([])
        assert mean == 0.0
        assert lo == 0.0
        assert hi == 0.0


# ── Reconciliation contract tests ─────────────────────────────────────


class TestReconciliationContract:
    def test_active_targets_queued(self) -> None:
        from rag_platform.services.reconciliation_contract import active_targets
        payloads = [
            {"status": "queued", "version_id": "00000000-0000-0000-0000-000000000001"},
            {"status": "running", "version_id": "00000000-0000-0000-0000-000000000002"},
            {"status": "completed", "version_id": "00000000-0000-0000-0000-000000000003"},
        ]
        versions, documents = active_targets(payloads)
        assert len(versions) == 2
        assert len(documents) == 0

    def test_active_targets_empty(self) -> None:
        from rag_platform.services.reconciliation_contract import active_targets
        versions, documents = active_targets([])
        assert len(versions) == 0
        assert len(documents) == 0


# ── Embedding registry tests ──────────────────────────────────────────


class TestEmbeddingRegistry:
    def test_both_models_registered(self) -> None:
        from rag_platform.core.embedding_registry import registry
        assert "bge-m3" in registry
        assert "multilingual-e5-small" in registry

    def test_dimensions(self) -> None:
        from rag_platform.core.embedding_registry import registry
        assert registry["bge-m3"].dimension == 1024
        assert registry["multilingual-e5-small"].dimension == 384

    def test_padding(self) -> None:
        from rag_platform.core.embedding_registry import registry
        e5 = registry["multilingual-e5-small"]
        vec = [1.0] * 384
        padded = e5.pad_vector(vec)
        assert len(padded) == 1024
        assert padded[384:] == [0.0] * (1024 - 384)

    def test_query_prefix(self) -> None:
        from rag_platform.core.embedding_registry import registry
        assert registry["multilingual-e5-small"].query_prefix == "query: "
        assert registry["bge-m3"].query_prefix == ""


# ── Blind assignment tests ────────────────────────────────────────────


class TestBlindAssignment:
    def test_deterministic_randomization(self) -> None:
        """Same seed produces same A/B assignment."""
        import random
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        assignments1 = [rng1.random() < 0.5 for _ in range(40)]
        assignments2 = [rng2.random() < 0.5 for _ in range(40)]
        assert assignments1 == assignments2

    def test_balanced_assignment(self) -> None:
        """Roughly 50/50 split over many queries."""
        import random
        rng = random.Random(42)
        assignments = [rng.random() < 0.5 for _ in range(100)]
        n_true = sum(assignments)
        assert 35 < n_true < 65  # ~50% ± 15%

    def test_no_model_leakage_in_pairs(self) -> None:
        """Pair data should not contain model names."""
        # Simulate a pair
        pair = {
            "pair_id": "abc123",
            "query": "Python developer",
            "set_a": [{"rank": 1, "chunk_id": "x", "content": "..."}],
            "set_b": [{"rank": 1, "chunk_id": "y", "content": "..."}],
        }
        pair_str = json.dumps(pair)
        assert "e5" not in pair_str.lower()
        assert "bge" not in pair_str.lower()


# ── Spearman correlation test ─────────────────────────────────────────


class TestSpearmanCorrelation:
    def test_perfect_correlation(self) -> None:
        from scipy import stats as scipy_stats
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        corr, p = scipy_stats.spearmanr(x, y)
        assert corr == pytest.approx(1.0)
        assert p < 0.05

    def test_negative_correlation(self) -> None:
        from scipy import stats as scipy_stats
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        corr, p = scipy_stats.spearmanr(x, y)
        assert corr == pytest.approx(-1.0)

    def test_no_correlation(self) -> None:
        from scipy import stats as scipy_stats
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [3.0, 1.0, 5.0, 2.0, 4.0]
        corr, _p = scipy_stats.spearmanr(x, y)
        assert abs(corr) < 0.8  # not strong


# ── Per-query comparison tests ────────────────────────────────────────


class TestPerQueryComparison:
    def test_win_counting(self) -> None:
        """Count wins correctly with threshold."""
        deltas = [0.5, -0.3, 0.0, -0.01, 0.02, 0.0, 0.4, -0.5]
        threshold = 0.01
        e5_wins = sum(1 for d in deltas if d < -threshold)
        bge_wins = sum(1 for d in deltas if d > threshold)
        ties = sum(1 for d in deltas if abs(d) <= threshold)
        assert e5_wins == 2
        assert bge_wins == 3
        assert ties == 3

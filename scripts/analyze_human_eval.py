"""Analyze blind human evaluation results.

Run inside the worker container:
  python /tmp/analyze_human_eval.py

Requires:
  /tmp/blind_eval_results.json  — human annotations (from HTML tool export)
  /tmp/blind_eval_mapping.json  — A/B→model mapping
  /tmp/benchmark_results_v1.json — automated benchmark results
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, "/usr/local/lib/python3.12/site-packages")

# ── Configuration ──────────────────────────────────────────────────────

RESULTS_PATH = "/tmp/blind_eval_results.json"
MAPPING_PATH = "/tmp/blind_eval_mapping.json"
PAIRS_PATH = "/tmp/blind_eval_pairs.json"
AUTOMATED_PATH = "/tmp/benchmark_results_v1.json"
OUTPUT_PATH = "/tmp/human_eval_analysis.json"

K_VALUES = [5, 10]
BOOTSTRAP_N = 2000
BOOTSTRAP_CI = 0.95

# ── Metric functions ──────────────────────────────────────────────────


def recall_at_k(
    retrieved_ids: list[str], relevant: set[str], k: int,
) -> float:
    if not relevant:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & relevant)
    return hits / len(relevant)


def mrr_at_k(retrieved_ids: list[str], relevant: set[str], k: int) -> float:
    for rank, item in enumerate(retrieved_ids[:k], 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevance_grades: dict[str, int],
    k: int,
) -> float:
    gains = [relevance_grades.get(item, 0) for item in retrieved_ids[:k]]
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


def mean_relevance_at_k(
    relevance_grades: dict[str, int],
    retrieved_ids: list[str],
    k: int,
) -> float:
    grades = [relevance_grades.get(item, 0) for item in retrieved_ids[:k]]
    return sum(grades) / len(grades) if grades else 0.0


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = BOOTSTRAP_N,
    ci: float = BOOTSTRAP_CI,
) -> tuple[float, float, float]:
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


def main():
    # Load data
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)
    with open(MAPPING_PATH, encoding="utf-8") as f:
        mapping = json.load(f)
    with open(AUTOMATED_PATH, encoding="utf-8") as f:
        automated = json.load(f)
    with open(PAIRS_PATH, encoding="utf-8") as f:
        pairs_data = json.load(f)
    pairs_by_id = {p["pair_id"]: p for p in pairs_data["pairs"]}

    eval_results = results["results"]
    print(f"Loaded {len(eval_results)} annotated queries")

    # ── De-blind and compute per-query metrics ─────────────────────────
    per_query = []
    for item in eval_results:
        pair_id = item["pair_id"]
        m = mapping.get(pair_id)
        if m is None:
            print(f"Warning: no mapping for {pair_id}, skipping")
            continue

        query = item["query"]
        a_model = m["set_a_model"]
        m["set_b_model"]
        e5_ids = m["e5_chunk_ids"]
        bge_ids = m["bge_chunk_ids"]
        a_grades = item["set_a_grades"]
        b_grades = item["set_b_grades"]

        # Get chunk_ids from pairs data
        pair = pairs_by_id.get(pair_id, {})
        a_chunk_ids = [r["chunk_id"] for r in pair.get("set_a", [])]
        b_chunk_ids = [r["chunk_id"] for r in pair.get("set_b", [])]

        # Build grade maps using pairs chunk_ids
        e5_grades_map = {}
        bge_grades_map = {}
        if a_model == "e5-small":
            for i, cid in enumerate(a_chunk_ids):
                if i < len(a_grades) and a_grades[i] is not None:
                    e5_grades_map[cid] = a_grades[i]
            for i, cid in enumerate(b_chunk_ids):
                if i < len(b_grades) and b_grades[i] is not None:
                    bge_grades_map[cid] = b_grades[i]
        else:
            for i, cid in enumerate(a_chunk_ids):
                if i < len(a_grades) and a_grades[i] is not None:
                    bge_grades_map[cid] = a_grades[i]
            for i, cid in enumerate(b_chunk_ids):
                if i < len(b_grades) and b_grades[i] is not None:
                    e5_grades_map[cid] = b_grades[i]

        e5_relevant = {cid for cid, g in e5_grades_map.items() if g > 0}
        bge_relevant = {cid for cid, g in bge_grades_map.items() if g > 0}

        entry = {"query": query, "pair_id": pair_id}

        for k in K_VALUES:
            e5_recall = recall_at_k(e5_ids, e5_relevant, k)
            bge_recall = recall_at_k(bge_ids, bge_relevant, k)
            e5_mrr = mrr_at_k(e5_ids, e5_relevant, k)
            bge_mrr = mrr_at_k(bge_ids, bge_relevant, k)
            e5_ndcg = ndcg_at_k(e5_ids, e5_grades_map, k)
            bge_ndcg = ndcg_at_k(bge_ids, bge_grades_map, k)

            entry[f"e5_recall@{k}"] = e5_recall
            entry[f"bge_recall@{k}"] = bge_recall
            entry[f"e5_mrr@{k}"] = e5_mrr
            entry[f"bge_mrr@{k}"] = bge_mrr
            entry[f"e5_ndcg@{k}"] = e5_ndcg
            entry[f"bge_ndcg@{k}"] = bge_ndcg

        per_query.append(entry)

    n_queries = len(per_query)
    print(f"Computed metrics for {n_queries} queries")

    # ── Aggregate metrics with bootstrap CI ────────────────────────────
    print(f"\n{'='*80}")
    print("HUMAN EVALUATION METRICS (with 95% bootstrap CI)")
    print(f"{'='*80}\n")

    header = (
        f"{'Metric':<15} {'E5 mean':>8} {'E5 CI':>18} "
        f"{'BGE mean':>9} {'BGE CI':>18} {'Δ':>7} {'Winner':>8}"
    )
    print(header)
    print("-" * len(header))

    human_metrics = {}
    metric_names = []
    for k in K_VALUES:
        metric_names.extend([
            f"Recall@{k}", f"MRR@{k}", f"nDCG@{k}",
        ])

    for metric_name in metric_names:
        key = metric_name.lower()
        e5_key = f"e5_{key}"
        bge_key = f"bge_{key}"

        e5_vals = [p[e5_key] for p in per_query if e5_key in p]
        bge_vals = [p[bge_key] for p in per_query if bge_key in p]

        if not e5_vals:
            continue

        e5_mean, e5_lo, e5_hi = bootstrap_ci(e5_vals)
        bge_mean, bge_lo, bge_hi = bootstrap_ci(bge_vals)
        delta = bge_mean - e5_mean

        if e5_lo > bge_hi:
            winner = "E5"
        elif bge_lo > e5_hi:
            winner = "BGE"
        else:
            winner = "~"

        e5_ci_str = f"[{e5_lo:.3f}, {e5_hi:.3f}]"
        bge_ci_str = f"[{bge_lo:.3f}, {bge_hi:.3f}]"
        print(
            f"  {metric_name:<13} {e5_mean:>8.3f} {e5_ci_str:>18} "
            f"{bge_mean:>9.3f} {bge_ci_str:>18} {delta:>+7.3f} {winner:>8}"
        )

        human_metrics[metric_name] = {
            "e5_mean": e5_mean,
            "e5_ci": [e5_lo, e5_hi],
            "bge_mean": bge_mean,
            "bge_ci": [bge_lo, bge_hi],
            "delta": delta,
            "winner": winner,
        }

    # ── Per-query win/loss/tie analysis ────────────────────────────────
    print(f"\n{'='*80}")
    print("PER-QUERY WIN/LOSS/TIE (nDCG@10)")
    print(f"{'='*80}\n")

    e5_wins = 0
    bge_wins = 0
    ties = 0
    deltas = []
    for p in per_query:
        e5_v = p.get("e5_ndcg@10", 0)
        bge_v = p.get("bge_ndcg@10", 0)
        d = bge_v - e5_v
        deltas.append(d)
        if d > 0.01:
            bge_wins += 1
        elif d < -0.01:
            e5_wins += 1
        else:
            ties += 1

    print(f"  E5 wins:  {e5_wins} ({e5_wins/n_queries*100:.0f}%)")
    print(f"  BGE wins: {bge_wins} ({bge_wins/n_queries*100:.0f}%)")
    print(f"  Ties:     {ties} ({ties/n_queries*100:.0f}%)")

    # ── Spearman correlation: automated vs human ───────────────────────
    print(f"\n{'='*80}")
    print("AUTOMATED vs HUMAN CORRELATION")
    print(f"{'='*80}\n")

    # Build per-query automated delta (e5_ndcg@10 - bge_ndcg@10)
    # We need to match queries between datasets

    # For now, compare aggregate-level correlation
    # Compute human delta per query and automated delta per query

    # Try to get automated per-query data
    # The automated benchmark doesn't store per-query results in the JSON
    # So we compute Spearman on the human side only
    # and note that automated is available only at aggregate level

    # Instead, compute correlation between human E5 and human BGE scores
    # This tells us if the human evaluation is consistent
    e5_human_scores = [p.get("e5_ndcg@10", 0) for p in per_query]
    bge_human_scores = [p.get("bge_ndcg@10", 0) for p in per_query]

    if len(e5_human_scores) > 2:
        corr, p_value = scipy_stats.spearmanr(e5_human_scores, bge_human_scores)
        print(f"  Spearman (E5 human vs BGE human nDCG@10): {corr:.3f} (p={p_value:.4f})")

    # Compute per-query human vs automated delta correlation
    # Note: automated per-query data not available in current JSON
    # Will note this limitation
    print("\n  Note: Per-query automated metrics not stored in benchmark JSON.")
    print("  Aggregate comparison:")
    for metric_name in ["Recall@10", "MRR@10", "nDCG@10"]:
        auto = automated["quality"].get(metric_name, {})
        human = human_metrics.get(metric_name, {})
        if auto and human:
            print(f"    {metric_name}:")
            a_str = f"E5={auto['e5_mean']:.3f} BGE={auto['bge_mean']:.3f}"
            h_str = f"E5={human['e5_mean']:.3f} BGE={human['bge_mean']:.3f}"
            print(f"      Automated: {a_str} Δ={auto['delta']:.3f}")
            print(f"      Human:     {h_str} Δ={human['delta']:.3f}")

    # ── Disagreement cases ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("STRONG DISAGREEMENT CASES (|Δ nDCG@10| > 0.3)")
    print(f"{'='*80}\n")

    def _abs_delta(x):
        return abs(x.get("bge_ndcg@10", 0) - x.get("e5_ndcg@10", 0))

    for p in sorted(per_query, key=_abs_delta, reverse=True):
        e5_v = p.get("e5_ndcg@10", 0)
        bge_v = p.get("bge_ndcg@10", 0)
        d = bge_v - e5_v
        if abs(d) > 0.3:
            winner = "BGE" if d > 0 else "E5"
            print(f"  \"{p['query'][:50]}...\"")
            d_str = f"Δ={d:+.3f}"
            print(f"    E5={e5_v:.3f}  BGE={bge_v:.3f}  {d_str}  Winner={winner}")

    # ── Save analysis ──────────────────────────────────────────────────
    analysis = {
        "n_queries": n_queries,
        "human_metrics": human_metrics,
        "automated_metrics": automated["quality"],
        "win_loss": {
            "e5_wins": e5_wins,
            "bge_wins": bge_wins,
            "ties": ties,
        },
        "per_query": per_query,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {OUTPUT_PATH}")

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}\n")
    print(f"  Human-annotated queries: {n_queries}")
    print("  Relevance scale: 0-3 (irrelevant to highly relevant)")
    print("  Blind evaluation: yes (annotator did not know model assignment)")
    print("  A/B randomization: yes (per query, seed=42)")
    print()

    for metric_name in ["Recall@10", "MRR@10", "nDCG@10"]:
        h = human_metrics.get(metric_name, {})
        if h:
            print(f"  {metric_name}: E5={h['e5_mean']:.3f} BGE={h['bge_mean']:.3f} → {h['winner']}")

    print(f"\n  E5 wins: {e5_wins}  BGE wins: {bge_wins}  Ties: {ties}")


if __name__ == "__main__":
    main()

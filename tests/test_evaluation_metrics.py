import json
import unittest
from pathlib import Path

from rag_platform.api.schemas import EvaluationDatasetCreate
from rag_platform.services.evaluation_metrics import aggregate_metrics, case_metrics


class EvaluationMetricsTests(unittest.TestCase):
    def test_checked_in_hard_negative_dataset_matches_api_schema(self) -> None:
        path = Path(__file__).parents[1] / "examples" / "evaluation-hard-negatives.json"
        dataset = EvaluationDatasetCreate.model_validate(json.loads(path.read_text("utf-8")))
        self.assertEqual(dataset.cases[0].difficulty, "hard")
        self.assertIn("opportunity-sous-chef-003", dataset.cases[0].forbidden_results)

    def test_perfect_ranking_scores_one_for_primary_metrics(self) -> None:
        metrics = case_metrics(["a", "b"], {"a": 3, "b": 1})
        self.assertEqual(metrics["Recall@3"], 1.0)
        self.assertEqual(metrics["MRR"], 1.0)
        self.assertEqual(metrics["NDCG@5"], 1.0)
        self.assertEqual(metrics["HitRate@5"], 1.0)

    def test_empty_retrieval_is_reported(self) -> None:
        metrics = case_metrics([], {"expected": 1})
        self.assertEqual(metrics["EmptyRetrievalRate"], 1.0)
        self.assertEqual(metrics["Recall@1"], 0.0)
        self.assertEqual(metrics["MRR"], 0.0)

    def test_duplicate_rate_counts_repeated_identifiers(self) -> None:
        metrics = case_metrics(["a", "a", "b"], {"a": 1})
        self.assertAlmostEqual(metrics["DuplicateRetrievalRate"], 1 / 3)

    def test_aggregate_is_arithmetic_mean(self) -> None:
        result = aggregate_metrics([{"MRR": 1.0}, {"MRR": 0.0}])
        self.assertEqual(result, {"MRR": 0.5})


if __name__ == "__main__":
    unittest.main()

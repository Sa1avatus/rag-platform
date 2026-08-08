import unittest
import uuid

from rag_platform.services.fusion import reciprocal_rank_fusion


class FusionTests(unittest.TestCase):
    def test_shared_result_outranks_single_source_results(self) -> None:
        shared = uuid.uuid4()
        vector_only = uuid.uuid4()
        lexical_only = uuid.uuid4()
        scores = reciprocal_rank_fusion(
            [[shared, vector_only], [shared, lexical_only]]
        )
        self.assertGreater(scores[shared], scores[vector_only])
        self.assertGreater(scores[shared], scores[lexical_only])

    def test_duplicate_in_one_ranking_is_counted_once(self) -> None:
        item = uuid.uuid4()
        scores = reciprocal_rank_fusion([[item, item]], k=60)
        self.assertEqual(scores[item], 1 / 61)

    def test_non_positive_constant_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], k=0)


if __name__ == "__main__":
    unittest.main()

import unittest

from rag_platform.services.embedding_contract import validate_embedding_dimension


class EmbeddingContractTests(unittest.TestCase):
    def test_matching_dimension_is_accepted(self) -> None:
        validate_embedding_dimension(1024, 1024)

    def test_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "dimension mismatch"):
            validate_embedding_dimension(768, 1024)

    def test_invalid_detected_dimension_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "invalid dimension"):
            validate_embedding_dimension(0, 1024)


if __name__ == "__main__":
    unittest.main()

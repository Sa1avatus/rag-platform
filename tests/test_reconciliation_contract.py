import unittest
import uuid

from rag_platform.services.reconciliation_contract import active_targets


class ReconciliationContractTests(unittest.TestCase):
    def test_only_active_valid_targets_are_returned(self) -> None:
        version_id = uuid.uuid4()
        document_id = uuid.uuid4()
        versions, documents = active_targets(
            [
                {"status": "queued", "version_id": str(version_id)},
                {"status": "running", "document_id": str(document_id)},
                {"status": "failed", "version_id": str(uuid.uuid4())},
                {"status": "queued", "version_id": "invalid"},
            ]
        )
        self.assertEqual(versions, {version_id})
        self.assertEqual(documents, {document_id})


if __name__ == "__main__":
    unittest.main()

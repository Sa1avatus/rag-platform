import unittest

from rag_platform.services.outbox_contract import event_message


class OutboxContractTests(unittest.TestCase):
    def test_index_event_routes_version_and_job(self) -> None:
        task, arguments = event_message(
            {
                "type": "document.index",
                "version_id": "version-id",
                "job_id": "job-id",
            }
        )
        self.assertEqual(task, "rag_platform.worker.tasks.index_document")
        self.assertEqual(arguments, ["version-id", "job-id"])

    def test_delete_event_routes_document_and_job(self) -> None:
        task, arguments = event_message(
            {
                "type": "document.delete",
                "document_id": "document-id",
                "job_id": "job-id",
            }
        )
        self.assertEqual(
            task,
            "rag_platform.worker.tasks.delete_document_derivatives",
        )
        self.assertEqual(arguments, ["document-id", "job-id"])

    def test_unknown_event_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            event_message({"type": "unknown", "job_id": "job-id"})

    def test_evaluation_event_routes_run(self) -> None:
        task, arguments = event_message({"type": "evaluation.run", "run_id": "run-id"})
        self.assertEqual(task, "rag_platform.worker.tasks.run_evaluation_task")
        self.assertEqual(arguments, ["run-id"])


if __name__ == "__main__":
    unittest.main()

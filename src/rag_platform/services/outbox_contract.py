from typing import Any


def event_message(payload: dict[str, Any]) -> tuple[str, list[str]]:
    job_id = payload.get("job_id")
    event_type = payload.get("type")
    if not isinstance(event_type, str):
        raise ValueError("outbox event type is missing")
    if event_type == "evaluation.run":
        run_id = payload.get("run_id")
        if not isinstance(run_id, str):
            raise ValueError("outbox event is missing identifiers")
        return "rag_platform.worker.tasks.run_evaluation_task", [run_id]
    identifier_field = {
        "document.index": "version_id",
        "document.delete": "document_id",
    }.get(event_type)
    if identifier_field is None:
        raise ValueError("unsupported outbox event type")
    identifier = payload.get(identifier_field)
    if not isinstance(identifier, str) or not isinstance(job_id, str):
        raise ValueError("outbox event is missing identifiers")
    task_name = {
        "document.index": "rag_platform.worker.tasks.index_document",
        "document.delete": "rag_platform.worker.tasks.delete_document_derivatives",
    }[event_type]
    return task_name, [identifier, job_id]

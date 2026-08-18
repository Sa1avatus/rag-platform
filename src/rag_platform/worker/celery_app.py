from celery import Celery

from rag_platform.core.config import get_settings

settings = get_settings()
app = Celery("rag-platform", broker=settings.redis_url, backend=settings.redis_url)
app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    task_routes={
        "rag_platform.worker.tasks.embed_query": {"queue": "search"},
        "rag_platform.worker.tasks.*": {"queue": "indexing"},
    },
    task_default_retry_delay=5,
    task_time_limit=1800,
    beat_schedule={
        "publish-rag-outbox": {
            "task": "rag_platform.worker.tasks.dispatch_outbox",
            "schedule": 2.0,
        },
        "reconcile-rag-indexes": {
            "task": "rag_platform.worker.tasks.reconcile_indexes",
            "schedule": 300.0,
        },
    },
)
app.autodiscover_tasks(["rag_platform.worker"])

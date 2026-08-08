from prometheus_client import Counter, Gauge, Histogram

DOCUMENTS_RECEIVED = Counter(
    "rag_documents_received_total",
    "Document versions accepted for indexing",
)
DOCUMENTS_INDEXED = Counter(
    "rag_documents_indexed_total",
    "Document versions fully indexed",
)
DOCUMENTS_FAILED = Counter(
    "rag_documents_failed_total",
    "Document indexing failures",
)
INDEXING_QUEUE_SIZE = Gauge(
    "rag_indexing_queue_size",
    "Queued indexing jobs observed by reconciliation",
)
EMBEDDING_DURATION = Histogram(
    "rag_embedding_duration_seconds",
    "Embedding batch duration",
)
EMBEDDING_BATCH_SIZE = Histogram(
    "rag_embedding_batch_size",
    "Number of texts per embedding batch",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
)
VECTOR_SEARCH_DURATION = Histogram(
    "rag_vector_search_duration_seconds",
    "pgvector query duration",
)
BM25_SEARCH_DURATION = Histogram(
    "rag_bm25_search_duration_seconds",
    "OpenSearch BM25 query duration",
)
FUSION_DURATION = Histogram(
    "rag_fusion_duration_seconds",
    "Rank fusion duration",
)
RERANKER_DURATION = Histogram(
    "rag_reranker_duration_seconds",
    "External reranker duration",
)
RERANKER_ERRORS = Counter(
    "rag_reranker_errors_total",
    "External reranker errors",
)
RERANKER_DEGRADED = Counter(
    "rag_reranker_degraded_total",
    "Retrieval requests degraded because reranking failed",
)
RETRIEVAL_EMPTY = Counter(
    "rag_retrieval_empty_total",
    "Retrieval requests with no results",
)
RETRIEVAL_RESULTS_COUNT = Histogram(
    "rag_retrieval_results_count",
    "Results returned per retrieval request",
    buckets=(0, 1, 3, 5, 10, 20, 50, 100),
)
DUPLICATE_CHUNKS_REMOVED = Counter(
    "rag_duplicate_chunks_removed_total",
    "Duplicate chunks removed during fusion",
)
CACHE_HITS = Counter("rag_cache_hits_total", "Embedding cache hits")
CACHE_MISSES = Counter("rag_cache_misses_total", "Embedding cache misses")

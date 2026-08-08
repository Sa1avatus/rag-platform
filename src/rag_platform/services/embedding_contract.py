def validate_embedding_dimension(detected: int, configured: int) -> None:
    if detected < 1:
        raise RuntimeError("embedding model reported an invalid dimension")
    if detected != configured:
        raise RuntimeError(
            f"embedding dimension mismatch: model={detected}, database_contract={configured}"
        )

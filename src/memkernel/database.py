from pathlib import Path

from memkernel.backend.sqlite_adapter import SQLiteBackend
from memkernel.embedding import EmbeddingProvider


def initialize_database(
    database_path: str | Path,
    embedding_provider: EmbeddingProvider,
) -> tuple[int, int]:
    """Create or migrate the database and ensure every memory has an embedding."""
    resolved_path = Path(database_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    backend = SQLiteBackend(
        resolved_path,
        embedding_provider=embedding_provider,
    )
    memory_count = len(backend.list_memories())
    embedding_count = backend.rebuild_embeddings()
    return memory_count, embedding_count

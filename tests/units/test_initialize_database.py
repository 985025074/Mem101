import sqlite3

from memkernel.backend.sqlite_adapter import SQLiteBackend
from memkernel.database import initialize_database


class StaticEmbeddingProvider:
    def embed_document(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_initialize_database_creates_schema(tmp_path) -> None:
    database_path = tmp_path / "nested" / "memkernel.db"

    memory_count, embedding_count = initialize_database(
        database_path,
        StaticEmbeddingProvider(),
    )

    assert database_path.is_file()
    assert memory_count == 0
    assert embedding_count == 0
    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {
        "memories",
        "memory_embeddings",
        "source_events",
        "memory_sources",
    } <= tables


def test_initialize_database_backfills_legacy_memories(tmp_path) -> None:
    database_path = tmp_path / "memkernel.db"
    legacy_backend = SQLiteBackend(database_path)
    memory_id = legacy_backend.insert("User likes Rust.")

    memory_count, embedding_count = initialize_database(
        database_path,
        StaticEmbeddingProvider(),
    )

    assert memory_count == 1
    assert embedding_count == 1
    initialized_backend = SQLiteBackend(
        database_path,
        embedding_provider=StaticEmbeddingProvider(),
    )
    matches = initialized_backend.search_current("Rust preference")
    assert [match.memory.id for match in matches] == [memory_id]

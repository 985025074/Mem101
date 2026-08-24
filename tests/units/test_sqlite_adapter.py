import sqlite3

import pytest

from memkernel.backend.sqlite_adapter import SQLiteBackend
from memkernel.embedding import OpenAIEmbeddingProvider
from memkernel.extractor.extractor import SimpleExtractedResult


class StaticEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FailingNewMemoryEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        if text == "New memory.":
            return []
        return [1.0, 0.0]


@pytest.fixture
def backend(tmp_path):
    client = OpenAIEmbeddingProvider(OpenAIEmbeddingProvider.get_client())
    return SQLiteBackend(tmp_path / "memkernel.db", client)


def test_query_by_content(backend: SQLiteBackend) -> None:
    memory_id = backend.remember(SimpleExtractedResult("User likes Rust."))

    memory = backend.query_by({"content": "User likes Rust."})

    assert memory is not None
    assert memory.id == memory_id
    assert memory.content == "User likes Rust."


def test_query_by_multiple_columns(backend: SQLiteBackend) -> None:
    memory_id = backend.remember(SimpleExtractedResult("User is building MemKernel."))

    memory = backend.query_by(
        {"id": memory_id, "content": "User is building MemKernel."}
    )

    assert memory is not None
    assert memory.id == memory_id


def test_query_by_returns_none_when_no_memory_matches(
    backend: SQLiteBackend,
) -> None:
    backend.remember(SimpleExtractedResult("User likes Rust."))

    assert backend.query_by({"content": "User likes Python."}) is None


def test_query_by_rejects_empty_query(backend: SQLiteBackend) -> None:
    with pytest.raises(ValueError, match="at least one field"):
        backend.query_by({})


def test_query_by_rejects_unknown_columns(backend: SQLiteBackend) -> None:
    with pytest.raises(ValueError, match="Unsupported query column"):
        backend.query_by({"unknown": "value"})


def test_remove_existing_memory(tmp_path) -> None:
    backend = SQLiteBackend(tmp_path / "remove.db")
    memory_id = backend.remember(SimpleExtractedResult("User likes Rust."))

    assert backend.remove(memory_id) is True
    assert backend.get(memory_id) is None


def test_remove_missing_memory(tmp_path) -> None:
    backend = SQLiteBackend(tmp_path / "remove.db")

    assert backend.remove("missing-memory-id") is False


def test_remove_rejects_empty_id(tmp_path) -> None:
    backend = SQLiteBackend(tmp_path / "remove.db")

    with pytest.raises(ValueError, match="non-empty string"):
        backend.remove("")


def test_remove_also_removes_embedding(tmp_path) -> None:
    backend = SQLiteBackend(
        tmp_path / "remove.db",
        embedding_provider=StaticEmbeddingProvider(),
    )
    memory_id = backend.remember(SimpleExtractedResult("User likes Rust."))

    assert backend.remove(memory_id) is True
    assert backend.search_similar("Rust preference") == []


def test_supersede_keeps_the_old_memory_and_embedding(tmp_path) -> None:
    backend = SQLiteBackend(
        tmp_path / "supersede.db",
        embedding_provider=StaticEmbeddingProvider(),
    )
    old_id = backend.insert("User lives in Shanghai.")

    new_id = backend.supersede(old_id, "User lives in Beijing.")

    old_memory = backend.get(old_id)
    new_memory = backend.get(new_id)
    assert old_memory is not None
    assert old_memory.state == "SUPERSEDED"
    assert old_memory.superseded_by_id == new_id
    assert old_memory.superseded_at is not None
    assert new_memory is not None
    assert new_memory.state == "ACTIVE"
    assert new_memory.superseded_by_id is None
    assert [result.memory.id for result in backend.search_current("where user lives")] == [
        new_id
    ]
    assert [result.memory.id for result in backend.search_history("where user lived")] == [
        old_id
    ]


def test_current_and_history_search_have_independent_limits(tmp_path) -> None:
    backend = SQLiteBackend(
        tmp_path / "search-state.db",
        embedding_provider=StaticEmbeddingProvider(),
    )
    old_id = backend.insert("Old preference.")
    backend.insert("Another active memory.")
    backend.supersede(old_id, "New preference.")

    current = backend.search_current("preference", top_k=1)
    history = backend.search_history("preference", top_k=1)

    assert len(current) == 1
    assert current[0].memory.state == "ACTIVE"
    assert len(history) == 1
    assert history[0].memory.id == old_id
    assert history[0].memory.state == "SUPERSEDED"


def test_supersede_does_not_write_when_target_is_invalid(tmp_path) -> None:
    backend = SQLiteBackend(
        tmp_path / "invalid-supersede.db",
        embedding_provider=StaticEmbeddingProvider(),
    )

    with pytest.raises(ValueError, match="not found"):
        backend.supersede("missing-id", "New memory.")

    assert backend.list_memories() == []


def test_supersede_rolls_back_both_state_and_new_memory_on_failure(tmp_path) -> None:
    backend = SQLiteBackend(
        tmp_path / "rollback-supersede.db",
        embedding_provider=FailingNewMemoryEmbeddingProvider(),
    )
    old_id = backend.insert("Old memory.")

    with pytest.raises(sqlite3.IntegrityError):
        backend.supersede(old_id, "New memory.")

    old_memory = backend.get(old_id)
    assert old_memory is not None
    assert old_memory.state == "ACTIVE"
    assert old_memory.superseded_by_id is None
    assert len(backend.list_memories()) == 1


def test_a_superseded_memory_cannot_be_superseded_again(tmp_path) -> None:
    backend = SQLiteBackend(
        tmp_path / "repeat-supersede.db",
        embedding_provider=StaticEmbeddingProvider(),
    )
    old_id = backend.insert("Old memory.")
    backend.supersede(old_id, "Current memory.")

    with pytest.raises(ValueError, match="active memory"):
        backend.supersede(old_id, "Another memory.")

    assert len(backend.list_memories()) == 2


def test_existing_database_is_migrated_with_active_memories(tmp_path) -> None:
    database_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        "INSERT INTO memories (id, content) VALUES (?, ?)",
        ("legacy-id", "Legacy memory."),
    )
    connection.commit()
    connection.close()

    backend = SQLiteBackend(database_path)

    memory = backend.get("legacy-id")
    assert memory is not None
    assert memory.state == "ACTIVE"
    assert memory.superseded_by_id is None
    assert memory.superseded_at is None


def test_embedding(backend: SQLiteBackend):
    list_1 = [
        "A hamburger is a fast food made with a meat patty inside a round bun.",
        "Hamburg is Germany's second-largest city and its largest port city.",
    ]
    for info in list_1:
        backend.remember(SimpleExtractedResult(info))
    result = backend.search_similar("I want to visit Hamburg for sightseeing.")

    print(result[0].similarity)
    assert result[0].memory.content == list_1[1]

    result = backend.search_similar("I want to eat a hamburger for lunch.")
    assert result[0].memory.content == list_1[0]
    print(result[0].similarity)

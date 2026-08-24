import pytest

from memkernel.backend.sqlite_adapter import SQLiteBackend
from memkernel.embedding import OpenAIEmbeddingProvider
from memkernel.extractor.extractor import SimpleExtractedResult


class StaticEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
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

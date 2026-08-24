import pytest

from memkernel.backend.sqlite_adapter import SQLiteBackend
from memkernel.extractor import ExtractedResult


@pytest.fixture
def backend(tmp_path):
    return SQLiteBackend(tmp_path / "memkernel.db")


def test_query_by_content(backend: SQLiteBackend) -> None:
    memory_id = backend.remember(ExtractedResult("User likes Rust."))

    memory = backend.query_by({"content": "User likes Rust."})

    assert memory is not None
    assert memory.id == memory_id
    assert memory.content == "User likes Rust."


def test_query_by_multiple_columns(backend: SQLiteBackend) -> None:
    memory_id = backend.remember(ExtractedResult("User is building MemKernel."))

    memory = backend.query_by(
        {"id": memory_id, "content": "User is building MemKernel."}
    )

    assert memory is not None
    assert memory.id == memory_id


def test_query_by_returns_none_when_no_memory_matches(
    backend: SQLiteBackend,
) -> None:
    backend.remember(ExtractedResult("User likes Rust."))

    assert backend.query_by({"content": "User likes Python."}) is None


def test_query_by_rejects_empty_query(backend: SQLiteBackend) -> None:
    with pytest.raises(ValueError, match="at least one field"):
        backend.query_by({})


def test_query_by_rejects_unknown_columns(backend: SQLiteBackend) -> None:
    with pytest.raises(ValueError, match="Unsupported query column"):
        backend.query_by({"unknown": "value"})

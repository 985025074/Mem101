import pytest

from memkernel.backend.backend import MemoryRecord, ScoredMemory
from memkernel.backend.backend_v2 import BackendV2
from memkernel.retriever_v2 import RetrievalResult, SemanticRetriever


def memory(memory_id: str, content: str) -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        content=content,
        created_at="2026-08-24 12:00:00",
    )


class FakeSemanticBackend:
    def __init__(
        self,
        matches: list[ScoredMemory],
        historical_matches: list[ScoredMemory] | None = None,
    ):
        self.matches = matches
        self.historical_matches = historical_matches or []
        self.calls: list[tuple[str, str, int]] = []

    def search_current(
        self, content: str, top_k: int = 5
    ) -> list[ScoredMemory]:
        self.calls.append(("current", content, top_k))
        return self.matches

    def search_history(
        self, content: str, top_k: int = 5
    ) -> list[ScoredMemory]:
        self.calls.append(("history", content, top_k))
        return self.historical_matches


class StaticEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


class UnusedAI:
    def get_client(self) -> object:
        return object()

    def get_ai_response(
        self, client: object, inst: str, input_text: str
    ) -> str:
        raise AssertionError("Retrieval must not call the LLM")


def test_retrieve_filters_and_ranks_semantic_matches() -> None:
    backend = FakeSemanticBackend(
        [
            ScoredMemory(memory("medium", "Medium match"), 0.72),
            ScoredMemory(memory("best", "Best match"), 0.95),
            ScoredMemory(memory("weak", "Weak match"), 0.49),
        ]
    )
    retriever = SemanticRetriever(backend)

    results = retriever.retrieve("  Rust preference  ", top_k=3, threshold=0.7)

    assert results == [
        RetrievalResult(memory("best", "Best match"), 0.95),
        RetrievalResult(memory("medium", "Medium match"), 0.72),
    ]
    assert backend.calls == [("current", "Rust preference", 3)]


def test_retrieve_enforces_top_k_even_if_backend_returns_more() -> None:
    backend = FakeSemanticBackend(
        [
            ScoredMemory(memory("second", "Second"), 0.8),
            ScoredMemory(memory("first", "First"), 0.9),
        ]
    )

    results = SemanticRetriever(backend).retrieve("query", top_k=1)

    assert [result.memory.id for result in results] == ["first"]


def test_retrieve_includes_a_score_equal_to_threshold() -> None:
    backend = FakeSemanticBackend(
        [ScoredMemory(memory("match", "Match"), 0.5)]
    )

    results = SemanticRetriever(backend).retrieve("query", threshold=0.5)

    assert len(results) == 1


def test_retrieve_returns_empty_list_when_nothing_matches() -> None:
    retriever = SemanticRetriever(FakeSemanticBackend([]))

    assert retriever.retrieve("query") == []


def test_current_and_history_use_independent_top_k_values() -> None:
    backend = FakeSemanticBackend(
        [
            ScoredMemory(memory("current-1", "Current one"), 0.9),
            ScoredMemory(memory("current-2", "Current two"), 0.8),
        ],
        historical_matches=[
            ScoredMemory(memory("history-1", "History one"), 0.95),
            ScoredMemory(memory("history-2", "History two"), 0.85),
        ],
    )
    retriever = SemanticRetriever(backend)

    current = retriever.retrieve_current("preference", top_k=1)
    history = retriever.retrieve_history("preference", top_k=2)

    assert [result.memory.id for result in current] == ["current-1"]
    assert [result.memory.id for result in history] == ["history-1", "history-2"]
    assert backend.calls == [
        ("current", "preference", 1),
        ("history", "preference", 2),
    ]


def test_retrieve_integrates_with_backend_v2_without_calling_llm(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=UnusedAI(),
        embedding_provider=StaticEmbeddingProvider(),
    )
    memory_id = backend.sqlite_backend.insert("User likes Rust.")

    results = SemanticRetriever(backend).retrieve(
        "Rust preference",
        threshold=0.9,
    )

    assert [result.memory.id for result in results] == [memory_id]


def test_history_retrieval_integrates_with_backend_v2(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=UnusedAI(),
        embedding_provider=StaticEmbeddingProvider(),
    )
    old_id = backend.sqlite_backend.insert("User lived in Shanghai.")
    new_id = backend.sqlite_backend.supersede(old_id, "User lives in Beijing.")
    retriever = SemanticRetriever(backend)

    current = retriever.retrieve_current("where user lives", threshold=0.9)
    history = retriever.retrieve_history("where user lived", threshold=0.9)

    assert [result.memory.id for result in current] == [new_id]
    assert [result.memory.id for result in history] == [old_id]


@pytest.mark.parametrize("query", ["", "   ", None])
def test_retrieve_rejects_empty_queries(query) -> None:
    retriever = SemanticRetriever(FakeSemanticBackend([]))

    with pytest.raises(ValueError, match="non-empty string"):
        retriever.retrieve(query)


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_retrieve_rejects_invalid_top_k(top_k) -> None:
    retriever = SemanticRetriever(FakeSemanticBackend([]))

    with pytest.raises(ValueError, match="positive integer"):
        retriever.retrieve("query", top_k=top_k)


@pytest.mark.parametrize("threshold", [-1.1, 1.1, "0.5", True])
def test_retrieve_rejects_invalid_threshold(threshold) -> None:
    retriever = SemanticRetriever(FakeSemanticBackend([]))

    with pytest.raises(ValueError, match="threshold"):
        retriever.retrieve("query", threshold=threshold)

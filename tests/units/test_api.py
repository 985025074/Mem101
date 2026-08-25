import json

from fastapi.testclient import TestClient

from memkernel.api import create_app
from memkernel.backend.backend import MemoryDecision, MemoryRecord, MemoryState
from memkernel.backend.backend_v2 import BackendV2
from memkernel.extractor.extractor_v2 import (
    ExtractionValidationError,
    ExtractedFact,
    JsonExtractedResult,
)
from memkernel.kernel import MemKernel, PostMemory
from memkernel.provenance import (
    MemorySourceRecord,
    SourceEvent,
    SourceEventRecord,
)
from memkernel.retriever_v2 import (
    RecallResults,
    RetrievalResult,
)


def memory(
    memory_id: str,
    content: str,
    *,
    created_at: str = "2026-08-24 12:00:00",
    state: MemoryState = "ACTIVE",
    superseded_by_id: str | None = None,
    superseded_at: str | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        content=content,
        created_at=created_at,
        state=state,
        superseded_by_id=superseded_by_id,
        superseded_at=superseded_at,
    )


def memory_source(source_id: str = "source-id") -> MemorySourceRecord:
    return MemorySourceRecord(
        source=SourceEventRecord(
            id=source_id,
            content="I like Rust",
            source_type="message",
            role="user",
            observed_at="2026-08-24T12:00:00+00:00",
            created_at="2026-08-24 12:00:01",
            metadata={"channel": "api"},
        ),
        evidence_quote="I like Rust",
        link_type="DERIVED",
        linked_at="2026-08-24 12:00:02",
    )


class FakeRetriever:
    def __init__(self, results: RecallResults):
        self.results = results
        self.calls: list[tuple[str, int, int, float]] = []

    def recall(
        self,
        query: str,
        *,
        current_top_k: int = 5,
        history_top_k: int = 0,
        threshold: float = 0.5,
    ) -> RecallResults:
        self.calls.append((query, current_top_k, history_top_k, threshold))
        return self.results


class FakeKernel:
    def __init__(
        self,
        results: RecallResults,
        histories: dict[str, list[MemoryRecord]] | None = None,
        decisions: list[MemoryDecision] | None = None,
        sources: dict[str, list[MemorySourceRecord]] | None = None,
    ):
        self.results = results
        self.histories = histories or {}
        self.decisions = decisions or []
        self.sources = sources or {}
        self.calls: list[tuple[str, int, int, float]] = []
        self.remember_calls: list[PostMemory] = []

    def recall(
        self,
        query: str,
        *,
        current_top_k: int = 5,
        history_top_k: int = 0,
        threshold: float = 0.5,
    ) -> RecallResults:
        self.calls.append((query, current_top_k, history_top_k, threshold))
        return self.results

    def get_history(self, memory_id: str) -> list[MemoryRecord] | None:
        return self.histories.get(memory_id)

    def get_sources(self, memory_id: str) -> list[MemorySourceRecord] | None:
        return self.sources.get(memory_id)

    def remember(self, content: PostMemory) -> list[MemoryDecision]:
        self.remember_calls.append(content)
        return self.decisions


class FakeMemoryStore:
    def __init__(self, memories: list[MemoryRecord]):
        self.memories = memories

    def list_memories(self) -> list[MemoryRecord]:
        return self.memories


class InvalidExtractionKernel(FakeKernel):
    def remember(self, content: PostMemory) -> list[MemoryDecision]:
        raise ExtractionValidationError("evidence is invalid")


class StaticEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


class UnusedAI:
    def get_client(self) -> object:
        return object()

    def get_ai_response(
        self, client: object, inst: str, input_text: str
    ) -> str:
        raise AssertionError("Recall must not call the LLM")


class UnusedExtractor:
    def extract_with_source(self, source: SourceEvent):
        raise AssertionError("Recall must not call the extractor")


class StaticExtractor:
    def extract_with_source(self, source: SourceEvent) -> JsonExtractedResult:
        payload = {
            "facts": [
                {
                    "content": source.content,
                    "evidence": source.content,
                }
            ]
        }
        return JsonExtractedResult(
            json.dumps(payload),
            payload,
            (ExtractedFact(content=source.content, evidence=source.content),),
        )


def test_recall_uses_current_only_defaults() -> None:
    kernel = FakeKernel(
        RecallResults(
            current=[RetrievalResult(memory("current", "Current fact."), 0.91)],
            history=[],
        )
    )
    client = TestClient(create_app(kernel=kernel))

    response = client.post("/v1/recall", json={"query": "  user preference  "})

    assert response.status_code == 200
    assert kernel.calls == [("user preference", 5, 0, 0.5)]
    assert response.json() == {
        "current": [
            {
                "id": "current",
                "content": "Current fact.",
                "created_at": "2026-08-24 12:00:00",
                "state": "ACTIVE",
                "superseded_by_id": None,
                "superseded_at": None,
                "score": 0.91,
            }
        ],
        "history": [],
    }


def test_recall_exposes_independent_history_parameters() -> None:
    old_memory = memory(
        "old",
        "Old fact.",
        state="SUPERSEDED",
        superseded_by_id="current",
        superseded_at="2026-08-24 12:00:00",
    )
    kernel = FakeKernel(
        RecallResults(
            current=[RetrievalResult(memory("current", "Current fact."), 0.9)],
            history=[RetrievalResult(old_memory, 0.8)],
        )
    )
    client = TestClient(create_app(kernel=kernel))

    response = client.post(
        "/v1/recall",
        json={
            "query": "previous preference",
            "current_top_k": 2,
            "history_top_k": 3,
            "threshold": 0.7,
        },
    )

    assert response.status_code == 200
    assert kernel.calls == [("previous preference", 2, 3, 0.7)]
    assert [item["id"] for item in response.json()["current"]] == ["current"]
    assert [item["id"] for item in response.json()["history"]] == ["old"]


def test_recall_validates_history_parameters_before_calling_kernel() -> None:
    kernel = FakeKernel(RecallResults(current=[], history=[]))
    client = TestClient(create_app(kernel=kernel))

    response = client.post(
        "/v1/recall",
        json={"query": "preference", "history_top_k": -1},
    )

    assert response.status_code == 422
    assert kernel.calls == []


def test_recall_returns_503_when_kernel_is_not_configured() -> None:
    client = TestClient(create_app())

    response = client.post("/v1/recall", json={"query": "preference"})

    assert response.status_code == 503


def test_remember_adds_memory_through_kernel() -> None:
    decision = MemoryDecision(
        action="ADD",
        fact="User likes Rust.",
        memory_id="memory-id",
    )
    kernel = FakeKernel(
        RecallResults(current=[], history=[]),
        decisions=[decision],
    )
    client = TestClient(create_app(kernel=kernel))

    response = client.post(
        "/v1/memories",
        json={"content": "  User likes Rust.  "},
    )

    assert response.status_code == 200
    assert len(kernel.remember_calls) == 1
    assert kernel.remember_calls[0].content == "User likes Rust."
    assert kernel.remember_calls[0].source_type == "message"
    assert kernel.remember_calls[0].role == "user"
    assert response.json() == {
        "decisions": [
            {
                "action": "ADD",
                "fact": "User likes Rust.",
                "memory_id": "memory-id",
                "matched_memory_id": None,
            }
        ]
    }


def test_remember_rejects_empty_content_before_calling_kernel() -> None:
    kernel = FakeKernel(RecallResults(current=[], history=[]))
    client = TestClient(create_app(kernel=kernel))

    response = client.post("/v1/memories", json={"content": "   "})

    assert response.status_code == 422
    assert kernel.remember_calls == []


def test_remember_returns_bad_gateway_for_invalid_extractor_output() -> None:
    kernel = InvalidExtractionKernel(RecallResults(current=[], history=[]))
    client = TestClient(create_app(kernel=kernel))

    response = client.post("/v1/memories", json={"content": "I like Rust"})

    assert response.status_code == 502
    assert response.json()["detail"] == "evidence is invalid"


def test_remember_accepts_generic_source_metadata() -> None:
    kernel = FakeKernel(RecallResults(current=[], history=[]))
    client = TestClient(create_app(kernel=kernel))

    response = client.post(
        "/v1/memories",
        json={
            "content": "Build completed successfully.",
            "source_type": "tool",
            "role": "tool",
            "observed_at": "2026-08-24T12:00:00+00:00",
            "metadata": {"tool_name": "builder"},
        },
    )

    assert response.status_code == 200
    source = kernel.remember_calls[0]
    assert source.source_type == "tool"
    assert source.role == "tool"
    assert source.date == "2026-08-24T12:00:00+00:00"
    assert source.metadata == {"tool_name": "builder"}


def test_memory_history_returns_the_complete_supersession_chain() -> None:
    oldest = memory(
        "oldest",
        "User lived in Shanghai.",
        created_at="2024-01-01 00:00:00",
        state="SUPERSEDED",
        superseded_by_id="middle",
        superseded_at="2025-01-01 00:00:00",
    )
    middle = memory(
        "middle",
        "User lived in Beijing.",
        created_at="2025-01-01 00:00:00",
        state="SUPERSEDED",
        superseded_by_id="current",
        superseded_at="2026-01-01 00:00:00",
    )
    current = memory(
        "current",
        "User lives in Shenzhen.",
        created_at="2026-01-01 00:00:00",
    )
    store = FakeMemoryStore([current, oldest, middle])
    kernel = MemKernel(
        extractor=UnusedExtractor(),
        memory_backend=store,
        retriever=FakeRetriever(RecallResults(current=[], history=[])),
    )
    client = TestClient(create_app(kernel=kernel))

    response = client.get("/v1/memories/current/history")

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["memories"]] == [
        "oldest",
        "middle",
        "current",
    ]


def test_memory_history_returns_404_for_unknown_memory() -> None:
    kernel = FakeKernel(RecallResults(current=[], history=[]))
    client = TestClient(create_app(kernel=kernel))

    response = client.get("/v1/memories/missing/history")

    assert response.status_code == 404


def test_memory_sources_returns_provenance_only_on_explicit_lookup() -> None:
    kernel = FakeKernel(
        RecallResults(current=[], history=[]),
        sources={"memory-id": [memory_source()]},
    )
    client = TestClient(create_app(kernel=kernel))

    response = client.get("/v1/memories/memory-id/sources")

    assert response.status_code == 200
    assert response.json() == {
        "sources": [
            {
                "id": "source-id",
                "content": "I like Rust",
                "source_type": "message",
                "role": "user",
                "observed_at": "2026-08-24T12:00:00+00:00",
                "created_at": "2026-08-24 12:00:01",
                "metadata": {"channel": "api"},
                "evidence_quote": "I like Rust",
                "link_type": "DERIVED",
                "linked_at": "2026-08-24 12:00:02",
            }
        ]
    }


def test_memory_sources_returns_404_for_unknown_memory() -> None:
    kernel = FakeKernel(RecallResults(current=[], history=[]))
    client = TestClient(create_app(kernel=kernel))

    response = client.get("/v1/memories/missing/sources")

    assert response.status_code == 404


def test_api_integrates_with_semantic_retriever_and_backend_v2(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=UnusedAI(),
        embedding_provider=StaticEmbeddingProvider(),
    )
    old_id = backend.sqlite_backend.insert("User lived in Shanghai.")
    current_id = backend.sqlite_backend.supersede(
        old_id,
        "User lives in Beijing.",
    )
    client = TestClient(
        create_app(
            kernel=MemKernel(
                extractor=UnusedExtractor(),
                memory_backend=backend,
            )
        )
    )

    recall_response = client.post(
        "/v1/recall",
        json={"query": "where user lives", "history_top_k": 1},
    )
    history_response = client.get(f"/v1/memories/{current_id}/history")

    assert recall_response.status_code == 200
    assert [item["id"] for item in recall_response.json()["current"]] == [
        current_id
    ]
    assert [item["id"] for item in recall_response.json()["history"]] == [old_id]
    assert history_response.status_code == 200
    assert [item["id"] for item in history_response.json()["memories"]] == [
        old_id,
        current_id,
    ]


def test_remember_api_integrates_with_memkernel_and_backend_v2(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=UnusedAI(),
        embedding_provider=StaticEmbeddingProvider(),
    )
    kernel = MemKernel(extractor=StaticExtractor(), memory_backend=backend)
    client = TestClient(create_app(kernel=kernel))

    response = client.post(
        "/v1/memories",
        json={"content": "User likes Rust."},
    )

    assert response.status_code == 200
    assert response.json()["decisions"][0]["action"] == "ADD"
    assert [memory.content for memory in backend.list_memories()] == [
        "User likes Rust."
    ]
    memory_id = response.json()["decisions"][0]["memory_id"]
    source_response = client.get(f"/v1/memories/{memory_id}/sources")
    assert source_response.status_code == 200
    assert source_response.json()["sources"][0]["evidence_quote"] == (
        "User likes Rust."
    )

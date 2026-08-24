from memkernel.backend.backend import MemoryDecision, MemoryRecord
from memkernel.extractor.extractor import SimpleExtractedResult
from memkernel.kernel import MemKernel, PostMemory
from memkernel.provenance import SourceEvent
from memkernel.retriver import RecallResults, RetrievalResult


class RecordingExtractor:
    def __init__(self):
        self.inputs: list[str] = []

    def extract_source(self, source: SourceEvent) -> SimpleExtractedResult:
        self.inputs.append(source.content)
        return SimpleExtractedResult(source.content)


class RecordingBackend:
    def __init__(self):
        self.remembered: list[SimpleExtractedResult] = []
        self.sources: list[SourceEvent] = []

    def remember(
        self,
        extracted: SimpleExtractedResult,
        source_event: SourceEvent | None = None,
    ) -> list[MemoryDecision]:
        self.remembered.append(extracted)
        assert source_event is not None
        self.sources.append(source_event)
        return [
            MemoryDecision(
                action="ADD",
                fact=extracted.content,
                memory_id="memory-id",
            )
        ]

    def list_memories(self) -> list[MemoryRecord]:
        return []

    def get(self, memory_id: str) -> MemoryRecord | None:
        return None

    def get_sources(self, memory_id: str) -> list:
        return []


class RecordingRetriever:
    def __init__(self):
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
        memory = MemoryRecord("memory-id", "User likes Rust.", "2026-08-24")
        return RecallResults(
            current=[RetrievalResult(memory=memory, score=0.9)],
            history=[],
        )


def test_memkernel_composes_extraction_storage_and_retrieval() -> None:
    extractor = RecordingExtractor()
    backend = RecordingBackend()
    retriever = RecordingRetriever()
    kernel = MemKernel(extractor, backend, retriever)

    decisions = kernel.remember(PostMemory("2026-08-24", "I like Rust coding."))
    recalled = kernel.recall(
        "Rust preference",
        current_top_k=3,
        history_top_k=1,
        threshold=0.7,
    )

    assert decisions[0].memory_id == "memory-id"
    assert extractor.inputs == ["I like Rust coding."]
    assert backend.remembered == [SimpleExtractedResult("I like Rust coding.")]
    assert backend.sources[0].observed_at == "2026-08-24T00:00:00+00:00"
    assert backend.sources[0].source_type == "message"
    assert backend.sources[0].role == "user"
    assert retriever.calls == [("Rust preference", 3, 1, 0.7)]
    assert recalled.current[0].memory.content == "User likes Rust."


def test_memkernel_sanitizes_source_before_extraction_and_storage() -> None:
    extractor = RecordingExtractor()
    backend = RecordingBackend()
    kernel = MemKernel(extractor, backend, RecordingRetriever())

    kernel.remember(
        PostMemory(
            "2026-08-24",
            "I like Rust; api_key=super-secret",
            metadata={"password": "hidden", "safe": "visible"},
        )
    )

    assert extractor.inputs == ["I like Rust; api_key=[REDACTED:SECRET]"]
    assert backend.sources[0].content == extractor.inputs[0]
    assert backend.sources[0].metadata == {
        "password": "[REDACTED:SECRET]",
        "safe": "visible",
    }

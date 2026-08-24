from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from memkernel.backend.backend import MemoryRecord, ScoredMemory
from memkernel.extractor import ExtractedResult, Extractor
from memkernel.retriever_v2 import SemanticRetriever
from memkernel.retriver import RecallResults, Retriever


class KernelBackend(Protocol):
    def remember(self, extracted: ExtractedResult) -> object: ...

    def search_current(self, content: str, top_k: int = 5) -> list[ScoredMemory]: ...

    def search_history(self, content: str, top_k: int = 5) -> list[ScoredMemory]: ...

    def list_memories(self) -> list[MemoryRecord]: ...


@dataclass(slots=True, frozen=True)
class PostMemory:
    date: str
    content: str


class MemKernel:
    """Facade that composes extraction, reconciliation, storage, and recall."""

    def __init__(
        self,
        extractor: Extractor,
        memory_backend: KernelBackend,
        retriever: Retriever | None = None,
    ):
        self.extractor = extractor
        self.memory_backend = memory_backend
        self.retriever = retriever or SemanticRetriever(memory_backend)

    def remember(self, raw: PostMemory | str) -> object:
        content = raw.content if isinstance(raw, PostMemory) else raw
        extracted = self.extractor.extract(content)
        return self.memory_backend.remember(extracted)

    def recall(
        self,
        query: str,
        *,
        current_top_k: int = 5,
        history_top_k: int = 0,
        threshold: float = 0.5,
    ) -> RecallResults:
        return self.retriever.recall(
            query,
            current_top_k=current_top_k,
            history_top_k=history_top_k,
            threshold=threshold,
        )

    def get_history(self, memory_id: str) -> list[MemoryRecord] | None:
        """Return a memory's supersession chain from oldest to newest."""
        memories = self.memory_backend.list_memories()
        by_id = {memory.id: memory for memory in memories}
        if memory_id not in by_id:
            return None

        predecessor_by_id = {
            memory.superseded_by_id: memory.id
            for memory in memories
            if memory.superseded_by_id in by_id
        }

        oldest_id = memory_id
        while oldest_id in predecessor_by_id:
            oldest_id = predecessor_by_id[oldest_id]

        history: list[MemoryRecord] = []
        current_id: str | None = oldest_id
        while current_id in by_id:
            memory = by_id[current_id]
            history.append(memory)
            current_id = memory.superseded_by_id

        return history

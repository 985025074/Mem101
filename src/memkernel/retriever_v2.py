from dataclasses import dataclass
from typing import Protocol

from memkernel.backend.backend import MemoryRecord, ScoredMemory


class SemanticSearchBackend(Protocol):
    def search_similar(self, content: str, top_k: int = 5) -> list[ScoredMemory]: ...


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    memory: MemoryRecord
    score: float


class SemanticRetriever:
    """A very simple retrive"""

    def __init__(self, memory_backend: SemanticSearchBackend):
        self.memory_backend = memory_backend

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> list[RetrievalResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a number")
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between -1 and 1")

        matches = self.memory_backend.search_similar(
            query.strip(),
            top_k=top_k,
        )
        matches = sorted(
            (match for match in matches if match.similarity >= float(threshold)),
            key=lambda match: match.similarity,
            reverse=True,
        )[:top_k]

        return [
            RetrievalResult(memory=match.memory, score=match.similarity)
            for match in matches
        ]

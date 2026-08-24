from collections.abc import Callable
from typing import Protocol

from memkernel.backend.backend import ScoredMemory
from memkernel.retriver import RecallResults, RetrievalResult


class SearchBackend(Protocol):
    def search_current(self, content: str, top_k: int = 5) -> list[ScoredMemory]: ...

    def search_history(self, content: str, top_k: int = 5) -> list[ScoredMemory]: ...


class SemanticRetriever:
    """A very simple retrive"""

    def __init__(self, memory_backend: SearchBackend):
        self.memory_backend = memory_backend

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> list[RetrievalResult]:
        """Retrieve current memories using the legacy convenience API."""
        return self.retrieve_current(query, top_k=top_k, threshold=threshold)

    def recall(
        self,
        query: str,
        *,
        current_top_k: int = 5,
        history_top_k: int = 0,
        threshold: float = 0.5,
    ) -> RecallResults:
        """Retrieve current and historical memories with independent limits."""
        if (
            isinstance(history_top_k, bool)
            or not isinstance(history_top_k, int)
            or history_top_k < 0
        ):
            raise ValueError("history_top_k must be a non-negative integer")

        current = self.retrieve_current(
            query,
            top_k=current_top_k,
            threshold=threshold,
        )
        history = (
            self.retrieve_history(
                query,
                top_k=history_top_k,
                threshold=threshold,
            )
            if history_top_k > 0
            else []
        )
        return RecallResults(current=current, history=history)

    def retrieve_current(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> list[RetrievalResult]:
        return self._retrieve(
            self.memory_backend.search_current,
            query,
            top_k=top_k,
            threshold=threshold,
        )

    def retrieve_history(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> list[RetrievalResult]:
        return self._retrieve(
            self.memory_backend.search_history,
            query,
            top_k=top_k,
            threshold=threshold,
        )

    @staticmethod
    def _retrieve(
        search: Callable[..., list[ScoredMemory]],
        query: str,
        *,
        top_k: int,
        threshold: float,
    ) -> list[RetrievalResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a number")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        matches = search(
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

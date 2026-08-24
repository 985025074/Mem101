# duplicated version
from dataclasses import dataclass
from typing import Protocol

from memkernel.backend.backend import MemoryRecord


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    memory: MemoryRecord
    score: float


@dataclass(frozen=True, slots=True)
class RecallResults:
    current: list[RetrievalResult]
    history: list[RetrievalResult]


class Retriever(Protocol):
    def recall(
        self,
        query: str,
        *,
        current_top_k: int = 5,
        history_top_k: int = 0,
        threshold: float = 0.5,
    ) -> RecallResults: ...


# deprecated
# class SimpleRetriver:
#     def retrive(self, memory_backend: Backend, query: str) -> Optional[RetriveResult]:
#         for m in memory_backend.list_memories():
#             if m.content.find(query) != -1:
#                 return RetriveResult(m.content)
#         return None

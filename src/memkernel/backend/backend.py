from dataclasses import dataclass
from typing import Any, Dict, Literal, Protocol

from memkernel.extractor import ExtractedResult


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    content: str
    created_at: str


# semantic  search result
@dataclass(frozen=True, slots=True)
class ScoredMemory:
    memory: MemoryRecord
    similarity: float


MemoryAction = Literal["ADD", "NOOP", "SUPERSEDE"]


@dataclass(frozen=True, slots=True)
class MemoryDecision:
    action: MemoryAction
    fact: str
    # Populated by _apply_decision after the action has been executed.
    memory_id: str | None = None
    # Existing memory selected by reconciliation, if any.
    matched_memory_id: str | None = None


class Backend(Protocol):
    # return memory id
    def remember(self, extracted: ExtractedResult) -> str: ...

    def get(self, memory_id: str) -> MemoryRecord | None: ...

    def remove(self, memory_id: str) -> bool: ...

    def query_by(self, query_dict: Dict[str, Any]) -> MemoryRecord | None: ...

    def search_similar(self, content: str, top_k: int = 5) -> list[ScoredMemory]: ...

    def list_memories(self) -> list[MemoryRecord]: ...

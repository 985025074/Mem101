from dataclasses import dataclass
from typing import Any, Dict, Literal, Protocol

from memkernel.extractor import ExtractedResult


MemoryState = Literal["ACTIVE", "SUPERSEDED"]


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    content: str
    created_at: str
    state: MemoryState = "ACTIVE"
    # old memory will have this.
    superseded_by_id: str | None = None
    # what time did  it die
    superseded_at: str | None = None


# semantic  search result
@dataclass(frozen=True, slots=True)
class ScoredMemory:
    memory: MemoryRecord
    similarity: float


MemoryAction = Literal["ADD", "NOOP", "SUPERSEDE"]
MemoryRelation = Literal["EQUIVALENT", "SUPERSEDES", "DISTINCT"]


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

    def search_current(self, content: str, top_k: int = 5) -> list[ScoredMemory]: ...

    def search_history(self, content: str, top_k: int = 5) -> list[ScoredMemory]: ...

    def list_memories(self) -> list[MemoryRecord]: ...

from dataclasses import dataclass
from typing import Any, Dict, Literal, Protocol

from memkernel.extractor import ExtractedResult
from memkernel.provenance import MemorySourceRecord, SourceEvent


# this is used to describe whether a memory is valid for now.
MemoryState = Literal["ACTIVE", "SUPERSEDED"]
# Tier is used to describe whether a memory is accessed often.
MemoryTier = Literal["HOT", "WARM", "COLD"]


@dataclass(frozen=True, slots=True)
class MemoryPolicy:
    """Lifecycle policy applied when a memory is created. How hard we want to remember somthing."""

    tier: MemoryTier = "HOT"
    importance: float = 0.5
    # When this is outdated automatically
    expires_at: str | None = None
    # Something important we can't forget
    pinned: bool = False

    # Used to check whether value is valid
    def __post_init__(self) -> None:
        if self.tier not in {"HOT", "WARM", "COLD"}:
            raise ValueError("tier must be HOT, WARM, or COLD")
        if isinstance(self.importance, bool) or not isinstance(
            self.importance,
            (int, float),
        ):
            raise ValueError("importance must be a number")
        if not 0.0 <= float(self.importance) <= 1.0:
            raise ValueError("importance must be between 0 and 1")
        if self.expires_at is not None and not isinstance(self.expires_at, str):
            raise ValueError("expires_at must be an ISO-8601 string or null")
        if not isinstance(self.pinned, bool):
            raise ValueError("pinned must be a boolean")


@dataclass(frozen=True, slots=True)
class MemoryUsage:
    """This is a memory'timeline state.Used to determin a memory's level in Our system(Hot,Cold warm)"""

    tier: MemoryTier = "HOT"
    importance: float = 0.5
    last_accessed_at: str | None = None
    access_count: int = 0
    last_confirmed_at: str | None = None
    confirmation_count: int = 0
    pinned: bool = False


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
    # Expiration is a soft delete: expired memories remain auditable.
    expires_at: str | None = None


# semantic  search result
@dataclass(frozen=True, slots=True)
class ScoredMemory:
    memory: MemoryRecord
    similarity: float
    usage: MemoryUsage | None = None
    # Retrieval may rerank without changing the raw similarity contract.
    rank_score: float | None = None


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
    def remember(
        self,
        extracted: ExtractedResult,
        source_event: SourceEvent,
        *,
        policy: MemoryPolicy | None = None,
    ) -> object: ...

    def get(self, memory_id: str) -> MemoryRecord | None: ...

    def remove(self, memory_id: str) -> bool: ...

    def query_by(self, query_dict: Dict[str, Any]) -> MemoryRecord | None: ...

    def search_similar(self, content: str, top_k: int = 5) -> list[ScoredMemory]: ...

    def search_current(self, content: str, top_k: int = 5) -> list[ScoredMemory]: ...

    def search_history(self, content: str, top_k: int = 5) -> list[ScoredMemory]: ...

    def list_memories(self) -> list[MemoryRecord]: ...

    def get_sources(self, memory_id: str) -> list[MemorySourceRecord]: ...

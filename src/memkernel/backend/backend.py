from dataclasses import dataclass
from typing import Any, Dict, Protocol

from memkernel.extractor import ExtractedResult


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    content: str
    created_at: str


class Backend(Protocol):
    def remember(self, extracted: ExtractedResult) -> str: ...

    def get(self, memory_id: str) -> MemoryRecord | None: ...

    def query_by(self, query_dict: Dict[str, Any]) -> MemoryRecord | None: ...

    def list_memories(self) -> list[MemoryRecord]: ...

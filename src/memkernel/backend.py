from dataclasses import dataclass
from typing import Protocol

from memkernel.extractor import ExtractedResult


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    content: str
    created_at: str


class Backend(Protocol):
    def remember(self, extracted: ExtractedResult) -> str: ...

    def get(self, memory_id: str) -> MemoryRecord | None: ...

    def list_memories(self) -> list[MemoryRecord]: ...

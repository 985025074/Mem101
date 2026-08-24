from dataclasses import dataclass
from typing import Optional, Protocol

from memkernel.backend import Backend


@dataclass(frozen=True, slots=True)
class RetriveResult:
    content: str


class Retriver(Protocol):
    def retrive(
        self, memory_backend: Backend, query: str
    ) -> Optional[RetriveResult]: ...


class SimpleRetriver:
    def retrive(self, memory_backend: Backend, query: str) -> Optional[RetriveResult]:
        for m in memory_backend.list_memories():
            if m.content.find(query) != -1:
                return RetriveResult(m.content)
        return None

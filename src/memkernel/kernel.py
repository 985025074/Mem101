# MemKernel
# Consolidation of multi part


from dataclasses import dataclass, field
from typing import Optional

from memkernel.backend import Backend
from memkernel.extractor import Extractor, LLMExtractor
from memkernel.retriver import Retriver, SimpleRetriver
from memkernel.sqlite_adapter import SQLiteBackend


@dataclass(slots=True)
class RecallResult:
    content: str


@dataclass(slots=True)
class PostMemory:
    # TODO:  proper type
    date: str
    content: str


@dataclass(slots=True)
class MemKernel:
    extractor: Extractor = LLMExtractor()
    memory_backend: Backend = field(default=SQLiteBackend())
    retriver: Retriver = SimpleRetriver()

    def recall(self, query: str) -> Optional[RecallResult]:
        content = self.retriver.retrive(self.memory_backend, query)
        if content is None:
            return None
        content = content.content
        return RecallResult(content=content)

    def remember(self, raw: PostMemory):
        extracted = self.extractor.extract(raw.content)
        self.memory_backend.remember(extracted)

from pathlib import Path

from memkernel.backend.backend import MemoryRecord
from memkernel.backend.sqlite_adapter import SQLiteBackend
from memkernel.extractor.extractor import ExtractedResult


# Sqlite bakcend. But with check of the memory
class BackendV2:
    def __init__(self, memory_path: Path):
        self.sqlite_backend = SQLiteBackend(memory_path)

    def remember(self, extracted: ExtractedResult) -> str: ...

    def get(self, memory_id: str) -> MemoryRecord | None: ...

    def list_memories(self) -> list[MemoryRecord]: ...
